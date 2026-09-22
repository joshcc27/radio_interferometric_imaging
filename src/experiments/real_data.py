"""Image a real dataset with this project's own code.

The goal here is modest and should stay modest: produce a recognisable image from
archival visibilities using nothing but the code in ``src/``, and compare it
against the published one. Not beat it. Real data arrives with residual station
gain errors, and the standard remedy -- self-calibration -- is a whole second
project (see :mod:`src.realdata`). A reconstruction from real data should be
expected to be worse than the published reconstruction, and if it is not, be
suspicious.

**No archival dataset has been imaged yet.** This script is written and its
loading path is tested (``tests/test_realdata.py``), but running it needs a
download and an optional dependency. Usage:

    pip install astropy
    python -m src.experiments.real_data --uvfits path/to/data.uvfits

    # or, with no extra dependencies, from the interchange format
    python -m src.experiments.real_data --npz path/to/vis.npz

Data sources: the EHT Collaboration released calibrated M87 visibilities
publicly, and the NRAO archive serves VLA measurement sets. Use
``--rescale-weights`` unless you trust the archive's weight column, and read the
note in :func:`src.realdata.rescale_weights_from_scatter` about why.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import matplotlib.pyplot as plt

from .. import clean as cleanmod
from .. import forward as fw
from .. import imaging as im
from .. import metrics as mt
from .. import plotting as pl
from .. import realdata as rd
from .. import regularised as rg


def load(args):
    if args.uvfits:
        return rd.load_uvfits(args.uvfits, average_channels=args.average_channels)
    if args.ms:
        return rd.load_measurement_set(args.ms,
                                       average_channels=args.average_channels)
    if args.npz:
        return rd.load_npz(args.npz)
    raise SystemExit("give one of --uvfits, --ms or --npz (see the module "
                     "docstring); no dataset ships with this repository")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uvfits")
    ap.add_argument("--ms")
    ap.add_argument("--npz")
    ap.add_argument("--npix", type=int, default=None)
    ap.add_argument("--pixels-per-beam", type=float, default=4.0)
    ap.add_argument("--field-factor", type=float, default=8.0,
                    help="field of view in nominal beams; keep it tight, the "
                         "field is itself a strong prior")
    ap.add_argument("--average-channels", action="store_true")
    ap.add_argument("--rescale-weights", action="store_true",
                    help="re-derive sigma from the scatter of the data")
    ap.add_argument("--scheme", default="briggs")
    ap.add_argument("--robust", type=float, default=0.0)
    ap.add_argument("--label", default="real")
    args = ap.parse_args()

    data = load(args)
    print(rd.describe(data))
    if args.rescale_weights:
        before = float(np.median(data.sigma))
        data = rd.rescale_weights_from_scatter(data)
        print("sigma rescaled from the data: median "
              + format(before, ".4g") + " -> "
              + format(float(np.median(data.sigma)), ".4g"))

    npix, cell = rd.suggest_grid(data, pixels_per_beam=args.pixels_per_beam,
                                 field_factor=args.field_factor)
    npix = args.npix or npix
    arc = 180.0 * 3600.0 / np.pi
    print("imaging on " + str(npix) + "^2 pixels of "
          + format(cell * arc * 1e6, ".3g") + " uas, field "
          + format(npix * cell * arc * 1e6, ".3g") + " uas")

    grid_op = fw.MeasurementOperator(data, npix, cell, scheme=args.scheme,
                                     robust=args.robust)
    acc_op = fw.DegridOperator(data, npix, cell, scheme=args.scheme,
                               robust=args.robust)
    dirty, psf = im.dirty_image(grid_op), im.dirty_beam(grid_op)
    beam = im.fit_clean_beam(psf, cell)
    print("occupied uv cells " + str(grid_op.n_cells) + " of " + str(npix ** 2)
          + ", restoring beam " + repr(beam))

    results = {}

    thr = 3.0 * cleanmod.theoretical_image_noise(grid_op)
    cl = cleanmod.clean(dirty, psf, gain=0.1, n_iter=20000, threshold=thr,
                        frac_threshold=1e-4, cell=cell, fit_beam=False)
    cl.beam = beam
    results["clean"] = cl.restored(cell, add_residual=True)
    print("CLEAN: " + str(cl.n_iter) + " components, " + cl.stop_reason
          + ", model flux " + format(cl.model_flux, ".4g")
          + ", chi2r " + format(acc_op.reduced_chi2(cl.components), ".3f"))

    kern = beam.image(npix, cell)
    for kind in ("tv_flux", "mem"):
        out = rg.solve_at_chi2(acc_op, kind=kind, target_chi2=1.0, n_iter=300,
                               n_bisect=7)
        results[kind] = im.convolve_periodic(out.image, kern)
        print(kind + ": lam " + format(out.lam, ".3g") + ", chi2r "
              + format(out.final_chi2, ".3f") + ", flux "
              + format(float(out.image.sum()), ".4g"))

    # Residual calibration errors show up as a chi-squared that no image can
    # reach. If every method plateaus well above 1 here while the same methods
    # reach 1 on simulated data, suspect the calibration rather than the imaging
    # -- and check the closure phases, which station gains cannot corrupt.
    summary = {"n_vis": len(data), "n_cells": grid_op.n_cells, "npix": npix,
               "cell_rad": cell, "beam": repr(beam),
               "clean_iterations": cl.n_iter,
               "clean_model_flux": cl.model_flux,
               "source": (data.meta or {}).get("source", "")}
    with open(pl.result_path("real_data_" + args.label + ".json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8))
    pl.show_image(axes[0], dirty, cell, unit="uas", title="dirty image",
                  label="Jy/beam")
    for ax, key, title in zip(axes[1:], ("clean", "tv_flux", "mem"),
                              ("CLEAN", "TV + flux penalty", "maximum entropy")):
        pl.show_image(ax, results[key], cell, unit="uas", title=title,
                      label="Jy/beam")
        pl.add_beam_marker(ax, beam, cell, npix, unit="uas")
    fig.suptitle("Real data, this project's code: " + str(len(data))
                 + " visibilities. No self-calibration -- expect worse than the "
                   "published image.", fontsize=12)
    fig.tight_layout()
    pl.save(fig, "12_real_data_" + args.label)


if __name__ == "__main__":
    main()
