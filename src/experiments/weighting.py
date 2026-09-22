"""The resolution-versus-sensitivity trade-off that imaging weights buy you.

A cheap, real result. The same visibilities, imaged three ways:

- **Natural** weighting (inverse variance) gives the best point-source
  sensitivity and the worst beam. The short baselines are crowded, so they
  dominate the sum and the beam grows a broad skirt.
- **Uniform** weighting divides by the local uv density, so every occupied cell
  gets an equal say. The beam narrows and its sidelobes fall, and the noise goes
  up -- sometimes by a lot, because cells with one lonely visibility get the same
  weight as cells with fifty.
- **Briggs robust** is the one-parameter family between them, and in practice the
  only one anybody uses.

The figure plots beam area and image noise against robustness, which is the
trade-off curve, and shows the beams themselves so the shape change is visible
rather than merely tabulated.

Run with ``python -m src.experiments.weighting``.
"""

from __future__ import annotations

import json

import numpy as np

import matplotlib.pyplot as plt

from .. import clean as cleanmod
from .. import forward as fw
from .. import imaging as im
from .. import metrics as mt
from .. import plotting as pl
from ..forward import ARCSEC
from . import config


def main():
    setup = config.vla_setup(npix=128)
    print(setup.describe())
    truth = config.truth_for(setup, "point_gaussian")
    data = fw.observe(truth, setup.cell, setup.coverage,
                      snr=config.DEFAULT_SNR, rng=np.random.default_rng(11))

    robusts = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    rows = []
    schemes = [("uniform", None), ("natural", None)]
    schemes += [("briggs", r) for r in robusts]

    for scheme, robust in schemes:
        op = fw.MeasurementOperator(data, setup.npix, setup.cell, scheme=scheme,
                                    robust=0.0 if robust is None else robust)
        psf = im.dirty_beam(op)
        beam = im.fit_clean_beam(psf, setup.cell)
        # sidelobes outside 1.5 main-lobe widths: what deconvolution has to fight
        ax = (np.arange(setup.npix) - setup.npix // 2) * setup.cell
        mm, ll = np.meshgrid(ax, ax, indexing="ij")
        far = np.hypot(ll, mm) > 1.5 * beam.bmaj
        rows.append(dict(
            scheme=scheme, robust=robust,
            bmaj_arcsec=beam.bmaj / ARCSEC, bmin_arcsec=beam.bmin / ARCSEC,
            beam_area_px=beam.area_pixels(setup.cell),
            noise=cleanmod.theoretical_image_noise(op),
            max_sidelobe=float(np.abs(psf[far]).max()),
            sidelobe_rms=float(psf[far].std()),
            n_cells=op.n_cells,
        ))
        print("  %-8s robust=%-5s bmaj=%5.1f\" bmin=%5.1f\" area=%6.1f px "
              "noise=%.3e sidelobe=%.3f"
              % (scheme, str(robust), rows[-1]["bmaj_arcsec"],
                 rows[-1]["bmin_arcsec"], rows[-1]["beam_area_px"],
                 rows[-1]["noise"], rows[-1]["max_sidelobe"]))

    nat = [r for r in rows if r["scheme"] == "natural"][0]
    uni = [r for r in rows if r["scheme"] == "uniform"][0]
    brg = [r for r in rows if r["scheme"] == "briggs"]

    with open(pl.result_path("weighting.json"), "w") as fh:
        json.dump(rows, fh, indent=1)

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(14.5, 8.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.85], hspace=0.32,
                          wspace=0.38)

    zoom = 40
    lo, hi = setup.npix // 2 - zoom, setup.npix // 2 + zoom
    axg = (np.arange(setup.npix) - setup.npix // 2) * setup.cell
    gmm, gll = np.meshgrid(axg, axg, indexing="ij")
    for k, (scheme, robust, label) in enumerate([
            ("natural", 0.0, "natural"),
            ("briggs", 0.5, "Briggs robust = 0.5"),
            ("uniform", 0.0, "uniform")]):
        op = fw.MeasurementOperator(data, setup.npix, setup.cell, scheme=scheme,
                                    robust=robust)
        psf = im.dirty_beam(op)
        beam = im.fit_clean_beam(psf, setup.cell)
        # the sidelobe level that matters is outside the main lobe; quoting the
        # peak would just report the normalisation
        far = np.hypot(gll, gmm) > 1.5 * beam.bmaj
        ax = fig.add_subplot(gs[0, k])
        pl.show_image(ax, psf[lo:hi, lo:hi], setup.cell, unit="arcsec",
                      title=(label + "\nbmaj " + format(beam.bmaj / ARCSEC, ".1f")
                             + "\", peak sidelobe "
                             + format(100 * np.abs(psf[far]).max(), ".1f") + "%"),
                      cmap="RdBu_r", vmin=-0.3, vmax=0.3,
                      label="dirty beam" if k == 2 else None)
        if k > 0:
            ax.set_ylabel("")

    # radial cuts through the three beams, on one axis
    ax = fig.add_subplot(gs[0, 3])
    for scheme, robust, label, c in [("natural", 0.0, "natural", "#c1121f"),
                                     ("briggs", 0.5, "robust 0.5", "#f4a261"),
                                     ("uniform", 0.0, "uniform", "#1f5fa8")]:
        op = fw.MeasurementOperator(data, setup.npix, setup.cell, scheme=scheme,
                                    robust=robust)
        psf = im.dirty_beam(op)
        cut = psf[setup.npix // 2, :]
        x = (np.arange(setup.npix) - setup.npix // 2) * setup.cell / ARCSEC
        ax.plot(x, cut, lw=1.2, color=c, label=label)
    ax.set_xlim(-250, 250)
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.set_xlabel("$l$ (arcsec)")
    ax.set_ylabel("dirty beam")
    ax.set_title("cut through the beam", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.4)

    # the trade-off curve
    ax = fig.add_subplot(gs[1, :2])
    r = np.array([b["robust"] for b in brg])
    area = np.array([b["beam_area_px"] for b in brg])
    noise = np.array([b["noise"] for b in brg])
    ax.plot(r, area / nat["beam_area_px"], "o-", color="#1f5fa8",
            label="beam area / natural")
    ax.plot(r, noise / nat["noise"], "s-", color="#c1121f",
            label="image noise / natural")
    ax.axhline(1.0, color="0.6", lw=0.6)
    ax.set_xlabel("Briggs robust parameter")
    ax.set_ylabel("relative to natural weighting")
    ax.set_title("the trade-off: robust $-2$ is uniform-like, $+2$ natural-like",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.4)

    ax = fig.add_subplot(gs[1, 2:])
    ax.plot(area / nat["beam_area_px"], noise / nat["noise"], "o-",
            color="#2a9d8f")
    for b in brg:
        ax.annotate(format(b["robust"], "+.1f"),
                    (b["beam_area_px"] / nat["beam_area_px"],
                     b["noise"] / nat["noise"]),
                    fontsize=7, textcoords="offset points", xytext=(4, 4))
    ax.scatter([uni["beam_area_px"] / nat["beam_area_px"]],
               [uni["noise"] / nat["noise"]], marker="*", s=130, color="#1f5fa8",
               zorder=5, label="uniform")
    ax.scatter([1.0], [1.0], marker="*", s=130, color="#c1121f", zorder=5,
               label="natural")
    ax.set_xlabel("beam area (relative to natural)")
    ax.set_ylabel("image noise (relative to natural)")
    ax.set_title("you buy resolution with noise, and the exchange rate is steep "
                 "at the uniform end", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.4)

    fig.suptitle("Imaging weights on one VLA dataset: "
                 + str(len(setup.coverage)) + " visibilities, "
                 + str(nat["n_cells"]) + " occupied uv cells", fontsize=12)
    pl.save(fig, "03_weighting")

    print("\nnatural -> uniform: beam area x%.2f, noise x%.2f"
          % (uni["beam_area_px"] / nat["beam_area_px"],
             uni["noise"] / nat["noise"]))


if __name__ == "__main__":
    main()
