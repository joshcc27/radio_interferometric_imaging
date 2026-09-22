"""The first sanity check: look at the uv coverage before believing anything else.

A Y array tracking a source for hours gives smooth elliptical arcs, one per
baseline, all concentric and all with the same axial ratio ``sin(dec)``. A global
array gives short disconnected arcs, because most baselines only have the source
above the horizon at both ends for part of the track.

If this figure shows straight lines, or a blob at the origin, the projection in
:func:`src.array.baselines_to_uvw` is wrong and nothing downstream can be
trusted.

Run with ``python -m src.experiments.uv_coverage``.
"""

from __future__ import annotations

import numpy as np

import matplotlib.pyplot as plt

from .. import array as arraymod
from ..forward import ARCSEC, UAS
from .. import plotting as pl
from . import config


def main():
    vla = config.vla_setup()
    eht = config.eht_setup()
    print(vla.describe())
    print(eht.describe())

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9))

    for row, setup, unit, scale, lab in (
        (0, vla, "kilo", ARCSEC, "arcsec"),
        (1, eht, "mega", UAS, "$\\mu$as"),
    ):
        pl.show_array(axes[row, 0], setup.array,
                      title=setup.array.name + " layout")
        pl.show_uv(axes[row, 1], setup.coverage, unit=unit,
                   title=(setup.name + ": " + str(2 * len(setup.coverage))
                          + " uv points (Hermitian pairs included)"),
                   s=0.7 if row == 0 else 3.0)

        # radial density: what sets which angular scales are constrained
        r = setup.coverage.radius
        ax = axes[row, 2]
        ax.hist(r / (1e3 if unit == "kilo" else 1e6), bins=40,
                color="#1f5fa8", alpha=0.85)
        ax.set_xlabel("$|uv|$ (" + ("k$\\lambda$" if unit == "kilo" else "M$\\lambda$") + ")")
        ax.set_ylabel("visibilities per bin")
        ax.set_title("radial uv density", fontsize=10)
        ax.grid(alpha=0.25, lw=0.4)
        res = setup.coverage.resolution_rad() / scale
        las = setup.coverage.largest_angular_scale_rad() / scale
        ax.text(0.97, 0.95,
                "resolution " + format(res, ".3g") + " " + lab
                + "\nlargest scale " + format(las, ".3g") + " " + lab,
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(fc="white", ec="0.7", alpha=0.9))

    fig.suptitle("Aperture synthesis: array geometry, uv coverage, radial density",
                 fontsize=12)
    fig.tight_layout()
    pl.save(fig, "01_uv_coverage")

    # a second figure making the "coverage thins as antennas are dropped" point,
    # which is the axis the whole experiment in Part 4 sweeps along
    counts = [27, 15, 9, 6]
    pin = arraymod.outermost(vla.array, 3)
    lim = 1.05 * vla.coverage.max_baseline / 1e3
    fig, axes = plt.subplots(1, len(counts), figsize=(4.0 * len(counts), 4.2))
    for ax, n in zip(np.atleast_1d(axes), counts):
        thinned = arraymod.thin(vla.array, n, seed=1, pin=pin)
        cov = vla.uv_track(thinned)
        frac = len(cov) / len(vla.coverage)
        pl.show_uv(ax, cov, unit="kilo", s=1.2,
                   title=(str(n) + " antennas, " + str(thinned.n_baselines)
                          + " baselines\n" + format(100 * frac, ".1f")
                          + "% of full coverage"))
        ax.set_xlim(lim, -lim)   # common axes, so thinning shows up as an
        ax.set_ylim(-lim, lim)   # emptier plane rather than a rescaled one
    fig.suptitle("Thinning with the three arm-tip antennas pinned: $u_{max}$, "
                 "and hence the nominal resolution, is held fixed so that only "
                 "the filling factor changes", fontsize=11)
    fig.tight_layout()
    pl.save(fig, "02_coverage_thinning")


if __name__ == "__main__":
    main()
