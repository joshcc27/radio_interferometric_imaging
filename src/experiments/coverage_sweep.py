"""The experiment that makes this a result rather than a demonstration.

Fix a ground truth. Thin the uv coverage from dense to sparse. At every level run
every reconstruction, and measure how each one degrades. The claim being tested
is quantitative: the methods agree where coverage is good and diverge exactly
where the data run out, and the divergence tracks the prior rather than the
noise.

Three sweeps, because one of them on its own would be ambiguous:

``thin``
    Drop antennas, with the three arm tips pinned so ``u_max`` and hence the
    nominal resolution stay fixed. Per-visibility noise is held constant, so
    total sensitivity falls with the visibility count -- which is what really
    happens when you lose antennas.

``thin_fixed_sensitivity``
    The same thinning, but with the per-visibility noise scaled as
    ``sigma ~ sqrt(n_vis)`` so that ``sigma / sqrt(n_vis)``, and therefore the
    thermal noise in the image, is constant along the sweep. This is the sweep
    that isolates *coverage* from *sensitivity*. Any degradation here is missing
    Fourier information and nothing else, which is the quantity the project
    claims to be measuring.

``track``
    Keep all 27 antennas and shorten the observation instead. Coverage thins
    along the uv ellipses rather than by losing whole baselines, which is a
    different kind of sparsity and a check that the result is not an artefact of
    how the thinning was done.

Comparing the first two is the point. If the curves look the same, the
degradation is loss of Fourier coverage; if only the first one degrades, it was
really a sensitivity effect wearing a coverage costume.

Every level shares the imaging grid and the restoring beam of the *full* array,
so all the numbers are on one yardstick. Lambda is chosen at every level by
:func:`~src.regularised.solve_at_chi2`, never by hand and never using the truth.

Run with ``python -m src.experiments.coverage_sweep``. It takes a while; results
are cached to ``results/coverage_sweep.json`` and the figures can be rebuilt
from that with ``--plot-only``.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import matplotlib.pyplot as plt

from .. import array as arraymod
from .. import forward as fw
from .. import metrics as mt
from .. import plotting as pl
from . import config, pipeline as pp

METHODS = ("dirty", "clean", "positivity", "tv", "tv_flux", "mem")
PLOT_METHODS = ("clean", "positivity", "tv", "tv_flux", "mem")
COLORS = {"dirty": "0.55", "clean": "#c1121f", "positivity": "#6a4c93",
          "tv": "#1f5fa8", "tv_flux": "#7fb3e8", "mem": "#2a9d8f"}
MARKERS = {"dirty": "x", "clean": "o", "positivity": "^", "tv": "s",
           "tv_flux": "v", "mem": "D"}

ANTENNA_LEVELS = (27, 21, 16, 12, 9, 7, 5)
TRACK_LEVELS = (8.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.25)
SEEDS = (0, 1)
"""Two noise realisations per level. Enough to show a spread band on the curves
without the sweep taking hours; the spread is plotted as a min-max band rather
than a standard error, so two seeds is not pretending to be a distribution."""


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------
def sweep_levels(setup, kind):
    """The coverage levels of one sweep, as ``(label, coverage, detail)``."""
    out = []
    if kind in ("thin", "thin_fixed_sensitivity"):
        pin = arraymod.outermost(setup.array, 3)
        for n in ANTENNA_LEVELS:
            arr = arraymod.thin(setup.array, n, seed=17, pin=pin)
            out.append((n, setup.uv_track(arr), {"n_ant": n,
                                                 "n_baselines": arr.n_baselines}))
    elif kind == "track":
        for hours in TRACK_LEVELS:
            half = 0.5 * hours
            n_t = max(int(round(setup.n_times * hours / 8.0)), 4)
            cov = arraymod.uv_track(setup.array, dec_deg=setup.dec_deg,
                                    ha_hours=(-half, half), n_times=n_t,
                                    freq_hz=setup.freq_hz)
            out.append((hours, cov, {"track_hours": hours, "n_times": n_t}))
    else:
        raise ValueError("unknown sweep: " + str(kind))
    return out


def run_sweep(setup, truth, kind, seeds=SEEDS, n_iter=250, n_bisect=6,
              snr=config.DEFAULT_SNR):
    """Run one sweep and return a flat list of result rows."""
    # the yardstick: grid and restoring beam from the full array
    full = pp.observe(setup, truth, snr=snr, seed=seeds[0])
    beam = full.beam
    n_full = len(setup.coverage)
    sigma_full = full.sigma
    truth_blurred = full.truth_blurred

    levels = sweep_levels(setup, kind)
    fixed_sens = kind == "thin_fixed_sensitivity"
    rows = []
    for label, cov, detail in levels:
        frac = len(cov) / n_full
        # hold sigma / sqrt(n_vis) fixed, so image thermal noise is constant and
        # only the Fourier coverage changes
        sigma = sigma_full * np.sqrt(len(cov) / n_full) if fixed_sens else sigma_full
        for seed in seeds:
            t0 = time.time()
            obs = pp.observe(setup, truth, coverage=cov, sigma=sigma, seed=seed,
                             beam=beam)
            recons = pp.run_all(obs, n_iter=n_iter, n_bisect=n_bisect)
            for name, rec in recons.items():
                row = rec.scores(truth, truth_blurred)
                row.update(detail)
                row.update(sweep=kind, level=float(label), seed=seed,
                           coverage_fraction=frac, n_vis=len(cov),
                           n_cells=obs.grid_op.n_cells, sigma=float(sigma),
                           uv_fill=obs.grid_op.n_cells / setup.npix ** 2)
                # did the feature survive? criterion fixed in metrics.py, in
                # advance, and applied to the common-resolution image
                res_rad = setup.coverage.resolution_rad()
                sc = mt.crescent_features(rec.blurred, setup.cell,
                                          radius=2.0 * res_rad,
                                          width=1.0 * res_rad, pa_true_deg=90.0)
                row.update({"feat_" + k: v for k, v in sc.as_dict().items()})
                rows.append(row)
            print("  %-22s level=%-5s frac=%.3f seed=%d  %5.0fs  "
                  % (kind, str(label), frac, seed, time.time() - t0)
                  + "  ".join(m + " " + format(
                      [r for r in rows if r["method"] == m][-1]["common_ssim"],
                      ".3f") for m in PLOT_METHODS))
    return rows


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _aggregate(rows, sweep, method, key):
    """Mean and spread of ``key`` over seeds, ordered by coverage fraction."""
    sel = [r for r in rows if r["sweep"] == sweep and r["method"] == method]
    fracs = sorted({r["coverage_fraction"] for r in sel}, reverse=True)
    mean, lo, hi = [], [], []
    for f in fracs:
        vals = np.array([r[key] for r in sel if r["coverage_fraction"] == f],
                        dtype=float)
        vals = vals[np.isfinite(vals)]
        mean.append(np.mean(vals) if vals.size else np.nan)
        lo.append(np.min(vals) if vals.size else np.nan)
        hi.append(np.max(vals) if vals.size else np.nan)
    return np.array(fracs), np.array(mean), np.array(lo), np.array(hi)


def plot_sweeps(rows, setup):
    sweeps = [s for s in ("thin", "thin_fixed_sensitivity", "track")
              if any(r["sweep"] == s for r in rows)]
    titles = {"thin": "antennas dropped\n(sensitivity falls too)",
              "thin_fixed_sensitivity":
                  "antennas dropped, image noise held fixed\n(pure coverage loss)",
              "track": "observation shortened\n(27 antennas throughout)"}

    # --- THE plot: error against coverage fraction, all methods on one axis
    fig, axes = plt.subplots(2, len(sweeps), figsize=(5.2 * len(sweeps), 8.4),
                             squeeze=False)
    for col, sweep in enumerate(sweeps):
        for key, row, ylab in ((("common_nrmse"), 0, "normalised RMS error"),
                               (("common_ssim"), 1, "structural similarity")):
            ax = axes[row][col]
            for m in PLOT_METHODS:
                f, mu, lo, hi = _aggregate(rows, sweep, m, key)
                ax.plot(f, mu, marker=MARKERS[m], color=COLORS[m], lw=1.6,
                        ms=5, label=pp.METHOD_LABELS[m])
                ax.fill_between(f, lo, hi, color=COLORS[m], alpha=0.16, lw=0)
            f, mu, _, _ = _aggregate(rows, sweep, "dirty", key)
            ax.plot(f, mu, ls=":", color=COLORS["dirty"], lw=1.3,
                    label="dirty image")
            ax.set_xscale("log")
            if key == "common_nrmse":
                ax.set_yscale("log")
            ax.set_xlabel("fraction of the full uv coverage")
            ax.set_ylabel(ylab)
            ax.grid(alpha=0.25, lw=0.4, which="both")
            if row == 0:
                ax.set_title(titles[sweep], fontsize=10)
            if col == 0 and row == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Reconstruction error against uv coverage: the methods agree "
                 "where the data are good and diverge where they run out",
                 fontsize=12)
    fig.tight_layout()
    pl.save(fig, "06_coverage_sweep")

    # --- feature survival, total flux, and dynamic range
    fig, axes = plt.subplots(3, len(sweeps), figsize=(5.2 * len(sweeps), 11.4),
                             squeeze=False)
    for col, sweep in enumerate(sweeps):
        ax = axes[0][col]
        for m in PLOT_METHODS:
            f, mu, _, _ = _aggregate(rows, sweep, m, "feat_survives")
            ax.plot(f, mu, marker=MARKERS[m], color=COLORS[m], lw=1.6, ms=5,
                    label=pp.METHOD_LABELS[m])
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel("fraction of the full uv coverage")
        ax.set_ylabel("fraction of realisations where\nthe crescent survives")
        ax.set_title(titles[sweep], fontsize=10)
        ax.grid(alpha=0.25, lw=0.4, which="both")
        if col == 0:
            ax.legend(fontsize=8)

        ax = axes[1][col]
        for m in PLOT_METHODS:
            f, mu, _, _ = _aggregate(rows, sweep, m, "flux_error")
            ax.plot(f, np.abs(mu), marker=MARKERS[m], color=COLORS[m], lw=1.6,
                    ms=5, label=pp.METHOD_LABELS[m])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("fraction of the full uv coverage")
        ax.set_ylabel("|fractional total flux error|")
        ax.grid(alpha=0.25, lw=0.4, which="both")
        if col == 0:
            ax.text(0.03, 0.95, "total flux: the quantity nothing measures",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(fc="white", ec="0.7", alpha=0.85))

        # dynamic range is the only one of these computable without a truth,
        # which is why it is the number papers quote -- and it says nothing
        # about whether the structure is right, so it is worth plotting beside
        # the metrics that do
        ax = axes[2][col]
        for m in PLOT_METHODS:
            f, mu, _, _ = _aggregate(rows, sweep, m, "dynamic_range")
            ax.plot(f, mu, marker=MARKERS[m], color=COLORS[m], lw=1.6, ms=5,
                    label=pp.METHOD_LABELS[m])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("fraction of the full uv coverage")
        ax.set_ylabel("dynamic range (peak / residual RMS)")
        ax.grid(alpha=0.25, lw=0.4, which="both")
        if col == 0:
            ax.text(0.03, 0.95, "computable without a truth --\nand no guide to "
                    "whether the structure is right",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(fc="white", ec="0.7", alpha=0.85))

    fig.suptitle("Feature survival by a criterion fixed in advance, total flux, "
                 "and dynamic range", fontsize=12)
    fig.tight_layout()
    pl.save(fig, "07_feature_survival")


def plot_montage(setup, truth, kind="thin_fixed_sensitivity", levels=(27, 12, 7, 5),
                 seed=0, n_iter=250, n_bisect=6):
    """Images down the sweep: what the numbers in figure 6 actually look like."""
    full = pp.observe(setup, truth, seed=seed)
    beam, tb = full.beam, full.truth_blurred
    pin = arraymod.outermost(setup.array, 3)
    n_full = len(setup.coverage)
    show = ("clean", "tv", "tv_flux", "mem")

    fig, axes = plt.subplots(len(levels), len(show) + 2,
                             figsize=(3.0 * (len(show) + 2), 3.0 * len(levels)),
                             squeeze=False)
    vmax = float(tb.max())
    for row, n in enumerate(levels):
        arr = arraymod.thin(setup.array, n, seed=17, pin=pin)
        cov = setup.uv_track(arr)
        sigma = full.sigma * np.sqrt(len(cov) / n_full)
        obs = pp.observe(setup, truth, coverage=cov, sigma=sigma, seed=seed,
                         beam=beam)
        recons = pp.run_all(obs, methods=show, n_iter=n_iter, n_bisect=n_bisect)

        pl.show_uv(axes[row][0], cov, unit="kilo", s=0.8,
                   title=(str(n) + " antennas, "
                          + format(100 * len(cov) / n_full, ".1f") + "% coverage"))
        lim = 1.05 * setup.coverage.max_baseline / 1e3
        axes[row][0].set_xlim(lim, -lim)
        axes[row][0].set_ylim(-lim, lim)
        pl.show_image(axes[row][1], tb, setup.cell, unit=setup.unit,
                      title="truth (blurred)" if row == 0 else None,
                      vmin=0, vmax=vmax, colorbar=False)
        for k, m in enumerate(show):
            img = recons[m].blurred
            pl.show_image(axes[row][k + 2], img, setup.cell, unit=setup.unit,
                          title=(pp.METHOD_LABELS[m] if row == 0 else None),
                          vmin=0, vmax=vmax, colorbar=False)
            axes[row][k + 2].text(
                0.03, 0.96, "SSIM " + format(mt.ssim(img, tb), ".3f"),
                transform=axes[row][k + 2].transAxes, fontsize=8, va="top",
                color="white")
        for ax in axes[row][1:]:
            ax.set_ylabel("")
            if row < len(levels) - 1:
                ax.set_xlabel("")
    fig.suptitle("Down the sweep at constant image noise: only the Fourier "
                 "coverage is changing", fontsize=12)
    fig.tight_layout()
    pl.save(fig, "08_sweep_montage")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true",
                    help="rebuild figures from results/coverage_sweep.json")
    ap.add_argument("--quick", action="store_true",
                    help="one seed and a small iteration budget, for checking")
    ap.add_argument("--sweeps", default="thin,thin_fixed_sensitivity,track")
    args = ap.parse_args()

    setup = config.vla_setup(npix=128)
    truth = config.truth_for(setup, "crescent")
    path = pl.result_path("coverage_sweep.json")
    print(setup.describe())

    if args.plot_only:
        with open(path) as fh:
            rows = json.load(fh)
    else:
        seeds = (0,) if args.quick else SEEDS
        n_iter, n_bisect = (120, 4) if args.quick else (250, 6)
        rows = []
        for kind in args.sweeps.split(","):
            t0 = time.time()
            rows += run_sweep(setup, truth, kind, seeds=seeds, n_iter=n_iter,
                              n_bisect=n_bisect)
            print(kind + " done in " + format((time.time() - t0) / 60, ".1f")
                  + " min")
            with open(path, "w") as fh:
                json.dump(rows, fh, indent=1)
        print("wrote " + path)

    plot_sweeps(rows, setup)
    if not args.plot_only:
        plot_montage(setup, truth,
                     n_iter=120 if args.quick else 250,
                     n_bisect=4 if args.quick else 6)

    # --- the headline numbers, printed
    print("\nSSIM at the sparse end (pure coverage loss):")
    for m in PLOT_METHODS:
        f, mu, _, _ = _aggregate(rows, "thin_fixed_sensitivity", m, "common_ssim")
        if len(f):
            print("  %-16s full %.4f -> sparsest (%.1f%%) %.4f"
                  % (pp.METHOD_LABELS[m], mu[0], 100 * f[-1], mu[-1]))


if __name__ == "__main__":
    main()
