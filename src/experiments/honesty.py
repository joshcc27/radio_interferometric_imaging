"""The honesty test: reconstruct ground truths that violate the prior.

Every paper in this area shows results on the kind of source its method suits, so
this experiment does the opposite and picks the two targets designed to break the
two regularisers:

- a **sharp-edged square** under **maximum entropy**. The entropy penalty is
  minimised by smooth images, so it should not be able to produce a hard edge
  however good the data are.
- a **smooth broad Gaussian** under **total variation**. TV is minimised by
  piecewise-flat images, so it should staircase a smooth gradient into plateaux.

Each truth is also run through the regulariser that *does* suit it, and through
CLEAN, so a failure can be attributed to the prior rather than to the data or the
solver.

**Run on both arrays, because the first version of this experiment was
misleading.** On the well-covered VLA array neither prior misbehaves at all: TV
even scores slightly *better* than maximum entropy on the smooth Gaussian, and
its staircase index is identical to entropy's. That is not a bug, it is the
point — lambda is chosen so the solution fits the data to reduced chi-squared 1,
and where the data are that good they leave the penalty almost nothing to do. The
prior's character only shows up where the data run out. So the interesting
comparison is VLA against EHT, and reporting only the VLA numbers would have
understated the effect the project is about.

Two criteria are fixed in advance, both measured on the *native* model:

- ``edge_sharpness`` -- mean gradient across the true boundary of the square.
- ``staircase_index`` -- how concentrated the brightness histogram is onto a few
  levels.

Both are read against the truth's own value, not against zero.

Run with ``python -m src.experiments.honesty``.
"""

from __future__ import annotations

import json

import numpy as np

import matplotlib.pyplot as plt

from .. import metrics as mt
from .. import plotting as pl
from . import config, pipeline as pp

# (truth, the prior it suits, the prior it breaks, blurb)
CASES = (
    ("square", "tv", "mem",
     "hard edges: TV's home ground, entropy cannot make one"),
    ("gaussian", "mem", "tv",
     "a smooth gradient: entropy's home ground, TV should staircase it"),
)
SHOW = ("clean", "tv", "mem")


def edge_sharpness(img, cell, side):
    """Mean |gradient| across the edge of the square, in units of the peak.

    A quantitative version of "does it have a hard edge": measured on the ring of
    pixels straddling the true boundary, so a rounded reconstruction scores low
    however well it does elsewhere.

    Measured on the *native* model, never on the common-resolution image.
    Convolving with the restoring beam removes the edge by construction, and a
    sharp and a soft reconstruction then score identically -- which is what the
    first version of this experiment did, reporting 0.125 for both TV and MEM
    and hiding the very effect it was built to measure.
    """
    from ..regularised import grad2

    npix = img.shape[0]
    ax = (np.arange(npix) - npix // 2) * cell
    mm, ll = np.meshgrid(ax, ax, indexing="ij")
    cheb = np.maximum(np.abs(ll), np.abs(mm))
    band = np.abs(cheb - 0.5 * side) <= 1.5 * cell
    gx, gy = grad2(img)
    g = np.hypot(gx, gy)
    peak = img.max()
    return float(g[band].mean() / peak) if peak > 0 and band.any() else 0.0


def staircase_index(img, n_bins=60):
    """How concentrated the brightness histogram is onto a few levels.

    TV's signature failure on smooth emission is flat plateaux, which show up as
    spikes in the histogram of pixel values: a smooth image spreads its values
    out, a staircased one piles them up. Reported as the fraction of on-source
    pixels in the three most populated bins, which is ``3/n_bins`` for a perfectly
    uniform spread and approaches 1 for a hard staircase.
    """
    img = np.asarray(img, dtype=float)
    on = img > 0.05 * img.max()
    if on.sum() < 10:
        return np.nan
    vals = img[on]
    hist, _ = np.histogram(vals, bins=n_bins,
                           range=(float(vals.min()), float(vals.max())))
    return float(np.sort(hist)[-3:].sum() / hist.sum())


def run_array(name, setup, n_iter=350, n_bisect=7):
    """All cases on one array. Returns ``(rows, panels)``."""
    print("\n" + "=" * 70)
    print(setup.describe())   # already carries the array name
    rows, panels = [], {}
    side = 4.0 * setup.coverage.resolution_rad()

    for kind, suited, broken, blurb in CASES:
        truth = config.truth_for(setup, kind)
        obs = pp.observe(setup, truth, seed=7)
        recons = pp.run_all(obs, methods=SHOW, n_iter=n_iter, n_bisect=n_bisect)
        panels[kind] = (obs, recons, suited, broken, blurb)

        ref = {"edge_sharpness": edge_sharpness(truth, setup.cell, side),
               "staircase_index": staircase_index(truth)}
        print("\n  " + kind + " -- " + blurb)
        print("    %-6s %36s edge=%.3f staircase=%.3f"
              % ("truth", "", ref["edge_sharpness"], ref["staircase_index"]))
        for r in pp.score_table(obs, recons):
            r["array"] = name
            r["truth"] = kind
            r["suited"] = r["method"] == suited
            r["broken"] = r["method"] == broken
            r["truth_edge_sharpness"] = ref["edge_sharpness"]
            r["truth_staircase_index"] = ref["staircase_index"]
            model = recons[r["method"]].model
            r["edge_sharpness"] = edge_sharpness(model, setup.cell, side)
            r["staircase_index"] = staircase_index(model)
            rows.append(r)
            tag = ("  <- suited" if r["suited"] else
                   ("  <- MISMATCHED" if r["broken"] else ""))
            print("    %-6s chi2r=%5.2f SSIM=%.4f nrmse=%.4f edge=%.3f "
                  "staircase=%.3f%s"
                  % (r["method"], r["reduced_chi2"], r["common_ssim"],
                     r["common_nrmse"], r["edge_sharpness"],
                     r["staircase_index"], tag))
    return rows, panels


def plot_array(name, setup, panels, filename):
    fig = plt.figure(figsize=(16.5, 8.6))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.25], hspace=0.32,
                          wspace=0.22)
    for row, (kind, _, _, _) in enumerate(CASES):
        obs, recons, suited, broken, blurb = panels[kind]
        tb = obs.truth_blurred
        vmax = float(tb.max())
        cols = [("truth", tb)] + [(m, recons[m].blurred) for m in SHOW]
        for k, (mname, img) in enumerate(cols):
            ax = fig.add_subplot(gs[row, k])
            if mname == "truth":
                title = kind + " truth (blurred)"
            else:
                mark = (" [suited]" if mname == suited
                        else (" [MISMATCHED]" if mname == broken else ""))
                title = (pp.METHOD_LABELS[mname] + mark + "\nSSIM "
                         + format(mt.ssim(img, tb), ".3f"))
            pl.show_image(ax, img, setup.cell, unit=setup.unit, title=title,
                          vmin=0, vmax=vmax, colorbar=False, bare=(k > 0))

        # a cut through the middle says more than any of the images: the native
        # models, because the blurred ones cannot show an edge or a plateau
        ax = fig.add_subplot(gs[row, 4])
        c = setup.npix // 2
        scale = pl.UNITS[setup.unit][0]
        x = (np.arange(setup.npix) - c) * setup.cell / scale
        ax.plot(x, obs.truth[c, :], color="k", lw=2.2, label="truth", alpha=0.75)
        for mname, color in (("clean", "#c1121f"), ("tv", "#1f5fa8"),
                             ("mem", "#2a9d8f")):
            lw = 2.0 if mname == broken else 1.2
            ax.plot(x, recons[mname].model[c, :], color=color, lw=lw,
                    label=pp.METHOD_LABELS[mname]
                    + (" (mismatched)" if mname == broken else ""))
        half = 0.35 * setup.npix * setup.cell / scale
        ax.set_xlim(half, -half)
        ax.set_xlabel("$l$ (" + pl.UNITS[setup.unit][1] + ", east left)")
        ax.set_ylabel("Jy/pixel")
        ax.set_title("cut through $m=0$, native resolution", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25, lw=0.4)

    fig.suptitle(name + ": each regulariser run on the source its prior cannot "
                 "represent", fontsize=13)
    pl.save(fig, filename)


def main():
    rows = []
    setups = (("VLA", config.vla_setup(npix=128), "09_honesty_vla"),
              ("EHT", config.eht_setup(), "10_honesty_eht"))
    for name, setup, filename in setups:
        r, panels = run_array(name, setup)
        rows += r
        plot_array(name, setup, panels, filename)

    with open(pl.result_path("honesty.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=float)

    # ---------------------------------------------------------- the comparison
    print("\n" + "=" * 70)
    print("matched vs mismatched prior, same data, same fit quality")
    print("%-5s %-9s %-24s %-24s" % ("array", "truth", "suited", "mismatched"))
    for name, _, _ in setups:
        for kind, suited, broken, _ in CASES:
            sel = {r["method"]: r for r in rows
                   if r["truth"] == kind and r["array"] == name}
            if not sel:
                continue
            s, b = sel[suited], sel[broken]
            rel = 100 * (b["common_ssim"] / s["common_ssim"] - 1)
            print("%-5s %-9s %-4s SSIM %.4f      %-4s SSIM %.4f   (%+.1f%%)"
                  % (name, kind, suited, s["common_ssim"], broken,
                     b["common_ssim"], rel))

    print("\nshape criteria, read against the truth's own value:")
    for name, _, _ in setups:
        sq = {r["method"]: r for r in rows
              if r["truth"] == "square" and r["array"] == name}
        ga = {r["method"]: r for r in rows
              if r["truth"] == "gaussian" and r["array"] == name}
        if sq:
            print("  %-4s square   edge sharpness: truth %.3f | TV %.3f | "
                  "MEM %.3f" % (name, sq["tv"]["truth_edge_sharpness"],
                                sq["tv"]["edge_sharpness"],
                                sq["mem"]["edge_sharpness"]))
        if ga:
            print("  %-4s gaussian staircase:     truth %.3f | MEM %.3f | "
                  "TV %.3f" % (name, ga["mem"]["truth_staircase_index"],
                               ga["mem"]["staircase_index"],
                               ga["tv"]["staircase_index"]))
    print("\nIf the mismatched column is no worse than the suited one, the data "
          "were\ngood enough that the prior never had to guess. That is the "
          "expected VLA\nresult, and it is why this is run on the sparse array "
          "as well.")


if __name__ == "__main__":
    main()
