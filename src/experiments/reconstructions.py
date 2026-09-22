"""Side-by-side reconstructions, and the difference between the two priors.

For each of the two arrays: the truth, the dirty image, and four
reconstructions, all at the same common resolution so they can be compared by
eye without being misled by units. Then the figure the project exists to
produce: the TV solution minus the MEM solution, from *identical data*, both
fitting it to the same reduced chi-squared of about 1.

Where that difference is small, the data have pinned the answer down. Where it
is large, they have not, and what fills the gap is the regulariser. The VLA panel
and the EHT panel differ mostly in how much of the image is in the second
category.

Run with ``python -m src.experiments.reconstructions``.
"""

from __future__ import annotations

import json

import numpy as np

import matplotlib.pyplot as plt

from .. import metrics as mt
from .. import plotting as pl
from . import config, pipeline as pp

SHOW = ("dirty", "clean", "positivity", "tv", "mem")


def _panel_row(fig, gs, row, obs, recons, unit, label_col=True):
    setup = obs.setup
    tb = obs.truth_blurred
    vmax = float(tb.max())
    cols = [("truth", tb)] + [(m, recons[m].blurred) for m in SHOW]
    for k, (name, img) in enumerate(cols):
        ax = fig.add_subplot(gs[row, k])
        title = "truth (blurred)" if name == "truth" else pp.METHOD_LABELS[name]
        if name != "truth":
            title += ("\nSSIM " + format(
                mt.ssim(img, tb), ".3f") + ", $\\chi^2_r$ "
                + format(recons[name].reduced_chi2, ".2f"))
        pl.show_image(ax, img, setup.cell, unit=unit, title=title,
                      vmin=0.0, vmax=vmax, colorbar=(k == len(cols) - 1),
                      label="Jy/beam", bare=(k > 0))
        if not label_col:
            ax.set_xlabel("")
        if name == "truth":
            pl.add_beam_marker(ax, obs.beam, setup.cell, setup.npix, unit=unit)


def main():
    rows = []
    figs = {}

    for name, setup, unit, seed in (("VLA", config.vla_setup(npix=128), "arcsec", 2),
                                    ("EHT", config.eht_setup(), "uas", 3)):
        print(setup.describe())
        truth = config.truth_for(setup, "crescent")
        obs = pp.observe(setup, truth, seed=seed)
        print("  sigma per visibility " + format(obs.sigma, ".4g")
              + ", restoring beam " + repr(obs.beam))
        recons = pp.run_all(obs, n_iter=300, n_bisect=7)
        for r in pp.score_table(obs, recons):
            r["array"] = name
            rows.append(r)
            print("  %-12s lam=%9.3g chi2r=%6.3f  common SSIM=%.4f "
                  "nrmse=%.4f  DR=%8.1f" % (r["method"], r["lam"],
                                            r["reduced_chi2"],
                                            r["common_ssim"], r["common_nrmse"],
                                            r["dynamic_range"]))
        figs[name] = (obs, recons, unit)

    with open(pl.result_path("reconstructions.json"), "w") as fh:
        json.dump(rows, fh, indent=1)

    # ------------------------------------------------------------ main figure
    fig = plt.figure(figsize=(3.05 * (len(SHOW) + 1), 7.4))
    gs = fig.add_gridspec(2, len(SHOW) + 1, hspace=0.30, wspace=0.12)
    for row, name in enumerate(("VLA", "EHT")):
        obs, recons, unit = figs[name]
        _panel_row(fig, gs, row, obs, recons, unit, label_col=(row == 1))
        fig.text(0.008, 0.74 - 0.48 * row,
                 name + "\n" + str(len(obs.coverage)) + " vis",
                 fontsize=11, rotation=90, va="center", ha="left")
    fig.suptitle("The same crescent, two arrays, four reconstructions -- all "
                 "convolved to the same restoring beam", fontsize=13)
    pl.save(fig, "04_reconstructions")

    # ------------------------------------------- the prior made visible figure
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 8.0))
    for row, name in enumerate(("VLA", "EHT")):
        obs, recons, unit = figs[name]
        setup = obs.setup
        tv = recons["tv"].model
        mem = recons["mem"].model
        scale = max(tv.max(), mem.max())
        pl.show_image(axes[row, 0], tv, setup.cell, unit=unit,
                      title=name + ": total variation (native resolution)",
                      vmin=0, vmax=scale, label="Jy/pixel")
        pl.show_image(axes[row, 1], mem, setup.cell, unit=unit,
                      title=name + ": maximum entropy (native resolution)",
                      vmin=0, vmax=scale, label="Jy/pixel")
        diff = tv - mem
        pl.show_image(axes[row, 2], diff, setup.cell, unit=unit,
                      title=("TV $-$ MEM, peak $\\pm$"
                             + format(100 * np.abs(diff).max() / scale, ".0f")
                             + "% of the image peak"),
                      symmetric=True, label="Jy/pixel")

        # How much of the disagreement sits on real emission, and how much on
        # empty sky? In absolute Jy/pixel the two priors differ most where the
        # source is brightest -- that is just scale. The informative quantity is
        # the *share* of the total disagreement that falls where there is nothing
        # to constrain it, which is what separates the two arrays.
        ax = axes[row, 3]
        t = obs.truth
        bright = t > 0.05 * t.max()
        total = float(np.abs(diff).sum())
        off_share = float(np.abs(diff[~bright]).sum()) / max(total, 1e-30)
        for mask, color, label in ((bright, "#1f5fa8", "on the source"),
                                   (~bright, "#c1121f", "off the source")):
            v = np.sort(np.abs(diff[mask]).ravel())[::-1]
            if v.size:
                ax.plot(np.arange(1, v.size + 1), np.cumsum(v) / total,
                        color=color, lw=1.8, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("pixels, brightest disagreement first")
        ax.set_ylabel("cumulative share of total |TV $-$ MEM|")
        ax.set_ylim(0, 1.02)
        ax.set_title(format(100 * off_share, ".0f") + "% of the disagreement is "
                     "on empty sky", fontsize=10)
        ax.legend(fontsize=8, loc="center left")
        ax.grid(alpha=0.25, lw=0.4, which="both")
        recons["tv"].extra["off_source_disagreement_share"] = off_share
        print("  " + name + ": TV-MEM peak difference "
              + format(100 * np.abs(diff).max() / scale, ".0f")
              + "% of the image peak, " + format(100 * off_share, ".0f")
              + "% of the total disagreement on empty sky")

    fig.suptitle("Identical data, identical fit quality, different images: "
                 "the difference is the prior", fontsize=13)
    fig.tight_layout()
    pl.save(fig, "05_prior_made_visible")


if __name__ == "__main__":
    main()
