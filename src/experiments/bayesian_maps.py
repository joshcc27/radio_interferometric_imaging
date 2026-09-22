"""Posterior sampling: mean image, uncertainty map, and a calibration check.

**Status: the sampler is verified, the posterior is not yet trustworthy.** See
the README section "The Bayesian layer". This script runs, produces the figure,
and prints the diagnostics -- including the ones that say the chain has not
mixed. It deliberately does *not* hide that: the acceptance criterion for an
uncertainty map is that its credible intervals contain the truth at close to the
nominal rate, and at the time of writing they do not.

What the figure shows, when it works:

- the posterior mean, which should look like the other reconstructions;
- the per-pixel posterior standard deviation, which is the actual deliverable;
- the ratio of the two, which says which features the data insist on and which
  the prior is carrying;
- the calibration curve: empirical coverage against nominal coverage, computed
  on-source and off-source separately, because coverage averaged over a mostly
  empty image is dominated by the background.

Run with ``python -m src.experiments.bayesian_maps [--npix 64] [--draws 2000]``.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import matplotlib.pyplot as plt

from .. import bayesian as by
from .. import imaging as im
from .. import metrics as mt
from .. import plotting as pl
from . import config, pipeline as pp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--array", default="eht", choices=("eht", "vla"))
    ap.add_argument("--npix", type=int, default=64)
    ap.add_argument("--draws", type=int, default=1500)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--leapfrog", type=int, default=25)
    ap.add_argument("--corr-length", type=float, default=4.0,
                    help="prior correlation length in pixels")
    ap.add_argument("--sigma", type=float, default=3.5,
                    help="prior std of log brightness, in e-folds")
    ap.add_argument("--mean-offset", type=float, default=0.0,
                    help="shift of the prior mean log brightness, in e-folds, "
                         "away from 'total flux spread uniformly'")
    args = ap.parse_args()

    setup = (config.eht_setup(npix=args.npix) if args.array == "eht"
             else config.vla_setup(npix=args.npix))
    print(setup.describe())
    truth = config.truth_for(setup, "crescent")
    obs = pp.observe(setup, truth, seed=3)
    op = obs.acc_op

    mean_log = float(np.log(1.0 / setup.npix ** 2)) + args.mean_offset
    prior = by.GPPrior(setup.npix, correlation_length=args.corr_length,
                       alpha=1.5, sigma=args.sigma, mean_log=mean_log)
    print("prior: correlation length " + format(args.corr_length, ".1f")
          + " px, sigma " + format(args.sigma, ".1f")
          + " e-folds, mean log brightness " + format(mean_log, ".2f"))
    print("sampling " + str(setup.npix ** 2) + " parameters from "
          + str(op.n_data) + " visibilities, " + str(args.chains) + " chains")

    x0 = np.clip(im.dirty_image(obs.grid_op), 0.0, None)
    t0 = time.time()
    summary = by.run_chains(op, prior, x0=x0, n_chains=args.chains,
                            n_draws=args.draws, n_warmup=args.warmup,
                            n_leapfrog=args.leapfrog, seed=11, verbose=True)
    print("sampled in " + format((time.time() - t0) / 60.0, ".1f") + " min")

    mean, std = summary.mean, summary.std
    bright = truth > 0.05 * truth.max()
    d = dict(summary.diagnostics)
    d.update(reduced_chi2_of_mean=op.reduced_chi2(mean),
             ssim_of_mean=mt.ssim(mean, truth),
             nrmse_of_mean=mt.nrmse(mean, truth),
             total_flux_of_mean=float(mean.sum()),
             npix=setup.npix, array=args.array,
             prior=dict(correlation_length=args.corr_length, sigma=args.sigma,
                        mean_log=mean_log))
    for k in (1.0, 2.0, 3.0):
        d["coverage_k%.0f_all" % k] = mt.credible_interval_coverage(
            truth, mean, std, k)
        d["coverage_k%.0f_onsource" % k] = mt.credible_interval_coverage(
            truth, mean, std, k, weight=bright)

    # ------------------------------------------------------- the verdict, first
    print("\ndiagnostics")
    print("  acceptance        " + ", ".join(format(a, ".2f")
                                             for a in d["accept_rate"]))
    print("  divergences       " + str(d["divergences"]))
    for name in ("potential", "total_flux", "peak"):
        print("  R-hat(%-10s) %.3f   ESS %.0f"
              % (name, d["rhat_" + name], d["ess_" + name]))
    print("  chi2r of mean     " + format(d["reduced_chi2_of_mean"], ".3f"))
    print("\ncalibration (nominal -> empirical)")
    for k, nom in ((1.0, 0.683), (2.0, 0.954), (3.0, 0.997)):
        print("  %.0f sigma: %.3f -> all %.3f, on-source %.3f"
              % (k, nom, d["coverage_k%.0f_all" % k],
                 d["coverage_k%.0f_onsource" % k]))

    ok_mixing = (max(d["rhat_potential"], d["rhat_total_flux"]) < 1.05)
    ok_calib = abs(d["coverage_k1_onsource"] - 0.683) < 0.12
    d["trustworthy"] = bool(ok_mixing and ok_calib)
    print("\nVERDICT: " + ("the chains have mixed and the intervals are "
                           "calibrated; the uncertainty map can be believed"
                           if d["trustworthy"] else
                           "NOT trustworthy -- "
                           + ("chains have not mixed (R-hat > 1.05); "
                              if not ok_mixing else "")
                           + ("credible intervals are miscalibrated"
                              if not ok_calib else "")))

    with open(pl.result_path("bayesian.json"), "w") as fh:
        json.dump(d, fh, indent=1, default=float)

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.6))
    unit = setup.unit
    vmax = float(max(truth.max(), mean.max()))
    pl.show_image(axes[0, 0], truth, setup.cell, unit=unit,
                  title="truth", vmin=0, vmax=vmax, label="Jy/pixel")
    pl.show_image(axes[0, 1], mean, setup.cell, unit=unit,
                  title="posterior mean, $\\chi^2_r$ = "
                        + format(d["reduced_chi2_of_mean"], ".2f"),
                  vmin=0, vmax=vmax, label="Jy/pixel")
    pl.show_image(axes[0, 2], std, setup.cell, unit=unit,
                  title="posterior standard deviation\n(the actual deliverable)",
                  cmap="viridis", label="Jy/pixel")

    snr = summary.snr_map()
    pl.show_image(axes[1, 0], np.clip(snr, 0, None), setup.cell, unit=unit,
                  title="mean / standard deviation\n(which features the data "
                        "insist on)", cmap="magma", label="posterior SNR")

    # calibration curve
    ax = axes[1, 1]
    ks, nominal, emp_all = mt.coverage_curve(truth, mean, std)
    _, _, emp_on = mt.coverage_curve(truth, mean, std, weight=bright)
    ax.plot([0, 1], [0, 1], ls="--", color="0.5", lw=1.0, label="perfect")
    ax.plot(nominal, emp_all, "o-", color="#1f5fa8", label="all pixels")
    ax.plot(nominal, emp_on, "s-", color="#c1121f", label="on source")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("calibration: does the truth fall inside\nthe credible "
                 "interval as often as claimed?", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.4)

    # trace of the potential: the fastest way to see a chain that has not settled
    ax = axes[1, 2]
    for c, ch in enumerate(summary.chains):
        ax.plot(ch.potential, lw=0.8, label="chain " + str(c))
    ax.set_xlabel("draw")
    ax.set_ylabel("$-\\log p$ (up to a constant)")
    ax.set_title("potential trace, R-hat = "
                 + format(d["rhat_potential"], ".3f"), fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, lw=0.4)

    fig.suptitle("Posterior mean and uncertainty, "
                 + str(setup.npix ** 2) + " parameters from "
                 + str(op.n_data) + " visibilities"
                 + ("" if d["trustworthy"] else
                    "  --  DIAGNOSTICS FAIL, see the README"), fontsize=12)
    fig.tight_layout()
    pl.save(fig, "11_bayesian_uncertainty")


if __name__ == "__main__":
    main()
