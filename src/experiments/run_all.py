"""Run every experiment in the order the README presents them.

    python -m src.experiments.run_all           # everything except real data
    python -m src.experiments.run_all --quick   # small budgets, for checking

The coverage sweep dominates the runtime; ``--skip coverage_sweep`` leaves it out
and ``python -m src.experiments.coverage_sweep --plot-only`` rebuilds its figures
from the cached results.

``real_data`` is not included: it needs a dataset that does not ship with this
repository. See :mod:`src.experiments.real_data`.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time

ORDER = ("uv_coverage", "weighting", "reconstructions", "coverage_sweep",
         "honesty", "bayesian_maps")

QUICK_ARGS = {
    "coverage_sweep": ["--quick", "--sweeps", "thin"],
    "bayesian_maps": ["--draws", "300", "--warmup", "300", "--chains", "2"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip", nargs="*", default=())
    args = ap.parse_args()

    names = list(args.only) if args.only else list(ORDER)
    names = [n for n in names if n not in args.skip]

    failures = []
    for name in names:
        print("\n" + "=" * 72)
        print("== " + name)
        print("=" * 72, flush=True)
        mod = importlib.import_module("src.experiments." + name)
        argv = sys.argv
        sys.argv = [name] + (QUICK_ARGS.get(name, []) if args.quick else [])
        t0 = time.time()
        try:
            mod.main()
        except SystemExit as exc:      # argparse inside an experiment
            failures.append((name, "exited: " + str(exc)))
        except Exception as exc:       # keep going; report at the end
            failures.append((name, type(exc).__name__ + ": " + str(exc)))
            print("FAILED: " + type(exc).__name__ + ": " + str(exc))
        finally:
            sys.argv = argv
        print("-- " + name + " took " + format((time.time() - t0) / 60.0, ".1f")
              + " min", flush=True)

    print("\n" + "=" * 72)
    if failures:
        print("failed: " + ", ".join(n + " (" + why + ")" for n, why in failures))
        return 1
    print("all " + str(len(names)) + " experiments completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
