# Code Review Handover

Review date: 2026-09-30 · Reviewed commit: `7bf5662` on `main` (clean working tree)

## Summary

The simulation and imaging core is sound. The real-data path has four confirmed
bugs, and each one would corrupt the first archival run without raising an
error. Fix them before imaging any real dataset.

| Item | State at review |
| --- | --- |
| Size | ~5,000 lines in `src/`, ~2,000 lines in `tests/` |
| Test suite | 122 passed, 3 skipped, 94 s (Python 3.12.10, numpy 2.5.0, scipy 1.18.0, matplotlib 3.11.0) |
| Skipped tests | The three UVFITS tests in `tests/test_realdata.py`, because `astropy` is not installed |
| README test count | Says "121 tests" and "118 passed, 3 skipped"; out of date |

The measurement operators, their adjoints, CLEAN, FISTA, the TV prox and the
HMC sampler are carefully derived and tested against the right invariants. The
README is candid about what is unfinished: the full coverage sweep, HMC
calibration and real data. The problems sit almost entirely in `src/realdata.py`,
the one module the test suite cannot exercise without astropy. No code was
changed during the review.

---

## Confirmed bugs (real-data path)

All four are in `src/realdata.py`. Bugs 1 and 2 were reproduced with the
scratch script described under [Verification](#verification). Bugs 3 and 4 are
clear from the code and from what the tests do and do not cover.

### 1. UVFITS loader mishandles the Stokes and IF axes

**Where:** `load_uvfits` (lines 118–134) and `_uvfits_frequencies`.

The loader calls `np.squeeze` on `DATA` and only collapses Stokes when
`vis.ndim >= 3`. With a single channel, the polarisation axis survives the
squeeze as axis 1, and the loader treats it as the channel axis.

| File layout | Expected | Got |
| --- | --- | --- |
| 2 channels, RR + LL | 100 vis, value 2.0 | 100 vis, value 2.0 (correct) |
| 1 channel, RR + LL | 50 vis, value 2.0 | 100 vis, values 1.0 and 3.0: RR and LL loaded as separate points |
| 1 channel, RR LL RL LR | 50 vis, value 2.0 | 200 vis: cross-hands (100.0) loaded as Stokes I |
| 4 IFs × 2 channels | 400 vis | `ValueError: cannot reshape array of size 400 into shape (200,)` |

Single-channel and multi-IF are the normal layouts for EHT release files. Two
related problems:

- `_uvfits_frequencies` never reads the FQ table, despite its docstring. Its
  `hdul` argument is unused.
- A mismatch between the frequency count and the channel count is replaced
  silently by the mean frequency (line 133) instead of raising an error.

`test_uvfits_reader_drops_flagged_samples` writes exactly the 1-channel, RR + LL
case and would fail if astropy were installed. It is skipped, so this has never
been caught.

**Fix:**

- Locate the COMPLEX, STOKES, FREQ and IF axes from the `CTYPEn` header keys
  rather than by position after a squeeze.
- Read the Stokes codes from the STOKES axis `CRVAL`/`CDELT` values and select
  the parallel hands by code.
- Read per-IF frequencies from the FQ table.
- Raise on any count mismatch.

### 2. `rescale_weights_from_scatter` underestimates sigma by √2

**Where:** line 338, `est = 1.4826 * mad / np.sqrt(2.0)`.

`1.4826 × MAD` of the real part already equals the per-component sigma. That
is the radio convention the project uses (`forward.add_thermal_noise`), so the
extra `/ √2` is wrong. On pure noise with a true sigma of 0.100, the function
returns 0.0704, a ratio of 0.704. Reduced χ² then comes out about 2× too high,
so `solve_at_chi2` picks the wrong λ.

Further problems in the same function:

- The loop over the real and imaginary parts overwrites `sigma[sel]` with the
  imaginary estimate instead of averaging the two.
- The median is taken over a whole |uv| annulus. For a resolved or off-centre
  source, the sky's own azimuthal variation inflates the estimate.
- The test tolerance, `0.5 < ratio < 2.0`, is loose enough to hide all of this.

**Fix:**

- Drop the `/ √2` and average the two parts.
- Tighten the test to a noise-only case at about ±5%.
- For real data, prefer a difference-based estimator: successive samples on one
  baseline within a scan, where the sky is constant. That is where a `/ √2`
  belongs.

### 3. Measurement-set loader averages the wrong polarisations

**Where:** `load_measurement_set` → `_to_stokes_i` (lines 179–194, 200–250).

`_to_stokes_i` assumes the first two correlations are the parallel hands. That
holds for UVFITS (RR, LL, RL, LR), but a full-polarisation VLA measurement set
is ordered RR, RL, LR, LL (`CORR_TYPE` 5, 6, 7, 8). The loader therefore
averages RR with RL.

Related problems:

- `DATA_DESC_ID` is compared directly with the spectral-window ID (line 220).
  The two differ in general and have to be mapped through the
  `DATA_DESCRIPTION` table.
- Neither table is ever closed (`finally: pass`, lines 217–218).

**Fix:**

- Read `CORR_TYPE` from the `POLARIZATION` subtable and select the parallel
  hands by type.
- Map the data description ID to its SPW and polarisation IDs.
- Close both tables in `finally`.

There is no test for this path: it needs `python-casacore` or `casatools`.

### 4. `closure_phase` mixes timestamps

**Where:** lines 364–373.

The lookup takes the first visibility on each baseline (`lookup[(a, b)][0]`).
Across a real track, the three legs of a triangle then come from different
times, so the station gains do not cancel and the result is not a closure
phase. The only test uses a single timestamp.

**Fix:**

- Take a per-visibility time index. `UVCoverage.time_index` already exists for
  simulations, and the loaders would need to carry one.
- Form the triangles within each timestamp and return one closure phase per
  triangle per time.

---

## Worth fixing (accuracy and robustness)

### 5. Kaiser–Bessel β is tuned for no oversampling

**Where:** `forward.py`, `DegridOperator._build_kernel` (line 590) and
`kb_grid_correction` (line 288).

Both use `β = π√(W²/4 − 0.8)`, which is Beatty et al.'s formula for an
oversampling factor α = 1. `DegridOperator` uses `osamp = 2`. The α = 2 form is
`β = π√((W/α)²(α − ½)² − 0.8)`: 13.86 for W = 6, against 9.00 now.

| β | Max relative model error (VLA crescent, 128²) |
| --- | --- |
| 9.00 (current) | 1.65e-4 |
| 13.86 (α = 2) | 2.96e-6 |

That is a 55× improvement from a one-line change in each place. Keep the
α = 1 value for `grid_visibilities(mode="kb")`, which has no oversampling. The
README's "model error 1.6e-4" and "300× better" figures would need updating.

### 6. `solve_at_chi2` can return an untested λ

**Where:** `regularised.py`, line 481.

If no bisection step reaches the χ² target, `lam_final` stays at the initial
lower bound, `scale * 1e-6`, which was never evaluated. The result comes back
with no flag.

**Fix:** record in `history` whether the target was bracketed, and warn when it
was not.

### 7. The TV prox restarts its dual variable from zero on every call

**Where:** `regularised.py`, `TotalVariation.prox`.

`px, py` start at zero on every call, and `n_inner = 40` leaves the prox
inexact. The result is biased towards *less* TV smoothing than λ implies. The
χ² bisection partly absorbs this, but the problem being solved is not exactly
the one advertised. The docstring says the prox is re-solved "from a nearby
point", which is true only of the primal variable.

**Fix:** keep `(px, py)` on the regulariser between FISTA iterations. This is
the standard warm start and costs nothing.

### 8. Result JSONs are not committed

The `.gitignore` comment says "the result JSONs are committed on purpose", but
the rule is `results/*` with no exception, and nothing under `results/` is
tracked. The README repeatedly points readers to `results/` as the record of
where its numbers came from.

**Fix:** add `!results/*.json`, then commit the JSONs once the full runs exist.

---

## Minor

- **HMC wasted transform** (`bayesian.py`, line 237): after a leapfrog run
  that did not diverge, the `for … else: U_new = posterior.potential(th)`
  branch recomputes a potential the last step already computed. That is one
  wasted forward transform per iteration, about 4% at `n_leapfrog = 25`.
- **`ess` docstring** (`bayesian.py`): it says "initial positive sequence", but
  the code truncates when the autocorrelation drops to 0.05 or below. Either
  fix the docstring or implement Geyer's estimator.
- **Clipped gradient** (`LogIntensityPosterior`): θ is clipped to [−40, 20]
  inside `image()`, but the gradient ignores the clip. This only matters if θ
  leaves that range.
- **Units in `real_data.py`**: the printout and figure axes are hard-coded to
  µas, which is wrong for VLA data. `realdata.describe` also always prints µas.
- **`forward.observe`** fails if `sigma` is passed as an array, because it calls
  `float(sigma)`.
- **`forward.thermal_sigma_for_snr`** silently treats any unrecognised `kind`
  as `"max"`.
- **README**: the test counts are out of date, and the Kaiser–Bessel figures
  depend on item 5.

---

## Verification

Two commands were run:

- `python -m pytest -q -rs` for the test suite (results in the Summary).
- A throwaway script, not committed, that did three things:
  1. It stubbed `astropy.io.fits` with a fake `open()` returning random-groups
     data. That drove `load_uvfits` through the four layouts in the table under
     bug 1.
  2. It called `rescale_weights_from_scatter` on 20,000 visibilities of pure
     complex Gaussian noise (σ = 0.1 per component) with a deliberately wrong
     stated sigma.
  3. It compared `DegridOperator.forward` against the exact
     `predict_visibilities` for the VLA crescent setup, under both β values.

Bugs 3 and 4 were not executed: no casacore is available, and the
closure-phase error is evident from the lookup.

To reproduce bug 1 properly, install astropy (`pip install astropy`) and run
`python -m pytest tests/test_realdata.py`. `test_uvfits_reader_drops_flagged_samples`
should then fail.

---

## Recommended order of work

1. **Make the UVFITS tests run.** Add `astropy` to a dev requirements file, or
   to CI. At least one test should then fail on bug 1, which confirms the
   harness works.
2. **Fix bug 1** and add test fixtures for 1-channel RR/LL, 1-channel 4-Stokes
   and multi-IF files.
3. **Fix bug 2** and tighten its test to a noise-only case at about ±5%.
4. **Fix bug 4** and add a multi-timestamp closure-phase test with per-station
   gains that vary in time.
5. **Fix bug 3.** It cannot be tested without casacore, so at least unit-test
   the `CORR_TYPE` selection logic with a stub table.
6. **Apply item 5** (the β change), re-run `test_operators.py`, and update the
   README figures.
7. **Items 6–8**, then the minor list.
8. **Then continue the README roadmap**: the full coverage sweep, HMC
   calibration and a first real dataset.
