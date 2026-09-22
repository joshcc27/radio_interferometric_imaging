"""Tests of the scoring functions.

A metric that is wrong is worse than no metric, because every number in the
results depends on it and nothing else will catch it.
"""

import numpy as np
import pytest

from src import metrics as mt
from src import sky
from src.forward import ARCSEC

NPIX, CELL = 64, 1.0 * ARCSEC


@pytest.fixture(scope="module")
def ring():
    return sky.crescent(NPIX, CELL, radius=12 * CELL, width=5 * CELL,
                        asymmetry=0.6, pa_deg=90.0)


# ---------------------------------------------------------------------------
# pixel error
# ---------------------------------------------------------------------------
def test_error_metrics_are_zero_for_a_perfect_match(ring):
    assert mt.rmse(ring, ring) == 0.0
    assert mt.nrmse(ring, ring) == 0.0
    assert mt.ssim(ring, ring) == pytest.approx(1.0)
    assert mt.flux_error(ring, ring) == pytest.approx(0.0)


def test_nrmse_is_scale_free(ring):
    """Doubling both images must not change a normalised error."""
    other = np.roll(ring, 2, axis=0)
    a = mt.nrmse(other, ring)
    b = mt.nrmse(2 * other, 2 * ring)
    assert np.isclose(a, b, rtol=1e-12)


def test_flux_matching_removes_a_pure_scale_error(ring):
    scaled = 1.7 * ring
    assert mt.nrmse(scaled, ring) > 0.5
    assert mt.nrmse(scaled, ring, flux_match=True) < 1e-12
    assert np.isclose(mt.flux_error(scaled, ring), 0.7, rtol=1e-12)


# ---------------------------------------------------------------------------
# structural similarity
# ---------------------------------------------------------------------------
def test_ssim_is_symmetric_and_bounded(ring):
    other = sky.gaussian_blob(NPIX, CELL, 14 * CELL)
    a = mt.ssim(ring, other)
    b = mt.ssim(other, ring, data_range=float(np.ptp(other)))
    assert -1.0 <= a <= 1.0
    # same data_range makes it symmetric; the default uses the truth's range,
    # which is the only asymmetry in the definition
    assert np.isclose(mt.ssim(ring, other, data_range=1.0),
                      mt.ssim(other, ring, data_range=1.0), rtol=1e-12)


def test_ssim_degrades_with_increasing_blur(ring):
    from scipy.ndimage import gaussian_filter

    vals = [mt.ssim(gaussian_filter(ring, s), ring) for s in (0.5, 1.5, 3.0, 6.0)]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))


def test_ssim_degrades_with_increasing_noise(ring):
    rng = np.random.default_rng(0)
    vals = []
    for amp in (0.01, 0.05, 0.2):
        vals.append(mt.ssim(ring + rng.normal(scale=amp * ring.max(),
                                              size=ring.shape), ring))
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))


def test_ssim_punishes_blur_harder_than_rmse_does(ring):
    """The reason SSIM is here at all.

    A one-pixel shift keeps the structure and moves it; a blur destroys the
    structure and keeps it in place. Both metrics prefer the shift, but SSIM
    prefers it far more strongly, which is why it tracks visual judgement better
    -- a blurred ring looks wrong in a way a shifted ring does not.
    """
    from scipy.ndimage import gaussian_filter

    shifted = np.roll(ring, 1, axis=1)
    blurred = gaussian_filter(ring, 3.0)
    blurred *= ring.sum() / blurred.sum()

    rmse_ratio = mt.nrmse(blurred, ring) / mt.nrmse(shifted, ring)
    ssim_ratio = (1.0 - mt.ssim(blurred, ring)) / (1.0 - mt.ssim(shifted, ring))
    assert rmse_ratio > 1.0
    assert ssim_ratio > 2.0 * rmse_ratio


def test_ssim_is_diluted_by_empty_field(ring):
    """A caveat worth pinning down, because it bites in practice.

    SSIM is a mean over pixels, and empty background agrees perfectly with empty
    background, so padding the *same* pair of images with blank sky drives the
    score towards 1. SSIM values are therefore only comparable at a fixed field
    size -- which is why every comparison in ``src/experiments`` fixes the grid
    before it varies anything else.
    """
    from scipy.ndimage import gaussian_filter

    blurred = gaussian_filter(ring, 3.0)
    blurred *= ring.sum() / blurred.sum()
    scores = []
    for size in (NPIX, 128, 192):
        pad = (size - NPIX) // 2
        scores.append(mt.ssim(np.pad(blurred, pad), np.pad(ring, pad)))
    assert all(scores[i] < scores[i + 1] for i in range(len(scores) - 1))
    assert scores[-1] - scores[0] > 0.1


# ---------------------------------------------------------------------------
# dynamic range
# ---------------------------------------------------------------------------
def test_dynamic_range_falls_as_the_residual_grows(ring):
    rng = np.random.default_rng(1)
    vals = [mt.dynamic_range(ring, rng.normal(scale=a, size=ring.shape))
            for a in (1e-5, 1e-4, 1e-3)]
    assert vals[0] > vals[1] > vals[2]


def test_dynamic_range_needs_something_to_measure_against(ring):
    with pytest.raises(ValueError):
        mt.dynamic_range(ring)


# ---------------------------------------------------------------------------
# feature survival
# ---------------------------------------------------------------------------
def test_crescent_criteria_accept_the_truth(ring):
    s = mt.crescent_features(ring, CELL, radius=12 * CELL, width=5 * CELL,
                             pa_true_deg=90.0)
    assert s.survives
    assert s.asymmetry_amplitude == pytest.approx(0.6, abs=0.02)
    assert s.pa_error_deg < 1.0


def test_crescent_criteria_reject_a_filled_blob():
    blob = sky.gaussian_blob(NPIX, CELL, 12 * CELL)
    s = mt.crescent_features(blob, CELL, radius=12 * CELL, width=5 * CELL,
                             pa_true_deg=90.0)
    assert not s.hole
    assert not s.ring
    assert not s.survives


def test_crescent_criteria_reject_the_wrong_asymmetry():
    flipped = sky.crescent(NPIX, CELL, 12 * CELL, 5 * CELL, asymmetry=0.6,
                           pa_deg=-90.0)
    s = mt.crescent_features(flipped, CELL, 12 * CELL, 5 * CELL, 90.0)
    assert s.hole and s.ring          # the ring itself is fine
    assert not s.asymmetry            # but it is bright on the wrong side
    assert not s.survives


def test_symmetric_ring_fails_the_asymmetry_test_on_amplitude():
    """Without the amplitude clause a symmetric ring passes whenever its
    arbitrary centroid angle lands close, which is luck, not measurement."""
    sym = sky.crescent(NPIX, CELL, 12 * CELL, 5 * CELL, asymmetry=0.0)
    s = mt.crescent_features(sym, CELL, 12 * CELL, 5 * CELL, 90.0)
    assert s.asymmetry_amplitude < 0.05
    assert not s.asymmetry


def test_resolved_peaks_counts_components():
    single = sky.gaussian_blob(NPIX, CELL, 4 * CELL)
    double = sky.double_source(NPIX, CELL, separation=14 * CELL, pa_deg=0.0)
    assert mt.resolved_peaks(single) == 1
    assert mt.resolved_peaks(double) == 2


# ---------------------------------------------------------------------------
# uncertainty calibration
# ---------------------------------------------------------------------------
def test_coverage_matches_the_nominal_rate_for_a_correct_posterior():
    """A synthetic, correctly-calibrated Gaussian posterior must score at the
    nominal coverage -- otherwise the calibration check itself is broken and
    would flatter the sampler."""
    rng = np.random.default_rng(2)
    shape = (96, 96)
    truth = rng.standard_normal(shape)
    std = np.full(shape, 0.3)
    mean = truth + rng.normal(scale=0.3, size=shape)
    ks, nominal, empirical = mt.coverage_curve(truth, mean, std)
    assert np.allclose(empirical, nominal, atol=0.02)


def test_coverage_detects_overconfidence():
    """Understate the error bars and coverage must fall below nominal, which is
    the failure mode that matters."""
    rng = np.random.default_rng(3)
    shape = (96, 96)
    truth = rng.standard_normal(shape)
    mean = truth + rng.normal(scale=0.3, size=shape)
    overconfident = np.full(shape, 0.1)
    got = mt.credible_interval_coverage(truth, mean, overconfident, k=1.0)
    assert got < 0.4


def test_coverage_can_be_restricted_to_bright_pixels(ring):
    """Coverage averaged over a mostly-empty image is dominated by background,
    so the on-source number has to be reportable separately."""
    rng = np.random.default_rng(4)
    std = np.full(ring.shape, 0.2 * ring.max())
    mean = ring + rng.normal(scale=0.2 * ring.max(), size=ring.shape)
    bright = ring > 0.05 * ring.max()
    all_px = mt.credible_interval_coverage(ring, mean, std, 1.0)
    on = mt.credible_interval_coverage(ring, mean, std, 1.0, weight=bright)
    assert 0.6 < all_px < 0.76
    assert 0.5 < on < 0.85
    assert np.isnan(mt.credible_interval_coverage(
        ring, mean, std, 1.0, weight=np.zeros_like(bright)))


def test_score_collects_the_standard_row(ring):
    resid = np.random.default_rng(5).normal(scale=1e-4, size=ring.shape)
    row = mt.score(ring, ring, residual=resid)
    for key in ("nrmse", "ssim", "flux_error", "dynamic_range", "residual_rms"):
        assert key in row
    assert row["nrmse"] == 0.0
