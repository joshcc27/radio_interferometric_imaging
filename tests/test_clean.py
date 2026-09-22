"""Tests of Hogbom CLEAN: bookkeeping, flux, and whether it actually works."""

import numpy as np
import pytest

from src import array as arraymod
from src import clean as cl
from src import forward as fw
from src import imaging as im
from src import sky
from src.forward import MeasurementOperator


@pytest.fixture(scope="module")
def observation():
    arr = arraymod.vla_y(n_per_arm=7)
    cov = arraymod.uv_track(arr, dec_deg=45.0, ha_hours=(-4.0, 4.0), n_times=40)
    npix = 64
    cell = cov.resolution_rad() / 4.0
    return cov, npix, cell


def _run(cov, npix, cell, truth, snr=None, seed=0, drop_zero=False, **kw):
    data = fw.observe(truth, cell, cov, snr=snr,
                      rng=np.random.default_rng(seed))
    op = MeasurementOperator(data, npix, cell, scheme="natural",
                             drop_zero_spacing=drop_zero)
    dirty, psf = im.dirty_image(op), im.dirty_beam(op)
    res = cl.clean(dirty, psf, cell=cell, **kw)
    return op, dirty, psf, res


def test_bookkeeping_identity_is_exact(observation):
    """``dirty == components * psf + residual`` to machine precision.

    This is the strong test. It fails for a sign error, an off-by-one in the psf
    shift, a wrong loop gain bookkeeping, or a psf that is not centred where the
    code thinks it is -- all of which otherwise produce a merely slightly worse
    image that looks fine.
    """
    cov, npix, cell = observation
    truth = sky.point_plus_gaussian(npix, cell, fwhm=6 * cell, point_fraction=0.5)
    op, dirty, psf, res = _run(cov, npix, cell, truth, snr=30.0,
                               gain=0.1, n_iter=400, frac_threshold=1e-4)
    left = dirty - cl.model_dirty(res.components, psf) - res.residual
    assert np.abs(left).max() < 1e-12 * np.abs(dirty).max()


def test_flux_conservation(observation):
    """``sum(dirty) == sum(psf) * model_flux + sum(residual)``, exactly.

    The gridded form of flux conservation. See the note in :mod:`src.clean`: the
    origin cell is not empty, because the shortest baselines fall inside it, so
    ``sum(psf)`` is non-zero and this identity has content.
    """
    cov, npix, cell = observation
    truth = sky.point_plus_gaussian(npix, cell, fwhm=6 * cell, point_fraction=0.5)
    op, dirty, psf, res = _run(cov, npix, cell, truth, snr=30.0,
                               gain=0.1, n_iter=400, frac_threshold=1e-4)
    assert abs(psf.sum()) > 1e-6
    lhs = dirty.sum()
    rhs = psf.sum() * res.model_flux + res.residual.sum()
    assert abs(lhs - rhs) <= 1e-10 * abs(lhs)


def test_residual_sum_is_invariant_without_a_zero_spacing(observation):
    """Empty the origin cell and CLEAN cannot change the residual sum at all.

    ``sum(psf)`` is exactly zero, so every subtraction removes zero net flux from
    the residual. The model's total flux is then not constrained by the dirty
    image in any way -- it comes entirely from the algorithm's prior.
    """
    cov, npix, cell = observation
    truth = sky.point_plus_gaussian(npix, cell, fwhm=6 * cell, point_fraction=0.5)
    op, dirty, psf, res = _run(cov, npix, cell, truth, snr=30.0, drop_zero=True,
                               gain=0.1, n_iter=400, frac_threshold=1e-4)
    assert abs(psf.sum()) < 1e-12
    assert abs(res.residual.sum() - dirty.sum()) < 1e-12 * max(abs(dirty).max(), 1e-30)
    assert res.model_flux > 0.5  # yet it still recovers most of the flux


def test_point_source_is_recovered_at_the_right_place_and_flux(observation):
    """A noiseless point source: CLEAN should put essentially all of the flux in
    one pixel, in the right pixel."""
    cov, npix, cell = observation
    truth = sky.point_source(npix, cell, flux=1.0)
    op, dirty, psf, res = _run(cov, npix, cell, truth, snr=None,
                               gain=0.1, n_iter=600, frac_threshold=1e-5)
    c = npix // 2
    assert np.isclose(res.model_flux, 1.0, rtol=0.02)
    assert res.components[c, c] / res.model_flux > 0.98
    assert np.abs(res.residual).max() < 1e-4


def test_offset_point_source_lands_in_the_right_pixel(observation):
    """Guards the sign of the psf shift, which a centred source cannot."""
    cov, npix, cell = observation
    off = (7 * cell, -5 * cell)
    truth = sky.point_source(npix, cell, flux=1.0, offset=off)
    op, dirty, psf, res = _run(cov, npix, cell, truth, snr=None,
                               gain=0.15, n_iter=600, frac_threshold=1e-5)
    i, j = np.unravel_index(np.argmax(res.components), res.components.shape)
    want_i, want_j = np.unravel_index(np.argmax(truth), truth.shape)
    assert (i, j) == (want_i, want_j)


def test_cleaning_reduces_the_residual_peak(observation):
    """Extended emission needs thousands of iterations, not hundreds: CLEAN has
    to represent a smooth ring as a speckle of delta functions, and each one only
    removes ``gain`` times its own peak."""
    cov, npix, cell = observation
    truth = sky.crescent(npix, cell, 5 * cell, 2.5 * cell)
    op, dirty, psf, res = _run(cov, npix, cell, truth, snr=25.0,
                               gain=0.1, n_iter=6000, frac_threshold=1e-3)
    assert np.abs(res.residual).max() < 0.2 * np.abs(dirty).max()
    assert res.stop_reason == "reached threshold"
    # the peak history should be broadly decreasing
    p = np.abs(res.peak_history)
    assert p[-1] < p[0]


def test_mask_confines_components(observation):
    cov, npix, cell = observation
    truth = sky.point_source(npix, cell)
    mask = np.zeros((npix, npix), dtype=bool)
    c = npix // 2
    mask[c - 3:c + 4, c - 3:c + 4] = True
    op, dirty, psf, res = _run(cov, npix, cell, truth, snr=20.0, seed=3,
                               gain=0.1, n_iter=200, mask=mask)
    assert np.all(res.components[~mask] == 0.0)


def test_negative_components_can_be_forbidden(observation):
    cov, npix, cell = observation
    truth = sky.crescent(npix, cell, 5 * cell, 2.5 * cell)
    op, dirty, psf, res = _run(cov, npix, cell, truth, snr=20.0, seed=4,
                               gain=0.1, n_iter=300, allow_negative=False)
    assert res.components.min() >= 0.0


def test_restoring_conserves_flux_in_jy_per_beam(observation):
    """Convolving with a peak-normalised beam multiplies the total by the beam
    area and leaves the peak of an isolated point source alone.

    That second part is the Jy/beam convention, not a bug: a 1 Jy point source
    has a restored peak of 1 Jy/beam.
    """
    cov, npix, cell = observation
    truth = sky.point_source(npix, cell)
    op, dirty, psf, res = _run(cov, npix, cell, truth, snr=None,
                               gain=0.1, n_iter=400, frac_threshold=1e-4)
    restored = res.restored(cell, add_residual=False)
    assert np.isclose(restored.sum(), res.model_flux * res.beam.area_pixels(cell),
                      rtol=1e-6)
    assert np.isclose(restored.max(), res.components.max(), rtol=1e-3)


def test_restoring_throws_away_super_resolution(observation):
    """Two sources one nominal resolution element apart.

    CLEAN's component list separates them -- two distinct peaks, at the right
    places, which is genuine super-resolution, because the deconvolution used the
    fact that the sky is a sum of points. Convolving with the restoring beam then
    merges them back into a single blob. The information was there and the last
    step discarded it, which is the standard objection to the CLEAN restored
    image and is worth being able to demonstrate rather than just assert.
    """
    cov, npix, cell = observation
    res_rad = cov.resolution_rad()
    truth = sky.double_source(npix, cell, separation=res_rad, pa_deg=0.0)
    op, dirty, psf, out = _run(cov, npix, cell, truth, snr=None,
                               gain=0.1, n_iter=3000, frac_threshold=1e-4)
    assert out.beam.bmaj > 1.5 * res_rad  # natural weighting: a fat beam

    def axis_peaks(profile, frac=0.2):
        inner = profile[1:-1]
        hi = (inner >= profile[:-2]) & (inner >= profile[2:])
        return int(np.sum(hi & (inner > frac * profile.max())))

    col = npix // 2
    assert axis_peaks(out.components[:, col]) == 2
    assert axis_peaks(out.restored(cell, add_residual=False)[:, col]) == 1


def test_theoretical_image_noise_matches_a_noise_only_simulation(observation):
    """The predicted dirty-image noise must match a Monte Carlo realisation.

    This is the assertion that catches the missing factor of ``sqrt(2)`` from the
    Hermitian partners sharing a noise realisation.
    """
    cov, npix, cell = observation
    n = len(cov)
    sigma = np.full(n, 0.01)
    emp = []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        noise = fw.add_thermal_noise(np.zeros(n, dtype=complex), sigma, rng)
        data = fw.hermitian_augment(
            fw.Visibilities(u=cov.u, v=cov.v, vis=noise, sigma=sigma))
        op = MeasurementOperator(data, npix, cell, scheme="natural")
        emp.append(im.dirty_image(op).std())
        pred = cl.theoretical_image_noise(op)
    assert np.isclose(np.mean(emp), pred, rtol=0.1)


def test_psf_must_be_peak_normalised(observation):
    cov, npix, cell = observation
    truth = sky.point_source(npix, cell)
    op, dirty, psf, _ = _run(cov, npix, cell, truth, snr=None, n_iter=1)
    with pytest.raises(ValueError):
        cl.clean(dirty, psf * 0.5, cell=cell)
