"""Tests of the dirty image, the dirty beam, and the restoring beam."""

import numpy as np
import pytest

from src import array as arraymod
from src import forward as fw
from src import imaging as im
from src import sky
from src.forward import ARCSEC, MeasurementOperator


@pytest.fixture(scope="module")
def setup():
    arr = arraymod.vla_y(n_per_arm=6)
    cov = arraymod.uv_track(arr, dec_deg=50.0, ha_hours=(-3.0, 3.0), n_times=30)
    npix = 64
    cell = cov.resolution_rad() / 4.0
    return cov, npix, cell


@pytest.mark.parametrize("scheme", ["natural", "uniform", "briggs"])
def test_dirty_image_of_point_source_is_the_dirty_beam(setup, scheme):
    """The sharpest check in the project.

    A unit point source at the phase centre has ``V = 1`` everywhere, so the
    gridded data are identical to the gridded weights and the dirty image must
    come out *exactly* equal to the dirty beam. It passes only if the gridding,
    the weighting, the FFT centring and both normalisations agree, and it fails
    for a half-pixel offset anywhere in the chain.
    """
    cov, npix, cell = setup
    truth = sky.point_source(npix, cell, flux=1.0)
    data = fw.observe(truth, cell, cov, sigma=0.0)
    op, dirty, beam = im.make_images(data, npix, cell, scheme=scheme, robust=0.0)
    assert np.abs(dirty - beam).max() < 1e-12


def test_dirty_beam_is_peak_normalised_and_symmetric(setup):
    """Peak exactly 1 at the centre, and point-symmetric because the sampling
    function is Hermitian.

    The symmetry is a consequence of augmenting the data with conjugates; drop
    that and the beam picks up an imaginary part and a lopsided real part.
    """
    cov, npix, cell = setup
    truth = sky.point_source(npix, cell)
    data = fw.observe(truth, cell, cov, sigma=0.0)
    op, _, beam = im.make_images(data, npix, cell, scheme="natural")
    c = npix // 2
    assert np.isclose(beam[c, c], 1.0, atol=1e-12)
    assert beam.max() <= 1.0 + 1e-12
    # B(-l, -m) = B(l, m); index 0 has no partner on an even grid, so skip it
    flipped = beam[1:, 1:][::-1, ::-1]
    assert np.abs(beam[1:, 1:] - flipped).max() < 1e-12


def test_uniform_weighting_gives_a_narrower_beam_than_natural(setup):
    """The resolution-versus-sensitivity trade-off, as an assertion.

    Uniform weighting equalises the weight per occupied uv cell, which stops the
    crowded short baselines from dominating: the beam narrows and its sidelobes
    drop, at the cost of throwing sensitivity away.
    """
    cov, npix, cell = setup
    truth = sky.point_source(npix, cell)
    data = fw.observe(truth, cell, cov, sigma=0.0)
    areas = {}
    for scheme in ("natural", "uniform"):
        op, _, beam = im.make_images(data, npix, cell, scheme=scheme)
        areas[scheme] = im.fit_clean_beam(beam, cell).area_pixels(cell)
    assert areas["uniform"] < areas["natural"]


def test_briggs_robust_interpolates_between_the_two(setup):
    cov, npix, cell = setup
    truth = sky.point_source(npix, cell)
    data = fw.observe(truth, cell, cov, sigma=0.0)
    got = []
    for robust in (-2.0, 0.0, 2.0):
        op = MeasurementOperator(data, npix, cell, scheme="briggs", robust=robust)
        got.append(im.fit_clean_beam(im.dirty_beam(op), cell).area_pixels(cell))
    assert got[0] < got[1] < got[2]  # uniform-like -> natural-like


def test_clean_beam_recovers_a_known_gaussian():
    """Feed the fitter a Gaussian and it must return that Gaussian.

    The fit is restricted to the main lobe, so this also checks that the
    restriction does not bias the width.
    """
    npix, cell = 96, 1.0 * ARCSEC
    want = im.GaussianBeam(bmaj=12.0 * ARCSEC, bmin=6.0 * ARCSEC, bpa_deg=30.0)
    psf = want.image(npix, cell, peak=1.0)
    got = im.fit_clean_beam(psf, cell)
    assert np.isclose(got.bmaj, want.bmaj, rtol=2e-3)
    assert np.isclose(got.bmin, want.bmin, rtol=2e-3)
    assert np.isclose(got.bpa_deg, want.bpa_deg, atol=0.5)


def test_beam_area_matches_direct_integration():
    npix, cell = 128, 0.5 * ARCSEC
    beam = im.GaussianBeam(10.0 * ARCSEC, 7.0 * ARCSEC, -20.0)
    assert np.isclose(beam.image(npix, cell).sum(), beam.area_pixels(cell), rtol=1e-6)


def test_shift_psf_is_a_circular_shift(setup):
    cov, npix, cell = setup
    truth = sky.point_source(npix, cell)
    data = fw.observe(truth, cell, cov, sigma=0.0)
    op, _, psf = im.make_images(data, npix, cell, scheme="natural")
    for (i, j) in ((npix // 2, npix // 2), (3, 7), (npix - 2, 1)):
        s = im.shift_psf(psf, i, j)
        assert np.isclose(s[i, j], 1.0, atol=1e-12)
        assert np.isclose(s.sum(), psf.sum(), rtol=1e-12)


def test_convolution_theorem_against_the_operator(setup):
    """``I_D = I * B`` -- the statement the whole project rests on.

    The dirty image of any sky must equal that sky circularly convolved with the
    dirty beam. Verifying it against the independently-computed operator output
    is what makes CLEAN's subtract step provably the right inverse to undo.
    """
    cov, npix, cell = setup
    truth = sky.crescent(npix, cell, 6 * cell, 3 * cell, asymmetry=0.7)
    data = fw.observe(truth, cell, cov, sigma=0.0)
    op, dirty, beam = im.make_images(data, npix, cell, scheme="natural")
    predicted = im.convolve_periodic(truth, beam)
    # the two differ only by the gridding of the exact visibilities onto cells,
    # which is a small effect but not zero
    err = np.abs(predicted - dirty).max() / np.abs(dirty).max()
    assert err < 0.05


def test_zero_spacing_can_be_dropped(setup):
    """With the origin cell emptied, both the beam and the dirty image sum to
    exactly zero: total flux is then unmeasured, as the physics says."""
    cov, npix, cell = setup
    truth = sky.gaussian_blob(npix, cell, 5 * cell)
    data = fw.observe(truth, cell, cov, sigma=0.0)
    op = MeasurementOperator(data, npix, cell, scheme="natural",
                             drop_zero_spacing=True)
    assert abs(im.dirty_beam(op).sum()) < 1e-12
    assert abs(im.dirty_image(op).sum()) < 1e-12
