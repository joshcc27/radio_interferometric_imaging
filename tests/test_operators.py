"""Tests of the measurement operator itself.

The adjoint test is the one that matters most. A wrong adjoint does not make the
optimiser diverge or blow up -- it makes it converge smoothly and confidently to
the wrong answer, with a residual that looks plausible. Nothing else in the
project would catch that.
"""

import numpy as np
import pytest

from src import array as arraymod
from src import forward as fw
from src import imaging as im
from src import sky
from src.forward import ARCSEC, F, Fi, MeasurementOperator


@pytest.fixture(scope="module")
def observation():
    """A modest but genuinely irregular observation, reused across tests."""
    arr = arraymod.vla_y(n_per_arm=5)
    cov = arraymod.uv_track(arr, dec_deg=40.0, ha_hours=(-3.0, 3.0), n_times=25)
    npix = 48
    cell = cov.resolution_rad() / 3.0
    truth = sky.crescent(npix, cell, radius=4.0 * cell, width=2.0 * cell)
    data = fw.observe(truth, cell, cov, snr=25.0,
                      rng=np.random.default_rng(7))
    return truth, data, npix, cell


# ---------------------------------------------------------------------------
# adjoint
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", ["natural", "uniform", "briggs"])
def test_adjoint_identity(observation, scheme):
    """<A x, y> == <x, A^H y> to 1e-10 relative, for random x and y.

    ``A`` maps a real image to a complex visibility vector, so the relevant
    identity is over the real inner product: ``Re<A x, y> = <x, A^H y>``. That is
    exactly the pairing the gradient of the misfit uses.
    """
    _, data, npix, cell = observation
    op = MeasurementOperator(data, npix, cell, scheme=scheme, robust=0.5)
    rng = np.random.default_rng(3)
    for _ in range(5):
        x = rng.standard_normal((npix, npix))
        y = rng.standard_normal(op.n_cells) + 1j * rng.standard_normal(op.n_cells)
        lhs = float(np.real(np.sum(op.forward(x) * np.conj(y))))
        rhs = float(np.sum(x * op.adjoint(y)))
        assert abs(lhs - rhs) <= 1e-10 * max(abs(lhs), abs(rhs))


def test_adjoint_of_gradient_matches_finite_differences(observation):
    """The misfit gradient really is the gradient of the misfit.

    A consistency check one level up from the adjoint test: it would also catch a
    sign error or a missing weight in :meth:`misfit_grad`, which the bilinear
    identity alone cannot.
    """
    _, data, npix, cell = observation
    op = MeasurementOperator(data, npix, cell, scheme="natural")
    rng = np.random.default_rng(11)
    x = np.abs(rng.standard_normal((npix, npix))) * 1e-3
    g = op.misfit_grad(x)
    d = rng.standard_normal((npix, npix))
    d /= np.linalg.norm(d)
    h = 1e-6 * max(np.abs(x).max(), 1e-8)
    num = (op.misfit(x + h * d) - op.misfit(x - h * d)) / (2 * h)
    ana = float(np.sum(g * d))
    assert abs(num - ana) <= 1e-5 * max(abs(num), abs(ana))


# ---------------------------------------------------------------------------
# Fourier conventions
# ---------------------------------------------------------------------------
def test_fft_pair_is_unitary_and_invertible():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((32, 32))
    assert np.abs(Fi(F(x)) - x).max() < 1e-12
    assert np.abs(F(Fi(x)) - x).max() < 1e-12


def test_parseval():
    """Energy is consistent between domains, which fixes the FFT normalisation."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((64, 64))
    assert np.isclose(np.sum(np.abs(F(x)) ** 2), np.sum(x ** 2), rtol=1e-12)


def test_zero_spacing_is_total_flux():
    """``V(0, 0)`` is the total flux -- the normalisation the simulator promises.

    Nothing in a real array measures this point, which is precisely why the total
    flux of a reconstruction is set by the prior rather than the data.
    """
    npix, cell = 64, 1.0 * ARCSEC
    img = sky.crescent(npix, cell, 8 * cell, 3 * cell, total_flux=2.5)
    v = fw.predict_visibilities(img, cell, [0.0], [0.0])
    assert np.isclose(v[0].real, 2.5, rtol=1e-12)
    assert abs(v[0].imag) < 1e-12


def test_centred_point_source_has_unit_visibility_everywhere():
    npix, cell = 64, 1.0 * ARCSEC
    img = sky.point_source(npix, cell, flux=1.0)
    rng = np.random.default_rng(2)
    u = rng.uniform(-5e3, 5e3, 50)
    v = rng.uniform(-5e3, 5e3, 50)
    got = fw.predict_visibilities(img, cell, u, v)
    assert np.abs(got - 1.0).max() < 1e-12


# ---------------------------------------------------------------------------
# Hermitian symmetry
# ---------------------------------------------------------------------------
def test_hermitian_augment_conjugates():
    npix, cell = 32, 2.0 * ARCSEC
    img = sky.gaussian_blob(npix, cell, 6 * cell)
    cov = arraymod.uv_track(arraymod.vla_y(n_per_arm=3), dec_deg=50.0, n_times=5)
    raw = fw.observe(img, cell, cov, sigma=0.0, hermitian=False)
    aug = fw.hermitian_augment(raw)
    n = len(raw)
    assert len(aug) == 2 * n
    assert np.allclose(aug.u[n:], -raw.u)
    assert np.allclose(aug.vis[n:], np.conj(raw.vis))
    # idempotent, so downstream code can call it freely
    assert len(fw.hermitian_augment(aug)) == 2 * n


def test_dirty_image_has_negligible_imaginary_part(observation):
    """The gridded, Hermitian-augmented visibilities invert to a real image.

    :meth:`MeasurementOperator.adjoint` takes the real part, so this test has to
    look at the complex transform before that happens -- otherwise it would pass
    no matter what.
    """
    _, data, npix, cell = observation
    op = MeasurementOperator(data, npix, cell, scheme="natural")
    grid = np.zeros((npix, npix), dtype=complex)
    grid[op.mask] = op.W * op.d
    img = Fi(grid)
    assert np.abs(img.imag).max() < 1e-12 * np.abs(img.real).max()


# ---------------------------------------------------------------------------
# round trip
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["point", "crescent", "square"])
def test_round_trip_full_coverage_no_noise(kind):
    """Sample every cell of the uv grid once, noiselessly, and the dirty image
    must be the truth back to numerical precision.

    Uniform weighting is the right choice here: it equalises the weight per
    occupied cell, so the sampling function is flat and the dirty beam is a delta
    function. Under natural weighting the Hermitian partners double up in every
    cell except the grid edges, the sampling function is not quite flat, and the
    round trip is only approximate -- which is itself worth knowing.
    """
    npix, cell = 32, 1.5 * ARCSEC
    truth = {
        "point": lambda: sky.point_source(npix, cell),
        "crescent": lambda: sky.crescent(npix, cell, 7 * cell, 3 * cell),
        "square": lambda: sky.sharp_square(npix, cell, 8 * cell),
    }[kind]()
    cov = arraymod.full_coverage(npix, cell)
    data = fw.observe(truth, cell, cov, sigma=0.0, hermitian=False)
    op, dirty, beam = im.make_images(data, npix, cell, scheme="uniform")

    assert op.n_cells == npix * npix
    assert np.abs(dirty - truth).max() < 1e-12 * max(truth.max(), 1e-30)
    # and the beam is then a delta function, as a filled aperture demands
    delta = np.zeros((npix, npix))
    delta[npix // 2, npix // 2] = 1.0
    assert np.abs(beam - delta).max() < 1e-12


def test_lipschitz_bounds_the_operator(observation):
    """Power iteration returns a real upper bound on the curvature FISTA steps at."""
    _, data, npix, cell = observation
    op = MeasurementOperator(data, npix, cell, scheme="natural")
    L = op.lipschitz()
    rng = np.random.default_rng(5)
    for _ in range(20):
        x = rng.standard_normal((npix, npix))
        x /= np.linalg.norm(x)
        rayleigh = float(np.sum(x * op.adjoint(op.W * op.forward(x))))
        assert rayleigh <= L * (1.0 + 1e-8)
