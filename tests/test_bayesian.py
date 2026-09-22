"""Tests of the posterior sampler.

Sampler bugs are the worst kind: the output always looks like a posterior. So
these tests check the sampler against cases where the answer is known
independently -- the prior on its own, and a linear-Gaussian problem with a
closed-form posterior -- rather than checking that it produces plausible images.
"""

import numpy as np
import pytest

from src import array as arraymod
from src import bayesian as by
from src import forward as fw
from src import metrics as mt
from src import sky
from src.forward import F, Fi, DegridOperator


# ---------------------------------------------------------------------------
# the prior
# ---------------------------------------------------------------------------
def test_prior_marginal_std_matches_its_parameter():
    """Because the field is stationary the marginal variance is the mean of the
    spectrum, so the normalisation is exact and worth asserting exactly."""
    rng = np.random.default_rng(0)
    for sigma in (0.5, 2.0):
        pr = by.GPPrior(32, correlation_length=5.0, sigma=sigma)
        draws = np.array([pr.sample(rng) for _ in range(300)])
        assert np.isclose(draws.std(), sigma, rtol=0.05)


def test_prior_samples_are_real_and_centred():
    pr = by.GPPrior(24, correlation_length=4.0, sigma=1.0, mean_log=-7.0)
    rng = np.random.default_rng(1)
    draws = np.array([pr.sample(rng) for _ in range(200)])
    assert draws.dtype == np.float64
    assert np.isclose(draws.mean(), -7.0, atol=0.1)


def test_fourier_multiplier_keeps_the_field_real():
    """A real, radially symmetric multiplier preserves Hermitian symmetry, so the
    discarded imaginary part must be round-off and not signal."""
    pr = by.GPPrior(32, correlation_length=3.0, sigma=1.5)
    rng = np.random.default_rng(2)
    x = rng.standard_normal((32, 32))
    full = Fi(F(x) * pr.power)
    assert np.abs(full.imag).max() < 1e-10 * np.abs(full.real).max()


def test_prior_gradient_matches_finite_differences():
    pr = by.GPPrior(16, correlation_length=3.0, sigma=1.2, mean_log=-5.0)
    rng = np.random.default_rng(3)
    theta = pr.sample(rng)
    g = pr.neg_log_density_grad(theta)
    for _ in range(4):
        d = rng.standard_normal(theta.shape)
        h = 1e-6
        num = (pr.neg_log_density(theta + h * d)
               - pr.neg_log_density(theta - h * d)) / (2 * h)
        assert np.isclose(num, float(np.sum(g * d)), rtol=1e-5)


def test_prior_correlation_length_does_what_it_says():
    """A longer correlation length must make neighbouring pixels more alike."""
    rng = np.random.default_rng(4)
    corrs = []
    for length in (2.0, 8.0):
        pr = by.GPPrior(32, correlation_length=length, sigma=1.0)
        d = np.array([pr.sample(rng) for _ in range(200)])
        a = d[:, 16, 16]
        b = d[:, 16, 20]      # four pixels away
        corrs.append(float(np.corrcoef(a, b)[0, 1]))
    assert corrs[1] > corrs[0]


def test_mass_matrix_is_the_prior_precision():
    """``E[kinetic] = dim/2`` for ``p ~ N(0, M)`` and ``K = 0.5 p M^-1 p``,
    whatever ``M`` is -- which checks that momentum sampling and the kinetic
    energy use consistent operators. Get these out of step and HMC still runs,
    accepts, and samples the wrong distribution."""
    n = 24
    pr = by.GPPrior(n, correlation_length=4.0, sigma=2.0)
    rng = np.random.default_rng(5)
    k = np.array([pr.kinetic(pr.sample_momentum(rng)) for _ in range(400)])
    assert np.isclose(k.mean(), 0.5 * n * n, rtol=0.08)


# ---------------------------------------------------------------------------
# the posterior gradient
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_problem():
    arr = arraymod.vla_y(n_per_arm=4)
    cov = arraymod.uv_track(arr, dec_deg=45.0, ha_hours=(-2.0, 2.0), n_times=12)
    npix = 24
    cell = cov.resolution_rad() / 3.0
    truth = sky.gaussian_blob(npix, cell, 4 * cell)
    data = fw.observe(truth, cell, cov, snr=15.0, rng=np.random.default_rng(6))
    op = DegridOperator(data, npix, cell, scheme="natural")
    prior = by.GPPrior(npix, correlation_length=3.0, sigma=2.0,
                       mean_log=float(np.log(1.0 / npix ** 2)))
    return truth, op, prior


def test_posterior_gradient_matches_finite_differences(small_problem):
    """The ``exp`` reparametrisation puts a chain-rule factor in the gradient;
    drop it and HMC samples a subtly wrong distribution while looking healthy."""
    truth, op, prior = small_problem
    post = by.LogIntensityPosterior(op, prior)
    rng = np.random.default_rng(7)
    theta = prior.sample(rng)
    U, g = post.potential_grad(theta)
    assert np.isclose(U, post.potential(theta), rtol=1e-12)
    for _ in range(4):
        d = rng.standard_normal(theta.shape)
        d /= np.linalg.norm(d)
        h = 1e-6
        num = (post.potential(theta + h * d) - post.potential(theta - h * d)) / (2 * h)
        assert np.isclose(num, float(np.sum(g * d)), rtol=2e-4)


def test_posterior_images_are_strictly_positive(small_problem):
    truth, op, prior = small_problem
    post = by.LogIntensityPosterior(op, prior)
    rng = np.random.default_rng(8)
    assert post.image(prior.sample(rng)).min() > 0.0


# ---------------------------------------------------------------------------
# the sampler, against known answers
# ---------------------------------------------------------------------------
class _NoData:
    """A zero-weight operator, so the posterior is exactly the prior."""

    def __init__(self, npix):
        self.npix = npix
        self.cell = 1.0

    def misfit(self, x):
        return 0.0

    def misfit_grad(self, x):
        return np.zeros_like(x)


def test_hmc_recovers_the_prior_when_there_is_no_data():
    """The strongest available check on the sampler.

    With no likelihood the posterior *is* the prior, and the prior's mean,
    marginal variance and spatial covariance function are all known in closed
    form from its power spectrum. Any error in the leapfrog, the mass matrix, the
    accept step or the adaptation shows up here.
    """
    n = 20
    pr = by.GPPrior(n, correlation_length=4.0, sigma=1.5, mean_log=-2.0)
    post = by.LogIntensityPosterior(_NoData(n), pr)
    rng = np.random.default_rng(9)
    ch = by.hmc(post, pr.sample(rng), n_draws=2500, n_warmup=700,
                n_leapfrog=12, rng=rng)
    theta = np.log(ch.images)

    assert 0.5 < ch.accept_rate < 0.98
    assert ch.divergences == 0
    assert np.isclose(theta.mean(), pr.mean_log, atol=0.12)
    assert np.isclose(theta.std(axis=0).mean(), pr.sigma, rtol=0.08)

    # the covariance function, C[0, r], is the inverse transform of the spectrum
    exact = np.real(np.fft.ifft2(np.fft.ifftshift(pr.power)))
    centred = theta - theta.mean()
    for lag in (1, 2):
        emp = float(np.mean(centred[:, 0, 0] * centred[:, 0, lag]))
        assert np.isclose(emp, exact[0, lag], atol=0.15 * pr.sigma ** 2)


def test_hmc_concentrates_when_the_data_are_informative(small_problem):
    """With real data the posterior must be much tighter than the prior and must
    sit near the truth -- otherwise the likelihood is not being used."""
    truth, op, prior = small_problem
    post = by.LogIntensityPosterior(op, prior)
    rng = np.random.default_rng(10)
    ch = by.hmc(post, np.full(truth.shape, prior.mean_log), n_draws=600,
                n_warmup=600, n_leapfrog=20, rng=rng)
    assert ch.accept_rate > 0.4
    theta = np.log(ch.images)
    # posterior width below the prior width: the data have said something
    assert theta.std(axis=0).mean() < 0.8 * prior.sigma
    # and the mean image fits the data
    assert op.reduced_chi2(ch.mean) < 5.0


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------
def test_split_rhat_is_near_one_for_independent_draws():
    rng = np.random.default_rng(11)
    chains = [rng.standard_normal(2000) for _ in range(4)]
    assert abs(by.split_rhat(chains) - 1.0) < 0.05


def test_split_rhat_detects_chains_that_disagree():
    rng = np.random.default_rng(12)
    chains = [rng.standard_normal(2000) + offset for offset in (0.0, 3.0)]
    assert by.split_rhat(chains) > 1.5


def test_split_rhat_detects_a_drifting_single_chain():
    """Splitting each chain in half is what catches this; an unsplit R-hat on one
    chain cannot."""
    drift = np.linspace(0.0, 6.0, 2000) + np.random.default_rng(13).standard_normal(2000)
    assert by.split_rhat([drift]) > 1.3


def test_ess_is_near_n_for_independent_draws_and_small_for_correlated_ones():
    rng = np.random.default_rng(14)
    indep = rng.standard_normal(4000)
    assert by.ess(indep) > 0.7 * len(indep)

    # an AR(1) chain with rho = 0.9 has ESS ~ n (1-rho)/(1+rho) ~ n/19
    ar = np.zeros(4000)
    for i in range(1, len(ar)):
        ar[i] = 0.9 * ar[i - 1] + rng.standard_normal()
    assert by.ess(ar) < 0.2 * len(ar)


def test_run_chains_reports_usable_diagnostics(small_problem):
    truth, op, prior = small_problem
    summary = by.run_chains(op, prior, n_chains=2, n_draws=200, n_warmup=300,
                            n_leapfrog=15, seed=3)
    d = summary.diagnostics
    for key in ("accept_rate", "rhat_total_flux", "ess_total_flux",
                "rhat_potential"):
        assert key in d
    assert summary.mean.shape == truth.shape
    assert summary.std.shape == truth.shape
    assert np.all(summary.std >= 0.0)
    assert summary.mean.min() > 0.0
    # the credible-interval machinery has to accept these arrays
    cov = mt.credible_interval_coverage(truth, summary.mean, summary.std, 1.0)
    assert 0.0 <= cov <= 1.0


def test_snr_map_is_finite_everywhere(small_problem):
    truth, op, prior = small_problem
    summary = by.run_chains(op, prior, n_chains=1, n_draws=120, n_warmup=200,
                            n_leapfrog=12, seed=4)
    assert np.all(np.isfinite(summary.snr_map()))
