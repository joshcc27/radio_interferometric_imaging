"""Posterior sampling, so the answer comes with error bars.

The deliverable here is the *uncertainty map*, not the mean image. Being able to
say that the ring in a reconstructed crescent is solid while some secondary
feature is not is the one thing CLEAN cannot give you at all, and it is the point
of doing this the hard way.

The model
---------
Sample the log brightness, ``theta = log I``, which enforces ``I > 0``
automatically and with no boundary to get stuck against. The posterior is

    -log p(theta | d)  =  0.5 sum_k w_k |A exp(theta) - d_k|^2
                          + 0.5 (theta - mu)^T C^-1 (theta - mu)  + const

with a stationary Gaussian process prior ``C`` on ``theta``: a Whittle-Matern
spectrum ``P(k) ~ (k0^2 + k^2)^-alpha``, parametrised by a correlation length in
pixels and a marginal standard deviation in e-folds of brightness. Because that
prior is stationary it is diagonal in Fourier space, so drawing from it, applying
its inverse and computing its log density are all just FFTs.

Why positivity is what makes the map interesting. If you put a Gaussian prior
directly on ``I`` and drop positivity, the posterior is exactly Gaussian and --
for a stationary prior and a gridded operator -- diagonal in Fourier space, so
every pixel ends up with *identical* variance. A flat uncertainty map is
arithmetic, not information. The structure in a real uncertainty map comes from
the non-Gaussianity, and here that is positivity: a pixel pinned near zero has a
small, one-sided uncertainty, while a bright pixel in a poorly sampled direction
has a large one.

The sampler
-----------
Hamiltonian Monte Carlo, with two choices that matter at these dimensions
(a 64x64 image is 4096 parameters, a 128x128 one 16384):

- **The mass matrix is the prior precision**, ``M = C^-1``. Then the prior part
  of the Hamiltonian flow is an exact rotation, the step size is set by the
  likelihood alone, and the sampler's efficiency stops collapsing as the grid is
  refined. This is the standard construction for Bayesian inverse problems
  (Beskos et al.); with ``M = I`` instead, HMC here is unusable. Both momentum
  sampling and ``M^-1 p`` are FFTs, so it costs nothing.
- **Dual averaging** adapts the step size during warm-up to a target acceptance
  rate, and the trajectory length is jittered to avoid resonances.

This is a hand-written sampler, not NUTS. It is adequate, it is checked against
an analytically known Gaussian posterior in ``tests/test_bayesian.py``, and it
reports acceptance, split R-hat and effective sample size so that a badly mixed
run is visible rather than quietly wrong. Where a run is too expensive, the
answer is a coarser grid -- stated plainly in the results, never hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .forward import F, Fi


# ---------------------------------------------------------------------------
# the Gaussian process prior, diagonal in Fourier space
# ---------------------------------------------------------------------------
class GPPrior:
    """Stationary Gaussian prior on the log image, via its power spectrum.

    ``P(k) = A (k0^2 + k^2)^-alpha`` with ``k`` in cycles per pixel and
    ``k0 = 1 / correlation_length``. ``A`` is fixed so the marginal standard
    deviation of ``theta`` is ``sigma``; since the field is stationary, that
    marginal variance is just the mean of the spectrum, which makes the
    normalisation exact rather than approximate.

    ``sigma`` is in e-folds: ``sigma = 3`` allows brightness to range over about
    ``e^3`` either side of the prior mean, which is a genuinely weak prior on an
    image with a dynamic range of a few hundred.
    """

    def __init__(self, npix, correlation_length=6.0, alpha=1.5, sigma=3.0,
                 mean_log=None):
        self.npix = int(npix)
        self.correlation_length = float(correlation_length)
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        k = (np.arange(self.npix) - self.npix // 2) / float(self.npix)
        kv, ku = np.meshgrid(k, k, indexing="ij")
        k2 = ku ** 2 + kv ** 2
        k0 = 1.0 / max(self.correlation_length, 1e-6)
        raw = (k0 ** 2 + k2) ** (-self.alpha)
        self.power = raw * (self.sigma ** 2 / raw.mean())
        self.mean_log = 0.0 if mean_log is None else float(mean_log)

    # -- the three operations a sampler needs ----------------------------------
    def sample(self, rng):
        """One draw from the prior: white noise shaped by ``sqrt(P)``."""
        w = rng.standard_normal((self.npix, self.npix))
        return self.mean_log + self._mult(w, np.sqrt(self.power))

    def neg_log_density(self, theta):
        """``0.5 (theta - mu)^T C^-1 (theta - mu)``, up to a constant."""
        t = F(theta - self.mean_log)
        return 0.5 * float(np.sum(np.abs(t) ** 2 / self.power))

    def neg_log_density_grad(self, theta):
        """``C^-1 (theta - mu)``."""
        return self._mult(theta - self.mean_log, 1.0 / self.power)

    # -- mass-matrix pieces: M = C^-1 ------------------------------------------
    def sample_momentum(self, rng):
        """Draw ``p ~ N(0, C^-1)``."""
        w = rng.standard_normal((self.npix, self.npix))
        return self._mult(w, 1.0 / np.sqrt(self.power))

    def apply_mass_inverse(self, p):
        """``M^-1 p = C p``."""
        return self._mult(p, self.power)

    def kinetic(self, p):
        return 0.5 * float(np.sum(p * self.apply_mass_inverse(p)))

    @staticmethod
    def _mult(x, multiplier):
        """Apply a real, symmetric Fourier multiplier to a real image.

        The multiplier depends only on ``|k|`` so it preserves the Hermitian
        symmetry of the transform of a real field; the ``.real`` is there to
        discard round-off, and ``tests/test_bayesian.py`` checks the discarded
        part is negligible rather than assuming it.
        """
        return np.real(Fi(F(x) * multiplier))


# ---------------------------------------------------------------------------
# the posterior
# ---------------------------------------------------------------------------
class LogIntensityPosterior:
    """Negative log posterior over ``theta = log I``, and its gradient.

    ``exp`` makes the map from parameters to image nonlinear, so the chain rule
    picks up one factor: with ``I = exp(theta)``,

        d/dtheta [misfit] = exp(theta) * (A^H W (A I - d))

    and the second factor is exactly what the operator already provides as
    :meth:`~src.forward._OperatorBase.misfit_grad`.
    """

    def __init__(self, op, prior, clip=(-40.0, 20.0)):
        self.op = op
        self.prior = prior
        self.clip = clip

    def image(self, theta):
        return np.exp(np.clip(theta, *self.clip))

    def potential(self, theta):
        x = self.image(theta)
        return self.op.misfit(x) + self.prior.neg_log_density(theta)

    def potential_grad(self, theta):
        x = self.image(theta)
        g = x * self.op.misfit_grad(x) + self.prior.neg_log_density_grad(theta)
        return self.op.misfit(x) + self.prior.neg_log_density(theta), g


# ---------------------------------------------------------------------------
# HMC
# ---------------------------------------------------------------------------
@dataclass
class ChainResult:
    """Samples in image space, plus the diagnostics needed to trust them."""

    images: np.ndarray            # (n_draws, npix, npix), Jy/pixel
    theta_last: np.ndarray
    accept_rate: float
    step_size: float
    n_leapfrog: int
    potential: np.ndarray         # per draw, for trace plots
    divergences: int = 0

    @property
    def mean(self):
        return self.images.mean(axis=0)

    @property
    def std(self):
        return self.images.std(axis=0, ddof=1)

    def quantile(self, q):
        return np.quantile(self.images, q, axis=0)


def hmc(posterior, theta0, n_draws=500, n_warmup=500, n_leapfrog=25,
        step_size=None, target_accept=0.75, rng=None, jitter=0.2,
        thin=1, verbose=False, store_images=True):
    """Sample the posterior by Hamiltonian Monte Carlo.

    The mass matrix is the prior precision (see the module docstring), supplied
    by ``posterior.prior``. Warm-up adapts ``step_size`` by dual averaging
    (Hoffman & Gelman 2014, section 3.2.1) towards ``target_accept`` and then
    freezes it, so the sampling phase is a valid Markov chain.
    """
    rng = np.random.default_rng() if rng is None else rng
    prior = posterior.prior
    theta = np.array(theta0, dtype=float)
    U, grad = posterior.potential_grad(theta)

    eps = 1.0 / max(n_leapfrog, 1) if step_size is None else float(step_size)
    # dual-averaging state
    mu_da = np.log(10.0 * eps)
    log_eps_bar, h_bar = np.log(eps), 0.0
    gamma, t0, kappa = 0.05, 10.0, 0.75

    draws, pots = [], []
    n_accept = 0
    n_diverge = 0
    total = int(n_warmup) + int(n_draws) * int(thin)

    for it in range(total):
        warm = it < n_warmup
        p = prior.sample_momentum(rng)
        H0 = U + prior.kinetic(p)

        # jitter the trajectory length: a fixed one can resonate with the
        # posterior's natural periods and stall
        L = max(1, int(round(n_leapfrog * (1.0 + jitter * (2 * rng.random() - 1)))))
        eps_it = eps * (1.0 + 0.1 * (2 * rng.random() - 1))

        th, gr = theta.copy(), grad.copy()
        p = p - 0.5 * eps_it * gr
        for s in range(L):
            th = th + eps_it * prior.apply_mass_inverse(p)
            U_new, gr = posterior.potential_grad(th)
            if not np.isfinite(U_new):
                break
            p = p - (eps_it if s < L - 1 else 0.5 * eps_it) * gr
        else:
            U_new = posterior.potential(th)

        H1 = U_new + prior.kinetic(p)
        dH = H0 - H1
        if not np.isfinite(dH):
            dH = -np.inf
            n_diverge += 1
        a = min(1.0, float(np.exp(min(dH, 0.0))))
        if rng.random() < a:
            theta, U, grad = th, U_new, gr
            n_accept += 1

        if warm:
            # dual averaging on log step size
            k = it + 1
            h_bar = (1.0 - 1.0 / (k + t0)) * h_bar + (target_accept - a) / (k + t0)
            log_eps = mu_da - np.sqrt(k) / gamma * h_bar
            log_eps_bar = (k ** -kappa) * log_eps + (1 - k ** -kappa) * log_eps_bar
            eps = float(np.exp(log_eps))
            if it == n_warmup - 1:
                eps = float(np.exp(log_eps_bar))
                n_accept = 0
        elif (it - n_warmup) % thin == 0:
            pots.append(U)
            if store_images:
                draws.append(posterior.image(theta))

        if verbose and it % 100 == 0:
            print("  it " + str(it) + (" (warmup)" if warm else "")
                  + " U " + format(U, ".4e") + " eps " + format(eps, ".3e")
                  + " acc " + format(a, ".2f"))

    n_sampling = total - n_warmup
    return ChainResult(
        images=np.asarray(draws) if draws else np.empty((0, *theta.shape)),
        theta_last=theta,
        accept_rate=n_accept / max(n_sampling, 1),
        step_size=float(eps), n_leapfrog=int(n_leapfrog),
        potential=np.asarray(pots), divergences=n_diverge)


# ---------------------------------------------------------------------------
# multiple chains and diagnostics
# ---------------------------------------------------------------------------
@dataclass
class PosteriorSummary:
    mean: np.ndarray
    std: np.ndarray
    chains: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    @property
    def images(self):
        return np.concatenate([c.images for c in self.chains], axis=0)

    def quantile(self, q):
        return np.quantile(self.images, q, axis=0)

    def snr_map(self, floor=None):
        """Posterior mean over posterior standard deviation, per pixel.

        The figure worth looking at: it says which features the data insist on
        and which ones the prior is carrying.
        """
        s = self.std
        f = (s[s > 0].min() if floor is None else floor) if np.any(s > 0) else 1.0
        return self.mean / np.maximum(s, f)


def run_chains(op, prior, x0=None, n_chains=4, seed=0, n_draws=400,
               n_warmup=400, n_leapfrog=25, thin=1, verbose=False):
    """Run several independent chains and summarise them.

    Independent chains are the only cheap way to notice that a sampler has not
    explored the posterior: a single chain that has converged to the wrong place
    looks exactly like a single chain that has converged.
    """
    posterior = LogIntensityPosterior(op, prior)
    npix = op.npix
    if x0 is None:
        from .imaging import dirty_image

        x0 = np.clip(dirty_image(op), 0.0, None)
    base = np.log(np.maximum(np.asarray(x0, dtype=float), 1e-10))

    chains = []
    for c in range(int(n_chains)):
        rng = np.random.default_rng(seed + 1000 * c)
        # disperse the starting points, or R-hat cannot detect anything
        start = base + 0.5 * prior.sample(rng) - 0.5 * prior.mean_log
        ch = hmc(posterior, start, n_draws=n_draws, n_warmup=n_warmup,
                 n_leapfrog=n_leapfrog, rng=rng, thin=thin, verbose=verbose)
        chains.append(ch)
        if verbose:
            print(" chain " + str(c) + ": accept " + format(ch.accept_rate, ".2f")
                  + ", step " + format(ch.step_size, ".3e")
                  + ", divergences " + str(ch.divergences))

    allimg = np.concatenate([c.images for c in chains], axis=0)
    diag = {
        "accept_rate": [float(c.accept_rate) for c in chains],
        "step_size": [float(c.step_size) for c in chains],
        "divergences": [int(c.divergences) for c in chains],
        "n_draws_total": int(allimg.shape[0]),
    }
    # scalar summaries are what R-hat and ESS are computed on: a per-pixel R-hat
    # over 16k pixels is mostly noise, and these three are the quantities the
    # results actually quote
    scalars = {
        "total_flux": [c.images.sum(axis=(1, 2)) for c in chains],
        "peak": [c.images.max(axis=(1, 2)) for c in chains],
        "potential": [c.potential for c in chains],
    }
    for name, series in scalars.items():
        diag["rhat_" + name] = float(split_rhat(series))
        diag["ess_" + name] = float(np.sum([ess(s) for s in series]))

    return PosteriorSummary(mean=allimg.mean(axis=0),
                            std=allimg.std(axis=0, ddof=1),
                            chains=chains, diagnostics=diag)


def split_rhat(chains):
    """Split Gelman-Rubin statistic for a list of 1-D chains.

    Each chain is split in half first, so a single chain that drifts is caught
    as well as chains that disagree with each other. Values above about 1.05 mean
    the chains have not mixed and nothing downstream should be believed.
    """
    halves = []
    for c in chains:
        c = np.asarray(c, dtype=float)
        n = len(c) // 2
        if n < 2:
            return np.nan
        halves.append(c[:n])
        halves.append(c[n:2 * n])
    x = np.asarray(halves)
    m, n = x.shape
    means = x.mean(axis=1)
    within = x.var(axis=1, ddof=1).mean()
    between = means.var(ddof=1) * n
    if within <= 0:
        return np.nan
    var_hat = (n - 1) / n * within + between / n
    return float(np.sqrt(var_hat / within))


def ess(x, max_lag=None):
    """Effective sample size from the autocorrelation sum (initial positive
    sequence), which is the standard estimator and needs no extra dependency."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return float(n)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return float(n)
    max_lag = n // 2 if max_lag is None else min(max_lag, n // 2)
    total = 0.0
    for lag in range(1, max_lag):
        r = float(np.dot(x[:-lag], x[lag:])) / denom
        if r <= 0.05:
            break
        total += r
    return float(n / (1.0 + 2.0 * total))
