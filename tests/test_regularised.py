"""Tests of the regularised solver: the operators it needs, the proxes, FISTA."""

import numpy as np
import pytest

from src import array as arraymod
from src import forward as fw
from src import metrics as mt
from src import regularised as rg
from src import sky
from src.forward import DegridOperator, MeasurementOperator


@pytest.fixture(scope="module")
def problem():
    arr = arraymod.vla_y(n_per_arm=6)
    cov = arraymod.uv_track(arr, dec_deg=45.0, ha_hours=(-3.0, 3.0), n_times=25)
    npix = 48
    cell = cov.resolution_rad() / 3.0
    truth = sky.crescent(npix, cell, radius=2.0 * cov.resolution_rad(),
                         width=1.0 * cov.resolution_rad())
    data = fw.observe(truth, cell, cov, snr=20.0, rng=np.random.default_rng(4))
    op = DegridOperator(data, npix, cell, scheme="natural")
    return truth, op


# ---------------------------------------------------------------------------
# finite differences
# ---------------------------------------------------------------------------
def test_grad_and_div_are_an_exact_adjoint_pair():
    """``<grad u, p> == -<u, div p>``.

    The same class of bug as a wrong measurement adjoint, one level down: get
    this wrong and the TV prox minimises a slightly different functional, quite
    smoothly, without ever complaining.
    """
    rng = np.random.default_rng(0)
    for n in (8, 17):
        u = rng.standard_normal((n, n))
        px = rng.standard_normal((n, n))
        py = rng.standard_normal((n, n))
        gx, gy = rg.grad2(u)
        lhs = float(np.sum(gx * px) + np.sum(gy * py))
        rhs = -float(np.sum(u * rg.div2(px, py)))
        assert abs(lhs - rhs) <= 1e-12 * max(abs(lhs), abs(rhs))


def test_total_variation_values():
    """TV of a constant image is zero; of a step, the length of the step."""
    x = np.full((10, 10), 3.0)
    tv = rg.TotalVariation()
    assert tv.value(x) == 0.0
    y = np.zeros((10, 10))
    y[:, 5:] = 2.0
    assert np.isclose(tv.value(y), 2.0 * 10)


def test_total_variation_is_blind_to_a_constant_offset():
    """``TV(I + c) = TV(I)``: constants are in TV's null space.

    This is the mechanism behind the project's most surprising result. A flat
    pedestal costs total variation nothing, so TV cannot determine total flux;
    only the zero-spacing visibility can, and no baseline measures it. Lose the
    short spacings and the total flux is constrained by neither the data nor the
    penalty.
    """
    rng = np.random.default_rng(20)
    x = np.abs(rng.standard_normal((16, 16)))
    tv = rg.TotalVariation()
    base = tv.value(x)
    for c in (0.1, 5.0, 1e3):
        assert np.isclose(tv.value(x + c), base, rtol=1e-12)


def test_flux_penalty_removes_the_degeneracy():
    """The one-line fix: with a flux term the penalty is no longer flat along
    the constant direction, so there is a unique minimiser again."""
    rng = np.random.default_rng(21)
    x = np.abs(rng.standard_normal((16, 16)))
    fixed = rg.TotalVariation(flux_penalty=0.02)
    assert fixed.name == "tv_flux"
    base = fixed.value(x)
    assert fixed.value(x + 0.1) > base
    assert fixed.value(x + 5.0) > fixed.value(x + 0.1)


def test_flux_penalty_prox_shrinks_total_flux():
    rng = np.random.default_rng(22)
    v = np.abs(rng.standard_normal((24, 24))) + 1.0
    plain = rg.TotalVariation(n_inner=200).prox(v, 0.1)
    fixed = rg.TotalVariation(n_inner=200, flux_penalty=0.5).prox(v, 0.1)
    assert fixed.sum() < plain.sum()
    assert fixed.min() >= 0.0


def test_pure_tv_invents_flux_when_the_short_spacings_are_gone():
    """The behavioural version of the result, on a deliberately sparse array.

    With the three arm tips pinned and only five antennas the shortest baseline
    exceeds half a uv cell, so nothing stands in for the zero spacing. Pure TV
    then puts in excess total flux as a near-flat pedestal, and the flux penalty
    pins it back down.

    The assertions compare the two solvers against each other rather than
    against a fixed number, because the size of the effect is not universal: the
    unconstrained mode is a pedestal over the *whole field*, so the excess flux
    scales with the number of pixels. On this 64-pixel grid pure TV invents about
    1.3x the true flux; on the 128-pixel grid the experiments use, 6x.
    """
    from src import array as am
    from src.experiments import config as cfg, pipeline as pp

    setup = cfg.vla_setup(npix=64)
    truth = cfg.truth_for(setup, "crescent")
    thinned = am.thin(setup.array, 5, seed=17, pin=am.outermost(setup.array, 3))
    cov = setup.uv_track(thinned)
    obs = pp.observe(setup, truth, coverage=cov, seed=0)

    # no short spacing lands in the origin cell, so sum(psf) is exactly zero and
    # the dirty image carries no information about total flux at all
    assert abs(obs.psf.sum()) < 1e-9

    plain = rg.solve_at_chi2(obs.acc_op, kind="tv", n_iter=200, n_bisect=5)
    fixed = rg.solve_at_chi2(obs.acc_op, kind="tv_flux", n_iter=200, n_bisect=5)
    assert plain.image.sum() > 1.15                     # truth is 1.0
    assert fixed.image.sum() < plain.image.sum() - 0.1  # the penalty bites
    assert mt.nrmse(fixed.image, truth) < mt.nrmse(plain.image, truth)


# ---------------------------------------------------------------------------
# proxes
# ---------------------------------------------------------------------------
def test_tv_prox_solves_its_own_minimisation():
    """Check the Chambolle-Pock inner solve against the objective it claims to
    minimise, using a reference optimiser on a small grid."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(1)
    v = rng.standard_normal((8, 8))
    tv = rg.TotalVariation(n_inner=600)
    for t in (0.05, 0.3):
        got = tv.prox(v, t)
        assert got.min() >= 0.0

        def obj(z):
            z = z.reshape(8, 8)
            return 0.5 * np.sum((z - v) ** 2) + t * tv.value(z)

        ref = minimize(obj, np.clip(v, 0, None).ravel(), method="L-BFGS-B",
                       bounds=[(0, None)] * v.size,
                       options=dict(maxiter=4000, ftol=1e-16))
        # Both optimisers are approximate here: primal-dual converges sublinearly
        # on a non-smooth objective, and L-BFGS-B is not built for one at all
        # (at t = 0.05 the prox beats it outright). So compare objective values
        # with a relative tolerance rather than demanding strict dominance.
        assert obj(got.ravel()) <= abs(ref.fun) * (1.0 + 1e-4)


def test_tv_prox_is_the_identity_at_zero_strength():
    rng = np.random.default_rng(2)
    v = np.abs(rng.standard_normal((12, 12)))
    tv = rg.TotalVariation()
    assert np.allclose(tv.prox(v, 0.0), v)


def test_tv_prox_flattens_as_strength_grows():
    rng = np.random.default_rng(3)
    v = np.abs(rng.standard_normal((16, 16)))
    tv = rg.TotalVariation(n_inner=200)
    vals = [tv.value(tv.prox(v, t)) for t in (0.01, 0.1, 1.0)]
    assert vals[0] > vals[1] > vals[2]


def test_l1_prox_soft_thresholds():
    l1 = rg.L1Sparsity()
    v = np.array([[-1.0, 0.05, 0.5]])
    got = l1.prox(v, 0.1, positivity=True)
    assert np.allclose(got, [[0.0, 0.0, 0.4]])
    got = l1.prox(v, 0.1, positivity=False)
    assert np.allclose(got, [[-0.9, 0.0, 0.4]])


def test_entropy_gradient_matches_finite_differences():
    rng = np.random.default_rng(5)
    x = np.abs(rng.standard_normal((6, 6))) + 0.1
    mem = rg.MaximumEntropy()
    g = mem.grad(x)
    for _ in range(5):
        d = rng.standard_normal(x.shape)
        h = 1e-7
        num = (mem.value(x + h * d) - mem.value(x - h * d)) / (2 * h)
        assert np.isclose(num, float(np.sum(g * d)), rtol=1e-5)


def test_entropy_is_minimised_by_the_prior_image():
    """``sum I log(I/m)`` at fixed total flux is smallest when ``I`` matches the
    prior -- which is exactly why MEM produces featureless images where the data
    do not object."""
    npix = 12
    mem = rg.MaximumEntropy()
    flat = np.full((npix, npix), 1.0 / npix ** 2)
    assert np.isclose(mem.value(flat), 0.0, atol=1e-12)
    peaked = np.zeros((npix, npix))
    peaked[6, 6] = 1.0
    assert mem.value(peaked) > mem.value(flat)


# ---------------------------------------------------------------------------
# the degridding operator
# ---------------------------------------------------------------------------
def test_degrid_adjoint_identity(problem):
    truth, op = problem
    rng = np.random.default_rng(6)
    for _ in range(5):
        x = rng.standard_normal((op.npix, op.npix))
        y = rng.standard_normal(op.n_data) + 1j * rng.standard_normal(op.n_data)
        lhs = float(np.real(np.sum(op.forward(x) * np.conj(y))))
        rhs = float(np.sum(x * op.adjoint(y)))
        assert abs(lhs - rhs) <= 1e-10 * max(abs(lhs), abs(rhs))


def test_degrid_reproduces_the_exact_transform(problem):
    """The reason this operator exists: its model error must be far below the
    noise, where the nearest-cell operator's is comparable to it."""
    truth, op = problem
    exact = fw.predict_visibilities(truth, op.cell, op.data.u, op.data.v)
    err = np.abs(op.forward(truth) - exact).max() / np.abs(exact).max()
    assert err < 1e-3

    grid = MeasurementOperator(op.data, op.npix, op.cell, scheme="natural")
    grid_err = np.abs(grid.forward(truth) - grid.d).max() / np.abs(exact).max()
    assert grid_err > 10 * err


def test_degrid_of_a_centred_point_source_is_exactly_one(problem):
    truth, op = problem
    pt = sky.point_source(op.npix, op.cell, flux=1.0)
    assert np.abs(op.forward(pt) - 1.0).max() < 1e-12


def test_dropping_conjugates_changes_nothing_measurable(problem):
    """The redundant Hermitian half costs a factor of two and buys nothing."""
    truth, op = problem
    full = DegridOperator(op.data, op.npix, op.cell, scheme="natural",
                          deduplicate_hermitian=False)
    assert np.isclose(op.reduced_chi2(truth), full.reduced_chi2(truth), rtol=1e-12)
    ga, gb = op.misfit_grad(truth), full.misfit_grad(truth)
    cos = np.sum(ga * gb) / (np.linalg.norm(ga) * np.linalg.norm(gb))
    assert np.isclose(cos, 1.0, atol=1e-12)


def test_reduced_chi2_of_the_truth_is_about_one(problem):
    """The honest end-to-end check on the noise model, the weights and the
    chi-squared normalisation all at once."""
    truth, op = problem
    assert 0.8 < op.reduced_chi2(truth) < 1.25


def test_reduced_chi2_normalisation_on_pure_noise():
    """A zero model against noise-only data must give reduced chi-squared 1."""
    arr = arraymod.vla_y(n_per_arm=5)
    cov = arraymod.uv_track(arr, dec_deg=45.0, n_times=20)
    npix, cell = 48, cov.resolution_rad() / 3.0
    n = len(cov)
    sigma = np.full(n, 0.02)
    vals = []
    for seed in range(6):
        noise = fw.add_thermal_noise(np.zeros(n, dtype=complex), sigma,
                                     np.random.default_rng(seed))
        data = fw.hermitian_augment(
            fw.Visibilities(u=cov.u, v=cov.v, vis=noise, sigma=sigma))
        for op in (DegridOperator(data, npix, cell, scheme="natural"),
                   MeasurementOperator(data, npix, cell, scheme="natural")):
            vals.append(op.reduced_chi2(np.zeros((npix, npix))))
    assert np.isclose(np.mean(vals), 1.0, rtol=0.1)


# ---------------------------------------------------------------------------
# FISTA
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["positivity", "tv", "mem", "l1"])
def test_output_is_non_negative(problem, kind):
    """Positivity is a hard constraint, not a soft preference."""
    truth, op = problem
    res = rg.solve(op, kind=kind, lam=1e3, n_iter=60)
    assert res.image.min() >= 0.0


@pytest.mark.parametrize("kind", ["positivity", "tv", "mem"])
def test_objective_decreases_monotonically(problem, kind):
    """Adaptive restart is supposed to make FISTA monotone here; if the objective
    ever rises, either the prox or the step rule is wrong."""
    truth, op = problem
    res = rg.solve(op, kind=kind, lam=1e3, n_iter=80)
    obj = res.objective
    assert np.all(np.diff(obj) <= 1e-9 * np.abs(obj[:-1]))


def test_solving_beats_the_dirty_image(problem):
    """The whole point: deconvolution has to help."""
    from src.imaging import dirty_image

    truth, op = problem
    grid = MeasurementOperator(op.data, op.npix, op.cell, scheme="natural")
    dirty = np.clip(dirty_image(grid), 0.0, None)
    res = rg.solve_at_chi2(op, kind="tv", target_chi2=1.0, n_iter=120, n_bisect=5)
    assert mt.nrmse(res.image, truth) < mt.nrmse(dirty, truth)
    assert mt.ssim(res.image, truth) > mt.ssim(dirty, truth)


def test_lambda_bisection_lands_near_the_target_chi2(problem):
    """The lambda choice has to be reproducible and truth-free, or the
    comparison between regularisers means nothing."""
    truth, op = problem
    for kind in ("tv", "mem"):
        res = rg.solve_at_chi2(op, kind=kind, target_chi2=1.0, n_iter=150,
                               n_bisect=7)
        assert 0.9 < res.final_chi2 < 1.1
        assert res.lam > 0.0


def test_more_regularisation_means_a_worse_fit(problem):
    """Monotonicity of chi-squared in lambda, which is what makes the bisection
    in :func:`solve_at_chi2` legitimate."""
    truth, op = problem
    chi2 = [rg.solve(op, kind="tv", lam=lam, n_iter=250).final_chi2
            for lam in (1e2, 1e4, 1e6)]
    assert chi2[0] < chi2[1] < chi2[2]


def test_tv_and_mem_disagree_on_the_same_data(problem):
    """The central claim of the project, as an assertion.

    Two solutions that fit the same data equally well, to the same reduced
    chi-squared, are still visibly different images -- and the difference is the
    prior, not the data.
    """
    truth, op = problem
    tv = rg.solve_at_chi2(op, kind="tv", target_chi2=1.0, n_iter=150, n_bisect=6)
    mem = rg.solve_at_chi2(op, kind="mem", target_chi2=1.0, n_iter=150, n_bisect=6)
    assert abs(tv.final_chi2 - mem.final_chi2) < 0.15
    rel = (np.abs(tv.image - mem.image).max()
           / max(tv.image.max(), mem.image.max()))
    assert rel > 0.05
