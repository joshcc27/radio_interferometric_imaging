"""Regularised inversion: FISTA, total variation, maximum entropy.

The problem solved here is

    I_hat = argmin_I  0.5 * sum_c W_c |(A I)_c - d_c|^2  +  lam * R(I),
            subject to I >= 0

with ``A`` the gridded measurement operator from :mod:`src.forward`. Because
``A`` is an FFT and a mask, both the gradient and the adjoint are cheap and
exact, which is what makes a first-order method the right choice.

The three ingredients, and why each is here:

- **Positivity alone** is a surprisingly strong regulariser. Brightness cannot
  be negative, and enforcing that couples the unmeasured Fourier coefficients to
  the measured ones. :func:`solve` with ``lam=0`` isolates its effect, and it is
  worth looking at on its own before adding anything else.
- **Total variation**, ``R(I) = sum sqrt(|grad I|^2)``, penalises the total
  amount of change. It preserves edges and produces piecewise-flat images,
  because moving an edge costs nothing but blurring it costs something.
- **Maximum entropy**, ``R(I) = sum I log(I/m)`` against a prior image ``m``,
  is the classical radio choice. It is smooth, it forces positivity by itself,
  and it produces smooth positive images -- so it cannot make a sharp edge, ever,
  which is the honesty test in :mod:`src.experiments.honesty`.

The interesting comparison is not which one wins. It is that TV and MEM produce
visibly different images from *identical data*, and the difference is the prior
made visible. Where the data are good they agree; where the data run out they
diverge, and the divergence is the prior.

Choosing lambda
---------------
A regularised image with a hand-tuned ``lam`` proves nothing, because ``lam``
can be tuned until the answer looks like whatever you expected. :func:`solve_at_chi2`
instead picks ``lam`` so the solution fits the data to a target reduced
chi-squared of about 1 -- i.e. as well as the noise allows and no better. That is
a defensible, truth-free criterion, it is what the classical MEM literature does,
and it is what the coverage sweep uses so that every point on the curve is
chosen the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# finite differences: grad and its exact negative adjoint
# ---------------------------------------------------------------------------
def grad2(x):
    """Forward differences with Neumann (zero-flux) boundaries.

    Returns ``(gx, gy)`` where ``gx`` differences along axis 1 (l) and ``gy``
    along axis 0 (m).
    """
    gx = np.zeros_like(x)
    gy = np.zeros_like(x)
    gx[:, :-1] = x[:, 1:] - x[:, :-1]
    gy[:-1, :] = x[1:, :] - x[:-1, :]
    return gx, gy


def div2(px, py):
    """Discrete divergence: exactly ``-grad2^T``.

    Checked against the gradient by an adjoint test in
    ``tests/test_regularised.py``. If these two are not an exact adjoint pair the
    TV prox solves a slightly different problem than the one advertised and
    converges to the wrong image without complaint -- the same failure mode as a
    wrong measurement adjoint, one level down.
    """
    out = np.zeros_like(px)
    out[:, :-1] += px[:, :-1]
    out[:, 1:] -= px[:, :-1]
    out[:-1, :] += py[:-1, :]
    out[1:, :] -= py[:-1, :]
    return out


# ---------------------------------------------------------------------------
# regularisers
# ---------------------------------------------------------------------------
class Regulariser:
    """Base class. A regulariser is either smooth (provides :meth:`grad`) or
    prox-able (provides :meth:`prox`); FISTA dispatches on ``is_smooth``."""

    is_smooth = False
    name = "none"

    def value(self, x):
        return 0.0

    def grad(self, x):
        raise NotImplementedError

    def prox(self, v, t, positivity=True, floor=0.0):
        raise NotImplementedError


class NoRegulariser(Regulariser):
    """Positivity only: the data plus ``I >= 0`` and nothing else."""

    name = "positivity"

    def prox(self, v, t, positivity=True, floor=0.0):
        return np.clip(v, floor, None) if positivity else v


class TotalVariation(Regulariser):
    """Isotropic total variation, with the prox solved by Chambolle-Pock.

    ``prox`` solves ``argmin_{u >= 0} 0.5||u - v||^2 + t * TV(u)``. The positivity
    constraint is handled *inside* the inner solve rather than by projecting
    afterwards: projecting after a TV prox does not solve the constrained
    problem, and the difference shows up as a faint positive bias in the
    background.

    **Total variation has a null space, and it matters.** ``TV(I + c) = TV(I)``
    for any constant ``c``: a flat pedestal costs nothing. So TV cannot
    distinguish images that differ by a constant, and it falls to the *data* to
    do so. The data can only do that through the zero-spacing visibility, which
    is the total flux -- and no baseline measures it. In practice the shortest
    baselines land inside the origin uv cell and stand in for it, but only while
    ``min|uv| < du/2 = 1/(2 npix cell)``. Lose the short spacings and that stops
    being true: the origin cell empties, ``sum(psf)`` becomes exactly zero, and
    the total flux is then constrained by neither the data nor the penalty. The
    coverage sweep walks straight across that threshold, and pure TV responds by
    inventing a pedestal -- six times the true total flux at the sparse end.

    ``flux_penalty`` breaks the degeneracy by adding ``eta * sum(I)`` to the
    penalty, which picks the minimum-flux image among those TV cannot tell
    apart. It costs one line in the inner solve, because with positivity the
    primal step is just a shift: ``u = max(z - tau*t*eta, 0)``. It is also a
    prior choice and has to be declared as one -- "assume the least flux
    consistent with the data" is an assumption, not a measurement. Compare the
    ``tv`` and ``tv_flux`` curves in the sweep to see what it buys and what it
    costs.

    The saddle-point form is

        min_{u>=0} max_{|p| <= t}  0.5||u - v||^2 + <grad u, p>

    and the inner iteration is the standard primal-dual one with steps
    ``sigma = tau = 1/sqrt(8)``, since ``||grad|| <= sqrt(8)``.

    ``n_inner`` defaults to 40, which leaves the prox *inexact*. That is
    deliberate and normal: the outer FISTA loop re-solves it hundreds of times
    from a nearby point, so paying for a tight inner solve buys almost nothing.
    ``tests/test_regularised.py`` checks a tight one against an independent
    optimiser to confirm the prox is solving the advertised problem.
    """

    name = "tv"

    def __init__(self, n_inner=40, tol=1e-6, flux_penalty=0.0):
        self.n_inner = int(n_inner)
        self.tol = float(tol)
        self.flux_penalty = float(flux_penalty)
        if flux_penalty:
            self.name = "tv_flux"

    def value(self, x):
        gx, gy = grad2(x)
        tv = float(np.sum(np.sqrt(gx ** 2 + gy ** 2)))
        return tv + self.flux_penalty * float(np.sum(np.abs(x)))

    def prox(self, v, t, positivity=True, floor=0.0):
        if t <= 0:
            return np.clip(v, floor, None) if positivity else v
        step = 1.0 / np.sqrt(8.0)
        shift = step * t * self.flux_penalty   # the flux term's primal step
        u = np.clip(v, floor, None) if positivity else v.copy()
        ubar = u.copy()
        px = np.zeros_like(v)
        py = np.zeros_like(v)
        for _ in range(self.n_inner):
            gx, gy = grad2(ubar)
            px = px + step * gx
            py = py + step * gy
            norm = np.maximum(1.0, np.sqrt(px ** 2 + py ** 2) / t)
            px /= norm
            py /= norm
            u_new = (u + step * div2(px, py) + step * v - shift) / (1.0 + step)
            if positivity:
                u_new = np.clip(u_new, floor, None)
            delta = np.linalg.norm(u_new - u)
            ubar = 2.0 * u_new - u
            u = u_new
            if delta <= self.tol * max(np.linalg.norm(u), 1e-30):
                break
        return u


class MaximumEntropy(Regulariser):
    """``R(I) = sum_j I_j log(I_j / m_j)``, the classical radio regulariser.

    Smooth, so FISTA carries it in the gradient and the prox is just the
    positivity projection. The gradient ``log(I/m) + 1`` diverges as ``I -> 0``,
    so the projection uses a small strictly positive floor; that is a numerical
    device, not a prior choice, and the floor is orders of magnitude below any
    real brightness.

    ``m`` is the prior image -- what you believe before seeing data. A flat ``m``
    says "featureless", and then entropy maximisation pushes towards the flattest
    image compatible with the data. The classical form also constrains total
    flux to a measured value; there is no such measurement here (see the note on
    the zero spacing in :mod:`src.clean`), so it is omitted and the flux is left
    to the data plus the penalty.
    """

    is_smooth = True
    name = "mem"

    def __init__(self, prior=None, floor=1e-12):
        self.prior = prior
        self.floor = float(floor)

    def _m(self, x):
        if self.prior is None:
            return np.full(x.shape, 1.0 / x.size)
        m = np.asarray(self.prior, dtype=float)
        return np.maximum(m, self.floor)

    def value(self, x):
        m = self._m(x)
        xs = np.maximum(x, self.floor)
        return float(np.sum(xs * np.log(xs / m)))

    def grad(self, x):
        m = self._m(x)
        xs = np.maximum(x, self.floor)
        return np.log(xs / m) + 1.0

    def prox(self, v, t, positivity=True, floor=None):
        f = self.floor if floor is None else floor
        return np.clip(v, f, None)


class L1Sparsity(Regulariser):
    """``R(I) = sum |I|`` -- sparsity in the pixel basis, by soft thresholding.

    With positivity also imposed this is a pure flux penalty, which shrinks
    every pixel towards zero and produces the sparsest image that still fits.
    It is the closest explicit analogue of CLEAN's implicit point-source prior,
    which makes it a useful third point of comparison even though it is the least
    physically motivated of the three.
    """

    name = "l1"

    def value(self, x):
        return float(np.sum(np.abs(x)))

    def prox(self, v, t, positivity=True, floor=0.0):
        if positivity:
            return np.clip(v - t, floor, None)
        return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


def _tv_with_flux():
    """TV plus a small flux penalty, which breaks TV's constant-offset
    degeneracy. ``eta = 0.02`` is small enough to leave the reconstruction alone
    where the short spacings exist, and enough to pin the pedestal where they do
    not; see :class:`TotalVariation`."""
    return TotalVariation(flux_penalty=0.02)


REGULARISERS = {
    "positivity": NoRegulariser,
    "tv": TotalVariation,
    "tv_flux": _tv_with_flux,
    "mem": MaximumEntropy,
    "l1": L1Sparsity,
}


# ---------------------------------------------------------------------------
# FISTA
# ---------------------------------------------------------------------------
@dataclass
class SolveResult:
    image: np.ndarray
    objective: np.ndarray = None
    reduced_chi2: np.ndarray = None
    n_iter: int = 0
    lam: float = 0.0
    regulariser: str = ""
    converged: bool = False
    step: float = 0.0
    history: dict = field(default_factory=dict)

    @property
    def final_chi2(self):
        return float(self.reduced_chi2[-1]) if len(self.reduced_chi2) else np.nan


def fista(op, reg=None, lam=0.0, x0=None, n_iter=400, positivity=True,
          tol=1e-7, backtrack=True, eta=0.5, restart=True, verbose=False,
          callback=None):
    """Minimise ``misfit + lam * R`` subject to positivity, by FISTA.

    Backtracking (Beck & Teboulle 2009, section 4) sets the step from the local
    curvature rather than from the global Lipschitz constant, which matters here
    because a crowded uv cell can make the global bound loose by orders of
    magnitude and the fixed-step version then crawls.

    Adaptive restart resets the momentum whenever the objective goes up. Plain
    FISTA is not monotone and on these problems it oscillates visibly; restarting
    costs nothing and makes the objective history readable, which is what you
    need when deciding whether a solve has actually converged.
    """
    reg = NoRegulariser() if reg is None else reg
    npix = op.npix
    floor = getattr(reg, "floor", 0.0) if reg.is_smooth else 0.0

    smooth = reg if reg.is_smooth else None
    nonsmooth = None if reg.is_smooth else reg

    # One application of the operator is by far the most expensive thing here, so
    # the loop below is written to reuse each residual it computes rather than
    # asking the operator for the same thing twice.
    def f_val(x):
        r = op.forward(x) - op.d
        v = 0.5 * float(np.sum(op.W * np.abs(r) ** 2))
        return (v + lam * smooth.value(x)) if smooth is not None else v

    def f_val_grad(x):
        """``f(x)`` and ``grad f(x)`` from a single forward and adjoint."""
        r = op.forward(x) - op.d
        v = 0.5 * float(np.sum(op.W * np.abs(r) ** 2))
        g = op.adjoint(op.W * r)
        if smooth is not None:
            v += lam * smooth.value(x)
            g = g + lam * smooth.grad(x)
        return v, g

    def g_val(x):
        return lam * nonsmooth.value(x) if nonsmooth is not None else 0.0

    def prox(v, t):
        if nonsmooth is not None:
            return nonsmooth.prox(v, t * lam, positivity=positivity, floor=floor)
        return np.clip(v, floor, None) if positivity else v

    if x0 is None:
        # the dirty image, clipped: a sensible, cheap warm start
        from .imaging import dirty_image

        x = np.clip(dirty_image(op), floor, None)
    else:
        x = np.clip(np.asarray(x0, dtype=float).copy(), floor, None)

    L = op.lipschitz()
    t_step = 1.0 / L if L > 0 else 1.0
    y = x.copy()
    theta = 1.0
    obj_hist = [f_val(x) + g_val(x)]
    chi2_hist = [op.reduced_chi2(x)]
    converged = False

    for it in range(int(n_iter)):
        fy, gy = f_val_grad(y)
        # a step that backtracked once is often too small a few iterations
        # later, so creep back up; without this the solve stalls at whatever
        # step the worst early iteration demanded
        if backtrack:
            t_step = min(t_step / eta, 1e4 / max(L, 1e-30))

        f_cand = None
        while True:
            cand = prox(y - t_step * gy, t_step)
            if not backtrack:
                break
            diff = cand - y
            f_cand = f_val(cand)
            rhs = fy + float(np.sum(gy * diff)) + 0.5 / t_step * float(np.sum(diff ** 2))
            if f_cand <= rhs * (1.0 + 1e-12) or t_step < 1e-14 / max(L, 1e-30):
                break
            t_step *= eta
        x_new = cand
        f_at_x = f_cand if f_cand is not None else f_val(cand)
        obj_new = f_at_x + g_val(x_new)

        if restart and obj_new > obj_hist[-1]:
            # momentum overshot: drop it and retake the step from the last good
            # point, where the gradient is the one we already trust
            theta = 1.0
            y = x
            fy, gy = f_val_grad(y)
            x_new = prox(y - t_step * gy, t_step)
            f_at_x = f_val(x_new)
            obj_new = f_at_x + g_val(x_new)

        theta_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * theta ** 2))
        y = x_new + ((theta - 1.0) / theta_new) * (x_new - x)
        if positivity:
            y = np.clip(y, floor, None)

        rel = np.linalg.norm(x_new - x) / max(np.linalg.norm(x_new), 1e-30)
        x, theta = x_new, theta_new
        obj_hist.append(obj_new)
        # reduced chi-squared for free: f already contains the misfit, so take it
        # back out rather than paying for another forward transform
        misfit = f_at_x - (lam * smooth.value(x) if smooth is not None else 0.0)
        chi2_hist.append(misfit / max(op.n_data, 1))
        if callback is not None:
            callback(it, x, obj_new)
        if verbose and (it % 25 == 0 or it == n_iter - 1):
            print("  iter " + str(it) + " obj " + format(obj_new, ".6e")
                  + " chi2r " + format(chi2_hist[-1], ".4f")
                  + " step " + format(t_step, ".3e"))
        if rel < tol:
            converged = True
            break

    return SolveResult(image=x, objective=np.asarray(obj_hist),
                       reduced_chi2=np.asarray(chi2_hist), n_iter=it + 1,
                       lam=float(lam), regulariser=reg.name, converged=converged,
                       step=float(t_step))


def solve(op, kind="tv", lam=0.0, **kw):
    """Convenience wrapper: name a regulariser, get a solution."""
    if kind not in REGULARISERS:
        raise ValueError("unknown regulariser: " + str(kind))
    reg = REGULARISERS[kind]()
    return fista(op, reg=reg, lam=lam, **kw)


# ---------------------------------------------------------------------------
# choosing lambda without looking at the truth
# ---------------------------------------------------------------------------
def solve_at_chi2(op, kind="tv", target_chi2=1.0, lam_lo=None, lam_hi=None,
                  n_bisect=9, n_iter=300, verbose=False, **kw):
    """Pick ``lam`` so the solution fits the data to ``target_chi2``, by bisection.

    Reduced chi-squared decreases monotonically as ``lam`` falls -- less penalty,
    closer fit -- so a bisection in ``log lam`` is well behaved. Stopping at
    ``chi2 ~ 1`` is the standard statement of "fit the data as well as the noise
    allows and no better": going below it is fitting noise, and everything below
    it is prior anyway.

    Returns the :class:`SolveResult` at the chosen ``lam``, with the bracketing
    history attached.
    """
    reg = REGULARISERS[kind]()
    if kind == "positivity":
        return fista(op, reg=reg, lam=0.0, n_iter=n_iter, **kw)

    # a scale for lam: make the penalty term comparable to the misfit at the
    # dirty image, so the bracket is in the right decade to begin with
    from .imaging import dirty_image

    x_d = np.clip(dirty_image(op), 1e-12, None)
    r_d = max(reg.value(x_d), 1e-30)
    scale = op.misfit(x_d) / r_d
    lo = scale * 1e-6 if lam_lo is None else lam_lo
    hi = scale * 1e2 if lam_hi is None else lam_hi

    trail = []
    warm = kw.pop("x0", None)
    for _ in range(int(n_bisect)):
        mid = float(np.sqrt(lo * hi))
        res = fista(op, reg=reg, lam=mid, n_iter=n_iter, x0=warm, **kw)
        c = res.final_chi2
        # continuation: the neighbouring lam's solution is a far better start
        # than the dirty image, and chi2(lam) is only monotone for *converged*
        # solves -- an under-converged one reads high and breaks the bisection
        warm = res.image
        trail.append((mid, c))
        if verbose:
            print("  lam " + format(mid, ".3e") + " -> chi2r " + format(c, ".4f"))
        if c > target_chi2:
            hi = mid          # too much penalty: the fit is too loose
        else:
            lo = mid          # still fits: can afford more regularisation
    # take the largest lam that still met the target, i.e. the most regularised
    # image consistent with the data
    lam_final = lo
    out = fista(op, reg=reg, lam=lam_final, n_iter=max(n_iter, 400), x0=warm, **kw)
    out.history["bisection"] = trail
    out.history["target_chi2"] = float(target_chi2)
    return out
