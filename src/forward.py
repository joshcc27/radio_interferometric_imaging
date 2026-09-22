"""Sky to visibilities: the measurement operator, noise, gridding and weighting.

Conventions, fixed once here
----------------------------
An image is an ``(npix, npix)`` real array of brightness, with pixel size
``cell`` radians. Axis 1 is ``l`` (east), axis 0 is ``m`` (north):

    l_j = (j - npix//2) * cell,    m_i = (i - npix//2) * cell

The uv grid that the FFT implies then has spacing ``du = 1/(npix*cell)``, axis 1
is ``u``, axis 0 is ``v``, and the same centring applies.

The FFT pair used everywhere is *unitary*:

    F(x)  = fftshift(fft2(ifftshift(x),  norm="ortho"))
    Fi(y) = fftshift(ifft2(ifftshift(y), norm="ortho"))

They are exact mutual inverses and ``F^H = Fi``, which is what makes the adjoint
of the measurement operator free and exact. The physical van Cittert-Zernike
integral, discretised, is ``V = sum_lm I exp(-2 pi i (ul + vm))``, which is
``npix * F(I)`` in this normalisation -- so a point source of unit flux gives
``V = 1`` on every baseline, as it must. The factor of ``npix`` is carried
explicitly in :class:`MeasurementOperator` rather than being hidden in the FFT.

The sky is real, so ``V(-u,-v) = conj(V(u,v))``. That is enforced explicitly by
:func:`hermitian_augment` rather than left to the FFT to imply: it makes the
gridded weight function symmetric, which in turn makes the dirty beam exactly
real and the operator exactly real-to-real.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

ARCSEC = np.pi / (180.0 * 3600.0)
UAS = ARCSEC * 1e-6


# ---------------------------------------------------------------------------
# the FFT pair
# ---------------------------------------------------------------------------
def F(x):
    """Unitary centred forward FFT: image -> Fourier plane."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm="ortho"))


def Fi(y):
    """Unitary centred inverse FFT: Fourier plane -> image."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(y), norm="ortho"))


def uv_axes(npix, cell):
    """The u and v coordinate axes of the imaging grid, in wavelengths."""
    du = 1.0 / (npix * cell)
    return (np.arange(npix) - npix // 2) * du


def image_axes(npix, cell):
    """The l and m coordinate axes of the image, in radians."""
    return (np.arange(npix) - npix // 2) * cell


# ---------------------------------------------------------------------------
# exact forward model: sky -> visibilities at arbitrary (u, v)
# ---------------------------------------------------------------------------
def predict_visibilities(image, cell, u, v, chunk=512, flux_floor=0.0):
    """Evaluate the exact discrete van Cittert-Zernike integral at given (u, v).

    This is the *simulator*, deliberately not the gridded operator the solvers
    use: sampling the true visibility function at the true continuous uv points
    is what a real array does, and keeping the two distinct means the gridding
    error shows up in the results instead of cancelling out.

    Cost is ``n_vis * n_pixels``, so it is chunked over visibilities. Pixels
    at or below ``flux_floor`` are skipped, which makes point-source and
    crescent models very cheap.
    """
    image = np.asarray(image, dtype=float)
    npix = image.shape[0]
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()

    l = image_axes(npix, cell)
    m = image_axes(npix, cell)
    mm, ll = np.meshgrid(m, l, indexing="ij")  # match image indexing (i->m, j->l)

    keep = image > flux_floor
    bright = image[keep]
    lk = ll[keep]
    mk = mm[keep]

    out = np.zeros(u.size, dtype=complex)
    if bright.size == 0:
        return out
    for s in range(0, u.size, chunk):
        e = min(s + chunk, u.size)
        phase = -2.0 * np.pi * (np.outer(u[s:e], lk) + np.outer(v[s:e], mk))
        out[s:e] = np.exp(1j * phase) @ bright
    return out


# ---------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------
def thermal_sigma_for_snr(vis, snr, kind="mean_amp"):
    """Pick a per-component noise sigma to hit a target visibility SNR.

    ``kind="mean_amp"`` sets sigma so that ``mean|V| / sigma = snr``, which is
    the honest way to specify difficulty: it is the per-visibility SNR, and the
    image-plane dynamic range you can reach is roughly ``snr * sqrt(n_vis)``.
    """
    amp = np.abs(np.asarray(vis))
    scale = amp.mean() if kind == "mean_amp" else amp.max()
    return float(scale / snr)


def add_thermal_noise(vis, sigma, rng=None):
    """Add independent complex Gaussian noise.

    ``sigma`` is the standard deviation of *each* of the real and imaginary
    parts, which is the radio convention; the corresponding statistical weight
    is ``1/sigma**2``.
    """
    rng = np.random.default_rng() if rng is None else rng
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), np.shape(vis))
    noise = rng.normal(scale=sigma) + 1j * rng.normal(scale=sigma)
    return np.asarray(vis) + noise


# ---------------------------------------------------------------------------
# the observed dataset
# ---------------------------------------------------------------------------
@dataclass
class Visibilities:
    """Measured visibilities: positions, values, and per-visibility noise."""

    u: np.ndarray
    v: np.ndarray
    vis: np.ndarray
    sigma: np.ndarray
    meta: dict | None = None

    def __len__(self) -> int:
        return int(self.u.size)

    @property
    def natural_weight(self):
        """Inverse-variance weight, the maximum-sensitivity choice.

        Noiseless simulations carry ``sigma = 0``; there the inverse variance is
        meaningless and every visibility is given unit weight instead, which is
        what the round-trip test needs.
        """
        sigma = np.asarray(self.sigma, dtype=float)
        if np.all(sigma <= 0):
            return np.ones_like(sigma)
        return 1.0 / sigma ** 2

    @property
    def radius(self):
        return np.hypot(self.u, self.v)


def observe(image, cell, coverage, snr=None, sigma=None, rng=None, chunk=512,
            flux_floor=0.0, hermitian=True):
    """Simulate an observation: sample the sky, then add thermal noise.

    Returns a :class:`Visibilities` with the conjugate points appended when
    ``hermitian`` is set, so downstream code never has to remember to do it.
    """
    u = np.asarray(coverage.u, dtype=float)
    v = np.asarray(coverage.v, dtype=float)
    clean_vis = predict_visibilities(image, cell, u, v, chunk=chunk,
                                     flux_floor=flux_floor)
    if sigma is None:
        sigma = 0.0 if snr is None else thermal_sigma_for_snr(clean_vis, snr)
    sigma_arr = np.full(u.size, float(sigma))
    noisy = clean_vis if sigma == 0 else add_thermal_noise(clean_vis, sigma_arr, rng)

    data = Visibilities(
        u=u, v=v, vis=noisy, sigma=sigma_arr,
        meta={"array": getattr(coverage, "array_name", ""), "sigma": float(sigma),
              "n_measured": int(u.size)},
    )
    return hermitian_augment(data) if hermitian else data


def hermitian_augment(data):
    """Append the conjugate-symmetric partner of every visibility.

    The sky is real, so ``V(-u,-v) = conj(V(u,v))``. Every measurement therefore
    hands you a second one for free. Doing this explicitly -- rather than hoping
    the FFT sorts it out -- is what guarantees a real dirty beam and a
    real-to-real measurement operator.
    """
    if data.meta is not None and data.meta.get("hermitian"):
        return data
    meta = dict(data.meta or {})
    meta["hermitian"] = True
    return Visibilities(
        u=np.concatenate([data.u, -data.u]),
        v=np.concatenate([data.v, -data.v]),
        vis=np.concatenate([data.vis, np.conj(data.vis)]),
        sigma=np.concatenate([data.sigma, data.sigma]),
        meta=meta,
    )


# ---------------------------------------------------------------------------
# weighting
# ---------------------------------------------------------------------------
def _cell_index(u, v, npix, cell):
    """Nearest uv-grid cell for each visibility, and a validity mask."""
    du = 1.0 / (npix * cell)
    cu = np.rint(np.asarray(u) / du).astype(int) + npix // 2
    cv = np.rint(np.asarray(v) / du).astype(int) + npix // 2
    inside = (cu >= 0) & (cu < npix) & (cv >= 0) & (cv < npix)
    return cu, cv, inside


def density_count(data, npix, cell):
    """Total natural weight falling in each visibility's own uv cell."""
    cu, cv, inside = _cell_index(data.u, data.v, npix, cell)
    w = data.natural_weight
    acc = np.zeros((npix, npix))
    np.add.at(acc, (cv[inside], cu[inside]), w[inside])
    out = np.zeros(len(data))
    out[inside] = acc[cv[inside], cu[inside]]
    return out


def weights(data, npix, cell, scheme="natural", robust=0.0):
    """Per-visibility imaging weights.

    - ``natural``: inverse variance. Best point-source sensitivity, worst beam --
      the short baselines are crowded, so they dominate and the beam grows a
      broad skirt.
    - ``uniform``: divide by the local uv density, so every *occupied cell* gets
      equal say. Best resolution, worst sensitivity.
    - ``briggs``: the one-parameter family between them,
      ``w = w_nat / (1 + W_cell * f2)`` with
      ``f2 = (5 * 10**-robust)**2 / (sum W_cell**2 / sum W_cell)``.
      ``robust=-2`` tends to uniform, ``robust=+2`` to natural.
    """
    w_nat = data.natural_weight
    if scheme == "natural":
        return w_nat
    dens = density_count(data, npix, cell)
    safe = np.where(dens > 0, dens, 1.0)
    if scheme == "uniform":
        return np.where(dens > 0, w_nat / safe, 0.0)
    if scheme == "briggs":
        cu, cv, inside = _cell_index(data.u, data.v, npix, cell)
        acc = np.zeros((npix, npix))
        np.add.at(acc, (cv[inside], cu[inside]), w_nat[inside])
        occupied = acc[acc > 0]
        f2 = (5.0 * 10.0 ** (-float(robust))) ** 2 / (
            np.sum(occupied ** 2) / np.sum(occupied)
        )
        return w_nat / (1.0 + dens * f2)
    raise ValueError("unknown weighting scheme: " + str(scheme))


# ---------------------------------------------------------------------------
# gridding
# ---------------------------------------------------------------------------
def _kaiser_bessel(x, width, beta):
    """Kaiser-Bessel gridding kernel, evaluated in units of uv cells."""
    from scipy.special import i0

    t = 2.0 * np.asarray(x) / width
    inside = np.abs(t) <= 1.0
    out = np.zeros_like(t, dtype=float)
    arg = beta * np.sqrt(np.clip(1.0 - t[inside] ** 2, 0.0, None))
    out[inside] = i0(arg) / i0(beta)
    return out


def kb_grid_correction(npix, width=6.0, beta=None):
    """Image-space deapodisation for the Kaiser-Bessel kernel.

    The kernel convolves in the uv plane, so it multiplies in the image plane by
    its own Fourier transform. Dividing that out is what makes convolutional
    gridding unbiased instead of merely smooth.
    """
    beta = np.pi * np.sqrt(width ** 2 / 4.0 - 0.8) if beta is None else beta
    s = (np.arange(npix) - npix // 2) / float(npix)
    a = (np.pi * width * s) ** 2
    z = beta ** 2 - a
    prof = np.empty(npix)
    pos = z > 0
    rp = np.sqrt(z[pos])
    prof[pos] = np.sinh(rp) / rp
    rn = np.sqrt(-z[~pos])
    prof[~pos] = np.sinc(rn / np.pi)
    prof = prof / prof[npix // 2]
    return np.outer(prof, prof)


def grid_visibilities(data, npix, cell, w=None, mode="nearest", width=6.0):
    """Grid weighted visibilities and weights onto the imaging uv grid.

    Returns ``(vis_grid, weight_grid)`` where ``vis_grid`` holds ``sum w*V`` and
    ``weight_grid`` holds ``sum w`` per cell.

    ``mode="nearest"`` is plain binning -- five lines, and the self-consistent
    partner of :class:`MeasurementOperator`, which evaluates the model at cell
    centres. ``mode="kb"`` spreads each visibility over a Kaiser-Bessel kernel,
    which suppresses the aliasing that nearest-neighbour binning folds in from
    outside the field; use it for imaging, and pair it with
    :func:`kb_grid_correction`.
    """
    w = np.ones(len(data)) if w is None else np.asarray(w, dtype=float)
    du = 1.0 / (npix * cell)
    vg = np.zeros((npix, npix), dtype=complex)
    wg = np.zeros((npix, npix), dtype=float)

    if mode == "nearest":
        cu, cv, inside = _cell_index(data.u, data.v, npix, cell)
        np.add.at(vg, (cv[inside], cu[inside]), w[inside] * data.vis[inside])
        np.add.at(wg, (cv[inside], cu[inside]), w[inside])
        return vg, wg

    if mode != "kb":
        raise ValueError("unknown gridding mode: " + str(mode))

    beta = np.pi * np.sqrt(width ** 2 / 4.0 - 0.8)
    half = int(np.ceil(width / 2.0))
    fu = np.asarray(data.u) / du + npix // 2
    fv = np.asarray(data.v) / du + npix // 2
    for offu in range(-half, half + 1):
        iu = np.rint(fu).astype(int) + offu
        du_off = iu - fu
        ku = _kaiser_bessel(du_off, width, beta)
        for offv in range(-half, half + 1):
            iv = np.rint(fv).astype(int) + offv
            kv = _kaiser_bessel(iv - fv, width, beta)
            k = ku * kv
            ok = (iu >= 0) & (iu < npix) & (iv >= 0) & (iv < npix) & (k > 0)
            if not np.any(ok):
                continue
            np.add.at(vg, (iv[ok], iu[ok]), (w * k)[ok] * data.vis[ok])
            np.add.at(wg, (iv[ok], iu[ok]), (w * k)[ok])
    return vg, wg


# ---------------------------------------------------------------------------
# operators: image -> visibilities, with exact adjoints
# ---------------------------------------------------------------------------
class _OperatorBase:
    """Shared data-fit machinery for the two measurement operators.

    A subclass supplies ``forward``, ``adjoint``, ``d``, ``W`` and ``n_data``;
    everything a solver needs -- the weighted misfit, its gradient, chi-squared
    and the Lipschitz constant -- follows from those and is written once here, so
    both operators are guaranteed to define "misfit" the same way.
    """

    npix = 0
    cell = 0.0

    def residual_vis(self, x):
        return self.forward(x) - self.d

    def misfit(self, x):
        """``0.5 * sum W |A x - d|^2``."""
        return 0.5 * float(np.sum(self.W * np.abs(self.residual_vis(x)) ** 2))

    def misfit_grad(self, x):
        """Gradient of :meth:`misfit` with respect to a real image."""
        return self.adjoint(self.W * self.residual_vis(x))

    def chi2(self, x):
        """Weighted chi-squared, summed over all measurements."""
        return float(np.sum(self.W * np.abs(self.residual_vis(x)) ** 2))

    def reduced_chi2(self, x):
        """Chi-squared per real degree of freedom.

        Each measurement is one complex residual, so two real numbers, and the
        divisor is ``2 * n_data``. The Hermitian partners are perfectly
        correlated and so are the pairs of cells they fall in -- but they also
        contribute twice to the sum, and the two factors of two cancel, leaving
        ``2 * n_data`` either way. Verified against a noise-only simulation in
        ``tests/test_operators.py``, which is the only way to be sure of a
        normalisation like this.
        """
        return self.chi2(x) / max(2 * self.n_data, 1)

    def lipschitz(self, n_iter=60, seed=0):
        """Operator norm of ``A^H W A`` by power iteration.

        The analytic bound ``npix**2 * max(W)`` is valid but can be loose by
        orders of magnitude when one cell is crowded, and FISTA steps at ``1/L``.
        """
        if self._lipschitz is not None:
            return self._lipschitz
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((self.npix, self.npix))
        x /= np.linalg.norm(x)
        lam = 0.0
        for _ in range(n_iter):
            y = self.adjoint(self.W * self.forward(x))
            lam = float(np.linalg.norm(y))
            if lam == 0.0:
                break
            x = y / lam
        self._lipschitz = lam
        return lam


class MeasurementOperator(_OperatorBase):
    """Image -> gridded visibilities, with an exact adjoint.

    Visibilities are consolidated into uv cells first: cell ``c`` carries the
    weighted mean ``d_c = sum_k w_k V_k / sum_k w_k`` and total weight
    ``W_c = sum_k w_k``. The operator is then

        A x = npix * F(x)  restricted to occupied cells

    and the weighted misfit ``0.5 * sum_c W_c |A x - d|^2`` equals the
    per-visibility chi-squared ``0.5 * sum_k w_k |model - V_k|^2`` up to a
    constant, so nothing is lost by gridding the data first.

    Because the operator is an FFT and a mask, the adjoint is the inverse FFT
    and the same mask -- exact, and free. That matters more than it sounds: a
    wrong adjoint makes a gradient method converge smoothly to the wrong answer,
    which is the hardest class of bug to notice. See ``tests/test_operators.py``.

    ``drop_zero_spacing`` addresses a subtlety that is easy to miss. Nothing
    physically samples ``(u, v) = (0, 0)``, but the grid cell *containing* the
    origin is a square of side ``du = 1/(npix*cell)``, and the shortest
    baselines fall inside it whenever the field of view is smaller than the
    largest angular scale the array is sensitive to -- which is the normal
    situation. Those baselines then act as a zero-spacing measurement, and the
    total flux becomes weakly constrained by the data after all. That is what
    real gridded imaging does too, so it is the default; setting the flag empties
    the origin cell instead, which is the harsher and more instructive case,
    because then the total flux of every reconstruction is set purely by the
    prior.
    """

    def __init__(self, data, npix, cell, w=None, scheme=None, robust=0.0,
                 drop_zero_spacing=False):
        self.npix = int(npix)
        self.cell = float(cell)
        self.data = data
        if w is None:
            w = data.natural_weight if scheme is None else weights(
                data, npix, cell, scheme=scheme, robust=robust
            )
        self.w = np.asarray(w, dtype=float)

        vg, wg = grid_visibilities(data, npix, cell, w=self.w, mode="nearest")
        if drop_zero_spacing:
            c = self.npix // 2
            vg[c, c] = 0.0
            wg[c, c] = 0.0
        self.drop_zero_spacing = bool(drop_zero_spacing)
        self.weight_grid = wg
        self.mask = wg > 0
        self.n_cells = int(self.mask.sum())
        self.data_grid = np.zeros_like(vg)
        self.data_grid[self.mask] = vg[self.mask] / wg[self.mask]
        self.sum_weight = float(wg.sum())
        self._lipschitz = None

    # -- basic linear algebra -------------------------------------------------
    def forward(self, x):
        """A x: model visibilities at the occupied cells (complex, packed)."""
        return (self.npix * F(x))[self.mask]

    def adjoint(self, y):
        """A^H y, projected back to the real image domain."""
        g = np.zeros((self.npix, self.npix), dtype=complex)
        g[self.mask] = y
        return self.npix * np.real(Fi(g))

    def forward_grid(self, x):
        """A x scattered back onto the full uv grid (zeros where unsampled)."""
        g = np.zeros((self.npix, self.npix), dtype=complex)
        g[self.mask] = self.forward(x)
        return g

    # -- data-fit pieces ------------------------------------------------------
    @property
    def d(self):
        """The gridded data vector, packed over occupied cells."""
        return self.data_grid[self.mask]

    @property
    def W(self):
        """Cell weights, packed over occupied cells."""
        return self.weight_grid[self.mask]

    @property
    def n_data(self):
        return self.n_cells


# ---------------------------------------------------------------------------
# the accurate operator: proper degridding, model error far below the noise
# ---------------------------------------------------------------------------
class DegridOperator(_OperatorBase):
    """Image -> visibilities at the *exact* measured ``(u, v)``, by degridding.

    Why this exists. :class:`MeasurementOperator` compares a model evaluated at
    uv *cell centres* against data measured up to half a cell away. The phase
    error is ``2 pi (du/2) l``, so it grows with distance from the phase centre,
    and for a source a couple of resolution elements off centre it reaches a few
    per cent of the visibility amplitude -- comparable to or larger than thermal
    noise at any decent SNR. That is a systematic in the forward model, and it
    makes "fit to reduced chi-squared of 1" meaningless, which in turn corrupts
    any principled choice of the regularisation weight. Measured for this
    project's setups in ``tests/test_operators.py``.

    The cure is the standard one. To evaluate
    ``V(u,v) = sum_lm I exp(-2 pi i (u l + v m))`` at arbitrary ``(u, v)``:

    1. divide the image by the apodisation ``psi_hat``, the Fourier transform of
       the gridding kernel;
    2. zero-pad by ``osamp`` and take the FFT, giving the visibility function on
       a grid ``osamp`` times finer than the image implies;
    3. interpolate onto the measured points with the Kaiser-Bessel kernel
       ``psi``, normalised per visibility so that a centred unit point source
       returns exactly 1.

    Step 1 is what cancels the smoothing step 3 introduces; without it the
    operator is smooth and biased rather than accurate. With ``osamp=2`` and a
    width-6 kernel the residual model error is a few times 1e-5 of the peak
    amplitude, which is far below any noise level used here.

    The adjoint is the exact transpose -- scatter with the same coefficients,
    inverse FFT, crop, divide by the apodisation -- so the adjoint test passes to
    machine precision, and the operator can be handed straight to FISTA or HMC.

    The cost is one FFT on an ``osamp*npix`` grid plus ``(width+1)^2`` taps per
    visibility, which is a few milliseconds for the datasets here.
    """

    def __init__(self, data, npix, cell, w=None, scheme=None, robust=0.0,
                 osamp=2, width=6, deduplicate_hermitian=True):
        self.npix = int(npix)
        self.cell = float(cell)
        self.osamp = int(osamp)
        self.width = float(width)
        self.M = int(osamp) * int(npix)
        if w is None:
            w = data.natural_weight if scheme is None else weights(
                data, npix, cell, scheme=scheme, robust=robust
            )
        w = np.asarray(w, dtype=float)

        # The conjugate half of a Hermitian-augmented dataset is exactly
        # redundant for this operator: it models the continuous transform of a
        # real image, so it already satisfies V(-u,-v) = conj(V(u,v)) and the
        # partner's residual is the conjugate of the original's. Keeping it
        # doubles the misfit, the weight sum and the cost, and changes neither
        # the gradient direction nor the reduced chi-squared. (It is *not*
        # redundant for MeasurementOperator, where augmenting is what makes the
        # gridded sampling function symmetric and the dirty beam real.)
        self.n_dropped_conjugates = 0
        if deduplicate_hermitian and (data.meta or {}).get("hermitian"):
            n = len(data) // 2
            if (np.allclose(data.u[n:], -data.u[:n])
                    and np.allclose(data.v[n:], -data.v[:n])):
                data = Visibilities(u=data.u[:n], v=data.v[:n], vis=data.vis[:n],
                                    sigma=data.sigma[:n],
                                    meta=dict(data.meta or {}, hermitian=False,
                                              deduplicated=True))
                w = w[:n]
                self.n_dropped_conjugates = n
        self.data = data
        self.w = w
        self.sum_weight = float(self.w.sum())
        self._lipschitz = None

        # apodisation over the image region of the padded grid
        apod_full = kb_grid_correction(self.M, width=self.width)
        lo = self.M // 2 - self.npix // 2
        self._crop = slice(lo, lo + self.npix)
        self.apod = apod_full[self._crop, self._crop]

        self._build_kernel()

    # -- interpolation weights, precomputed once --------------------------------
    def _build_kernel(self):
        beta = np.pi * np.sqrt(self.width ** 2 / 4.0 - 0.8)
        du = 1.0 / (self.M * self.cell)
        fu = np.asarray(self.data.u, dtype=float) / du + self.M // 2
        fv = np.asarray(self.data.v, dtype=float) / du + self.M // 2
        half = int(np.ceil(self.width / 2.0))
        offs = np.arange(-half, half + 1)

        cu = np.rint(fu).astype(np.int64)[:, None] + offs[None, :]
        cv = np.rint(fv).astype(np.int64)[:, None] + offs[None, :]
        ku = _kaiser_bessel(cu - fu[:, None], self.width, beta)
        kv = _kaiser_bessel(cv - fv[:, None], self.width, beta)

        if cu.min() < 0 or cu.max() >= self.M or cv.min() < 0 or cv.max() >= self.M:
            raise ValueError(
                "a visibility falls within half a kernel width of the uv grid "
                "edge; the image is too coarse for this array (reduce cell or "
                "raise npix)")

        # outer product of the separable kernel, flattened per visibility
        coef = (kv[:, :, None] * ku[:, None, :]).reshape(len(ku), -1)
        idx = (cv[:, :, None] * self.M + cu[:, None, :]).reshape(len(ku), -1)
        # per-visibility normalisation: makes a centred unit point source return
        # exactly 1, which is the only calibration this operator needs
        norm = coef.sum(axis=1, keepdims=True)
        self._coef = coef / norm
        self._idx = idx.astype(np.int64)

    # -- basic linear algebra ---------------------------------------------------
    def forward(self, x):
        """A x: model visibilities at the measured ``(u, v)``."""
        pad = np.zeros((self.M, self.M), dtype=float)
        pad[self._crop, self._crop] = np.asarray(x, dtype=float) / self.apod
        g = (self.M * F(pad)).ravel()
        return np.einsum("kj,kj->k", self._coef, g[self._idx])

    def adjoint(self, y):
        """A^H y, back in the real image domain."""
        y = np.asarray(y)
        n = self.M * self.M
        wr = (self._coef * np.real(y)[:, None]).ravel()
        wi = (self._coef * np.imag(y)[:, None]).ravel()
        flat = self._idx.ravel()
        g = (np.bincount(flat, weights=wr, minlength=n)
             + 1j * np.bincount(flat, weights=wi, minlength=n))
        out = self.M * Fi(g.reshape(self.M, self.M))
        return np.real(out[self._crop, self._crop]) / self.apod

    # -- data-fit pieces --------------------------------------------------------
    @property
    def d(self):
        return np.asarray(self.data.vis)

    @property
    def W(self):
        return self.w

    @property
    def n_data(self):
        return int(self.w.size)
