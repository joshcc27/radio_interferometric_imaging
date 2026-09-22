"""Hogbom CLEAN.

The algorithm the field actually ran on for forty years, and it is about thirty
lines::

    residual = dirty_image
    loop:
        find the peak of the residual
        subtract gain * peak * (dirty beam shifted to that peak)
        record (position, gain * peak) as a component
    until the peak drops below threshold, or the iteration cap is hit
    restored = components convolved with the clean beam, plus the residual

Two things worth noticing, both of which are discussed in the README:

1. CLEAN's implicit prior is that the sky is a sum of delta functions. It is not
   a regulariser you can write down and tune, but it is a prior all the same, and
   it is why CLEAN handles extended emission badly -- a smooth halo gets
   decomposed into a speckle of points that only looks smooth again after the
   restoring convolution.
2. The restoring step convolves the component list back up to the nominal
   resolution, which quietly throws away whatever super-resolution the
   deconvolution actually achieved. The component list is sharper than the image
   CLEAN hands you, and nobody can tell you how much of that sharpness to trust.

A note on flux, which is less straightforward than it first looks. Summing a
centred image is the same as reading off its Fourier transform at the origin, so
``sum(psf)`` is proportional to the gridded sampling weight in the *origin cell*
and ``sum(dirty)`` to the weighted visibility there. No baseline physically sits
at ``(u, v) = (0, 0)``, but the origin cell is a square of side
``du = 1/(npix*cell)``, and the shortest baselines land inside it whenever the
field of view is smaller than the largest angular scale the array is sensitive
to -- the usual situation. So in practice ``sum(psf) != 0``, the short spacings
stand in for a zero-spacing measurement, and

    sum(dirty) = sum(psf) * model_flux + sum(residual)

holds exactly, which is the flux conservation ``tests/test_clean.py`` asserts.
Empty the origin cell instead (``drop_zero_spacing=True`` on the operator) and
``sum(psf)`` becomes exactly zero: the identity degenerates, the residual sum is
invariant, and the total flux of the model is set entirely by the algorithm
rather than by the data. Both cases are tested, because the second one is the
honest picture of how much of a reconstruction's flux is a prior.

Stronger than either, and exact in both cases, is the bookkeeping identity
``dirty == components * psf + residual``, which is what actually pins down the
shift-and-subtract indexing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .imaging import convolve_periodic, fit_clean_beam, restore, shift_psf


@dataclass
class CleanResult:
    """Everything CLEAN produces, kept separate so nothing is silently merged."""

    components: np.ndarray          # Jy/pixel, the model
    residual: np.ndarray            # Jy/beam, what is left in the dirty image
    component_list: list = field(default_factory=list)  # (i, j, flux) in order
    peak_history: np.ndarray = None
    n_iter: int = 0
    stop_reason: str = ""
    beam: object = None             # the fitted restoring beam, if requested

    @property
    def model_flux(self):
        return float(self.components.sum())

    def restored(self, cell, add_residual=True):
        """Components convolved with the restoring beam, plus the residual."""
        if self.beam is None:
            raise ValueError("no restoring beam was fitted; pass psf to clean()")
        return restore(self.components, self.residual, self.beam, cell,
                       add_residual=add_residual)


def clean(dirty, psf, gain=0.1, n_iter=2000, threshold=None, frac_threshold=1e-3,
          cell=None, mask=None, allow_negative=True, fit_beam=True):
    """Deconvolve ``dirty`` by ``psf`` with the Hogbom algorithm.

    Parameters
    ----------
    dirty, psf : (npix, npix) arrays
        Dirty image and peak-normalised dirty beam, on the same grid and from the
        same weighting -- see :mod:`src.imaging`. Both must be centred at
        ``npix//2``.
    gain : float
        Loop gain. 0.1 is the usual choice: small enough that the sidelobes of
        one component are removed before the next is misidentified, large enough
        to finish.
    threshold, frac_threshold : float
        Stop when the residual peak falls below ``threshold`` (absolute), or
        below ``frac_threshold`` times the initial peak. Whichever is larger
        wins, so passing a noise-based ``threshold`` does the sensible thing.
    mask : (npix, npix) bool array, optional
        A clean window: components may only be placed where it is True. This is
        the crudest possible prior and it is remarkably effective, which is worth
        remembering when comparing against the explicit regularisers.
    allow_negative : bool
        Search on ``|residual|``, so negative components are allowed. Real CLEAN
        does this and it matters: forbidding them turns sidelobe overshoot into a
        positive bias. Set False for a non-negative model.

    The subtraction uses a circular shift of the psf, which is exact on a gridded
    uv plane -- see :mod:`src.imaging`.
    """
    dirty = np.asarray(dirty, dtype=float)
    psf = np.asarray(psf, dtype=float)
    if dirty.shape != psf.shape:
        raise ValueError("dirty image and psf must share a grid")
    npix = dirty.shape[0]
    c = npix // 2
    if not np.isclose(psf[c, c], 1.0, atol=1e-8):
        raise ValueError("psf must be peak-normalised at the image centre")

    residual = dirty.copy()
    components = np.zeros_like(dirty)
    comp_list = []
    peaks = []

    search = np.ones(dirty.shape, dtype=bool) if mask is None else np.asarray(mask, bool)
    if not search.any():
        raise ValueError("clean mask excludes every pixel")

    initial = np.abs(residual[search]).max()
    thr = float(frac_threshold) * initial
    if threshold is not None:
        thr = max(thr, float(threshold))

    stop = "hit iteration cap"
    for it in range(int(n_iter)):
        work = np.abs(residual) if allow_negative else residual
        work = np.where(search, work, -np.inf)
        idx = int(np.argmax(work))
        i, j = divmod(idx, npix)
        peak = residual[i, j]
        peaks.append(peak)
        if abs(peak) <= thr:
            stop = "reached threshold"
            break
        step = gain * peak
        residual -= step * shift_psf(psf, i, j)
        components[i, j] += step
        comp_list.append((i, j, step))
    else:
        it = int(n_iter)

    beam = fit_clean_beam(psf, cell) if (fit_beam and cell is not None) else None
    return CleanResult(components=components, residual=residual,
                       component_list=comp_list, peak_history=np.asarray(peaks),
                       n_iter=len(comp_list), stop_reason=stop, beam=beam)


def model_dirty(components, psf):
    """The dirty image a component model predicts: ``components * psf``.

    The exact partner of the subtraction loop, so
    ``dirty - model_dirty(components, psf) - residual`` is zero to machine
    precision. That identity is the real conservation law here.
    """
    return convolve_periodic(components, psf)


def clean_from_operator(op, gain=0.1, n_iter=2000, sigma_stop=3.0, **kw):
    """Run CLEAN straight from a :class:`~src.forward.MeasurementOperator`.

    The stopping threshold defaults to ``sigma_stop`` times the theoretical
    image-plane noise, which is the honest place to stop: below that CLEAN is
    fitting noise, and its components stop meaning anything.
    """
    from .imaging import dirty_beam, dirty_image

    dirty = dirty_image(op)
    psf = dirty_beam(op)
    thr = sigma_stop * theoretical_image_noise(op)
    return clean(dirty, psf, gain=gain, n_iter=n_iter, threshold=thr,
                 cell=op.cell, **kw)


def theoretical_image_noise(op):
    """Expected noise in the weighted dirty image, in Jy/beam.

    The dirty image is ``sum_k w_k V_k exp(2 pi i (u l + v m)) / sum_k w_k``, so
    a pixel's noise is a weighted sum of the visibility noise. The factor that is
    easy to get wrong is the Hermitian augmentation: a visibility and its
    conjugate partner carry the *same* noise realisation, not an independent one,
    so the pair contributes ``2 w Re(n e^{i phi})`` with variance ``4 w^2
    sigma^2`` rather than two independent terms. Summing over the ``N``
    independent measurements,

        var = sum_k 4 w_k^2 sigma_k^2 / (sum_aug w)^2
            = 2 * sum_aug w^2 sigma^2 / (sum_aug w)^2

    where ``sum_aug`` runs over the augmented list. Dropping that factor of two
    underestimates the noise by ``sqrt(2)``, which is exactly the kind of error
    that quietly sets a CLEAN threshold too low and lets it chew into the noise.
    Verified against a noise-only simulation in ``tests/test_clean.py``.
    """
    w = op.w
    sig = np.asarray(op.data.sigma, dtype=float)
    denom = w.sum()
    if denom <= 0:
        return 0.0
    hermitian = bool((op.data.meta or {}).get("hermitian"))
    var = np.sum(w ** 2 * sig ** 2) / denom ** 2
    return float(np.sqrt(2.0 * var if hermitian else var))
