"""Scoring reconstructions against a known truth.

Four kinds of number, deliberately kept separate because they disagree:

- **Pixel error** (:func:`rmse`, :func:`nrmse`). Easy to compute, and a poor
  description of what the eye objects to -- it punishes a one-pixel positional
  shift as hard as a missing feature.
- **Structural similarity** (:func:`ssim`). Correlates much better with visual
  judgement, because it compares local means, variances and covariance rather
  than pixel values.
- **Dynamic range** (:func:`dynamic_range`). The only one of the four you can
  compute without a truth, which is why it is the one quoted in papers -- and
  it says nothing about whether the structure is right.
- **Feature survival** (:func:`crescent_features`, :func:`resolved_peaks`).
  Did the specific thing you cared about survive? The criterion has to be fixed
  in advance, in code, or it is not a measurement.

Plus :func:`credible_interval_coverage`, which scores the Bayesian method
against its own claimed uncertainty rather than against the truth alone.

Everything here is scale-aware: a reconstruction with the wrong total flux but
the right structure should not be scored as structurally wrong, so the
brightness-sensitive metrics offer an optional flux match.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .forward import image_axes


# ---------------------------------------------------------------------------
# scaling helpers
# ---------------------------------------------------------------------------
def match_flux(img, truth):
    """Rescale ``img`` to the total flux of ``truth``.

    Total flux is barely constrained by an interferometer -- nothing samples the
    zero spacing -- so comparing shapes usually means taking the scale out
    first. Which comparison you want depends on the question, so it is an
    explicit option everywhere rather than a hidden default.
    """
    s = np.sum(img)
    return img * (np.sum(truth) / s) if s != 0 else img


def _prepare(img, truth, flux_match):
    img = np.asarray(img, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if img.shape != truth.shape:
        raise ValueError("image and truth must share a grid")
    return (match_flux(img, truth) if flux_match else img), truth


# ---------------------------------------------------------------------------
# pixel error
# ---------------------------------------------------------------------------
def rmse(img, truth, flux_match=False):
    img, truth = _prepare(img, truth, flux_match)
    return float(np.sqrt(np.mean((img - truth) ** 2)))


def nrmse(img, truth, flux_match=False):
    """RMS error as a fraction of the truth's own RMS.

    Normalising by the truth rather than by its peak keeps the number meaningful
    for both a point source and a broad ring. A value of 1 means the error is as
    large as the signal.
    """
    img, truth = _prepare(img, truth, flux_match)
    denom = np.sqrt(np.mean(truth ** 2))
    return float(np.sqrt(np.mean((img - truth) ** 2)) / denom) if denom > 0 else np.inf


def flux_error(img, truth):
    """Fractional error in total flux -- almost pure prior, so worth its own row."""
    t = float(np.sum(truth))
    return float((np.sum(img) - t) / t) if t != 0 else np.inf


# ---------------------------------------------------------------------------
# structural similarity
# ---------------------------------------------------------------------------
def ssim(img, truth, sigma=1.5, data_range=None, flux_match=False,
         k1=0.01, k2=0.03, full=False):
    """Structural similarity index, Wang et al. (2004), Gaussian-windowed.

    Implemented here rather than imported so the project keeps to numpy and
    scipy. Local statistics come from a Gaussian filter of width ``sigma``
    (1.5 px, the standard choice); the stabilising constants use
    ``data_range``, which defaults to the peak-to-peak range of the *truth* so
    that the same physical error scores the same across methods.

    Returns the mean over pixels, or ``(mean, map)`` if ``full``.
    """
    from scipy.ndimage import gaussian_filter

    img, truth = _prepare(img, truth, flux_match)
    if data_range is None:
        data_range = float(np.ptp(truth))
    if data_range <= 0:
        return (1.0, np.ones_like(truth)) if full else 1.0
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    def blur(a):
        return gaussian_filter(a, sigma, mode="nearest")

    mx, my = blur(img), blur(truth)
    sxx = blur(img * img) - mx * mx
    syy = blur(truth * truth) - my * my
    sxy = blur(img * truth) - mx * my
    smap = (((2 * mx * my + c1) * (2 * sxy + c2))
            / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2)))
    return (float(smap.mean()), smap) if full else float(smap.mean())


# ---------------------------------------------------------------------------
# dynamic range
# ---------------------------------------------------------------------------
def dynamic_range(image, residual=None, op=None):
    """Peak brightness over the RMS of what is left unexplained.

    With a residual image (CLEAN) that is the usual definition. For a
    regularised solve, pass the operator and the residual is formed as the dirty
    image of the data minus the model, which is the same quantity.
    """
    image = np.asarray(image, dtype=float)
    if residual is None:
        if op is None:
            raise ValueError("need either a residual image or an operator")
        residual = residual_image(op, image)
    rms = float(np.std(residual))
    return float(image.max() / rms) if rms > 0 else np.inf


def residual_image(op, model):
    """Dirty image of the data-minus-model visibilities, in Jy/beam.

    The natural generalisation of CLEAN's residual to any method, which is what
    makes dynamic range comparable across all three reconstructions.
    """
    r = op.forward(model) - op.d
    return op.adjoint(op.W * r) / op.sum_weight


# ---------------------------------------------------------------------------
# feature survival
# ---------------------------------------------------------------------------
@dataclass
class CrescentScore:
    """Did the ring, the hole and the asymmetry each survive?

    The three criteria are fixed here, before any reconstruction is run:

    - ``hole``: the mean brightness inside ``0.4 * radius`` is below half the
      mean on the ring annulus. A filled blob fails this.
    - ``ring``: the azimuthally averaged profile peaks within half a ring width
      of the true radius. A ring at the wrong size fails.
    - ``asymmetry``: the ring carries a first azimuthal harmonic of amplitude at
      least 0.2, *and* its position angle is within 30 degrees of the truth. The
      amplitude clause matters -- without it a symmetric reconstruction passes
      whenever its arbitrary centroid angle happens to land close, which is a
      coin toss rather than a measurement.

    ``survives`` requires all three.
    """

    hole_ratio: float
    ring_radius_error: float
    pa_error_deg: float
    asymmetry_amplitude: float
    hole: bool
    ring: bool
    asymmetry: bool

    @property
    def survives(self):
        return bool(self.hole and self.ring and self.asymmetry)

    def as_dict(self):
        d = asdict(self)
        d["survives"] = self.survives
        return d


def crescent_features(img, cell, radius, width, pa_true_deg,
                      hole_threshold=0.5, pa_tolerance_deg=30.0,
                      min_asymmetry=0.2):
    """Score a reconstruction of a :func:`~src.sky.crescent` against the criteria
    in :class:`CrescentScore`."""
    img = np.clip(np.asarray(img, dtype=float), 0.0, None)
    npix = img.shape[0]
    ax = image_axes(npix, cell)
    mm, ll = np.meshgrid(ax, ax, indexing="ij")
    r = np.hypot(ll, mm)
    phi = np.arctan2(ll, mm)

    annulus = np.abs(r - radius) <= 0.5 * width
    inner = r <= 0.4 * radius
    ring_mean = float(img[annulus].mean()) if annulus.any() else 0.0
    inner_mean = float(img[inner].mean()) if inner.any() else 0.0
    hole_ratio = inner_mean / ring_mean if ring_mean > 0 else np.inf

    # azimuthally averaged radial profile, to locate the ring
    nbin = max(npix // 2, 8)
    edges = np.linspace(0.0, ax.max(), nbin + 1)
    idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, nbin - 1)
    tot = np.bincount(idx, weights=img.ravel(), minlength=nbin)
    cnt = np.bincount(idx, minlength=nbin)
    prof = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
    centres = 0.5 * (edges[:-1] + edges[1:])
    r_peak = float(centres[int(np.argmax(prof))])

    # first azimuthal harmonic of the ring brightness. For a profile
    # 1 + a cos(phi - pa) sampled uniformly in phi, sum(w e^{i phi}) / sum(w)
    # equals (a/2) e^{i pa}, so one complex moment gives both the strength of the
    # asymmetry and its direction.
    if annulus.any() and img[annulus].sum() > 0:
        wgt = img[annulus]
        m1 = np.sum(wgt * np.exp(1j * phi[annulus])) / np.sum(wgt)
        amp = 2.0 * float(np.abs(m1))
        pa = np.rad2deg(np.angle(m1))
    else:
        amp, pa = 0.0, np.nan
    pa_err = abs((pa - pa_true_deg + 180.0) % 360.0 - 180.0)

    return CrescentScore(
        hole_ratio=hole_ratio,
        ring_radius_error=(r_peak - radius) / radius,
        pa_error_deg=float(pa_err),
        asymmetry_amplitude=amp,
        hole=bool(hole_ratio < hole_threshold),
        ring=bool(abs(r_peak - radius) <= 0.5 * width),
        asymmetry=bool(amp >= min_asymmetry and pa_err <= pa_tolerance_deg),
    )


def resolved_peaks(img, axis_index=None, frac=0.2):
    """Count local maxima along a one-dimensional cut, above ``frac`` of the peak.

    The survival criterion for :func:`~src.sky.double_source`: either the two
    components are there as two, or they are not.
    """
    img = np.asarray(img, dtype=float)
    col = img.shape[1] // 2 if axis_index is None else int(axis_index)
    p = img[:, col]
    inner = p[1:-1]
    hi = (inner >= p[:-2]) & (inner >= p[2:]) & (inner > frac * p.max())
    return int(np.sum(hi))


# ---------------------------------------------------------------------------
# uncertainty calibration
# ---------------------------------------------------------------------------
def credible_interval_coverage(truth, mean, std, k=1.0, weight=None):
    """Fraction of pixels whose truth lies within ``k`` standard deviations.

    A calibration check on the Bayesian method's own error bars, not on its mean
    image. For a well-calibrated Gaussian posterior the answer should be 0.683 at
    ``k=1`` and 0.954 at ``k=2``. Below that the posterior is overconfident,
    which is the failure mode that matters: an uncertainty map nobody can trust
    is worse than no uncertainty map.

    ``weight`` restricts the count, e.g. to pixels where the truth is bright --
    coverage averaged over a mostly-empty image is dominated by the background.
    """
    truth = np.asarray(truth, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    inside = np.abs(truth - mean) <= k * np.maximum(std, 0.0)
    if weight is None:
        return float(inside.mean())
    weight = np.asarray(weight, dtype=bool)
    return float(inside[weight].mean()) if weight.any() else np.nan


def coverage_curve(truth, mean, std, ks=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
                   weight=None):
    """Empirical coverage against nominal coverage, for a calibration plot."""
    from math import erf, sqrt

    ks = np.asarray(ks, dtype=float)
    nominal = np.array([erf(k / sqrt(2.0)) for k in ks])
    empirical = np.array([credible_interval_coverage(truth, mean, std, k, weight)
                          for k in ks])
    return ks, nominal, empirical


# ---------------------------------------------------------------------------
# one call for the sweep
# ---------------------------------------------------------------------------
def score(img, truth, op=None, residual=None, flux_match=False):
    """The standard row of numbers, for the coverage sweep table."""
    out = {
        "nrmse": nrmse(img, truth, flux_match=flux_match),
        "nrmse_fluxmatched": nrmse(img, truth, flux_match=True),
        "ssim": ssim(img, truth, flux_match=flux_match),
        "flux_error": flux_error(img, truth),
    }
    if residual is None and op is not None:
        residual = residual_image(op, img)
    if residual is not None:
        out["dynamic_range"] = dynamic_range(img, residual)
        out["residual_rms"] = float(np.std(residual))
    if op is not None:
        out["reduced_chi2"] = op.reduced_chi2(img)
    return out
