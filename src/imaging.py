"""Dirty image, dirty beam, and the restoring beam.

The two lines the whole project hangs off. With sampling function
``S(u, v) = sum_k w_k delta(u - u_k, v - v_k)``,

    B   = F^-1[S]          the dirty beam, i.e. the point spread function
    I_D = F^-1[S . V]      the dirty image

and by the convolution theorem ``I_D = I * B``. Deconvolution is the entire
problem, and it is hard only because ``S`` is sparse, which gives ``B``
sidelobes tens of percent of its peak instead of the gentle skirt of a filled
aperture.

Normalisation
-------------
Both are divided by ``sum_k w_k``, which makes ``B(0, 0) = 1`` exactly and puts
the dirty image in units of Jy/beam. Because the same divisor is used for both,
the dirty image of a unit point source at the phase centre is *exactly* the
dirty beam -- see ``tests/test_imaging.py``, which is the single best check that
the gridding, weighting and FFT conventions all agree.

Wrap-around
-----------
Once visibilities are binned onto a uv grid of spacing ``du = 1/(npix*cell)``,
the implied dirty beam is periodic with period ``npix*cell`` -- the image FOV.
So the relation ``I_D = I * B`` holds with a *circular* convolution, and
:func:`shift_psf` uses :func:`numpy.roll`, which is exact rather than merely
convenient. It also means a source close to the edge folds its sidelobes back in
from the opposite side; the cure is the usual one, keep the field comfortably
bigger than the emission. All the experiments here do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .forward import F, Fi, MeasurementOperator, image_axes

FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


# ---------------------------------------------------------------------------
# the two images
# ---------------------------------------------------------------------------
def dirty_beam(op):
    """Point spread function on the imaging grid, peak normalised to 1."""
    return op.adjoint(op.W) / op.sum_weight


def dirty_image(op):
    """Weighted inverse transform of the sampled visibilities, in Jy/beam."""
    return op.adjoint(op.W * op.d) / op.sum_weight


def make_images(data, npix, cell, scheme="natural", robust=0.0):
    """Convenience wrapper: operator, dirty image and dirty beam in one call."""
    op = MeasurementOperator(data, npix, cell, scheme=scheme, robust=robust)
    return op, dirty_image(op), dirty_beam(op)


def shift_psf(psf, i, j):
    """The psf recentred on pixel ``(i, j)``.

    Exact, because the gridded psf is periodic on the image grid; see the module
    docstring.
    """
    c = psf.shape[0] // 2
    return np.roll(psf, (int(i) - c, int(j) - c), axis=(0, 1))


def convolve_periodic(image, kernel):
    """Circular convolution of two centred, equal-sized arrays."""
    n = image.shape[0]
    return np.real(Fi(F(image) * F(kernel))) * n


# ---------------------------------------------------------------------------
# restoring beam
# ---------------------------------------------------------------------------
@dataclass
class GaussianBeam:
    """An elliptical Gaussian beam. FWHMs in radians, position angle east of north."""

    bmaj: float
    bmin: float
    bpa_deg: float

    def area_pixels(self, cell):
        """Beam solid angle in pixels -- the Jy/beam to Jy/pixel conversion."""
        return np.pi * self.bmaj * self.bmin / (4.0 * np.log(2.0)) / float(cell) ** 2

    def image(self, npix, cell, peak=1.0):
        """Render the beam as a centred image with the given peak value."""
        ax = image_axes(npix, cell)
        mm, ll = np.meshgrid(ax, ax, indexing="ij")
        a = np.deg2rad(self.bpa_deg)
        maj = mm * np.cos(a) + ll * np.sin(a)
        minr = -mm * np.sin(a) + ll * np.cos(a)
        sm = self.bmaj / FWHM_PER_SIGMA
        sn = self.bmin / FWHM_PER_SIGMA
        return float(peak) * np.exp(-0.5 * ((maj / sm) ** 2 + (minr / sn) ** 2))

    def __repr__(self):
        arc = 180.0 * 3600.0 / np.pi
        maj = self.bmaj * arc
        # pick a unit that reads as a number rather than an exponent: a VLA beam
        # is tens of arcseconds, an EHT beam tens of microarcseconds
        unit, scale = ("arcsec", 1.0)
        if maj < 1e-3:
            unit, scale = ("uas", 1e6)
        elif maj < 1.0:
            unit, scale = ("mas", 1e3)
        return ("GaussianBeam(bmaj=" + format(maj * scale, ".3g")
                + " " + unit + ", bmin=" + format(self.bmin * arc * scale, ".3g")
                + " " + unit + ", bpa=" + format(self.bpa_deg, ".1f") + " deg)")


def fit_clean_beam(psf, cell, level=0.35):
    """Fit an elliptical Gaussian to the main lobe of the dirty beam.

    The fit is restricted to the connected patch of pixels above ``level``
    containing the peak, which is what keeps sidelobes out of it -- a global fit
    to a sparse-array psf is dragged badly by the sidelobe structure.

    The result is the *restoring* beam: CLEAN convolves its component list with
    this, which smooths the answer back to the nominal resolution and throws away
    whatever super-resolution the deconvolution actually achieved. That is a real
    cost, not a formality, and is discussed in the README.
    """
    from scipy import ndimage, optimize

    npix = psf.shape[0]
    c = npix // 2
    mask = psf >= level * psf[c, c]
    lab, _ = ndimage.label(mask)
    mask = lab == lab[c, c]

    ii, jj = np.nonzero(mask)
    dl = (jj - c) * float(cell)
    dm = (ii - c) * float(cell)
    vals = psf[ii, jj]

    def model(p):
        sm, sn, a = p
        maj = dm * np.cos(a) + dl * np.sin(a)
        minr = -dm * np.sin(a) + dl * np.cos(a)
        return np.exp(-0.5 * ((maj / sm) ** 2 + (minr / sn) ** 2))

    # second moments of the main lobe give a starting guess that is already close
    wsum = vals.sum()
    s_mm = np.sqrt(max(np.sum(vals * dm ** 2) / wsum, (0.1 * cell) ** 2))
    s_ll = np.sqrt(max(np.sum(vals * dl ** 2) / wsum, (0.1 * cell) ** 2))
    p0 = [s_mm, s_ll, 0.0]
    lo = [0.05 * cell, 0.05 * cell, -np.pi]
    hi = [npix * cell, npix * cell, np.pi]
    fit = optimize.least_squares(lambda p: model(p) - vals, p0, bounds=(lo, hi))
    sm, sn, a = fit.x

    bmaj, bmin = sm * FWHM_PER_SIGMA, sn * FWHM_PER_SIGMA
    bpa = np.rad2deg(a)
    if bmin > bmaj:  # keep major/minor ordered, rotate the position angle to match
        bmaj, bmin = bmin, bmaj
        bpa += 90.0
    return GaussianBeam(float(bmaj), float(bmin), float((bpa + 90.0) % 180.0 - 90.0))


def restore(components_image, residual, beam, cell, add_residual=True):
    """Convolve a component image with the restoring beam and add the residual.

    The residual is added in Jy/beam and the components in Jy/pixel, which is the
    long-standing unit inconsistency of the CLEAN restored image: the two terms
    are only commensurate because the beam is peak-normalised to 1. Everyone does
    it; it is worth knowing that you are doing it.
    """
    kern = beam.image(components_image.shape[0], cell, peak=1.0)
    out = convolve_periodic(components_image, kern)
    return out + residual if add_residual else out
