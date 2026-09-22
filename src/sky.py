"""Ground-truth sky models.

Every model returns an ``(npix, npix)`` non-negative float array using the
indexing convention fixed in :mod:`src.forward`: axis 0 is ``m`` (north), axis 1
is ``l`` (east), pixel size ``cell`` radians, phase centre at
``[npix//2, npix//2]``.

Unless stated otherwise each model is normalised to unit total flux, so "flux
error" means the same thing across models and the unmeasured zero-spacing value
is 1 everywhere.

The models are ordered by how much they hurt:

- :func:`point_source` -- the calibration of the whole pipeline. Its dirty image
  *is* the dirty beam, exactly, which is the sharpest test in ``tests/``.
- :func:`double_source` -- resolution. Two points at a controlled separation,
  which either survive deconvolution as two or do not.
- :func:`point_plus_gaussian` -- dynamic range plus extended emission, the
  combination CLEAN is worst at.
- :func:`crescent` -- the black-hole test case: a ring with a brightness
  gradient, so a reconstruction can get the ring right and the asymmetry wrong.
- :func:`sharp_square` and :func:`smooth_gaussian` -- the honesty test. A square
  violates the smoothness maximum entropy assumes; a broad Gaussian violates the
  piecewise-flat structure total variation assumes.
"""

from __future__ import annotations

import numpy as np

from .forward import image_axes


def _grid(npix, cell):
    """``(m, l)`` coordinate grids matching the image indexing convention."""
    ax = image_axes(npix, cell)
    return np.meshgrid(ax, ax, indexing="ij")


def _normalise(img, total_flux):
    s = img.sum()
    if s <= 0:
        raise ValueError("sky model has no flux")
    return img * (float(total_flux) / s)


def _pixel_of(l, m, npix, cell):
    """Nearest pixel indices ``(i, j)`` for an offset in radians."""
    j = int(np.rint(l / cell)) + npix // 2
    i = int(np.rint(m / cell)) + npix // 2
    if not (0 <= i < npix and 0 <= j < npix):
        raise ValueError("source at l=" + str(l) + ", m=" + str(m) + " is off the image")
    return i, j


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def point_source(npix, cell, flux=1.0, offset=(0.0, 0.0)):
    """A single delta function, by default exactly at the phase centre.

    A centred unit point source has ``V(u, v) = 1`` on every baseline, which is
    why it pins down the forward model's normalisation with no room to hide.
    ``offset`` is ``(l, m)`` in radians.
    """
    img = np.zeros((npix, npix))
    i, j = _pixel_of(offset[0], offset[1], npix, cell)
    img[i, j] = float(flux)
    return img


def double_source(npix, cell, separation, flux_ratio=1.0, pa_deg=0.0,
                  total_flux=1.0):
    """Two point sources separated by ``separation`` radians.

    ``pa_deg`` is the position angle of the pair east of north, so ``pa_deg=0``
    separates them in ``m`` and ``pa_deg=90`` in ``l``. Sweeping ``separation``
    against the nominal resolution ``1/u_max`` is the cheapest quantitative
    statement about super-resolution the project can make.
    """
    a = np.deg2rad(float(pa_deg))
    half = 0.5 * float(separation)
    dl, dm = half * np.sin(a), half * np.cos(a)
    img = np.zeros((npix, npix))
    i1, j1 = _pixel_of(-dl, -dm, npix, cell)
    i2, j2 = _pixel_of(dl, dm, npix, cell)
    img[i1, j1] += 1.0
    img[i2, j2] += float(flux_ratio)
    return _normalise(img, total_flux)


def gaussian_blob(npix, cell, fwhm, centre=(0.0, 0.0), axial_ratio=1.0,
                  pa_deg=0.0, total_flux=1.0):
    """An elliptical Gaussian with the given FWHM (radians) on its major axis."""
    mm, ll = _grid(npix, cell)
    dl = ll - float(centre[0])
    dm = mm - float(centre[1])
    a = np.deg2rad(float(pa_deg))
    # rotate into the source frame; major axis lies along m at pa = 0
    maj = dm * np.cos(a) + dl * np.sin(a)
    minr = -dm * np.sin(a) + dl * np.cos(a)
    sig = float(fwhm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    img = np.exp(-0.5 * ((maj / sig) ** 2 + (minr / (sig * axial_ratio)) ** 2))
    return _normalise(img, total_flux)


def point_plus_gaussian(npix, cell, fwhm, point_fraction=0.5, offset=(0.0, 0.0),
                        total_flux=1.0):
    """A compact core on top of a broad halo.

    ``point_fraction`` of the flux sits in the delta function. The pairing is
    deliberate: CLEAN's point-source prior nails the core and shreds the halo
    into a speckle of components, and you can watch that happen.
    """
    img = gaussian_blob(npix, cell, fwhm, total_flux=1.0 - float(point_fraction))
    i, j = _pixel_of(offset[0], offset[1], npix, cell)
    img[i, j] += float(point_fraction)
    return _normalise(img, total_flux)


def crescent(npix, cell, radius, width, asymmetry=0.6, pa_deg=90.0,
             floor_fraction=0.0, total_flux=1.0):
    """A ring of the given radius and Gaussian cross-section ``width`` (FWHM),
    carrying a brightness gradient around it.

    The azimuthal profile is ``1 + asymmetry * cos(phi - pa)`` with ``phi``
    measured east of north, so ``asymmetry=0`` is a symmetric ring and
    ``asymmetry -> 1`` puts nearly everything on one side. This is the standard
    black-hole test image: a hole, a resolved ring, and an asymmetry, and a
    reconstruction can plausibly recover two of the three while losing the third.

    ``floor_fraction`` adds a faint uniform disc inside the ring, which gives a
    deliberately low-contrast feature for survival tests.
    """
    mm, ll = _grid(npix, cell)
    r = np.hypot(ll, mm)
    phi = np.arctan2(ll, mm)  # east of north
    sig = float(width) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    ring = np.exp(-0.5 * ((r - float(radius)) / sig) ** 2)
    mod = 1.0 + float(asymmetry) * np.cos(phi - np.deg2rad(float(pa_deg)))
    img = ring * np.clip(mod, 0.0, None)
    if floor_fraction > 0:
        disc = (r < float(radius)).astype(float)
        if disc.sum() > 0:
            img = img / img.sum() * (1.0 - float(floor_fraction))
            img = img + disc / disc.sum() * float(floor_fraction)
    return _normalise(img, total_flux)


def sharp_square(npix, cell, side, centre=(0.0, 0.0), total_flux=1.0):
    """A flat-topped square with hard edges.

    The honesty-test target for maximum entropy, whose penalty is minimised by
    smooth images and which therefore cannot produce a hard edge however good
    the data are.
    """
    mm, ll = _grid(npix, cell)
    half = 0.5 * float(side)
    img = ((np.abs(ll - centre[0]) <= half) & (np.abs(mm - centre[1]) <= half))
    return _normalise(img.astype(float), total_flux)


def smooth_gaussian(npix, cell, fwhm=None, total_flux=1.0):
    """A single broad Gaussian: the honesty-test target for total variation,
    which prefers piecewise-flat images and tends to staircase a smooth gradient.
    """
    fwhm = (npix * cell) / 6.0 if fwhm is None else fwhm
    return gaussian_blob(npix, cell, fwhm, total_flux=total_flux)


MODELS = {
    "point": point_source,
    "double": double_source,
    "point_gaussian": point_plus_gaussian,
    "crescent": crescent,
    "square": sharp_square,
    "gaussian": smooth_gaussian,
}
