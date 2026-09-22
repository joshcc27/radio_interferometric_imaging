"""The two standard observing setups every experiment shares.

Fixing these in one place is what lets the coverage sweep, the weighting
comparison and the Bayesian run be compared against each other rather than
against their own private choices of pixel size.

The pixel size is always a fixed fraction of the nominal resolution
``1/u_max``, so "four pixels across the beam" means the same thing for the VLA
at 1.4 GHz and the EHT at 230 GHz.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import array as arraymod
from .. import sky
from ..forward import ARCSEC, UAS


@dataclass
class Setup:
    """An array, a source, and the imaging grid that follows from them."""

    name: str
    array: object
    dec_deg: float
    ha_hours: tuple
    n_times: int
    freq_hz: float
    npix: int
    pixels_per_beam: float
    unit: str
    cell: float = field(init=False)
    coverage: object = field(init=False)

    def __post_init__(self):
        self.coverage = self.uv_track(self.array)
        self.cell = self.coverage.resolution_rad() / float(self.pixels_per_beam)

    def uv_track(self, arr):
        """The uv track of an arbitrary (possibly thinned) array, same schedule."""
        return arraymod.uv_track(arr, dec_deg=self.dec_deg, ha_hours=self.ha_hours,
                                 n_times=self.n_times, freq_hz=self.freq_hz)

    @property
    def fov(self):
        return self.npix * self.cell

    def describe(self):
        scale, lab = (ARCSEC, "arcsec") if self.unit == "arcsec" else (UAS, "uas")
        c = self.coverage
        return (self.name + ": " + self.array.name + ", " + str(len(c)) + " vis, "
                + "res " + format(c.resolution_rad() / scale, ".3g") + " " + lab
                + ", cell " + format(self.cell / scale, ".3g") + " " + lab
                + ", FOV " + format(self.fov / scale, ".3g") + " " + lab)


def vla_setup(npix=128, pixels_per_beam=4.0, n_times=60, ha_hours=(-4.0, 4.0)):
    """VLA-like Y array, 27 antennas, 1.4 GHz, source at dec +45.

    Well-behaved coverage: smooth elliptical arcs, a beam with ~12 per cent
    sidelobes. This is the case where all three reconstruction methods should
    agree, which is what makes the sparse case interesting.
    """
    return Setup(name="VLA", array=arraymod.vla_y(), dec_deg=45.0,
                 ha_hours=ha_hours, n_times=n_times, freq_hz=1.4e9,
                 npix=npix, pixels_per_beam=pixels_per_beam, unit="arcsec")


def eht_setup(npix=64, pixels_per_beam=4.0, n_times=80, ha_hours=(-6.0, 6.0)):
    """EHT-like global array, 9 stations, 230 GHz, source at the declination of M87.

    Genuinely sparse: a few hundred visibilities, strong elevation gaps, and a
    psf whose sidelobes are a large fraction of its peak. This is where the
    prior does the work.

    ``npix=64`` gives a field about 400 microarcseconds across, four times the
    diameter of the test crescent. That is a deliberate choice, not a
    convenience: the field of view is itself a prior, and a strong one. With a
    field eight or sixteen times the source, nothing in the data says the outer
    region is empty, every method scatters flux into it, and the reconstructions
    look far worse than published EHT images do -- for a reason that is about the
    field and not about the algorithms. Any real observer picks a field a few
    times the emission, so that is what is done here, and the effect of widening
    it is reported in the README rather than buried.
    """
    return Setup(name="EHT", array=arraymod.eht_like(), dec_deg=12.391,
                 ha_hours=ha_hours, n_times=n_times, freq_hz=230e9,
                 npix=npix, pixels_per_beam=pixels_per_beam, unit="uas")


# ---------------------------------------------------------------------------
# the standard ground truths, sized relative to the beam of each setup
# ---------------------------------------------------------------------------
def truth_for(setup, kind="crescent"):
    """Build a ground truth scaled to the setup's resolution.

    Sizes are given in beams, not pixels, so the same call gives a comparably
    difficult source for both arrays.
    """
    npix, cell = setup.npix, setup.cell
    res = setup.coverage.resolution_rad()
    if kind == "point":
        return sky.point_source(npix, cell)
    if kind == "double":
        return sky.double_source(npix, cell, separation=1.5 * res, pa_deg=30.0)
    if kind == "point_gaussian":
        return sky.point_plus_gaussian(npix, cell, fwhm=4.0 * res,
                                       point_fraction=0.4)
    if kind == "crescent":
        return sky.crescent(npix, cell, radius=2.0 * res, width=1.0 * res,
                            asymmetry=0.6, pa_deg=90.0)
    if kind == "square":
        return sky.sharp_square(npix, cell, side=4.0 * res)
    if kind == "gaussian":
        return sky.smooth_gaussian(npix, cell, fwhm=4.0 * res)
    raise ValueError("unknown truth: " + str(kind))


TRUTH_KINDS = ("point", "double", "point_gaussian", "crescent", "square", "gaussian")

# The per-visibility SNR used throughout. mean|V| / sigma = 20 is comfortable
# but not generous: the image-plane dynamic range ceiling is roughly
# snr * sqrt(n_vis), which is a few thousand for the VLA and a few hundred for
# the EHT setup -- so the EHT case is noise-limited as well as coverage-limited,
# exactly as the real thing is.
DEFAULT_SNR = 20.0
