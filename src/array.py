"""Antenna layouts and uv-track generation.

Coordinate conventions
----------------------
Antenna positions are carried internally as geocentric Cartesian (ITRF-like)
coordinates in metres: x through the Greenwich meridian at the equator, y 90
degrees east of it, z through the north pole. A spherical Earth is assumed --
this is a simulator, and the ellipsoid correction is far below the level of
anything measured here.

For a baseline vector ``b = (bx, by, bz)`` in that frame, a source at
declination ``dec`` observed at Greenwich hour angle ``H`` projects onto the
plane perpendicular to the source direction as (Thompson, Moran & Swenson,
"Interferometry and Synthesis in Radio Astronomy", eq. 4.1):

    u =  sin(H) bx + cos(H) by
    v = -sin(dec) cos(H) bx + sin(dec) sin(H) by + cos(dec) bz
    w =  cos(dec) cos(H) bx - cos(dec) sin(H) by + sin(dec) bz

Dividing by the observing wavelength puts ``(u, v)`` in wavelengths, which is
the unit the van Cittert-Zernike relation wants.

As H advances, each baseline traces an ellipse in the uv plane -- semi-major
axis set by the baseline length, axial ratio by sin(dec). That is aperture
synthesis, and it is the only reason a handful of antennas gives usable
coverage at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C_LIGHT = 299792458.0
EARTH_RADIUS = 6371000.0


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def enu_to_itrf(enu, lat_deg, lon_deg, radius=EARTH_RADIUS):
    """Convert local East-North-Up offsets at a site to geocentric XYZ.

    Parameters
    ----------
    enu : (n, 3) array
        East, North, Up offsets in metres relative to the site.
    lat_deg, lon_deg : float
        Latitude and east longitude of the site, degrees.

    Returns
    -------
    (n, 3) array of geocentric XYZ positions in metres.
    """
    enu = np.atleast_2d(np.asarray(enu, dtype=float))
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)

    # rows of the ENU -> ECEF rotation
    rot = np.array(
        [
            [-so, -sl * co, cl * co],
            [co, -sl * so, cl * so],
            [0.0, cl, sl],
        ]
    )
    site = radius * np.array([cl * co, cl * so, sl])
    return site[None, :] + enu @ rot.T


def geodetic_to_itrf(lat_deg, lon_deg, height=0.0, radius=EARTH_RADIUS):
    """Geocentric XYZ for arrays specified station-by-station on the globe."""
    lat = np.deg2rad(np.atleast_1d(np.asarray(lat_deg, dtype=float)))
    lon = np.deg2rad(np.atleast_1d(np.asarray(lon_deg, dtype=float)))
    r = radius + np.atleast_1d(np.asarray(height, dtype=float)) * np.ones_like(lat)
    return np.stack(
        [r * np.cos(lat) * np.cos(lon), r * np.cos(lat) * np.sin(lon), r * np.sin(lat)],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# array layouts
# ---------------------------------------------------------------------------
@dataclass
class Array:
    """An interferometer: antenna positions plus the site geometry we need."""

    name: str
    xyz: np.ndarray  # (n_ant, 3) geocentric metres
    lat_deg: np.ndarray  # (n_ant,) per-antenna latitude, for elevation limits
    lon_deg: np.ndarray  # (n_ant,) per-antenna east longitude
    station_names: list | None = None

    @property
    def n_ant(self) -> int:
        return self.xyz.shape[0]

    @property
    def n_baselines(self) -> int:
        return self.n_ant * (self.n_ant - 1) // 2

    def baseline_vectors(self):
        """All ``n(n-1)/2`` baseline vectors and the antenna index pairs."""
        i, j = np.triu_indices(self.n_ant, k=1)
        return self.xyz[j] - self.xyz[i], i, j


def vla_y(n_per_arm=9, r_min=50.0, r_max=1000.0, lat_deg=34.079, lon_deg=-107.618,
          arm_pa_deg=(5.0, 125.0, 245.0), scale=1.0):
    """A scaled VLA-like Y configuration.

    Three arms at position angles ``arm_pa_deg`` measured east of north. Station
    radii follow the ``r ~ i**1.716`` power law the real VLA uses, which is what
    gives it a roughly uniform radial uv density.

    ``n_per_arm=9`` gives 27 antennas and 351 baselines.
    """
    idx = np.arange(1, n_per_arm + 1, dtype=float)
    radii = r_min + (r_max - r_min) * (
        (idx ** 1.716 - 1.0) / (n_per_arm ** 1.716 - 1.0)
    )
    radii = radii * scale

    enu = []
    names = []
    arm_labels = "WEN"
    for arm, pa in enumerate(arm_pa_deg):
        a = np.deg2rad(pa)
        for k, r in enumerate(radii):
            enu.append([r * np.sin(a), r * np.cos(a), 0.0])
            names.append(arm_labels[arm] + str(k + 1))
    enu = np.asarray(enu)

    xyz = enu_to_itrf(enu, lat_deg, lon_deg)
    n = xyz.shape[0]
    return Array(
        name="VLA-Y(" + str(n) + ")",
        xyz=xyz,
        lat_deg=np.full(n, float(lat_deg)),
        lon_deg=np.full(n, float(lon_deg)),
        station_names=names,
    )


# Approximate geodetic positions of the 2017 EHT stations (lat, lon, height).
# Close enough for coverage experiments; not for astrometry.
EHT_STATIONS = {
    "ALMA": (-23.029, -67.755, 5070.0),
    "APEX": (-23.006, -67.759, 5105.0),
    "SMT": (32.702, -109.891, 3159.0),
    "LMT": (18.985, -97.315, 4593.0),
    "SMA": (19.824, -155.478, 4115.0),
    "JCMT": (19.823, -155.477, 4120.0),
    "PV": (37.066, -3.393, 2920.0),
    "SPT": (-90.000, 0.000, 2800.0),
    "GLT": (76.531, -68.703, 80.0),
}


def eht_like(stations=None):
    """An EHT-like global array: a few stations, genuinely sparse uv coverage."""
    stations = list(EHT_STATIONS) if stations is None else list(stations)
    lat = np.array([EHT_STATIONS[s][0] for s in stations])
    lon = np.array([EHT_STATIONS[s][1] for s in stations])
    hgt = np.array([EHT_STATIONS[s][2] for s in stations])
    return Array(
        name="EHT-like(" + str(len(stations)) + ")",
        xyz=geodetic_to_itrf(lat, lon, hgt),
        lat_deg=lat,
        lon_deg=lon,
        station_names=stations,
    )


def outermost(array, n=3):
    """Indices of the ``n`` antennas furthest from the array centroid.

    The coverage sweep pins these. Thinning at random shortens the longest
    baseline as well as thinning the plane, which would confound "sparser
    coverage" with "poorer resolution" -- two different reasons for a
    reconstruction to get worse. Keeping the extremes holds ``u_max``, and hence
    the nominal resolution, roughly fixed while the filling factor drops.
    """
    r = np.linalg.norm(array.xyz - array.xyz.mean(axis=0), axis=1)
    return np.sort(np.argsort(r)[::-1][:int(n)])


def thin(array, n_keep, seed=0, pin=None):
    """Drop antennas at random, keeping ``n_keep`` of them.

    ``pin`` is a set of antenna indices that are always retained; pass
    ``outermost(array, 3)`` to hold the resolution fixed across a sweep. It
    defaults to antenna 0, so every sub-array at least shares a reference
    station and the sweep does not wander the array centre around as well.
    """
    if n_keep >= array.n_ant:
        return array
    pin = np.array([0]) if pin is None else np.unique(np.asarray(pin, dtype=int))
    if n_keep < pin.size:
        raise ValueError("n_keep=" + str(n_keep) + " is smaller than the "
                         + str(pin.size) + " pinned antennas")
    rng = np.random.default_rng(seed)
    pool = np.setdiff1d(np.arange(array.n_ant), pin)
    picked = rng.choice(pool, size=n_keep - pin.size, replace=False)
    keep = np.sort(np.concatenate([pin, picked]))
    names = None
    if array.station_names is not None:
        names = [array.station_names[k] for k in keep]
    return Array(
        name=array.name + "-thin" + str(n_keep),
        xyz=array.xyz[keep],
        lat_deg=array.lat_deg[keep],
        lon_deg=array.lon_deg[keep],
        station_names=names,
    )


# ---------------------------------------------------------------------------
# uv sampling
# ---------------------------------------------------------------------------
def baselines_to_uvw(b_xyz, hour_angle, dec):
    """Project geocentric baseline vectors onto the uvw frame of the source.

    ``b_xyz`` is ``(n_bl, 3)`` in metres, ``hour_angle`` scalar or ``(n_t,)`` in
    radians, ``dec`` a scalar in radians. Returns ``(n_t, n_bl, 3)`` in metres.
    """
    b_xyz = np.atleast_2d(b_xyz)
    H = np.atleast_1d(np.asarray(hour_angle, dtype=float))
    sH, cH = np.sin(H)[:, None], np.cos(H)[:, None]
    sd, cd = np.sin(dec), np.cos(dec)
    bx, by, bz = b_xyz[None, :, 0], b_xyz[None, :, 1], b_xyz[None, :, 2]
    ones = np.ones_like(sH)

    u = sH * bx + cH * by
    v = -sd * cH * bx + sd * sH * by + cd * bz * ones
    w = cd * cH * bx - cd * sH * by + sd * bz * ones
    return np.stack([u, v, w], axis=-1)


def elevation(lat_deg, lon_deg, hour_angle, dec):
    """Source elevation in radians at each station, shape ``(n_t, n_ant)``.

    ``hour_angle`` is the *Greenwich* hour angle; the local hour angle at a
    station is ``H + lon``.
    """
    H = np.atleast_1d(np.asarray(hour_angle, dtype=float))[:, None]
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))[None, :]
    lha = H + np.deg2rad(np.asarray(lon_deg, dtype=float))[None, :]
    sin_el = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(lha)
    return np.arcsin(np.clip(sin_el, -1.0, 1.0))


@dataclass
class UVCoverage:
    """Sampled uv points, in wavelengths, with the bookkeeping to trace them back."""

    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    ant1: np.ndarray
    ant2: np.ndarray
    time_index: np.ndarray
    freq_hz: float
    dec_rad: float
    array_name: str = ""

    def __len__(self) -> int:
        return int(self.u.size)

    @property
    def uv(self):
        return np.stack([self.u, self.v], axis=-1)

    @property
    def radius(self):
        return np.hypot(self.u, self.v)

    @property
    def max_baseline(self):
        return float(self.radius.max()) if len(self) else 0.0

    @property
    def min_baseline(self):
        return float(self.radius.min()) if len(self) else 0.0

    def resolution_rad(self):
        """Nominal diffraction-limited resolution ~ 1 / (longest baseline)."""
        return 1.0 / self.max_baseline

    def largest_angular_scale_rad(self):
        """Biggest structure the array is sensitive to ~ 1 / (shortest baseline)."""
        return 1.0 / self.min_baseline


def uv_track(array, dec_deg, ha_hours=(-4.0, 4.0), n_times=60,
             freq_hz=1.4e9, min_elevation_deg=15.0):
    """Generate a uv track by sweeping hour angle.

    A baseline contributes at a timestep only if *both* its antennas see the
    source above ``min_elevation_deg``. For a compact array that just trims the
    ends of the track; for a global array it is the main reason coverage is
    patchy, and switching it off would make the EHT case look far better than
    it really is.
    """
    dec = np.deg2rad(float(dec_deg))
    lam = C_LIGHT / float(freq_hz)
    ha = np.deg2rad(np.linspace(ha_hours[0], ha_hours[1], int(n_times)) * 15.0)

    b, i, j = array.baseline_vectors()
    uvw = baselines_to_uvw(b, ha, dec) / lam  # (n_t, n_bl, 3) in wavelengths

    el = elevation(array.lat_deg, array.lon_deg, ha, dec)  # (n_t, n_ant)
    up = el > np.deg2rad(min_elevation_deg)
    visible = up[:, i] & up[:, j]  # (n_t, n_bl)

    t_idx = np.broadcast_to(np.arange(len(ha))[:, None], visible.shape)
    a1 = np.broadcast_to(i[None, :], visible.shape)
    a2 = np.broadcast_to(j[None, :], visible.shape)

    return UVCoverage(
        u=uvw[..., 0][visible],
        v=uvw[..., 1][visible],
        w=uvw[..., 2][visible],
        ant1=a1[visible],
        ant2=a2[visible],
        time_index=t_idx[visible],
        freq_hz=float(freq_hz),
        dec_rad=dec,
        array_name=array.name,
    )


def full_coverage(npix, cell_rad):
    """Every cell of the imaging uv grid sampled exactly once.

    Not physical -- this is the reference case for the round-trip test, where a
    perfectly sampled Fourier plane has to invert back to the truth.
    """
    du = 1.0 / (npix * cell_rad)
    k = (np.arange(npix) - npix // 2) * du
    uu, vv = np.meshgrid(k, k, indexing="xy")  # axis 1 -> u, axis 0 -> v
    n = uu.size
    return UVCoverage(
        u=uu.ravel(),
        v=vv.ravel(),
        w=np.zeros(n),
        ant1=np.zeros(n, dtype=int),
        ant2=np.zeros(n, dtype=int),
        time_index=np.zeros(n, dtype=int),
        freq_hz=1.4e9,
        dec_rad=np.pi / 2,
        array_name="full-coverage",
    )
