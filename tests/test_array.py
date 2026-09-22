"""Tests of the geometry: baselines, projection, and the uv ellipses.

These check physics rather than code. If the projection is wrong the uv coverage
plot looks wrong too, but only if you know what to look for -- these assertions
know what to look for.
"""

import numpy as np
import pytest

from src import array as arraymod


def test_baseline_count_is_quadratic():
    for n_per_arm, expect_ant in ((9, 27), (5, 15)):
        arr = arraymod.vla_y(n_per_arm=n_per_arm)
        assert arr.n_ant == expect_ant
        assert arr.n_baselines == expect_ant * (expect_ant - 1) // 2
        b, i, j = arr.baseline_vectors()
        assert b.shape == (arr.n_baselines, 3)
        assert np.all(i < j)  # each pair once, no autocorrelations


def test_enu_round_trip():
    """A station at zero ENU offset sits at the site position, and offsets are
    orthonormal in the local frame."""
    lat, lon = 34.079, -107.618
    p = arraymod.enu_to_itrf([[0.0, 0.0, 0.0]], lat, lon)[0]
    assert np.isclose(np.linalg.norm(p), arraymod.EARTH_RADIUS)
    offs = arraymod.enu_to_itrf(np.eye(3) * 1000.0, lat, lon) - p
    for k in range(3):
        assert np.isclose(np.linalg.norm(offs[k]), 1000.0)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert abs(float(offs[a] @ offs[b])) < 1e-6


def test_pole_source_traces_a_circle():
    """At ``dec = 90`` a horizontal baseline keeps a constant ``|uv|``.

    The projected baseline just rotates as the Earth turns, which is the
    best-possible case for coverage and a clean check that the rotation in
    :func:`baselines_to_uvw` is a rotation.
    """
    b = np.array([[900.0, -400.0, 0.0]])
    ha = np.linspace(0.0, 2 * np.pi, 36, endpoint=False)
    uvw = arraymod.baselines_to_uvw(b, ha, np.pi / 2)
    r = np.hypot(uvw[:, 0, 0], uvw[:, 0, 1])
    assert np.allclose(r, np.hypot(900.0, 400.0))
    assert np.allclose(uvw[:, 0, 2], 0.0)


@pytest.mark.parametrize("dec_deg", [20.0, 45.0, 70.0])
def test_uv_ellipse_axial_ratio_is_sin_dec(dec_deg):
    """A baseline with no ``z`` component traces an ellipse with axial ratio
    ``sin(dec)`` centred on the origin -- the textbook result."""
    # a full turn sampled without repeating the endpoint: duplicating H = 0
    # biases the mean by one sample of the oscillating term, which is well above
    # the tolerance these assertions want
    b = np.array([[1200.0, 300.0, 0.0]])
    ha = np.linspace(0.0, 2 * np.pi, 720, endpoint=False)
    uvw = arraymod.baselines_to_uvw(b, ha, np.deg2rad(dec_deg))
    u, v = uvw[:, 0, 0], uvw[:, 0, 1]
    assert np.isclose(np.ptp(v) / np.ptp(u), np.sin(np.deg2rad(dec_deg)), rtol=1e-3)
    assert abs(u.mean()) < 1e-6 * np.ptp(u)
    assert abs(v.mean()) < 1e-6 * np.ptp(u)


def test_nonzero_bz_offsets_the_ellipse_in_v():
    """A north-south component shifts the ellipse centre to ``v = bz cos(dec)``,
    which is why an east-west-only array cannot image a low-declination source
    well however long it tracks."""
    bz = 500.0
    dec = np.deg2rad(35.0)
    b = np.array([[1000.0, 0.0, bz]])
    ha = np.linspace(0.0, 2 * np.pi, 720, endpoint=False)
    uvw = arraymod.baselines_to_uvw(b, ha, dec)
    assert np.isclose(uvw[:, 0, 1].mean(), bz * np.cos(dec), rtol=1e-6)


def test_elevation_limit_removes_baselines():
    """A global array loses most of its baselines most of the time, and the
    elevation cut is what encodes that."""
    eht = arraymod.eht_like()
    full = arraymod.uv_track(eht, dec_deg=12.4, ha_hours=(-6, 6), n_times=40,
                             freq_hz=230e9, min_elevation_deg=0.0)
    cut = arraymod.uv_track(eht, dec_deg=12.4, ha_hours=(-6, 6), n_times=40,
                            freq_hz=230e9, min_elevation_deg=15.0)
    assert len(cut) < len(full)
    assert len(cut) > 0


def test_thin_pins_requested_antennas_and_holds_umax():
    """The sweep needs thinning that does not also shorten the array."""
    arr = arraymod.vla_y()
    pin = arraymod.outermost(arr, 3)
    full = arraymod.uv_track(arr, dec_deg=45.0, n_times=20)
    for n in (20, 12, 6, 3):
        t = arraymod.thin(arr, n, seed=2, pin=pin)
        assert t.n_ant == n
        cov = arraymod.uv_track(t, dec_deg=45.0, n_times=20)
        assert np.isclose(cov.max_baseline, full.max_baseline, rtol=1e-12)
    with pytest.raises(ValueError):
        arraymod.thin(arr, 2, pin=pin)


def test_resolution_scales_with_frequency():
    """Same array, higher frequency, finer beam -- the baselines are measured in
    wavelengths, so this is the only place the frequency enters."""
    arr = arraymod.vla_y()
    lo = arraymod.uv_track(arr, dec_deg=45.0, n_times=10, freq_hz=1.4e9)
    hi = arraymod.uv_track(arr, dec_deg=45.0, n_times=10, freq_hz=14e9)
    assert np.isclose(lo.resolution_rad() / hi.resolution_rad(), 10.0, rtol=1e-9)


def test_nothing_samples_the_origin():
    """No baseline is zero length, so total flux is never measured."""
    cov = arraymod.uv_track(arraymod.vla_y(), dec_deg=45.0, n_times=30)
    assert cov.min_baseline > 0.0
