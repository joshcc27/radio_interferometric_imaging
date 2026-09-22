"""Tests of the real-data path.

No archival dataset is imaged here -- that needs a download and an optional
dependency, and the README says plainly that the step is outstanding. What these
tests *do* establish is that the loading path is correct and that the imaging
pipeline runs end to end on data that arrived through it: visibilities are
written to disk in the interchange format, read back, gridded, imaged and
deconvolved, and the result is checked against the sky that produced them.

The UVFITS reader is exercised against a file this test writes itself, with the
same random-groups layout and the same units (``UU``/``VV`` in seconds) that the
EHT release uses. That catches the mistakes that actually happen -- the
seconds-versus-wavelengths conversion, the frequency axis, the Stokes collapse,
the flag handling -- without needing the real file.
"""

import numpy as np
import pytest

from src import array as arraymod
from src import clean as cleanmod
from src import forward as fw
from src import imaging as im
from src import metrics as mt
from src import realdata as rd
from src import sky

astropy = pytest.importorskip


@pytest.fixture(scope="module")
def simulated():
    """A small observation to write out, read back and image."""
    arr = arraymod.vla_y(n_per_arm=6)
    cov = arraymod.uv_track(arr, dec_deg=45.0, ha_hours=(-3.0, 3.0), n_times=30)
    npix = 64
    cell = cov.resolution_rad() / 4.0
    truth = sky.point_plus_gaussian(npix, cell, fwhm=5 * cell,
                                    point_fraction=0.5)
    data = fw.observe(truth, cell, cov, snr=20.0, rng=np.random.default_rng(0),
                      hermitian=False)
    return truth, data, cov, npix, cell


# ---------------------------------------------------------------------------
# the interchange format
# ---------------------------------------------------------------------------
def test_npz_round_trip_preserves_the_data(tmp_path, simulated):
    truth, data, cov, npix, cell = simulated
    path = rd.save_npz(tmp_path / "vis.npz", data)
    back = rd.load_npz(path, hermitian=False)
    assert len(back) == len(data)
    assert np.allclose(back.u, data.u)
    assert np.allclose(back.vis, data.vis)
    assert np.allclose(back.sigma, data.sigma)


def test_loading_augments_hermitian_by_default(tmp_path, simulated):
    truth, data, cov, npix, cell = simulated
    path = rd.save_npz(tmp_path / "vis.npz", data)
    assert len(rd.load_npz(path)) == 2 * len(data)


def test_pipeline_runs_end_to_end_on_loaded_data(tmp_path, simulated):
    """The point of the whole module: data that came off disk must image.

    Everything after the loader is the same code the simulated experiments use,
    so if this passes, the only thing standing between the project and a real
    dataset is the file itself.
    """
    truth, data, cov, npix, cell = simulated
    path = rd.save_npz(tmp_path / "vis.npz", data)
    loaded = rd.load_npz(path)

    op = fw.MeasurementOperator(loaded, npix, cell, scheme="natural")
    dirty, psf = im.dirty_image(op), im.dirty_beam(op)
    res = cleanmod.clean(dirty, psf, gain=0.1, n_iter=2000, cell=cell,
                         frac_threshold=1e-3)
    restored = res.restored(cell, add_residual=False)
    truth_blurred = im.convolve_periodic(truth, res.beam.image(npix, cell))
    assert mt.ssim(restored, truth_blurred) > 0.9
    assert np.isclose(res.model_flux, 1.0, rtol=0.15)


def test_loader_rejects_an_empty_dataset(tmp_path):
    empty = fw.Visibilities(u=np.array([]), v=np.array([]),
                            vis=np.array([], dtype=complex),
                            sigma=np.array([]))
    path = rd.save_npz(tmp_path / "empty.npz", empty)
    with pytest.raises(ValueError):
        rd.load_npz(path)


def test_loader_drops_nonpositive_sigma(tmp_path):
    """A zero or negative sigma is an infinite weight, which would silently
    dominate every cell it lands in."""
    n = 40
    rng = np.random.default_rng(1)
    sigma = np.full(n, 0.1)
    sigma[:5] = 0.0
    sigma[5:8] = -1.0
    data = fw.Visibilities(u=rng.uniform(-1e3, 1e3, n),
                          v=rng.uniform(-1e3, 1e3, n),
                          vis=rng.standard_normal(n) + 0j, sigma=sigma)
    path = rd.save_npz(tmp_path / "bad.npz", data)
    back = rd.load_npz(path, hermitian=False)
    assert len(back) == n - 8
    assert np.all(back.sigma > 0)


# ---------------------------------------------------------------------------
# empirical weights
# ---------------------------------------------------------------------------
def test_weight_rescaling_recovers_a_wrong_sigma(simulated):
    """Archive weights are often in arbitrary units. If the stated sigma is off
    by a factor of three, reduced chi-squared is off by nine and every
    chi-squared-based decision downstream is wrong."""
    truth, data, cov, npix, cell = simulated
    true_sigma = float(np.median(data.sigma))
    wrong = fw.Visibilities(u=data.u, v=data.v, vis=data.vis,
                            sigma=data.sigma * 3.0, meta=data.meta)
    fixed = rd.rescale_weights_from_scatter(wrong, n_bins=16)
    ratio = float(np.median(fixed.sigma)) / true_sigma
    assert 0.5 < ratio < 2.0


# ---------------------------------------------------------------------------
# closure phase
# ---------------------------------------------------------------------------
def test_closure_phase_is_immune_to_station_gains():
    """The property that makes closure quantities worth computing: corrupt every
    station with an arbitrary phase and the closure phase does not move.

    That is how you tell a calibration problem from an imaging problem, and it is
    why self-calibration is possible at all.
    """
    n_ant = 5
    rng = np.random.default_rng(2)
    i, j = np.triu_indices(n_ant, k=1)
    true_phase = rng.uniform(-np.pi, np.pi, len(i))
    vis = np.exp(1j * true_phase)
    data = fw.Visibilities(u=rng.standard_normal(len(i)),
                           v=rng.standard_normal(len(i)),
                           vis=vis, sigma=np.full(len(i), 0.01))
    tri = [(0, 1, 2), (0, 2, 3), (1, 3, 4)]
    before = rd.closure_phase(data, tri, ant1=i, ant2=j)

    gains = rng.uniform(-np.pi, np.pi, n_ant)
    corrupted = fw.Visibilities(
        u=data.u, v=data.v,
        vis=vis * np.exp(1j * (gains[i] - gains[j])),
        sigma=data.sigma)
    after = rd.closure_phase(corrupted, tri, ant1=i, ant2=j)

    assert np.all(np.isfinite(before))
    diff = (after - before + np.pi) % (2 * np.pi) - np.pi
    assert np.abs(diff).max() < 1e-10


def test_closure_phase_needs_antenna_indices(simulated):
    truth, data, cov, npix, cell = simulated
    with pytest.raises(ValueError):
        rd.closure_phase(data, [(0, 1, 2)])


# ---------------------------------------------------------------------------
# grid suggestion and description
# ---------------------------------------------------------------------------
def test_suggest_grid_resolves_the_longest_baseline(simulated):
    truth, data, cov, npix, cell = simulated
    got_npix, got_cell = rd.suggest_grid(data, pixels_per_beam=4.0)
    assert np.isclose(got_cell, cov.resolution_rad() / 4.0, rtol=1e-9)
    assert got_npix >= 32


def test_describe_mentions_the_key_numbers(simulated):
    truth, data, cov, npix, cell = simulated
    text = rd.describe(data)
    for word in ("visibilities", "resolution", "largest scale", "sigma"):
        assert word in text


# ---------------------------------------------------------------------------
# UVFITS, against a file the test writes itself
# ---------------------------------------------------------------------------
def _write_uvfits(path, u_sec, v_sec, vis, weight, freqs):
    """Write a minimal random-groups UVFITS in the EHT release's layout."""
    fits = pytest.importorskip("astropy.io.fits")
    n_row = len(u_sec)
    n_chan = len(freqs)
    # (n_row, n_chan, n_stokes=2, 3) -- RR and LL
    data = np.zeros((n_row, 1, 1, n_chan, 2, 3), dtype=np.float32)
    for pol in range(2):
        data[:, 0, 0, :, pol, 0] = np.real(vis)
        data[:, 0, 0, :, pol, 1] = np.imag(vis)
        data[:, 0, 0, :, pol, 2] = weight
    pardata = [u_sec.astype(np.float32), v_sec.astype(np.float32),
               np.zeros(n_row, dtype=np.float32),
               np.zeros(n_row, dtype=np.float64)]
    gd = fits.GroupData(data, parnames=["UU", "VV", "WW", "DATE"],
                        pardata=pardata, bitpix=-32)
    hdu = fits.GroupsHDU(gd)
    h = hdu.header
    df = float(freqs[1] - freqs[0]) if n_chan > 1 else 1e8
    for axis, (ctype, crval, cdelt, crpix) in enumerate(
            [("COMPLEX", 1.0, 1.0, 1.0), ("STOKES", -1.0, -1.0, 1.0),
             ("FREQ", float(freqs[0]), df, 1.0), ("RA", 0.0, 0.0, 1.0),
             ("DEC", 0.0, 0.0, 1.0)], start=2):
        h["CTYPE" + str(axis)] = ctype
        h["CRVAL" + str(axis)] = crval
        h["CDELT" + str(axis)] = cdelt
        h["CRPIX" + str(axis)] = crpix
    hdu.writeto(path, overwrite=True)
    return path


def test_uvfits_reader_recovers_the_written_visibilities(tmp_path, simulated):
    """The unit conversion is the thing to get right: UVFITS stores ``UU`` in
    seconds, so ``u`` in wavelengths is ``UU * frequency``."""
    pytest.importorskip("astropy")
    truth, data, cov, npix, cell = simulated
    freqs = np.array([1.4e9, 1.5e9])
    n = 200
    u_sec = data.u[:n] / freqs[0]
    v_sec = data.v[:n] / freqs[0]
    vis = np.repeat(data.vis[:n, None], len(freqs), axis=1)
    weight = np.full((n, len(freqs)), 1.0 / data.sigma[0] ** 2)

    path = _write_uvfits(tmp_path / "test.uvfits", u_sec, v_sec, vis, weight,
                         freqs)
    back = rd.load_uvfits(path, hermitian=False)

    assert len(back) == n * len(freqs)
    # channel 0 must come back at exactly the u it was written with
    got = back.u[:n] if np.isclose(back.u[0], data.u[0], rtol=1e-4) else None
    assert got is not None, "channel frequency scaling is wrong"
    assert np.allclose(np.sort(back.u)[-1], np.max(u_sec * freqs[-1]), rtol=1e-4)
    assert np.allclose(back.sigma, data.sigma[0], rtol=1e-4)


def test_uvfits_reader_drops_flagged_samples(tmp_path, simulated):
    """Zero or negative weights mark flagged data and must be dropped, not
    down-weighted."""
    pytest.importorskip("astropy")
    truth, data, cov, npix, cell = simulated
    freqs = np.array([1.4e9])
    n = 120
    weight = np.full((n, 1), 100.0)
    weight[:20] = 0.0
    weight[20:30] = -1.0
    path = _write_uvfits(tmp_path / "flagged.uvfits", data.u[:n] / freqs[0],
                         data.v[:n] / freqs[0], data.vis[:n, None], weight,
                         freqs)
    back = rd.load_uvfits(path, hermitian=False)
    assert len(back) == n - 30
    assert back.meta["n_flagged"] == 30


def test_uvfits_channel_averaging_is_optional_and_announced(tmp_path, simulated,
                                                            capsys):
    pytest.importorskip("astropy")
    truth, data, cov, npix, cell = simulated
    freqs = np.array([1.4e9, 1.5e9, 1.6e9])
    n = 90
    vis = np.repeat(data.vis[:n, None], len(freqs), axis=1)
    weight = np.full((n, len(freqs)), 100.0)
    path = _write_uvfits(tmp_path / "chan.uvfits", data.u[:n] / freqs[0],
                         data.v[:n] / freqs[0], vis, weight, freqs)

    expanded = rd.load_uvfits(path, hermitian=False)
    averaged = rd.load_uvfits(path, hermitian=False, average_channels=True)
    assert len(expanded) == n * len(freqs)
    assert len(averaged) == n
    assert "smears" in capsys.readouterr().out
