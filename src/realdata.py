"""Reading real visibilities into the simulator's data structures.

Status
------
The readers here are written and unit-tested against synthetic files in the same
formats, and the full imaging pipeline runs end to end on data loaded through
them (``tests/test_realdata.py``, and
``python -m src.experiments.real_data --npz ...``). **No actual archival dataset
has been imaged yet**, because that needs either ``astropy`` (for UVFITS) or
``python-casacore`` (for a measurement set), neither of which this project
depends on, plus a download. The README says what to run once one is in hand,
and says plainly that this step is outstanding rather than implying otherwise.

Formats
-------
``load_npz``
    The interchange format, and the one with no dependencies: an ``.npz`` with
    ``u``, ``v`` (wavelengths), ``vis`` (complex Stokes I) and ``sigma``. Any
    dataset can be converted into it with a few lines, which keeps the imaging
    code independent of whichever reader a given archive needs.

``load_uvfits``
    What the EHT releases look like. Needs ``astropy``. Random groups with
    ``UU``/``VV``/``WW`` in *seconds*, visibilities as
    ``(real, imag, weight)`` per complex/Stokes/frequency axis.

``load_measurement_set``
    What the NRAO archive serves. Needs ``python-casacore`` or ``casatools``.
    ``UVW`` in metres, one row per time and baseline.

What real data brings that the simulator does not
-------------------------------------------------
- **Channels.** ``(u, v)`` is measured in wavelengths, so it differs per
  channel. Averaging channels before gridding smears the longest baselines
  radially ("bandwidth smearing"); :func:`load_uvfits` therefore expands
  channels by default rather than averaging them, and says so.
- **Polarisation.** Stokes I is ``(RR + LL) / 2`` for circular feeds,
  ``(XX + YY) / 2`` for linear.
- **Flags and zero weights.** Flagged samples must be dropped, not
  down-weighted, or they poison the gridded weight function.
- **Weights that are not inverse variances.** Archive weights are often in
  arbitrary units. :func:`rescale_weights_from_scatter` re-derives sigma
  empirically from the scatter of the data themselves, which is the only way to
  get a chi-squared that means anything.
- **Residual calibration errors.** This is the real obstacle. Station gains
  drift, so the measured visibilities are not the sky's Fourier transform times
  a known constant. The standard remedy, self-calibration, alternates between
  imaging and solving for per-station gains -- a whole second project, and
  deliberately not attempted here. The consequence is that a reconstruction from
  real data should be expected to be worse than the published one, and
  :func:`closure_phase` is provided because closure quantities are the part of
  the data that residual gains cannot corrupt, which is how you tell a
  calibration problem from an imaging problem.
"""

from __future__ import annotations

import numpy as np

from .array import C_LIGHT
from .forward import Visibilities, hermitian_augment


# ---------------------------------------------------------------------------
# the dependency-free interchange format
# ---------------------------------------------------------------------------
def save_npz(path, data, extra=None):
    """Write a :class:`~src.forward.Visibilities` to the interchange format."""
    payload = dict(u=np.asarray(data.u), v=np.asarray(data.v),
                   vis=np.asarray(data.vis), sigma=np.asarray(data.sigma))
    if extra:
        payload.update(extra)
    np.savez_compressed(path, **payload)
    return path


def load_npz(path, hermitian=True):
    """Read the interchange format. ``u``, ``v`` in wavelengths."""
    with np.load(path, allow_pickle=False) as z:
        data = Visibilities(
            u=np.asarray(z["u"], dtype=float),
            v=np.asarray(z["v"], dtype=float),
            vis=np.asarray(z["vis"], dtype=complex),
            sigma=np.asarray(z["sigma"], dtype=float),
            meta={"source": str(path), "format": "npz"})
    return _finish(data, hermitian)


# ---------------------------------------------------------------------------
# UVFITS
# ---------------------------------------------------------------------------
def load_uvfits(path, hermitian=True, average_channels=False,
                stokes="I", min_weight=0.0):
    """Read a random-groups UVFITS file (the EHT release format).

    ``UU``, ``VV`` are stored in seconds; multiplying by the channel frequency
    gives wavelengths, which is the unit everything here wants.

    ``average_channels=False`` (the default) expands every channel into its own
    visibility at its own ``(u, v)``. Averaging first is the usual shortcut and
    it smears the long baselines radially, so it is off by default and reported
    when on.
    """
    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "reading UVFITS needs astropy: pip install astropy") from exc

    with fits.open(path) as hdul:
        gp = hdul[0]
        hdr = gp.header
        freqs = _uvfits_frequencies(hdr, hdul)
        uu = np.asarray(gp.data["UU"], dtype=float)   # seconds
        vv = np.asarray(gp.data["VV"], dtype=float)
        vis_raw = np.asarray(gp.data["DATA"], dtype=float)

    # DATA has shape (n_rows, ..., n_freq, n_stokes, 3); squeeze the degenerate
    # axes UVFITS likes to carry and identify the last three
    vis_raw = np.squeeze(vis_raw)
    if vis_raw.shape[-1] != 3:
        raise ValueError("expected a trailing (real, imag, weight) axis, got "
                         + str(vis_raw.shape))
    re, imv, wt = vis_raw[..., 0], vis_raw[..., 1], vis_raw[..., 2]
    vis = re + 1j * imv

    # collapse the Stokes axis to I
    if vis.ndim >= 3:
        vis, wt = _to_stokes_i(vis, wt, stokes)

    n_rows = vis.shape[0]
    n_chan = vis.shape[1] if vis.ndim > 1 else 1
    if len(freqs) != n_chan:
        freqs = np.full(n_chan, float(np.mean(freqs)))

    if average_channels:
        good = wt > min_weight
        wsum = np.where(good, wt, 0.0).sum(axis=1)
        vis = np.where(good, vis * wt, 0.0).sum(axis=1) / np.maximum(wsum, 1e-30)
        wt = wsum
        fref = float(np.mean(freqs))
        u = uu * fref
        v = vv * fref
        print("averaged " + str(n_chan) + " channels before gridding: this "
              "smears the longest baselines radially")
    else:
        u = (uu[:, None] * freqs[None, :]).ravel()
        v = (vv[:, None] * freqs[None, :]).ravel()
        vis = np.asarray(vis).reshape(n_rows * n_chan)
        wt = np.asarray(wt).reshape(n_rows * n_chan)

    keep = np.isfinite(vis) & (wt > min_weight)
    data = Visibilities(u=u[keep], v=v[keep], vis=vis[keep],
                        sigma=1.0 / np.sqrt(wt[keep]),
                        meta={"source": str(path), "format": "uvfits",
                              "n_channels": int(n_chan),
                              "freq_hz": float(np.mean(freqs)),
                              "averaged_channels": bool(average_channels),
                              "n_flagged": int((~keep).sum())})
    return _finish(data, hermitian)


def _uvfits_frequencies(hdr, hdul):
    """Channel frequencies, from the FREQ axis and any FQ table."""
    n_ax = int(hdr.get("NAXIS", 0))
    ref_freq, n_chan, d_freq, ref_pix = None, 1, 0.0, 1.0
    for k in range(1, n_ax + 1):
        if str(hdr.get("CTYPE" + str(k), "")).strip().upper() == "FREQ":
            ref_freq = float(hdr.get("CRVAL" + str(k), 0.0))
            d_freq = float(hdr.get("CDELT" + str(k), 0.0))
            ref_pix = float(hdr.get("CRPIX" + str(k), 1.0))
            n_chan = int(hdr.get("NAXIS" + str(k), 1))
            break
    if ref_freq is None:
        raise ValueError("no FREQ axis in the UVFITS header")
    return ref_freq + (np.arange(n_chan) + 1.0 - ref_pix) * d_freq


def _to_stokes_i(vis, wt, stokes):
    """Average the two parallel-hand polarisations into Stokes I.

    UVFITS orders Stokes as RR, LL (or XX, YY) first, so the two parallel hands
    are the first two planes of that axis. Cross hands carry no Stokes I.
    """
    if stokes != "I":
        raise ValueError("only Stokes I is supported, asked for " + str(stokes))
    n_pol = vis.shape[-1]
    if n_pol == 1:
        return vis[..., 0], wt[..., 0]
    w = np.clip(wt[..., :2], 0.0, None)
    num = (vis[..., :2] * w).sum(axis=-1)
    den = w.sum(axis=-1)
    out = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)
    return out, den


# ---------------------------------------------------------------------------
# measurement set
# ---------------------------------------------------------------------------
def load_measurement_set(path, hermitian=True, data_column="DATA", spw=0,
                         field=0, average_channels=False):
    """Read a CASA measurement set via python-casacore or casatools.

    ``UVW`` is in metres here, not seconds, so it is divided by the channel
    wavelength rather than multiplied by the frequency.
    """
    tb, spw_tb = _open_ms(path)
    try:
        freqs = np.asarray(spw_tb["CHAN_FREQ"][spw], dtype=float).ravel()
        uvw = np.asarray(tb["UVW"], dtype=float)
        vis = np.asarray(tb[data_column])
        flag = np.asarray(tb["FLAG"], dtype=bool)
        wspec = tb.get("WEIGHT_SPECTRUM")
        weight = np.asarray(tb["WEIGHT"], dtype=float)
        fid = np.asarray(tb.get("FIELD_ID", np.zeros(len(uvw), dtype=int)))
        ddid = np.asarray(tb.get("DATA_DESC_ID", np.zeros(len(uvw), dtype=int)))
    finally:
        pass

    sel = (fid == field) & (ddid == spw)
    uvw, vis, flag = uvw[sel], vis[sel], flag[sel]
    weight = weight[sel]
    wt = (np.asarray(wspec)[sel] if wspec is not None
          else np.broadcast_to(weight[:, None, :], vis.shape).copy())

    # vis is (n_row, n_chan, n_pol); take Stokes I from the parallel hands
    visi, wti = _to_stokes_i(vis, np.where(flag, 0.0, wt), "I")
    lam = C_LIGHT / freqs
    u = (uvw[:, 0][:, None] / lam[None, :])
    v = (uvw[:, 1][:, None] / lam[None, :])

    if average_channels:
        wsum = wti.sum(axis=1)
        visi = (visi * wti).sum(axis=1) / np.maximum(wsum, 1e-30)
        wti = wsum
        fref = float(freqs.mean())
        u = uvw[:, 0] / (C_LIGHT / fref)
        v = uvw[:, 1] / (C_LIGHT / fref)
    else:
        u, v = u.ravel(), v.ravel()
        visi, wti = visi.ravel(), wti.ravel()

    keep = np.isfinite(visi) & (wti > 0)
    data = Visibilities(u=u[keep], v=v[keep], vis=visi[keep],
                        sigma=1.0 / np.sqrt(wti[keep]),
                        meta={"source": str(path), "format": "ms",
                              "freq_hz": float(freqs.mean()),
                              "n_channels": int(len(freqs)),
                              "n_flagged": int((~keep).sum())})
    return _finish(data, hermitian)


def _open_ms(path):  # pragma: no cover - needs an optional dependency
    try:
        from casacore.tables import table

        main = table(path, ack=False)
        spw = table(path + "/SPECTRAL_WINDOW", ack=False)
        return _CasacoreView(main), _CasacoreView(spw)
    except ImportError:
        pass
    try:
        import casatools

        tb = casatools.table()
        tb.open(path)
        main = _CasaToolsView(tb)
        sp = casatools.table()
        sp.open(path + "/SPECTRAL_WINDOW")
        return main, _CasaToolsView(sp)
    except ImportError as exc:
        raise ImportError("reading a measurement set needs python-casacore or "
                          "casatools") from exc


class _CasacoreView:  # pragma: no cover - needs an optional dependency
    """Dict-like access to a casacore table, so both backends look the same."""

    def __init__(self, tb):
        self.tb = tb

    def __getitem__(self, key):
        return self.tb.getcol(key)

    def get(self, key, default=None):
        try:
            return self.tb.getcol(key)
        except Exception:
            return default


class _CasaToolsView(_CasacoreView):  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# things real data needs
# ---------------------------------------------------------------------------
def _finish(data, hermitian):
    if len(data) == 0:
        raise ValueError("no unflagged visibilities survived loading")
    bad = ~np.isfinite(data.sigma) | (data.sigma <= 0)
    if bad.any():
        keep = ~bad
        data = Visibilities(u=data.u[keep], v=data.v[keep], vis=data.vis[keep],
                            sigma=data.sigma[keep], meta=data.meta)
    return hermitian_augment(data) if hermitian else data


def rescale_weights_from_scatter(data, n_bins=24):
    """Re-derive sigma from the scatter of the data, in annuli of ``|uv|``.

    Archive weights are frequently in arbitrary units, or are simply optimistic.
    A sigma that is wrong by a factor of three makes reduced chi-squared wrong by
    a factor of nine, and then any chi-squared-based choice of regularisation
    weight is meaningless. This estimates the noise from the visibilities
    themselves: within a narrow annulus the sky signal is smooth, so the
    high-frequency scatter about a local median is dominated by noise.

    Crude but honest, and far better than trusting a weight column blind. With
    real data, cross-check it against the scatter within a single scan on one
    baseline, where the sky contribution is genuinely constant.
    """
    r = np.hypot(data.u, data.v)
    edges = np.quantile(r, np.linspace(0.0, 1.0, int(n_bins) + 1))
    edges[-1] *= 1.0 + 1e-9
    sigma = np.array(data.sigma, dtype=float).copy()
    for k in range(int(n_bins)):
        sel = (r >= edges[k]) & (r < edges[k + 1])
        if sel.sum() < 8:
            continue
        v = data.vis[sel]
        # median absolute deviation of the real and imaginary parts about the
        # annulus median, converted to a Gaussian sigma
        for part in (np.real, np.imag):
            x = part(v)
            mad = np.median(np.abs(x - np.median(x)))
            est = 1.4826 * mad / np.sqrt(2.0)
            if est > 0:
                sigma[sel] = est
    return Visibilities(u=data.u, v=data.v, vis=data.vis, sigma=sigma,
                        meta=dict(data.meta or {}, weights_rescaled=True))


def closure_phase(data, triangles, ant1=None, ant2=None):
    """Closure phases, the quantity station gain errors cannot corrupt.

    For antennas ``i, j, k`` the sum
    ``arg V_ij + arg V_jk + arg V_ki`` is independent of every per-station phase
    error, because each station's phase enters twice with opposite sign. If a
    reconstruction reproduces the amplitudes but not the closure phases,
    the problem is the image; if it reproduces the closure phases but not the
    visibility phases, the problem is calibration. Telling those apart is the
    first thing to do with real data.

    ``triangles`` is a sequence of ``(i, j, k)`` antenna indices; ``ant1`` and
    ``ant2`` are the per-visibility antenna indices, needed to find the
    baselines.
    """
    if ant1 is None or ant2 is None:
        raise ValueError("closure phases need per-visibility antenna indices")
    ant1 = np.asarray(ant1)
    ant2 = np.asarray(ant2)
    lookup = {}
    for idx, (a, b) in enumerate(zip(ant1, ant2)):
        lookup.setdefault((int(a), int(b)), []).append(idx)

    def phase(a, b):
        if (a, b) in lookup:
            return np.angle(data.vis[lookup[(a, b)][0]])
        if (b, a) in lookup:
            return -np.angle(data.vis[lookup[(b, a)][0]])
        return np.nan

    out = []
    for (i, j, k) in triangles:
        p = phase(i, j) + phase(j, k) + phase(k, i)
        out.append(np.nan if not np.isfinite(p)
                   else (p + np.pi) % (2 * np.pi) - np.pi)
    return np.asarray(out)


def describe(data):
    """A one-line summary, and the numbers you need to pick an imaging grid."""
    r = np.hypot(data.u, data.v)
    rmin = float(r[r > 0].min()) if np.any(r > 0) else 0.0
    rmax = float(r.max())
    arc = 180.0 * 3600.0 / np.pi
    return (str(len(data)) + " visibilities, |uv| "
            + format(rmin / 1e6, ".3g") + " to " + format(rmax / 1e6, ".3g")
            + " Mlambda, resolution ~" + format(arc / rmax * 1e6, ".3g")
            + " uas, largest scale ~" + format(arc / rmin * 1e6, ".3g")
            + " uas, median sigma " + format(float(np.median(data.sigma)), ".3g"))


def suggest_grid(data, pixels_per_beam=4.0, field_factor=8.0):
    """A sensible ``(npix, cell)`` for a dataset: resolution over
    ``pixels_per_beam``, and a field a few times the nominal resolution.

    Deliberately conservative on the field of view, because the field is itself a
    strong prior -- see the note in :func:`src.experiments.config.eht_setup`.
    """
    rmax = float(np.hypot(data.u, data.v).max())
    res = 1.0 / rmax
    cell = res / float(pixels_per_beam)
    npix = int(2 ** np.ceil(np.log2(field_factor * pixels_per_beam)))
    return npix, cell
