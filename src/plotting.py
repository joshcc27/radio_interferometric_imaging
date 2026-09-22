"""Figure helpers, so every plot in the project uses the same conventions.

Images are drawn with east to the left and north up, which is the radio
convention and the opposite of what ``imshow`` does by default. Getting this
wrong mirrors every image in the project and is invisible until you try to
compare a crescent's asymmetry against the truth, so it is fixed in one place.
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib

if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .forward import ARCSEC, UAS  # noqa: E402

FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "figures")
RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results")

UNITS = {"arcsec": (ARCSEC, "arcsec"), "uas": (UAS, "$\\mu$as"),
         "mas": (ARCSEC * 1e-3, "mas")}


def save(fig, name, dpi=140):
    """Write a figure into ``figures/`` and return the path."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, name if name.endswith(".png") else name + ".png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote " + path)
    return path


def result_path(name):
    os.makedirs(RESULT_DIR, exist_ok=True)
    return os.path.join(RESULT_DIR, name)


def _extent(npix, cell, unit):
    scale = UNITS[unit][0]
    half = 0.5 * cell / scale
    lo = (0 - npix // 2) * cell / scale - half
    hi = (npix - 1 - npix // 2) * cell / scale + half
    return (hi, lo, lo, hi)  # x reversed: east to the left


def show_image(ax, img, cell, unit="arcsec", title=None, cmap="afmhot",
               vmin=None, vmax=None, symmetric=False, colorbar=True,
               label=None, bare=False):
    """Draw a sky image with the radio orientation and angular axes.

    ``bare`` drops the tick labels and axis labels, which is what a row of
    panels sharing one coordinate frame needs -- otherwise adjacent panels'
    tick labels run into each other.
    """
    img = np.asarray(img)
    npix = img.shape[0]
    if symmetric:
        v = np.abs(img).max()
        vmin, vmax = -v, v
        cmap = "RdBu_r"
    kw = dict(origin="lower", extent=_extent(npix, cell, unit), cmap=cmap,
              vmin=vmin, vmax=vmax, interpolation="nearest")
    hi = ax.imshow(img, **kw)
    if bare:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel("$l$ (" + UNITS[unit][1] + ", east left)")
        ax.set_ylabel("$m$ (" + UNITS[unit][1] + ")")
    if title:
        ax.set_title(title, fontsize=10)
    if colorbar:
        cb = ax.figure.colorbar(hi, ax=ax, fraction=0.046, pad=0.03)
        if label:
            cb.set_label(label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    return hi


def show_uv(ax, coverage, unit="kilo", title=None, s=0.6, color="#1f5fa8",
            hermitian=True):
    """Scatter the uv coverage, both halves of the Hermitian pair."""
    div, lab = (1e3, "k$\\lambda$") if unit == "kilo" else (1e6, "M$\\lambda$")
    u = np.asarray(coverage.u) / div
    v = np.asarray(coverage.v) / div
    if hermitian:
        u = np.concatenate([u, -u])
        v = np.concatenate([v, -v])
    ax.scatter(u, v, s=s, c=color, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("$u$ (" + lab + ")")
    ax.set_ylabel("$v$ (" + lab + ")")
    ax.invert_xaxis()  # u axis conventionally increases to the east, i.e. left
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, lw=0.4)
    return ax


def show_array(ax, arr, title=None):
    """Plot antenna positions, projected sensibly for local or global arrays.

    For a compact array the station latitudes and longitudes are all the site
    value, so the layout has to come from the geocentric positions: rotate the
    offsets from the array centre back into the local East-North frame.
    """
    if np.ptp(arr.lat_deg) < 1.0 and np.ptp(arr.lon_deg) < 1.0:
        lat0 = np.deg2rad(arr.lat_deg.mean())
        lon0 = np.deg2rad(arr.lon_deg.mean())
        sl, cl = np.sin(lat0), np.cos(lat0)
        so, co = np.sin(lon0), np.cos(lon0)
        rot = np.array([[-so, -sl * co, cl * co],
                        [co, -sl * so, cl * so],
                        [0.0, cl, sl]])           # ENU -> ECEF
        enu = (arr.xyz - arr.xyz.mean(axis=0)) @ rot  # ECEF -> ENU is rot^T, i.e. x @ rot
        ax.scatter(enu[:, 0] / 1e3, enu[:, 1] / 1e3, s=18, c="#1f5fa8")
        ax.set_xlabel("east (km)")
        ax.set_ylabel("north (km)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.25, lw=0.4)
    else:
        ax.scatter(arr.lon_deg, arr.lat_deg, s=26, c="#1f5fa8")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("east longitude (deg)")
        ax.set_ylabel("latitude (deg)")
        ax.grid(alpha=0.3, lw=0.4)
        if arr.station_names:
            # co-located stations (ALMA/APEX, SMA/JCMT) would print on top of
            # each other, so alternate the label offset
            for k, (nm, lo, la) in enumerate(
                    zip(arr.station_names, arr.lon_deg, arr.lat_deg)):
                dy = 4 if k % 2 == 0 else -9
                ax.annotate(nm, (lo, la), fontsize=6,
                            textcoords="offset points", xytext=(4, dy))
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def add_beam_marker(ax, beam, cell, npix, unit="arcsec", frac=0.12):
    """Draw the restoring beam ellipse in the lower-left corner of an image."""
    from matplotlib.patches import Ellipse

    scale = UNITS[unit][0]
    half = 0.5 * npix * cell / scale
    x = half - 2.2 * frac * half
    y = -half + 1.4 * frac * half
    e = Ellipse((x, y), width=beam.bmin / scale, height=beam.bmaj / scale,
                angle=-beam.bpa_deg, facecolor="none", edgecolor="#39d0ff", lw=1.0)
    ax.add_patch(e)
    return e
