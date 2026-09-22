"""One observation, three reconstructions, one set of scores.

Every experiment goes through here, so that the comparisons between methods are
comparisons of methods and not of incidental choices.

Two decisions are worth stating up front, because they are the ones that make or
break a fair comparison.

**Which operator.** The dirty image and the dirty beam are gridded quantities by
definition, so CLEAN runs against :class:`~src.forward.MeasurementOperator`.
The regularised solvers instead use :class:`~src.forward.DegridOperator`, whose
forward-model error is ~1e-4 of the peak visibility rather than ~5e-2, because
otherwise "fit to reduced chi-squared 1" would be fitting the gridding error.
Every method's chi-squared and residual is then reported through the *accurate*
operator, so the numbers are on one footing regardless of how the image was
produced.

**Which resolution.** CLEAN's model is a list of delta functions in Jy/pixel and
its restored image is in Jy/beam; TV and MEM produce Jy/pixel images at whatever
resolution the data and the prior support. Comparing those against each other
directly measures the units as much as the astronomy. So every method is scored
twice:

- ``native``: the model image against the truth, both in Jy/pixel. CLEAN scores
  badly here and that is a real result, not an artefact -- a delta-function
  model genuinely has a large pixel-wise error even when every component is in
  the right place.
- ``common``: every model *and the truth* convolved with the same restoring
  beam, fitted once from the full array's dirty beam. This is the apples-to-
  apples comparison, and it is the one the headline plots use.

Reporting both is the honest thing to do, since neither one alone is the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import clean as cleanmod
from .. import forward as fw
from .. import imaging as im
from .. import metrics as mt
from .. import regularised as rg
from . import config

METHODS = ("dirty", "clean", "positivity", "tv", "tv_flux", "mem")

METHOD_LABELS = {
    "dirty": "dirty image",
    "clean": "CLEAN",
    "positivity": "positivity only",
    "tv": "total variation",
    "tv_flux": "TV + flux penalty",
    "mem": "maximum entropy",
}


@dataclass
class Reconstruction:
    """One method's output plus everything needed to score it."""

    method: str
    model: np.ndarray             # Jy/pixel, comparable to the truth
    blurred: np.ndarray           # Jy/beam at the common resolution
    residual: np.ndarray          # Jy/beam
    reduced_chi2: float
    lam: float = 0.0
    n_iter: int = 0
    extra: dict = field(default_factory=dict)
    native_comparable: bool = True
    """False for the dirty image, which is in Jy/beam and so cannot be compared
    pixel-for-pixel against a Jy/pixel truth -- the ratio is the beam area, tens
    of pixels, and the resulting numbers are arithmetic rather than astronomy.
    Its common-resolution scores are meaningful and are reported."""

    def scores(self, truth, truth_blurred, flux_match=False):
        common = mt.score(self.blurred, truth_blurred, flux_match=flux_match)
        out = {"method": self.method, "lam": self.lam, "n_iter": self.n_iter,
               "reduced_chi2": self.reduced_chi2,
               "dynamic_range": mt.dynamic_range(self.blurred, self.residual),
               "residual_rms": float(np.std(self.residual))}
        if self.native_comparable:
            native = mt.score(self.model, truth, flux_match=flux_match)
            out["flux_error"] = mt.flux_error(self.model, truth)
        else:
            native = {k: np.nan for k in ("nrmse", "nrmse_fluxmatched", "ssim",
                                          "flux_error")}
            out["flux_error"] = np.nan
        for k, v in native.items():
            out["native_" + k] = v
        for k, v in common.items():
            out["common_" + k] = v
        return out


@dataclass
class Observation:
    """A simulated dataset with both operators and the common restoring beam."""

    setup: object
    truth: np.ndarray
    data: object
    grid_op: object
    acc_op: object
    dirty: np.ndarray
    psf: np.ndarray
    beam: object
    coverage: object
    sigma: float

    @property
    def truth_blurred(self):
        return im.convolve_periodic(self.truth,
                                    self.beam.image(self.setup.npix, self.setup.cell))


def observe(setup, truth, coverage=None, snr=config.DEFAULT_SNR, sigma=None,
            seed=0, scheme="natural", robust=0.0, beam=None,
            drop_zero_spacing=False):
    """Simulate one observation and build everything the solvers need.

    ``beam`` can be supplied so a whole sweep shares one restoring beam; that
    matters, because letting each thinned array fit its own beam would change the
    yardstick at every point of the sweep and mix a resolution change into what
    is supposed to be a coverage change.

    ``sigma`` overrides ``snr`` with an explicit per-visibility noise level,
    which is what the coverage sweep needs to hold total sensitivity fixed while
    the number of visibilities changes.
    """
    coverage = setup.coverage if coverage is None else coverage
    data = fw.observe(truth, setup.cell, coverage,
                      snr=None if sigma is not None else snr, sigma=sigma,
                      rng=np.random.default_rng(seed))
    grid_op = fw.MeasurementOperator(data, setup.npix, setup.cell, scheme=scheme,
                                     robust=robust,
                                     drop_zero_spacing=drop_zero_spacing)
    acc_op = fw.DegridOperator(data, setup.npix, setup.cell, scheme=scheme,
                               robust=robust)
    dirty = im.dirty_image(grid_op)
    psf = im.dirty_beam(grid_op)
    beam = im.fit_clean_beam(psf, setup.cell) if beam is None else beam
    return Observation(setup=setup, truth=truth, data=data, grid_op=grid_op,
                       acc_op=acc_op, dirty=dirty, psf=psf, beam=beam,
                       coverage=coverage,
                       sigma=float((data.meta or {}).get("sigma", 0.0)))


def _finish(obs, method, model, lam=0.0, n_iter=0, extra=None,
            residual=None):
    kern = obs.beam.image(obs.setup.npix, obs.setup.cell)
    blurred = im.convolve_periodic(model, kern)
    if residual is None:
        residual = mt.residual_image(obs.acc_op, model)
    return Reconstruction(method=method, model=model, blurred=blurred,
                          residual=residual,
                          reduced_chi2=obs.acc_op.reduced_chi2(model),
                          lam=lam, n_iter=n_iter, extra=extra or {})


def run_method(obs, method, n_iter=300, n_bisect=7, target_chi2=1.0,
               clean_gain=0.1, clean_iter=20000, verbose=False):
    """Run one reconstruction method on a prepared :class:`Observation`."""
    setup = obs.setup

    if method == "dirty":
        # the dirty image is not a reconstruction, but it is the baseline every
        # method has to beat, and it is already in Jy/beam at the dirty-beam
        # resolution -- so it is its own "blurred" version
        return Reconstruction(
            method="dirty", model=obs.dirty, blurred=obs.dirty,
            residual=obs.dirty, reduced_chi2=obs.acc_op.reduced_chi2(obs.dirty),
            native_comparable=False)

    if method == "clean":
        thr = 3.0 * cleanmod.theoretical_image_noise(obs.grid_op)
        res = cleanmod.clean(obs.dirty, obs.psf, gain=clean_gain,
                             n_iter=clean_iter, threshold=thr,
                             frac_threshold=1e-4, cell=setup.cell,
                             fit_beam=False)
        res.beam = obs.beam
        # CLEAN's own residual is the honest one for its dynamic range: it is
        # what the algorithm itself failed to explain
        return _finish(obs, "clean", res.components, n_iter=res.n_iter,
                       residual=res.residual,
                       extra={"stop_reason": res.stop_reason,
                              "model_flux": res.model_flux})

    if method == "positivity":
        out = rg.fista(obs.acc_op, n_iter=max(n_iter, 600))
        return _finish(obs, "positivity", out.image, lam=0.0, n_iter=out.n_iter)

    if method in ("tv", "tv_flux", "mem", "l1"):
        out = rg.solve_at_chi2(obs.acc_op, kind=method, target_chi2=target_chi2,
                               n_iter=n_iter, n_bisect=n_bisect, verbose=verbose)
        return _finish(obs, method, out.image, lam=out.lam, n_iter=out.n_iter,
                       extra={"bisection": out.history.get("bisection")})

    raise ValueError("unknown method: " + str(method))


def run_all(obs, methods=METHODS, **kw):
    """Run several methods and return ``{name: Reconstruction}``."""
    return {m: run_method(obs, m, **kw) for m in methods}


def score_table(obs, recons, flux_match=False):
    """A list of score dicts, one per method, ready for a DataFrame or CSV."""
    tb = obs.truth_blurred
    return [r.scores(obs.truth, tb, flux_match=flux_match)
            for r in recons.values()]
