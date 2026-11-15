"""Spectro-temporal measurements F(lambda, t) and their analysis (Phase Three).

WHAT A MEASUREMENT IS
  One SPHEREx exposure, one source: forced aperture photometry on the
  exposure's OWN native pixels (no resampling, so no resampling-correlated
  noise), at the source position propagated to the exposure's mid-time,
  with the wavelength and bandwidth that the linear-variable filter puts at
  that position (WCS-WAVE lookup). Every exposure gives ONE wavelength per
  source: a SPHEREx "spectrum" of a source is assembled from many exposures
  taken at different times. Exposures of the dichroic detector pairs
  (D1/D4, D2/D5, D3/D6) share an obs_id and are truly simultaneous; nothing
  else is.

  Units: flux density F_nu in mJy (aperture sum of MJy/sr x pixel solid angle).
  The aperture (r = 9") is NOT aperture-corrected in `flux_mjy`. An
  approximate Gaussian aperture correction from the header PSF_FWHM is given
  separately (`aperture_correction`), never applied silently.

MEASUREMENT STATES (distinct scientific meanings)
  valid             finite aperture, no saturation-related flag in it, S/N >= 3
  valid_flagged     finite aperture, but some aperture pixel carries OVERFLOW,
                    BLOOM, SUR_ERROR or NONLINEAR. The pipeline still reports a
                    value; what these flags guarantee could not be verified
                    (SPHEREx Explanatory Supplement, Table 16, not accessible).
                    Kept, marked, and every conclusion is also tested without them.
  below_threshold   finite, unflagged, S/N < 3: an upper limit (flux + 3 sigma)
  saturated         the aperture contains blanked (NaN) pixels together with
                    saturation-related flags: the pipeline could not measure it
  masked            the aperture contains pixels masked for other reasons
                    (NONFUNC, MISSING_DATA, HOT, COLD, or NaN without a
                    saturation flag)
  off_image         the aperture does not fit on the cutout
  no_background     too few valid sky pixels
  (retrieval_failed is a SERIES-level count: exposures that were requested
   but could not be fetched. They have no pixels, hence no measurement row.)
  A wavelength/epoch combination that SPHEREx did not observe is "missing":
  it appears in the coverage summary, never as a measurement.

ANALYSIS (per source, per detector; see docs/PHASE3.md for full definitions)
  Model   F_i = a_e(i) * (1 + s * (lambda_i - lambda0))
          one amplitude per epoch, one relative spectral slope s shared by
          the epochs, lambda0 = median wavelength of the detector's usable
          measurements. sigma_i^2 = sigma_stat,i^2 + (f_sys * model_i)^2.
  M1 grey variability   Delta chi^2 between "a_e free" and "a_e = a" (dof E-1)
  M2 slope change       Delta chi^2 between "s_e free per epoch" and "s shared"
  M3 colour change      ln(a_e(det A) / a_e(det B)) constant across epochs?
                        (both detectors from the same exposures where possible)
  M4 matched pairs      model-free: exposures of different epochs whose
                        wavelengths differ by < half a bandwidth
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist

from .pipeline import DEFAULT_BAD_FLAG_NAMES, assign_epochs, central_wavelength, read_flag_names

ARCSEC_PER_RAD = 206264.80624709636
R_AP_ARCSEC = 9.0
R_IN_ARCSEC = 18.0
R_OUT_ARCSEC = 30.0
DETECTION_SNR = 3.0
SATURATION_FLAGS = ("OVERFLOW", "BLOOM", "SUR_ERROR", "NONLINEAR")
USABLE_STATES = ("valid", "valid_flagged")

# Fractional systematic error per single-exposure measurement.
# Calibrated 2026-11-15 on REAL field stars in the cached fields
# (scripts/calibrate_spectrotemporal.py, docs/PHASE3.md): robust scatter of
# the residuals about the shared-shape model, minus the statistical error,
# for reference-star exposures with S/N >= 50 (206 exposures, D3/D5/D6
# pooled): 2.43 %. Per S/N bin: 2.3 % (50-100), 2.9 % (100-200), 1.8 % (>200).
# A first calibration (median reduced chi^2 = 1 per detector) gave 0.7 % for
# D6 because faint exposures dominate it; it was too small for bright stars.
# One floor for all detectors: too few bright reference exposures to separate them.
SYS_FLOOR: dict = {}
SYS_FLOOR_DEFAULT = 0.025


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

@dataclass
class Measurement:
    source_id: str
    obs_id: str
    detector: str
    collection: str
    time_mjd: float                  # mid-exposure (header MJD-AVG) when available
    time_source: str                 # "MJD-AVG" | "SIA t_min"
    epoch_index: Optional[int]
    ra_deg: float                    # where it was measured
    dec_deg: float
    wavelength_um: Optional[float]   # WCS-WAVE at the source position (~1 nm lookup accuracy)
    bandwidth_um: Optional[float]
    flux_mjy: Optional[float]        # aperture flux density, r = 9", NOT aperture-corrected
    flux_err_mjy: Optional[float]    # statistical (variance plane + sky), 1 sigma
    snr: Optional[float]
    upper_limit_mjy: Optional[float]  # flux + 3 sigma, for below_threshold
    state: str
    method: str                       # "forced_aperture"
    flags_in_aperture: list[str] = field(default_factory=list)
    n_invalid_pixels: int = 0
    psf_fwhm_arcsec: Optional[float] = None
    aperture_correction: Optional[float] = None   # Gaussian approximation, NOT applied
    background_mjy_sr: Optional[float] = None
    fits_ref: str = ""

    @property
    def usable(self) -> bool:
        return self.state in USABLE_STATES and self.flux_mjy is not None and self.wavelength_um is not None

    def to_dict(self) -> dict:
        return asdict(self)


def _flag_names_in(flags: np.ndarray, names: dict[int, str], sel: np.ndarray) -> list[str]:
    if not sel.any():
        return []
    ored = int(np.bitwise_or.reduce(flags[sel].astype(np.int64)))
    return sorted(n for b, n in names.items() if (ored >> b) & 1)


_CTX_CACHE: dict = {}


def _exposure_context(cutout) -> dict:
    """Per-exposure arrays and WCS objects, built once and reused for every
    source measured in that exposure (building the spectral WCS is slow)."""
    key = id(cutout.hdul)
    ctx = _CTX_CACHE.get(key)
    if ctx is not None and ctx["hdul"] is cutout.hdul:
        return ctx
    hdul = cutout.hdul
    header = hdul["IMAGE"].header
    wcs = WCS(header)
    image = hdul["IMAGE"].data.astype(np.float64)
    zodi = hdul["ZODI"].data.astype(np.float64) if "ZODI" in hdul else 0.0
    var = hdul["VARIANCE"].data.astype(np.float64) if "VARIANCE" in hdul else None
    sb = image - zodi
    names = read_flag_names(hdul["FLAGS"].header) if "FLAGS" in hdul else {}
    flags = hdul["FLAGS"].data if "FLAGS" in hdul else np.zeros(sb.shape, np.int32)
    bad_value = 0
    for b, n in names.items():
        if n in DEFAULT_BAD_FLAG_NAMES:
            bad_value |= 1 << b
    invalid = ~np.isfinite(sb) | ((flags.astype(np.int64) & bad_value) != 0)
    if var is not None:
        invalid |= ~np.isfinite(var)
    try:
        spec = WCS(header, fobj=hdul, key="W")
        spec.sip = None
    except Exception:  # noqa: BLE001 - no wavelength solution: wavelength stays unknown
        spec = None

    def wave_at(x, y):
        if spec is None:
            return None, None
        try:
            lam, bw = spec.pixel_to_world(x, y)
            lam = float(lam.to("um").value); bw = float(bw.to("um").value)
        except Exception:  # noqa: BLE001
            return None, None
        return (lam if math.isfinite(lam) else None), (bw if math.isfinite(bw) else None)

    psf = header.get("PSF_FWHM")
    ctx = {"hdul": hdul, "wcs": wcs, "sb": sb, "var": var, "flags": flags, "names": names, "invalid": invalid,
           "scale": float(np.mean(proj_plane_pixel_scales(wcs)) * 3600.0), "wave_at": wave_at,
           "psf": float(psf) if psf is not None else None}
    if len(_CTX_CACHE) > 2000:
        _CTX_CACHE.clear()
    _CTX_CACHE[key] = ctx
    return ctx


def measure_exposure(cutout, source_id: str, ra_deg: float, dec_deg: float,
                     r_ap_arcsec: float = R_AP_ARCSEC, fits_ref: str = "") -> Measurement:
    """Forced aperture photometry of one source in one exposure (native pixels)."""
    from photutils.aperture import CircularAnnulus, CircularAperture
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    hdul = cutout.hdul
    rec = cutout.record
    header = hdul["IMAGE"].header
    t_mid = header.get("MJD-AVG")
    time_mjd, time_src = (float(t_mid), "MJD-AVG") if t_mid is not None else (float(rec.time_mjd), "SIA t_min")
    base = dict(source_id=source_id, obs_id=rec.obs_id, detector=rec.detector, collection=rec.collection,
                time_mjd=time_mjd, time_source=time_src, epoch_index=None, ra_deg=ra_deg, dec_deg=dec_deg,
                method="forced_aperture", fits_ref=fits_ref)

    ctx = _exposure_context(cutout)
    wcs = ctx["wcs"]
    coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
    x, y = (float(v) for v in wcs.world_to_pixel(coord))
    lam, bw = ctx["wave_at"](x, y)
    psf = ctx["psf"]
    apcorr = None
    if psf:
        sigma = psf / 2.3548
        apcorr = 1.0 / (1.0 - math.exp(-(r_ap_arcsec ** 2) / (2 * sigma ** 2)))
    base.update(wavelength_um=lam, bandwidth_um=bw, psf_fwhm_arcsec=psf, aperture_correction=apcorr)

    sb, var, flags, names, invalid, scale = (ctx[k] for k in ("sb", "var", "flags", "names", "invalid", "scale"))
    h, w = sb.shape
    r_ap = r_ap_arcsec / scale

    empty = dict(flux_mjy=None, flux_err_mjy=None, snr=None, upper_limit_mjy=None)
    if not (r_ap - 0.5 <= x <= w - 0.5 - r_ap and r_ap - 0.5 <= y <= h - 0.5 - r_ap):
        return Measurement(**base, **empty, state="off_image")

    ap = CircularAperture((x, y), r=r_ap)
    wmap = ap.to_mask(method="exact").to_image(sb.shape)
    if wmap is None:
        return Measurement(**base, **empty, state="off_image")
    in_ap = wmap > 0
    ap_flags = _flag_names_in(flags, names, in_ap)
    sat_flags = [f for f in ap_flags if f in SATURATION_FLAGS]
    n_bad = int((invalid & in_ap).sum())
    base.update(flags_in_aperture=ap_flags, n_invalid_pixels=n_bad)
    if n_bad:
        nan_in_ap = bool((~np.isfinite(sb) & in_ap).any())
        state = "saturated" if (nan_in_ap and sat_flags) else "masked"
        return Measurement(**base, **empty, state=state)

    ann = CircularAnnulus((x, y), r_in=R_IN_ARCSEC / scale, r_out=R_OUT_ARCSEC / scale)
    amap = ann.to_mask(method="center").to_image(sb.shape)
    sky = sb[(amap > 0) & ~invalid]
    if sky.size < 8:
        return Measurement(**base, **empty, state="no_background")
    _, bkg, bkg_std = sigma_clipped_stats(sky, sigma=3.0, maxiters=5)

    omega_sr = (scale / ARCSEC_PER_RAD) ** 2
    to_mjy = omega_sr * 1e9
    # Sum over aperture pixels only: blanked pixels elsewhere in the cutout
    # must not leak in (0 * NaN is NaN).
    wv = wmap[in_ap]
    area = float(wv.sum())
    net = float(np.sum(wv * (sb[in_ap] - bkg)))
    var_src = float(np.sum(wv ** 2 * var[in_ap])) if var is not None else float(np.sum(wv ** 2)) * bkg_std ** 2
    var_bkg = area ** 2 * bkg_std ** 2 / sky.size
    flux = net * to_mjy
    err = math.sqrt(var_src + var_bkg) * to_mjy
    snr = flux / err if err > 0 else None
    base.update(background_mjy_sr=float(bkg))
    if snr is not None and snr < DETECTION_SNR and not sat_flags:
        return Measurement(**base, flux_mjy=flux, flux_err_mjy=err, snr=snr,
                           upper_limit_mjy=max(flux, 0.0) + DETECTION_SNR * err, state="below_threshold")
    return Measurement(**base, flux_mjy=flux, flux_err_mjy=err, snr=snr, upper_limit_mjy=None,
                       state="valid_flagged" if sat_flags else "valid")


def assign_measurement_epochs(meas: list[Measurement], gap_days: float = 45.0) -> list[dict]:
    """Group measurements into survey epochs by time gaps (same rule as the
    image pipeline). Returns epoch descriptors; sets m.epoch_index."""
    if not meas:
        return []
    order = sorted(range(len(meas)), key=lambda i: meas[i].time_mjd)

    class _R:  # minimal record shape for assign_epochs
        def __init__(self, t):
            self.time_mjd = t

    ep = assign_epochs([_R(meas[i].time_mjd) for i in order], gap_days)
    for i, e in zip(order, ep):
        meas[i].epoch_index = e
    out = []
    for e in sorted(set(ep)):
        ts = [m.time_mjd for m in meas if m.epoch_index == e]
        out.append({"index": e, "mjd_min": min(ts), "mjd_max": max(ts), "mjd_mean": float(np.mean(ts))})
    return out


# ---------------------------------------------------------------------------
# Per-detector spectro-temporal model
# ---------------------------------------------------------------------------
#
# F_i = a_e(i) * S(lambda_i) * (1 + t_e(i) * (lambda_i - lambda0))
#
#   S(lambda)  shared spectral SHAPE: piecewise-linear through K knots placed
#              at quantiles of the observed wavelengths (so every knot interval
#              holds data), normalised to S(lambda0) = 1 at the knot lambda0.
#              Flexible enough for brown-dwarf and M-dwarf molecular bands at
#              SPHEREx resolution; a straight line across a detector is not
#              (that produced false "variability", see docs/PHASE3.md).
#   a_e        flux density at lambda0 in epoch e (the band light curve).
#   t_e        optional per-epoch tilt (t_first = 0): change of spectral slope
#              relative to the shared shape.
# Null models: grey test fixes a_e = a; shape test fixes t_e = 0.
# Errors: sigma_i^2 = sigma_stat,i^2 + (f_sys * model_i)^2.

@dataclass
class DetectorFit:
    detector: str
    lambda0_um: float
    n_points: int
    epochs: list[int]
    amplitudes_mjy: list[float]                  # a_e at lambda0, per epoch in `epochs`
    amplitude_errs_mjy: list[float]
    knots_um: list[float]
    shape: list[float]                           # S at the knots (S(lambda0) = 1)
    chi2: float
    dof: int
    sys_floor: float
    errors_rescaled: bool
    epoch_lambda_ranges: list[list[float]]
    extrapolated_epochs: list[int]               # epochs with no data within one bandwidth of lambda0
    model_adequate: Optional[bool] = None        # set against the reference stars, see analyse_source


def _knots(lam: np.ndarray, n_free: int, bandwidth: float) -> np.ndarray:
    """Knots at quantiles of the observed wavelengths: at most one per three
    bandwidths, at most one per ~3 data points (n_free), at least 2. More
    knots than this let shape and amplitude trade off (seen on real data)."""
    span = float(lam.max() - lam.min())
    k = int(min(max(2, round(span / (3 * bandwidth)) + 1), max(2, n_free)))
    q = np.quantile(lam, np.linspace(0, 1, k))
    q = np.unique(np.round(q, 6))
    return q if q.size >= 2 else np.array([lam.min(), lam.max()])


def _hat(lam: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Piecewise-linear basis, shape (n_points, n_knots)."""
    B = np.zeros((lam.size, knots.size))
    for j in range(knots.size):
        e = np.zeros(knots.size); e[j] = 1.0
        B[:, j] = np.interp(lam, knots, e)
    return B


def _fit_model(lam, F, err, ei, n_ep, knots, i0, fsys, tilt=False, grey=False):
    """Returns dict(p, cov, chi2, dof, model, a_idx) or None if under-determined."""
    B = _hat(lam, knots)
    lam0 = knots[i0]
    free_knots = [j for j in range(knots.size) if j != i0]
    n_a = 1 if grey else n_ep
    n_c = len(free_knots)
    n_t = (n_ep - 1) if tilt else 0
    n_par = n_a + n_c + n_t
    if F.size - n_par < 1:
        return None
    if not (np.all(np.isfinite(F)) and np.all(np.isfinite(err)) and np.all(np.isfinite(lam))):
        raise ValueError("non-finite input to the spectro-temporal fit")

    def unpack(p):
        a = np.full(n_ep, p[0]) if grey else p[:n_ep]
        c = np.ones(knots.size); c[free_knots] = p[n_a:n_a + n_c]
        t = np.zeros(n_ep)
        if tilt:
            t[1:] = p[n_a + n_c:]
        return a, c, t

    def model(p):
        a, c, t = unpack(p)
        return a[ei] * (B @ c) * (1 + t[ei] * (lam - lam0))

    # starting point: shape from all data normalised at lambda0, amplitudes per epoch
    s0 = np.interp(knots, np.sort(lam), F[np.argsort(lam)])
    s0 = s0 / s0[i0] if s0[i0] != 0 else np.ones_like(s0)
    a0 = [np.median(F[ei == e] / np.maximum(B[ei == e] @ s0, 1e-6)) if np.any(ei == e) else np.median(F) for e in range(n_ep)]
    p0 = np.concatenate([[np.median(a0)] if grey else a0, s0[free_knots], np.zeros(n_t)])
    sig = np.sqrt(err ** 2 + (fsys * np.abs(F)) ** 2)
    res = None
    for _ in range(3):
        res = least_squares(lambda p: (F - model(p)) / sig, p0, method="trf", x_scale="jac", max_nfev=2000)
        p0 = res.x
        sig = np.sqrt(err ** 2 + (fsys * np.abs(model(p0))) ** 2)
    r = (F - model(p0)) / sig
    J = res.jac
    cov = np.linalg.pinv(J.T @ J)
    return {"p": p0, "cov": cov, "chi2": float(r @ r), "dof": int(F.size - n_par), "unpack": unpack, "n_a": n_a}


def _detector_data(meas, detector, min_points_per_epoch=2):
    pts = [m for m in meas if m.detector == detector and m.usable]
    by_e: dict[int, int] = {}
    for m in pts:
        by_e[m.epoch_index] = by_e.get(m.epoch_index, 0) + 1
    epochs = sorted(e for e, n in by_e.items() if n >= min_points_per_epoch)
    pts = [m for m in pts if m.epoch_index in epochs]
    if not pts:
        return None
    lam = np.array([m.wavelength_um for m in pts]); F = np.array([m.flux_mjy for m in pts])
    err = np.array([m.flux_err_mjy for m in pts]); remap = {e: i for i, e in enumerate(epochs)}
    ei = np.array([remap[m.epoch_index] for m in pts])
    bw = float(np.median([m.bandwidth_um for m in pts if m.bandwidth_um] or [0.04]))
    return {"pts": pts, "epochs": epochs, "lam": lam, "F": F, "err": err, "ei": ei, "bw": bw}


def _setup(d, fsys, extra_params=0):
    n_ep = len(d["epochs"])
    # knots allowed so that the richest model keeps dof >= 4 and >= ~3 points per knot
    n_free = min(d["F"].size // 3, d["F"].size - n_ep - extra_params - 4 + 1)
    knots = _knots(d["lam"], n_free, d["bw"])
    i0 = int(np.argmin(np.abs(knots - np.median(d["lam"]))))
    return n_ep, knots, i0


def fit_detector(meas: list[Measurement], detector: str, n_epochs: int = 0, sys_floor: Optional[float] = None,
                 min_points_per_epoch: int = 2) -> Optional[DetectorFit]:
    """Shared-shape model for one detector; None if not enough usable data."""
    fsys = SYS_FLOOR.get(detector, SYS_FLOOR_DEFAULT) if sys_floor is None else sys_floor
    d = _detector_data(meas, detector, min_points_per_epoch)
    if d is None or not d["epochs"]:
        return None
    n_ep, knots, i0 = _setup(d, fsys, extra_params=max(len(d["epochs"]) - 1, 0))
    f = _fit_model(d["lam"], d["F"], d["err"], d["ei"], n_ep, knots, i0, fsys)
    if f is None:
        return None
    cov = f["cov"]
    rescaled = False
    if f["dof"] > 0 and f["chi2"] / f["dof"] > 1:
        cov = cov * (f["chi2"] / f["dof"])
        rescaled = True
    errs = np.sqrt(np.clip(np.diag(cov), 0, None))
    a, c, _ = f["unpack"](f["p"])
    lam, ei = d["lam"], d["ei"]
    ranges = [[float(lam[ei == i].min()), float(lam[ei == i].max())] for i in range(n_ep)]
    lam0 = float(knots[i0])
    extrap = [d["epochs"][i] for i in range(n_ep) if np.min(np.abs(lam[ei == i] - lam0)) > d["bw"]]
    return DetectorFit(
        detector=detector, lambda0_um=lam0, n_points=int(lam.size), epochs=d["epochs"],
        amplitudes_mjy=[float(v) for v in a], amplitude_errs_mjy=[float(v) for v in errs[:n_ep]],
        knots_um=[float(k) for k in knots], shape=[float(v) for v in c],
        chi2=f["chi2"], dof=f["dof"], sys_floor=fsys, errors_rescaled=rescaled,
        epoch_lambda_ranges=ranges, extrapolated_epochs=extrap,
    )


def _p(chi2_val, dof):
    return float(chi2_dist.sf(chi2_val, dof)) if dof > 0 and chi2_val >= 0 else None


def metric_grey_variability(meas, detector, n_epochs=0, sys_floor=None) -> dict:
    """M1. Does the flux at lambda0 change between epochs, given a spectral
    shape shared by all epochs? Delta chi^2 (a_e free vs a_e = a), dof E - 1."""
    out = {"metric": "grey_variability", "detector": detector,
           "requires": ">= 2 epochs with >= 2 usable measurements each in this detector, and enough points to fit the shape"}
    fsys = SYS_FLOOR.get(detector, SYS_FLOOR_DEFAULT) if sys_floor is None else sys_floor
    d = _detector_data(meas, detector)
    if d is None or len(d["epochs"]) < 2:
        out.update(status="insufficient_data", reason="fewer than 2 epochs with >= 2 usable measurements")
        return out
    n_ep, knots, i0 = _setup(d, fsys, extra_params=n_ep_minus1(d))
    f1 = _fit_model(d["lam"], d["F"], d["err"], d["ei"], n_ep, knots, i0, fsys)
    f0 = _fit_model(d["lam"], d["F"], d["err"], d["ei"], n_ep, knots, i0, fsys, grey=True)
    if f1 is None or f0 is None:
        out.update(status="insufficient_data", reason="too few points for the shape model")
        return out
    a = f1["unpack"](f1["p"])[0]
    scale = math.sqrt(f1["chi2"] / f1["dof"]) if f1["chi2"] > f1["dof"] else 1.0
    ea = np.sqrt(np.clip(np.diag(f1["cov"])[:n_ep], 0, None)) * scale
    mean = float(np.mean(a))
    i_max, i_min = int(np.argmax(a)), int(np.argmin(a))
    rng = float((a[i_max] - a[i_min]) / mean) if mean > 0 else None
    rng_err = float(math.hypot(ea[i_max], ea[i_min]) / mean) if mean > 0 else None
    dchi2 = max(f0["chi2"] - f1["chi2"], 0.0)
    out["flagged_fraction"] = float(np.mean([m.state == "valid_flagged" for m in d["pts"]]))
    out.update(status="ok", lambda0_um=float(knots[i0]), epochs=d["epochs"], amplitudes_mjy=[float(v) for v in a],
               amplitude_errs_mjy=[float(v) for v in ea], delta_chi2=dchi2, dof=f0["dof"] - f1["dof"],
               p_value=_p(dchi2, f0["dof"] - f1["dof"]), fractional_range=rng, fractional_range_err=rng_err,
               sys_floor=fsys, model_reduced_chi2=f1["chi2"] / f1["dof"], n_knots=int(knots.size))
    return out


def n_ep_minus1(d):
    return max(len(d["epochs"]) - 1, 0)


def metric_slope_change(meas, detector, n_epochs=0, sys_floor=None) -> dict:
    """M2. Does the spectral slope within the detector change between epochs,
    relative to the shared shape? Delta chi^2 (tilts t_e free vs t_e = 0), dof E - 1."""
    out = {"metric": "slope_change", "detector": detector,
           "requires": ">= 2 epochs with >= 3 usable measurements spanning >= 0.1 um each"}
    fsys = SYS_FLOOR.get(detector, SYS_FLOOR_DEFAULT) if sys_floor is None else sys_floor
    d = _detector_data(meas, detector, min_points_per_epoch=3)
    if d is not None:
        ok = [e for i, e in enumerate(d["epochs"]) if np.ptp(d["lam"][d["ei"] == i]) >= 0.1]
        if len(ok) < len(d["epochs"]):
            d = _detector_data([m for m in meas if m.epoch_index in ok], detector, 3) if len(ok) >= 2 else None
    if d is None or len(d["epochs"]) < 2:
        out.update(status="insufficient_data", reason="fewer than 2 epochs sample enough of the detector's wavelength range")
        return out
    n_ep, knots, i0 = _setup(d, fsys, extra_params=2 * n_ep_minus1(d))
    f1 = _fit_model(d["lam"], d["F"], d["err"], d["ei"], n_ep, knots, i0, fsys, tilt=True)
    f0 = _fit_model(d["lam"], d["F"], d["err"], d["ei"], n_ep, knots, i0, fsys)
    if f1 is None or f0 is None:
        out.update(status="insufficient_data", reason="too few points for the shape-plus-tilt model")
        return out
    t = f1["unpack"](f1["p"])[2]
    n_c = knots.size - 1
    scale = math.sqrt(f1["chi2"] / f1["dof"]) if f1["chi2"] > f1["dof"] else 1.0
    et = np.concatenate([[0.0], np.sqrt(np.clip(np.diag(f1["cov"])[n_ep + n_c:], 0, None)) * scale])
    dchi2 = max(f0["chi2"] - f1["chi2"], 0.0)
    out["flagged_fraction"] = float(np.mean([m.state == "valid_flagged" for m in d["pts"]]))
    out.update(status="ok", lambda0_um=float(knots[i0]), epochs=d["epochs"], tilts_per_um=[float(v) for v in t],
               tilt_errs_per_um=[float(v) for v in et], delta_chi2=dchi2, dof=f0["dof"] - f1["dof"],
               p_value=_p(dchi2, f0["dof"] - f1["dof"]), sys_floor=fsys,
               model_reduced_chi2=f0["chi2"] / f0["dof"], n_knots=int(knots.size),
               note="tilts are relative to the first epoch (t = 0 there)")
    return out


def metric_colour_change(fit_a: Optional[DetectorFit], fit_b: Optional[DetectorFit],
                         simultaneous_fraction: Optional[float]) -> dict:
    """M3. Is the flux ratio between two detectors constant across epochs?
    C_e = ln(a_e(A) / a_e(B)); chi^2 of a constant (dof = n - 1)."""
    out = {"metric": "colour_change", "requires": "both detectors fitted, >= 2 common epochs, positive amplitudes"}
    if fit_a is None or fit_b is None:
        out.update(status="insufficient_data", reason="one of the detectors has no usable fit")
        return out
    out.update(detectors=[fit_a.detector, fit_b.detector], lambda0_um=[fit_a.lambda0_um, fit_b.lambda0_um])
    common = [e for e in fit_a.epochs if e in fit_b.epochs]
    rows = []
    for e in common:
        a = fit_a.amplitudes_mjy[fit_a.epochs.index(e)]; ea = fit_a.amplitude_errs_mjy[fit_a.epochs.index(e)]
        b = fit_b.amplitudes_mjy[fit_b.epochs.index(e)]; eb = fit_b.amplitude_errs_mjy[fit_b.epochs.index(e)]
        if a is None or b is None or a <= 0 or b <= 0:
            continue
        rows.append((e, math.log(a / b), math.hypot(ea / a, eb / b), a / b))
    if len(rows) < 2:
        out.update(status="insufficient_data", reason=f"{len(rows)} epoch(s) with positive amplitudes in both detectors")
        return out
    c = np.array([r[1] for r in rows]); s = np.array([r[2] for r in rows])
    w = 1 / s ** 2
    mean = float(np.sum(w * c) / np.sum(w))
    chi2 = float(np.sum(w * (c - mean) ** 2))
    out.update(status="ok", epochs=[r[0] for r in rows], ratio=[r[3] for r in rows], ln_ratio=[float(v) for v in c],
               ln_ratio_err=[float(v) for v in s], chi2=chi2, dof=len(rows) - 1, p_value=_p(chi2, len(rows) - 1),
               max_change=float(c.max() - c.min()), simultaneous_fraction=simultaneous_fraction)
    return out


def metric_matched_pairs(meas, detector, sys_floor=None, tolerance_fraction: float = 0.25) -> dict:
    """M4. Model-free: pairs of exposures in DIFFERENT epochs whose wavelengths
    differ by less than `tolerance_fraction` x bandwidth. For each pair,
    z = (F2 - F1) / sqrt(s1^2 + s2^2) with sigma including the floor."""
    fsys = SYS_FLOOR.get(detector, SYS_FLOOR_DEFAULT) if sys_floor is None else sys_floor
    pts = [m for m in meas if m.detector == detector and m.usable]
    out = {"metric": "matched_wavelength_pairs", "detector": detector,
           "requires": f"exposures in different epochs within {tolerance_fraction:g} x bandwidth of each other"}
    pairs = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            a, b = pts[i], pts[j]
            if a.epoch_index == b.epoch_index:
                continue
            tol = tolerance_fraction * min(a.bandwidth_um or 0.04, b.bandwidth_um or 0.04)
            if abs(a.wavelength_um - b.wavelength_um) <= tol:
                s1 = math.hypot(a.flux_err_mjy, fsys * a.flux_mjy); s2 = math.hypot(b.flux_err_mjy, fsys * b.flux_mjy)
                first, second = (a, b) if a.time_mjd < b.time_mjd else (b, a)
                pairs.append({"obs_ids": [first.obs_id, second.obs_id], "epochs": [first.epoch_index, second.epoch_index],
                              "wavelengths_um": [first.wavelength_um, second.wavelength_um],
                              "fluxes_mjy": [first.flux_mjy, second.flux_mjy],
                              "ratio": second.flux_mjy / first.flux_mjy if first.flux_mjy > 0 else None,
                              "z": (second.flux_mjy - first.flux_mjy) / math.hypot(s1, s2)})
    if not pairs:
        out.update(status="insufficient_data", reason="no wavelength-matched pairs across epochs")
        return out
    z = np.array([p["z"] for p in pairs])
    lr = [math.log(p["ratio"]) for p in pairs if p["ratio"] and p["ratio"] > 0]
    if lr:
        # effect size for the assessment text: typical fractional change between matched exposures
        out["fractional_range"] = float(abs(math.exp(np.median(np.abs(lr))) - 1.0))
        out["effect_label"] = "typical change between wavelength-matched exposures"
    out.update(median_ratio_later_over_earlier=float(math.exp(np.median(lr))) if lr else None,
               flagged_fraction=float(np.mean([m.state == "valid_flagged" for m in pts])) if pts else 0.0)
    out.update(status="ok", n_pairs=len(pairs), pairs=pairs, max_abs_z=float(np.max(np.abs(z))),
               chi2=float(np.sum(z ** 2)), dof=len(pairs), p_value=_p(float(np.sum(z ** 2)), len(pairs)),
               note="pairs sharing an exposure are not independent; the chi^2 p-value is then approximate")
    return out


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

# Model adequacy: a fit with reduced chi^2 above the limit is treated as not
# describing the spectrum, and the model-based metrics are not evaluated.
# Limit = reference stars' 84th percentile, clipped to [2, 3]; 2 without
# reference stars. Chosen on synthetic data: with 2.0, 0 of 900 metric
# evaluations of a constant source with a deep, resolved absorption band
# claimed a significant change; with 3.0, 9 did (docs/PHASE3.md). The 95th
# percentile was tried first and is not robust: one odd reference star in the
# RW Aur field pushed it to 119.
MODEL_ADEQUACY_FALLBACK = 2.0
MODEL_ADEQUACY_MAX = 3.0
# Empirical control for measurements with saturation-related flags: the quiet
# Barnard's Star (M4 V), all of whose usable D6 data are flagged, shows a
# 7.0 % range of epoch amplitudes (3274-3511 mJy at 4.59 um; docs/PHASE3.md).
# One star: an indication of the size of flagged-data systematics, not a calibration.
FLAGGED_CONTROL_RANGE = 0.07


def _assess(m: dict, ref: Optional[dict], adequacy_limit: Optional[float] = None) -> dict:
    """Plain-language assessment; never a probability of being astrophysical."""
    if m.get("status") != "ok":
        m["assessment"] = "inconclusive"
        m["assessment_text"] = f"Not evaluated: {m.get('reason', 'insufficient data')}."
        return m
    rc = m.get("model_reduced_chi2")
    if rc is not None:
        limit = adequacy_limit if adequacy_limit is not None else MODEL_ADEQUACY_FALLBACK
        m["model_adequacy_limit"] = limit
        if rc > limit:
            m["assessment"] = "inconclusive"
            m["assessment_text"] = (
                f"The shared-spectrum model does not describe this source's measurements (reduced chi^2 {rc:.1f}, "
                f"limit {limit:.1f} from the reference stars). Its spectrum has structure the model does not follow, "
                "or the measurements have errors the floor does not cover, so a change in time cannot be separated "
                "from a mismatch in wavelength. Use the model-free matched-wavelength comparison instead.")
            return m
    p = m.get("p_value")
    if p is None:
        m["assessment"] = "inconclusive"
        m["assessment_text"] = "No degrees of freedom: nothing can be tested."
        return m
    ref_txt = ""
    unusual = None
    if ref and ref.get("n", 0) >= 5:
        # compared by p-value: reference stars have different numbers of epochs (dof)
        k = int(np.sum(np.array(ref["values"]) <= p))
        n = int(ref["n"])
        frac = k / n
        bound = (k + 1) / (n + 1)
        unusual = k == 0
        ref_txt = (f" {k} of {n} stationary field stars measured the same way show a change at least this significant"
                   + (f" (so it is rarer than about 1 in {n + 1} here; {n} reference stars cannot show more)." if k == 0 else "."))
        m["reference_fraction_at_least_as_large"] = frac
        m["reference_rarity_upper_bound"] = bound
        m["reference_n"] = n
    elif ref is not None:
        ref_txt = " Too few reference stars in this field for an empirical comparison; the p-value rests on the error model alone."
    flagged = m.get("flagged_fraction", 0.0) or 0.0
    structured = m.get("structured_spectrum", False)
    if p < 1e-3 and (unusual is None or unusual) and (flagged > 0.5 or structured):
        why = []
        if flagged > 0.5:
            rng = m.get("fractional_range")
            size = ""
            if rng is not None:
                size = (f" The {m.get('effect_label', 'range of epoch amplitudes')} is {100 * rng:.0f}%, "
                        + ("well above" if rng > 3 * FLAGGED_CONTROL_RANGE else "comparable to")
                        + f" the {100 * FLAGGED_CONTROL_RANGE:.0f}% range shown by the quiet Barnard's Star in such data.")
            why.append(f"{100 * flagged:.0f}% of the measurements have OVERFLOW/BLOOM/SUR_ERROR pixels in the aperture, "
                       f"whose reliability could not be verified.{size}")
        if structured:
            why.append("The spectrum has structure that the shape model cannot follow, so even the small wavelength "
                       "offsets between matched exposures may change the flux.")
        m["assessment"] = "conditional"
        m["assessment_text"] = (f"Formally significant (p = {p:.1e}), but conditional: " + " ".join(why) + ref_txt)
    elif p < 1e-3 and (unusual is None or unusual):
        m["assessment"] = "significant_change"
        m["assessment_text"] = (f"Change detected beyond the errors (p = {p:.1e} with a {100 * m.get('sys_floor', 0):.1f}% "
                                f"systematic floor).{ref_txt} This is a measurement statement, not an interpretation "
                                "of its physical cause.")
    elif p < 1e-3:
        m["assessment"] = "not_unusual"
        m["assessment_text"] = (f"Formally significant (p = {p:.1e}), but field stars show comparable values, so it is "
                                f"not distinguishable from systematics.{ref_txt}")
    else:
        m["assessment"] = "no_significant_change"
        m["assessment_text"] = f"Consistent with no change within the errors (p = {p:.2f}).{ref_txt}"
    return m


def analyse_source(meas: list[Measurement], epochs: list[dict], reference: Optional[dict] = None) -> dict:
    """All per-source products: coverage, per-epoch SED points, detector fits,
    metrics M1-M4 (with and without flagged measurements)."""
    dets = sorted({m.detector for m in meas})
    states = {}
    for m in meas:
        states[m.state] = states.get(m.state, 0) + 1
    coverage = []
    for d in dets:
        for e in epochs:
            sel = [m for m in meas if m.detector == d and m.epoch_index == e["index"]]
            use = [m for m in sel if m.usable]
            coverage.append({
                "detector": d, "epoch": e["index"], "n_exposures": len(sel), "n_usable": len(use),
                "lambda_min_um": min((m.wavelength_um for m in use), default=None),
                "lambda_max_um": max((m.wavelength_um for m in use), default=None),
                "states": {s: sum(1 for m in sel if m.state == s) for s in sorted({m.state for m in sel})},
            })
    obs = {}
    for m in meas:
        obs.setdefault(m.obs_id, set()).add(m.detector)
    n_epochs = len(epochs)
    fits = {d: fit_detector(meas, d, n_epochs) for d in dets}
    limits = {}
    for d in dets:
        p84 = ((reference or {}).get(f"fit:{d}") or {}).get("p84")
        limits[d] = (min(MODEL_ADEQUACY_MAX, max(MODEL_ADEQUACY_FALLBACK, p84)) if p84 is not None
                     else MODEL_ADEQUACY_FALLBACK)
    for d, f in fits.items():
        if f is not None and f.dof > 0:
            lim = limits[d] if limits[d] is not None else MODEL_ADEQUACY_FALLBACK
            f.model_adequate = bool(f.chi2 / f.dof <= lim)
    metrics = []
    variants = [("all_usable", meas)]
    unflagged = [m for m in meas if m.state != "valid_flagged"]
    if any(m.state == "valid_flagged" for m in meas):
        variants.append(("unflagged_only", unflagged))
    for label, data in variants:
        for d in dets:
            ref = (reference or {}).get(f"grey:{d}") if label == "all_usable" else None
            metrics.append({**_assess(metric_grey_variability(data, d, n_epochs), ref, limits[d]), "data": label})
            ref = (reference or {}).get(f"slope:{d}") if label == "all_usable" else None
            metrics.append({**_assess(metric_slope_change(data, d, n_epochs), ref, limits[d]), "data": label})
            ref = (reference or {}).get(f"pairs:{d}") if label == "all_usable" else None
            pm = metric_matched_pairs(data, d)
            pm["structured_spectrum"] = bool(fits.get(d) is not None and fits[d].model_adequate is False)
            metrics.append({**_assess(pm, ref), "data": label})
        if len(dets) >= 2:
            fa = fit_detector(data, dets[-1], n_epochs) if label != "all_usable" else fits[dets[-1]]
            fb = fit_detector(data, dets[0], n_epochs) if label != "all_usable" else fits[dets[0]]
            pairs = [o for o, ds in obs.items() if dets[0] in ds and dets[-1] in ds]
            simfrac = len(pairs) / max(len(obs), 1)
            ref = (reference or {}).get(f"colour:{dets[-1]}/{dets[0]}") if label == "all_usable" else None
            cm = metric_colour_change(fa, fb, simfrac)
            used = [m for m in data if m.usable and m.detector in (dets[0], dets[-1])]
            cm["flagged_fraction"] = float(np.mean([m.state == "valid_flagged" for m in used])) if used else 0.0
            bad = [f.detector for f in (fa, fb) if f is not None and f.dof > 0 and
                   f.chi2 / f.dof > (limits.get(f.detector) or MODEL_ADEQUACY_FALLBACK)]
            if cm.get("status") == "ok" and bad:
                cm["model_reduced_chi2"] = max(f.chi2 / f.dof for f in (fa, fb) if f.detector in bad)
            metrics.append({**_assess(cm, ref, max((limits.get(b) or MODEL_ADEQUACY_FALLBACK) for b in bad) if bad else None),
                            "data": label})
    return {
        "detectors": dets, "epochs": epochs, "state_counts": states, "coverage": coverage,
        "simultaneous_pairs": sum(1 for ds in obs.values() if len(ds) > 1),
        "fits": {d: (asdict(f) if f else None) for d, f in fits.items()},
        "metrics": metrics,
        "measurements": [m.to_dict() for m in sorted(meas, key=lambda m: (m.detector, m.time_mjd))],
        "units": {"flux": "mJy (F_nu, r = 9\" aperture, not aperture-corrected)", "wavelength": "um", "time": "MJD (mid-exposure)"},
    }


def reference_distribution(per_star: list[dict]) -> dict:
    """Reference (field) star statistics, keyed like analyse_source's lookups:
    metric p-values ("grey:D6", ...) and the shape model's reduced chi^2 ("fit:D6")."""
    out: dict[str, dict] = {}
    for res in per_star:
        dets = res["detectors"]
        for d, f in (res.get("fits") or {}).items():
            if f and f["dof"] > 0:
                out.setdefault(f"fit:{d}", {"values": []})["values"].append(float(f["chi2"] / f["dof"]))
        for m in res["metrics"]:
            if m.get("data") != "all_usable" or m.get("status") != "ok":
                continue
            if m.get("p_value") is None:
                continue
            if m["metric"] == "grey_variability":
                key = f"grey:{m['detector']}"
            elif m["metric"] == "slope_change":
                key = f"slope:{m['detector']}"
            elif m["metric"] == "matched_wavelength_pairs":
                key = f"pairs:{m['detector']}"
            elif m["metric"] == "colour_change":
                key = f"colour:{dets[-1]}/{dets[0]}"
            else:
                continue
            val = m["p_value"]
            out.setdefault(key, {"values": []})["values"].append(float(val))
    for v in out.values():
        v["n"] = len(v["values"])
        if v["n"] >= 5:
            v["p84"] = float(np.percentile(v["values"], 84))
            v["p95"] = float(np.percentile(v["values"], 95))
    return out


def source_positions_at(ra0: float, dec0: float, ref_mjd: float, pm_e_mas: Optional[float],
                        pm_n_mas: Optional[float], mjd: float) -> tuple[float, float]:
    if not pm_e_mas and not pm_n_mas:
        return ra0, dec0
    dt = (mjd - ref_mjd) / 365.25
    dec = dec0 + (pm_n_mas or 0) * dt / 3.6e6
    ra = ra0 + (pm_e_mas or 0) * dt / 3.6e6 / math.cos(math.radians(0.5 * (dec + dec0)))
    return ra % 360.0, dec
