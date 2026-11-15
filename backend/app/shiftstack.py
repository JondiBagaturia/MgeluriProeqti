"""Trial-motion shift-and-stack for faint moving point sources.

Idea: a source too faint to detect in any single frame can still be found
if all frames are added along its (unknown) trajectory. For each trial
proper motion the frames are shifted so that a source with that motion
stays put, combined with matched-filter weights, and searched for peaks.

Statistic (per sky position x and trial velocity v), the optimal
point-source S/N for independent Gaussian noise:

    S/N(x, v) = sum_i [ (d_i / var_i) (*) P ](x_i)  /  sqrt( sum_i [ (1 / var_i) (*) P^2 ](x_i) )

where d_i is frame i minus its background, var_i its variance map, P the
PSF, (*) correlation, and x_i the position the source would have in frame i.
Frames with larger noise get less weight; missing pixels (NaN) get zero
weight, so missing observations are handled without special cases.

What is done about each known problem:
  * Correlated noise from resampling and imperfect variance maps make the
    formal S/N wrong by a factor. Every velocity plane is therefore
    rescaled by its own robust scatter (median/MAD), so "S/N" below always
    means "in units of the measured noise of that stack".
  * Static background sources are bright at every trial velocity (only
    smeared). A peak counts as MOVING only if its S/N^2 at the best
    velocity exceeds the best zero-motion S/N^2 nearby by `min_delta_chi2`.
    With a 9" PSF this needs a total displacement comparable to the PSF
    over the time baseline; slower motion is indistinguishable from none.
  * Searching many positions and velocities produces false peaks by
    chance. The search is repeated on control data sets in which the
    observation times are shuffled among the frames (same pixels, same
    noise, same static sources, no coherent motion). The fraction of
    controls whose best "moving" peak is at least as strong as the real
    one is the empirical false-alarm probability. With K controls it
    cannot be smaller than 1/(K+1), and is reported as such.
  * PSF differences between frames are ignored (one Gaussian for all).
    This costs sensitivity but does not create false signal.
  * Parallax is ignored. For WISE 0855 it is 0.44", well below a pixel.

Coarse-to-fine: velocities are first sampled on a grid whose step moves a
source by about a third of the PSF width over the baseline, then the best
peaks are refined on a 5x finer local grid.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage

DAYS_PER_YEAR = 365.25


@dataclass
class StackFrame:
    image: np.ndarray                 # MJy/sr (any linear unit), NaN = no data
    variance: Optional[np.ndarray]    # same shape; None -> estimated from the image scatter
    time_mjd: float


@dataclass
class ShiftStackConfig:
    psf_fwhm_arcsec: float = 9.0
    v_max_arcsec_yr: float = 12.0
    coarse_step_psf_fraction: float = 1.0 / 3.0   # coarse step shifts a source by this fraction of the FWHM over the baseline
    fine_factor: int = 5
    detect_snr: float = 5.0
    min_delta_chi2: float = 11.8                  # 3 sigma for the two extra velocity parameters
    min_snr_gain: float = 1.05                    # S/N at best motion must beat the zero-motion S/N by this factor
    min_shift_psf_fraction: float = 0.5           # total displacement over the baseline, in FWHM, needed to call it moving
    static_radius_psf: float = 1.0                # zero-motion comparison is taken within this many FWHM
    edge_margin_psf: float = 1.0
    n_controls: int = 40
    max_candidates: int = 10
    seed: int = 12345


@dataclass
class _Prepared:
    num: list[np.ndarray]             # (d/var) (*) P per frame
    den: list[np.ndarray]             # (1/var) (*) P^2 per frame
    times: np.ndarray
    scale: float
    shape: tuple[int, int]


def _psf_kernel(sigma_px: float) -> np.ndarray:
    r = int(math.ceil(4 * sigma_px))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_px ** 2))
    return k / k.sum()


def prepare(frames: list[StackFrame], grid_scale_arcsec: float, cfg: ShiftStackConfig) -> _Prepared:
    sigma_px = cfg.psf_fwhm_arcsec / 2.3548 / grid_scale_arcsec
    psf = _psf_kernel(sigma_px)
    num, den = [], []
    for f in frames:
        img = np.asarray(f.image, float)
        valid = np.isfinite(img)
        if valid.sum() < 20:
            num.append(np.zeros(img.shape)); den.append(np.zeros(img.shape))
            continue
        med = float(np.median(img[valid]))
        mad = 1.4826 * float(np.median(np.abs(img[valid] - med)))
        if f.variance is not None:
            var = np.asarray(f.variance, float)
            valid &= np.isfinite(var) & (var > 0)
        else:
            var = np.full(img.shape, max(mad, 1e-12) ** 2)
        w = np.where(valid, 1.0 / np.where(valid, var, 1.0), 0.0)
        d = np.where(valid, img - med, 0.0)
        num.append(ndimage.correlate(w * d, psf, mode="constant", cval=0.0))
        den.append(ndimage.correlate(w, psf ** 2, mode="constant", cval=0.0))
    return _Prepared(num=num, den=den, times=np.array([f.time_mjd for f in frames], float),
                     scale=grid_scale_arcsec, shape=frames[0].image.shape)


def _normalise(snr: np.ndarray, den: np.ndarray) -> np.ndarray:
    ok = den > 0
    if ok.sum() < 20:
        return np.zeros_like(snr)
    vals = snr[ok]
    med = np.median(vals)
    mad = 1.4826 * np.median(np.abs(vals - med))
    out = np.zeros_like(snr)
    if mad > 0:
        out[ok] = (vals - med) / mad
    return out


def stack_at(prep: _Prepared, ve: float, vn: float, times: Optional[np.ndarray] = None,
             t_ref: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
    """(normalised S/N map, formal flux map) for a source moving `ve` east and
    `vn` north (arcsec/yr), in the frame of epoch `t_ref` (default: mean time).
    The flux map is in image units summed over the PSF (multiply by the pixel
    solid angle for a flux density)."""
    t = prep.times if times is None else times
    t_ref = float(np.mean(prep.times)) if t_ref is None else t_ref
    num = np.zeros(prep.shape)
    den = np.zeros(prep.shape)
    for n_i, d_i, t_i in zip(prep.num, prep.den, t):
        dt = (t_i - t_ref) / DAYS_PER_YEAR
        # Source position in frame i relative to t_ref: east = -x, north = +y.
        # Move the frame's content back by that displacement.
        dx = ve * dt / prep.scale
        dy = -vn * dt / prep.scale
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            num += n_i; den += d_i
        else:
            num += ndimage.shift(n_i, (dy, dx), order=1, mode="constant", cval=0.0)
            den += ndimage.shift(d_i, (dy, dx), order=1, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = np.where(den > 0, num / np.sqrt(den), 0.0)
        flux = np.where(den > 0, num / den, np.nan)
    return _normalise(snr, den), flux


def _raw_snr(prep: _Prepared, ve: float, vn: float, times: np.ndarray, t_ref: float) -> np.ndarray:
    """Un-normalised formal S/N map (see stack_at for the conventions)."""
    num = np.zeros(prep.shape)
    den = np.zeros(prep.shape)
    for n_i, d_i, t_i in zip(prep.num, prep.den, times):
        dt = (t_i - t_ref) / DAYS_PER_YEAR
        dx, dy = ve * dt / prep.scale, -vn * dt / prep.scale
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            num += n_i; den += d_i
        else:
            num += ndimage.shift(n_i, (dy, dx), order=1, mode="constant", cval=0.0)
            den += ndimage.shift(d_i, (dy, dx), order=1, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.sqrt(den), 0.0)


def _noise_scale(prep: _Prepared, ve: float, vn: float, times: np.ndarray, t_ref: float) -> tuple[float, float]:
    """(median, robust sigma) of the full-frame formal S/N map at this velocity."""
    raw = _raw_snr(prep, ve, vn, times, t_ref)
    vals = raw[raw != 0]
    if vals.size < 20:
        return 0.0, 1.0
    med = float(np.median(vals))
    mad = 1.4826 * float(np.median(np.abs(vals - med)))
    return med, (mad if mad > 0 else 1.0)


def velocity_grid(prep: _Prepared, cfg: ShiftStackConfig) -> tuple[np.ndarray, float]:
    baseline = max((prep.times.max() - prep.times.min()) / DAYS_PER_YEAR, 1e-3)
    step = cfg.coarse_step_psf_fraction * cfg.psf_fwhm_arcsec / baseline
    n = max(1, int(math.ceil(cfg.v_max_arcsec_yr / step)))
    return np.arange(-n, n + 1) * step, step


def _search(prep: _Prepared, cfg: ShiftStackConfig, times: Optional[np.ndarray] = None,
            refine: bool = True) -> tuple[list[dict], dict]:
    axis, step = velocity_grid(prep, cfg)
    h, w = prep.shape
    fwhm_px = cfg.psf_fwhm_arcsec / prep.scale
    meta_baseline = max((prep.times.max() - prep.times.min()) / DAYS_PER_YEAR, 1e-3)
    margin = int(round(cfg.edge_margin_psf * fwhm_px))
    inner = np.zeros(prep.shape, bool)
    inner[margin:h - margin, margin:w - margin] = True

    s0, _ = stack_at(prep, 0.0, 0.0, times)
    size = 2 * int(round(cfg.static_radius_psf * fwhm_px)) + 1
    s0_near = ndimage.maximum_filter(s0, size=size)

    best = np.full(prep.shape, -np.inf)
    best_ve = np.zeros(prep.shape)
    best_vn = np.zeros(prep.shape)
    for ve in axis:
        for vn in axis:
            if math.hypot(ve, vn) > cfg.v_max_arcsec_yr * 1.0001:
                continue
            s = s0 if (ve == 0 and vn == 0) else stack_at(prep, ve, vn, times)[0]
            better = s > best
            best[better] = s[better]
            best_ve[better] = ve
            best_vn[better] = vn

    # local maxima of the best-over-velocity map
    peaks = (best == ndimage.maximum_filter(best, size=size)) & inner & (best >= cfg.detect_snr)
    cands = []
    for y, x in zip(*np.nonzero(peaks)):
        cands.append({"x": int(x), "y": int(y), "snr": float(best[y, x]), "ve": float(best_ve[y, x]),
                      "vn": float(best_vn[y, x]), "snr_static": float(s0_near[y, x])})
    cands.sort(key=lambda c: -c["snr"])
    cands = cands[: cfg.max_candidates * 3]

    if refine:
        fine = step / cfg.fine_factor
        offs = np.arange(-cfg.fine_factor, cfg.fine_factor + 1) * fine
        r = int(round(fwhm_px))
        t_all = prep.times if times is None else times
        t_ref = float(np.mean(prep.times))
        for c in cands:
            # Every peak is refined, including those whose best COARSE
            # velocity is zero: on real data (WISE 0855, D5) the coarse grid
            # can miss a mover's velocity by enough that zero motion wins, and
            # only the fine grid recovers it. Cost is kept down by working on
            # a window around the peak.
            y0, y1 = max(c["y"] - r, 0), min(c["y"] + r + 1, h)
            x0, x1 = max(c["x"] - r, 0), min(c["x"] + r + 1, w)
            # Refinement only needs the S/N near this peak: accumulate just
            # that window (the frames are shifted whole, then cropped; the
            # per-plane noise scale comes from the coarse plane's statistics).
            surface = np.full((offs.size, offs.size), -np.inf)
            top = (c["snr"], c["ve"], c["vn"], c["x"], c["y"])
            pad = r + int(math.ceil((cfg.v_max_arcsec_yr + step) * meta_baseline / prep.scale)) + 3
            Y0, Y1 = max(c["y"] - pad, 0), min(c["y"] + pad + 1, h)
            X0, X1 = max(c["x"] - pad, 0), min(c["x"] + pad + 1, w)
            sub_prep = _Prepared(num=[a[Y0:Y1, X0:X1] for a in prep.num], den=[a[Y0:Y1, X0:X1] for a in prep.den],
                                 times=prep.times, scale=prep.scale, shape=(Y1 - Y0, X1 - X0))
            scale_ref = _noise_scale(prep, c["ve"], c["vn"], t_all, t_ref)
            for i, de in enumerate(offs):
                for j, dn in enumerate(offs):
                    raw = _raw_snr(sub_prep, c["ve"] + de, c["vn"] + dn, t_all, t_ref)
                    s = (raw - scale_ref[0]) / scale_ref[1]
                    sub = s[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
                    k = np.unravel_index(np.argmax(sub), sub.shape)
                    surface[i, j] = sub[k]
                    if sub[k] > top[0]:
                        top = (float(sub[k]), c["ve"] + de, c["vn"] + dn, x0 + int(k[1]), y0 + int(k[0]))
            c["snr"], c["ve"], c["vn"], c["x"], c["y"] = top
            # 1-sigma velocity half-widths: where S/N^2 falls by 1 from its peak
            drop = top[0] ** 2 - surface ** 2
            within = drop <= 1.0
            ii, jj = np.nonzero(within)
            c["ve_err"] = float(max(fine, (offs[ii].max() - offs[ii].min()) / 2)) if ii.size else float(fine)
            c["vn_err"] = float(max(fine, (offs[jj].max() - offs[jj].min()) / 2)) if jj.size else float(fine)

    baseline_yr = max((prep.times.max() - prep.times.min()) / DAYS_PER_YEAR, 1e-3)
    min_speed = cfg.min_shift_psf_fraction * cfg.psf_fwhm_arcsec / baseline_yr
    for c in cands:
        c["delta_chi2"] = c["snr"] ** 2 - max(c["snr_static"], 0.0) ** 2
        c["speed"] = math.hypot(c["ve"], c["vn"])
        # Three conditions, all needed (the first alone wrongly flags bright
        # static stars, whose S/N^2 jitters by more than the threshold):
        c["moving"] = bool(
            c["delta_chi2"] >= cfg.min_delta_chi2
            and c["snr"] >= cfg.min_snr_gain * max(c["snr_static"], 0.0)
            and c["speed"] >= min_speed
        )
    meta = {"velocity_step_arcsec_yr": float(step), "n_velocities": int(sum(1 for a in axis for b in axis if math.hypot(a, b) <= cfg.v_max_arcsec_yr * 1.0001)),
            "baseline_yr": float((prep.times.max() - prep.times.min()) / DAYS_PER_YEAR),
            "min_distinguishable_speed_arcsec_yr": float(min_speed),
            "v_max_arcsec_yr": float(cfg.v_max_arcsec_yr),
            "position_quantum_arcsec": float(prep.scale)}
    return cands, meta


def _best_moving(cands: list[dict]) -> float:
    moving = [c["snr"] for c in cands if c["moving"]]
    return max(moving) if moving else 0.0


def search(frames: list[StackFrame], grid_scale_arcsec: float, cfg: Optional[ShiftStackConfig] = None) -> dict:
    """Blind search over position and velocity, with time-shuffled controls."""
    cfg = cfg or ShiftStackConfig()
    t0 = time.perf_counter()
    prep = prepare(frames, grid_scale_arcsec, cfg)
    cands, meta = _search(prep, cfg)
    t_search = time.perf_counter() - t0

    # Controls: same frames, observation times shuffled. Coarse search only
    # (refinement can only raise S/N a little, and it does so for the real
    # data too, so the real candidate is compared at its COARSE S/N).
    coarse, _ = _search(prep, cfg, refine=False)
    real_stat = _best_moving(coarse)
    rng = np.random.default_rng(cfg.seed)
    control_stats = []
    n_frames = len(frames)
    if n_frames >= 3 and cfg.n_controls > 0:
        seen = set()
        attempts = 0
        while len(control_stats) < cfg.n_controls and attempts < cfg.n_controls * 20:
            attempts += 1
            perm = tuple(rng.permutation(n_frames))
            if perm in seen or np.array_equal(prep.times[list(perm)], prep.times):
                continue
            seen.add(perm)
            c_cands, _ = _search(prep, cfg, times=prep.times[list(perm)], refine=False)
            control_stats.append(_best_moving(c_cands))
    k = len(control_stats)
    p_fa = (1 + sum(1 for s in control_stats if s >= real_stat)) / (k + 1) if (k and real_stat > 0) else None

    out_cands = []
    # Every moving candidate is returned, then the strongest static peaks up
    # to max_candidates. (Ranking by S/N alone let bright static stars push a
    # faint mover out of the list: WISE 0855 in D5.)
    moving_c = [c for c in cands if c["moving"]]
    static_c = [c for c in cands if not c["moving"]][: max(cfg.max_candidates - len(moving_c), 0)]
    for c in sorted(moving_c + static_c, key=lambda c: -c["snr"]):
        out_cands.append({
            "x": c["x"], "y": c["y"], "snr": c["snr"], "snr_static": c["snr_static"], "delta_chi2": c["delta_chi2"],
            "pm_east_arcsec_yr": c["ve"], "pm_north_arcsec_yr": c["vn"],
            "pm_east_err_arcsec_yr": c.get("ve_err"), "pm_north_err_arcsec_yr": c.get("vn_err"),
            "speed_arcsec_yr": c["speed"], "moving": c["moving"],
        })
    return {
        "n_frames": n_frames,
        "epoch_ref_mjd": float(np.mean(prep.times)),
        **meta,
        "candidates": out_cands,
        "best_moving_snr_coarse": real_stat,
        "controls": {"n": k, "best_moving_snr": control_stats,
                     "false_alarm_probability": p_fa,
                     "note": "Fraction of time-shuffled control searches whose best moving peak is at least as strong as the real one. Cannot be smaller than 1/(n+1)."},
        "seconds": {"search": t_search, "total": time.perf_counter() - t0},
    }


def forced(frames: list[StackFrame], grid_scale_arcsec: float, x: float, y: float, ve: float, vn: float,
           t_ref: Optional[float] = None, cfg: Optional[ShiftStackConfig] = None, n_control_positions: int = 200) -> dict:
    """S/N of a KNOWN trajectory (position x, y at `t_ref`, motion ve, vn),
    with three kinds of control so the number can be judged:
      * the same position stacked with no motion and with reversed motion,
      * the same motion stacked at many other positions (its noise distribution),
      * the same position with the motion direction rotated in 30-degree steps.
    """
    cfg = cfg or ShiftStackConfig()
    prep = prepare(frames, grid_scale_arcsec, cfg)
    h, w = prep.shape
    xi, yi = int(round(x)), int(round(y))

    def at(a, b):
        s, f = stack_at(prep, a, b, t_ref=t_ref)
        return s, f

    s_on, f_on = at(ve, vn)
    r = max(1, int(round(0.33 * cfg.psf_fwhm_arcsec / grid_scale_arcsec)))
    box = (slice(max(yi - r, 0), min(yi + r + 1, h)), slice(max(xi - r, 0), min(xi + r + 1, w)))
    snr_on = float(np.max(s_on[box]))
    k = np.unravel_index(np.argmax(s_on[box]), s_on[box].shape)
    flux_on = float(f_on[box][k])

    snr_static = float(np.max(at(0.0, 0.0)[0][box]))
    snr_reversed = float(np.max(at(-ve, -vn)[0][box]))
    speed = math.hypot(ve, vn)
    ang0 = math.atan2(vn, ve)
    rotated = []
    for step in range(1, 12):
        a = ang0 + step * math.pi / 6
        rotated.append(float(np.max(at(speed * math.cos(a), speed * math.sin(a))[0][box])))

    # off-position controls with the true motion: local maxima in boxes of the
    # same size at random places away from the target
    rng = np.random.default_rng(cfg.seed)
    margin = int(round(cfg.psf_fwhm_arcsec / grid_scale_arcsec)) + r
    offs = []
    tries = 0
    while len(offs) < n_control_positions and tries < n_control_positions * 20:
        tries += 1
        cx, cy = int(rng.integers(margin, w - margin)), int(rng.integers(margin, h - margin))
        if math.hypot(cx - xi, cy - yi) * grid_scale_arcsec < 2 * cfg.psf_fwhm_arcsec:
            continue
        offs.append(float(np.max(s_on[cy - r:cy + r + 1, cx - r:cx + r + 1])))
    offs_arr = np.array(offs) if offs else np.array([0.0])
    return {
        "snr_on_trajectory": snr_on,
        "flux_sum_image_units": flux_on,
        "snr_same_position_no_motion": snr_static,
        "snr_same_position_reversed_motion": snr_reversed,
        "snr_same_position_rotated_motion": rotated,
        "off_position": {"n": len(offs), "median": float(np.median(offs_arr)), "p95": float(np.percentile(offs_arr, 95)),
                         "max": float(offs_arr.max()),
                         "fraction_at_least_as_high": float(np.mean(offs_arr >= snr_on))},
        "n_frames": len(frames),
    }
