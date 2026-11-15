"""Source detection, cross-epoch tracking, proper-motion fitting, variability,
catalogue cross-matching and explainable ranking for ONE detector series
(frames of the same detector on the same grid, time-ordered).

Everything here is pure computation on arrays plus lists of catalogue
sources; no network. `analyze_series` is the entry point.

Four quantities are reported separately and never merged
(they answer different questions):
  * detection S/N          how strongly the source stands out of the noise
  * anomaly score          how unusual the measured behaviour is (0..1,
                           a weighted sum of the indicators listed in
                           `score_components`; NOT a probability)
  * reliability            how much the measurement itself can be trusted
                           (epochs, masks, edges, confusion)
  * class probability      NOT estimated. Always null. Nothing here says
                           what kind of object a source is.

Stated assumptions and limits:
  * Positional error per detection = quadrature sum of the centroid's
    statistical error FWHM/(2.355*S/N) and a systematic floor (default 1.0",
    for nearest-neighbour resampling and per-exposure WCS error). A
    detection whose core is masked/saturated gets a 3" floor instead.
  * Tracks are linked nearest-first inside a gate that grows with the time
    gap (max motion `pm_max_arcsec_yr`). In crowded fields a wrong link is
    possible; tracks with more than one candidate in a gate are flagged.
  * Proper motion = weighted linear fit of tangent-plane offsets vs time.
    Parallax is not part of this fit. For the nearest stars (Barnard's
    Star: 0.55") it is larger than the 0.4" positional floor, shows up as
    excess chi^2, and inflates the quoted motion error through the
    chi^2 rescaling. A separate 5-parameter fit (app/astrometry.py) is
    reported alongside, and only believed when it is constrained.
    With 2 epochs there are no degrees of freedom, so the fit cannot check
    itself; flagged.
  * Variability = chi^2 of a constant-flux model, with a fractional
    systematic error added to every point (default 10%, the frame-to-frame
    scatter measured on Barnard's Star at fixed wavelength, see
    docs/AUDIT.md). Frames of one detector still sample different
    wavelengths, so a flux change can be spectral; flagged when the frames'
    wavelengths differ by more than `wavelength_tolerance_um`.
  * "Unmatched" means no Gaia DR3 / CatWISE2020 source inside the
    uncertainty gate. It is not a discovery claim: faint, blended and
    artefact detections are all unmatched too.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from scipy.stats import chi2 as chi2_dist

from .astrometry import fit_motion_parallax, parallax_is_constrained
from .catalogs import CatalogSource, tangent_offset_arcsec
from .photometry import aperture_photometry, DEFAULT_R_AP_ARCSEC, DEFAULT_R_OUT_ARCSEC

DAYS_PER_YEAR = 365.25


@dataclass
class DiscoveryConfig:
    detect_nsigma: float = 5.0
    min_native_pixels: float = 1.5          # connected area, in native-pixel units
    psf_fwhm_arcsec: float = 9.0
    # Systematic floors, measured on REAL data (docs/PHASE2.md, 2026-11-15):
    # 34 field stars with S/N >= 30 in the five validation fields, Lanczos-3
    # resampling, 3-4 epochs each. Position repeats to 0.17-0.23" (median),
    # 0.31-0.39" at the 84th percentile; flux to 2.8-5.3% (median), 5.4-7.1%
    # at the 84th percentile (this includes the wavelength differences between
    # frames of one detector). The values below are the conservative,
    # 84th-percentile ones. They were 1.0" and 10% with nearest-neighbour
    # resampling.
    pos_sys_arcsec: float = 0.4
    pos_sys_masked_arcsec: float = 3.0
    pm_max_arcsec_yr: float = 12.0
    gate_nsigma: float = 4.0
    flux_sys_fraction: float = 0.06
    wavelength_tolerance_um: float = 0.05
    match_nsigma: float = 3.5
    registration_min_stars: int = 3
    registration_max_ref_pm_mas_yr: float = 50.0
    max_sources_per_frame: int = 300
    # Lanczos-3 resampling of an undersampled PSF leaves a negative trough
    # (~-3.5% of the peak at 12-18") and a positive lobe (+1.3-2.8% at
    # 18-24", +0.3% at 24-30") around a star (scripts/benchmark_resampling.py
    # geometry; docs/PHASE2.md). The trough can cut the lobe off as a separate
    # "source". Real case: RW Aur D3, a 2%-of-peak detection 19" away in all
    # three epochs, absent with bilinear resampling. A detection inside this
    # annulus of a star at least `ring_flux_ratio` times brighter (peak S/N)
    # is flagged; it is not removed, because real neighbours exist too.
    ring_inner_arcsec: float = 12.0
    ring_outer_arcsec: float = 32.0
    ring_flux_ratio: float = 25.0
    # Correction of aperture errors for correlated pixels. None = assume each
    # native pixel was simply duplicated (native/grid)^2 times.
    aperture_variance_inflation: Optional[float] = None


@dataclass
class SeriesFrame:
    image: np.ndarray               # MJy/sr, NaN = no data (before display hole-filling)
    variance: Optional[np.ndarray]
    time_mjd: float
    wavelength_um: Optional[float] = None
    label: str = ""


@dataclass
class Detection:
    frame_index: int
    time_mjd: float
    x: float                        # 0-based grid pixel
    y: float
    east_arcsec: float              # tangent-plane offset from grid centre
    north_arcsec: float
    pos_err_arcsec: float
    peak_snr: float
    flux_mjy: Optional[float]
    flux_err_mjy: Optional[float]
    phot_status: str
    core_masked: bool
    near_edge: bool
    near_bright_star: bool = False
    ra_deg: float = 0.0
    dec_deg: float = 0.0


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_sources_in_frame(
    frame: SeriesFrame,
    frame_index: int,
    wcs: WCS,
    grid_scale_arcsec: float,
    native_scale_arcsec: float,
    cfg: DiscoveryConfig,
) -> list[Detection]:
    from photutils.segmentation import detect_sources, deblend_sources, SourceCatalog

    image = frame.image
    finite = np.isfinite(image)
    if finite.sum() < 50:
        return []
    _, bkg, std = sigma_clipped_stats(image[finite], sigma=3.0, maxiters=10)
    if not np.isfinite(std) or std <= 0:
        return []

    data = np.where(finite, image - bkg, 0.0)
    oversample = (native_scale_arcsec / grid_scale_arcsec) ** 2
    npixels = max(3, int(round(cfg.min_native_pixels * oversample)))
    segm = detect_sources(data, cfg.detect_nsigma * std, n_pixels=npixels, mask=~finite)
    if segm is None:
        return []
    try:
        segm = deblend_sources(data, segm, n_pixels=npixels, n_levels=16, contrast=0.05)
    except Exception:  # noqa: BLE001 - deblending is an improvement, not a requirement
        pass
    cat = SourceCatalog(data, segm, mask=~finite)

    h, w = image.shape
    edge_pix = DEFAULT_R_OUT_ARCSEC / grid_scale_arcsec
    r_ap_pix = DEFAULT_R_AP_ARCSEC / grid_scale_arcsec
    yy, xx = np.mgrid[0:h, 0:w]

    order = np.argsort(-np.asarray(cat.max_value))[: cfg.max_sources_per_frame]
    xs, ys, peaks = np.atleast_1d(cat.x_centroid), np.atleast_1d(cat.y_centroid), np.atleast_1d(cat.max_value)
    out: list[Detection] = []
    for i in order:
        x, y, peak = float(xs[i]), float(ys[i]), float(peaks[i])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        core_masked = bool((~finite & (np.hypot(xx - x, yy - y) <= r_ap_pix)).any())
        near_edge = x < edge_pix or y < edge_pix or x > w - 1 - edge_pix or y > h - 1 - edge_pix
        phot = aperture_photometry(image, frame.variance, x, y, grid_scale_arcsec,
                                   variance_inflation=cfg.aperture_variance_inflation or oversample)
        peak_snr = peak / std
        # Astrometric precision from the best available S/N.
        snr_for_pos = max(peak_snr, phot.snr or 0.0, 1.0)
        stat = cfg.psf_fwhm_arcsec / (2.355 * snr_for_pos)
        sys = cfg.pos_sys_masked_arcsec if core_masked else cfg.pos_sys_arcsec
        east = -(x - (w - 1) / 2.0) * grid_scale_arcsec
        north = (y - (h - 1) / 2.0) * grid_scale_arcsec
        sky = wcs.pixel_to_world(x, y)
        out.append(
            Detection(
                frame_index=frame_index, time_mjd=frame.time_mjd, x=x, y=y,
                east_arcsec=east, north_arcsec=north,
                pos_err_arcsec=math.hypot(stat, sys),
                peak_snr=float(peak_snr),
                flux_mjy=phot.flux_mjy, flux_err_mjy=phot.flux_err_mjy, phot_status=phot.status,
                core_masked=core_masked, near_edge=bool(near_edge),
                ra_deg=float(sky.ra.deg), dec_deg=float(sky.dec.deg),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

@dataclass
class Track:
    detections: list[Detection] = field(default_factory=list)
    ambiguous_links: int = 0

    def predict(self, mjd: float) -> tuple[float, float]:
        d = self.detections
        if len(d) < 2 or d[-1].time_mjd == d[0].time_mjd:
            return d[-1].east_arcsec, d[-1].north_arcsec
        dt = d[-1].time_mjd - d[0].time_mjd
        vx = (d[-1].east_arcsec - d[0].east_arcsec) / dt
        vy = (d[-1].north_arcsec - d[0].north_arcsec) / dt
        return d[-1].east_arcsec + vx * (mjd - d[-1].time_mjd), d[-1].north_arcsec + vy * (mjd - d[-1].time_mjd)


def link_detections(per_frame: list[list[Detection]], cfg: DiscoveryConfig) -> list[Track]:
    """Greedy nearest-first association, frame by frame in time order."""
    tracks: list[Track] = []
    for dets in per_frame:
        if not dets:
            continue
        if not tracks:
            tracks = [Track([d]) for d in dets]
            continue
        t_now = dets[0].time_mjd
        pairs = []  # (distance, track_i, det_j)
        in_gate_count = np.zeros(len(tracks), dtype=int)
        for ti, tr in enumerate(tracks):
            last = tr.detections[-1]
            dt_yr = abs(t_now - last.time_mjd) / DAYS_PER_YEAR
            px, py = tr.predict(t_now)
            # A track with a measured velocity needs far less room than one
            # whose motion is still unknown.
            motion_room = cfg.pm_max_arcsec_yr * dt_yr * (1.0 if len(tr.detections) < 2 else 0.25)
            for dj, d in enumerate(dets):
                gate = cfg.gate_nsigma * math.hypot(last.pos_err_arcsec, d.pos_err_arcsec) + motion_room
                dist = math.hypot(d.east_arcsec - px, d.north_arcsec - py)
                if dist <= gate:
                    pairs.append((dist, ti, dj))
                    in_gate_count[ti] += 1
        used_t, used_d = set(), set()
        for dist, ti, dj in sorted(pairs):
            if ti in used_t or dj in used_d:
                continue
            used_t.add(ti)
            used_d.add(dj)
            tracks[ti].detections.append(dets[dj])
            if in_gate_count[ti] > 1:
                tracks[ti].ambiguous_links += 1
        for dj, d in enumerate(dets):
            if dj not in used_d:
                tracks.append(Track([d]))
    return tracks


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------

@dataclass
class MotionFit:
    n_epochs: int
    baseline_yr: float
    t0_mjd: float                    # weighted mean epoch (positions below refer to it)
    east0_arcsec: float
    north0_arcsec: float
    pos0_err_arcsec: float
    pmra_mas_yr: Optional[float] = None      # east component = mu_alpha*cos(dec)
    pmdec_mas_yr: Optional[float] = None
    pmra_err_mas_yr: Optional[float] = None
    pmdec_err_mas_yr: Optional[float] = None
    pm_total_mas_yr: Optional[float] = None
    pm_significance: Optional[float] = None  # sqrt(chi^2) of "no motion", 2 parameters
    chi2: Optional[float] = None
    dof: int = 0
    errors_rescaled: bool = False


def fit_motion(times_mjd, east, north, sigma) -> MotionFit:
    t = np.asarray(times_mjd, float)
    e, n, s = np.asarray(east, float), np.asarray(north, float), np.asarray(sigma, float)
    w = 1.0 / s ** 2
    t0 = float(np.sum(w * t) / np.sum(w))
    tau = (t - t0) / DAYS_PER_YEAR
    e0, n0 = float(np.sum(w * e) / np.sum(w)), float(np.sum(w * n) / np.sum(w))
    pos_err = float(1.0 / math.sqrt(np.sum(w)))
    baseline = float((t.max() - t.min()) / DAYS_PER_YEAR)
    fit = MotionFit(n_epochs=len(t), baseline_yr=baseline, t0_mjd=t0, east0_arcsec=e0, north0_arcsec=n0, pos0_err_arcsec=pos_err)
    stt = float(np.sum(w * tau ** 2))
    if len(t) < 2 or stt <= 0:
        return fit
    # tau is centred on the weighted mean, so slope and intercept decouple.
    vx = float(np.sum(w * tau * (e - e0)) / stt)
    vy = float(np.sum(w * tau * (n - n0)) / stt)
    v_err = 1.0 / math.sqrt(stt)
    chi2 = float(np.sum(w * ((e - e0 - vx * tau) ** 2 + (n - n0 - vy * tau) ** 2)))
    dof = 2 * len(t) - 4
    rescaled = False
    if dof > 0 and chi2 / dof > 1.0:
        v_err *= math.sqrt(chi2 / dof)
        rescaled = True
    fit.pmra_mas_yr, fit.pmdec_mas_yr = vx * 1000.0, vy * 1000.0
    fit.pmra_err_mas_yr = fit.pmdec_err_mas_yr = v_err * 1000.0
    fit.pm_total_mas_yr = math.hypot(vx, vy) * 1000.0
    fit.pm_significance = math.hypot(vx, vy) / v_err
    fit.chi2, fit.dof, fit.errors_rescaled = chi2, dof, rescaled
    return fit


@dataclass
class VariabilityFit:
    n_points: int
    mean_flux_mjy: Optional[float] = None
    chi2: Optional[float] = None
    dof: int = 0
    p_value: Optional[float] = None          # P(chi^2 >= observed | constant flux)
    amplitude_fraction: Optional[float] = None  # (max-min)/mean
    wavelength_span_um: Optional[float] = None
    wavelength_mismatch: bool = False
    sys_fraction: float = 0.06


def fit_variability(fluxes, errors, wavelengths, cfg: DiscoveryConfig) -> VariabilityFit:
    f, e = np.asarray(fluxes, float), np.asarray(errors, float)
    out = VariabilityFit(n_points=len(f), sys_fraction=cfg.flux_sys_fraction)
    waves = [w for w in wavelengths if w is not None and np.isfinite(w)]
    if len(waves) >= 2:
        out.wavelength_span_um = float(max(waves) - min(waves))
        out.wavelength_mismatch = out.wavelength_span_um > cfg.wavelength_tolerance_um
    if len(f) < 2:
        if len(f) == 1:
            out.mean_flux_mjy = float(f[0])
        return out
    sig = np.sqrt(e ** 2 + (cfg.flux_sys_fraction * np.abs(f)) ** 2)
    w = 1.0 / sig ** 2
    mean = float(np.sum(w * f) / np.sum(w))
    chi2 = float(np.sum(w * (f - mean) ** 2))
    dof = len(f) - 1
    out.mean_flux_mjy, out.chi2, out.dof = mean, chi2, dof
    out.p_value = float(chi2_dist.sf(chi2, dof))
    if mean != 0:
        out.amplitude_fraction = float((f.max() - f.min()) / abs(mean))
    return out


# ---------------------------------------------------------------------------
# Catalogue matching
# ---------------------------------------------------------------------------

@dataclass
class CatalogMatch:
    status: str                      # "matched" | "ambiguous" | "unmatched" | "catalog_unavailable"
    n_in_gate: int = 0
    n_within_psf: int = 0            # catalogue sources inside one PSF FWHM (blend indicator)
    separation_arcsec: Optional[float] = None
    normalized_separation: Optional[float] = None
    source: Optional[dict] = None
    pm_difference_sigma: Optional[float] = None


def match_to_catalog(
    east: float, north: float, pos_err: float, mjd: float,
    sources: Optional[list[CatalogSource]], centre_ra: float, centre_dec: float, cfg: DiscoveryConfig,
    fit: Optional[MotionFit] = None,
) -> CatalogMatch:
    if sources is None:
        return CatalogMatch(status="catalog_unavailable")
    best = None
    n_gate = n_psf = 0
    for s in sources:
        ra, dec, sig_cat = s.position_at(mjd)
        ce, cn = tangent_offset_arcsec(ra, dec, centre_ra, centre_dec)
        d = math.hypot(ce - east, cn - north)
        if d <= cfg.psf_fwhm_arcsec:
            n_psf += 1
        sigma = math.hypot(pos_err, sig_cat)
        if d <= cfg.match_nsigma * sigma:
            n_gate += 1
            if best is None or d / sigma < best[1]:
                best = (d, d / sigma, s)
    if best is None:
        return CatalogMatch(status="unmatched", n_within_psf=n_psf)
    d, dn, s = best
    m = CatalogMatch(status="matched" if n_gate == 1 else "ambiguous", n_in_gate=n_gate, n_within_psf=n_psf,
                     separation_arcsec=d, normalized_separation=dn, source=s.to_dict())
    if fit is not None and fit.pmra_mas_yr is not None and s.has_pm:
        err = math.hypot(fit.pmra_err_mas_yr, s.pm_err_mas_yr or 0.0)
        if err > 0:
            m.pm_difference_sigma = math.hypot(fit.pmra_mas_yr - s.pmra_mas_yr, fit.pmdec_mas_yr - s.pmdec_mas_yr) / err
    return m


def estimate_registration(tracks: list[Track], gaia: Optional[list[CatalogSource]], n_frames: int,
                          centre_ra: float, centre_dec: float, cfg: DiscoveryConfig) -> list[dict]:
    """Per-frame median offset (measured - Gaia) of slow, clean Gaia stars.

    This is the check that apparent motion is not image misalignment: if a
    frame is shifted, every star in it shifts together. Frames with fewer
    than `registration_min_stars` reference stars are reported as unchecked
    and left uncorrected.
    """
    offsets: list[list[tuple[float, float]]] = [[] for _ in range(n_frames)]
    if gaia:
        refs = [s for s in gaia if s.pm_total_mas_yr() <= cfg.registration_max_ref_pm_mas_yr]
        for tr in tracks:
            for d in tr.detections:
                if d.core_masked or d.near_edge:
                    continue
                cands = []
                for s in refs:
                    ra, dec, _ = s.position_at(d.time_mjd)
                    ce, cn = tangent_offset_arcsec(ra, dec, centre_ra, centre_dec)
                    sep = math.hypot(ce - d.east_arcsec, cn - d.north_arcsec)
                    if sep <= cfg.psf_fwhm_arcsec / 2.0:
                        cands.append((sep, d.east_arcsec - ce, d.north_arcsec - cn))
                if len(cands) == 1:  # isolated reference stars only
                    offsets[d.frame_index].append((cands[0][1], cands[0][2]))
    report = []
    for fi in range(n_frames):
        n = len(offsets[fi])
        if n >= cfg.registration_min_stars:
            arr = np.array(offsets[fi])
            de, dn = float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))
            report.append({"frame_index": fi, "n_reference_stars": n, "offset_east_arcsec": de,
                           "offset_north_arcsec": dn, "applied": True})
        else:
            report.append({"frame_index": fi, "n_reference_stars": n, "offset_east_arcsec": None,
                           "offset_north_arcsec": None, "applied": False})
    return report


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def _clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


def score_track(fit: MotionFit, var: VariabilityFit, gaia: CatalogMatch, catwise: CatalogMatch,
                flags: list[str], median_snr: float) -> dict:
    reasons: list[str] = []
    caveats: list[str] = []

    # --- reliability (trust in the measurement itself) ---
    penalty = 0
    if fit.n_epochs < 2:
        penalty += 3
        caveats.append("Seen in only one frame: no motion or variability can be measured, and a single detection can be an artefact.")
    elif fit.n_epochs == 2:
        penalty += 1
        caveats.append("Only two epochs: the motion fit has no redundancy to check itself.")
    if "core_masked_all_epochs" in flags:
        # No flux at all and only a 3" position floor in every epoch (e.g. a
        # saturated bright star such as R Dra): never a reliable candidate.
        penalty += 2
        caveats.append("Core is masked or saturated in every frame: no flux can be measured and the position is only "
                       "known to a few arcseconds.")
    elif "core_masked" in flags:
        penalty += 1
        caveats.append("Core is masked or saturated in at least one frame: position is less certain and flux is not measured there.")
    if "near_bright_star" in flags:
        penalty += 2
        caveats.append(
            "Faint detection 12-32\" from a star at least 25 times brighter: it may be a resampling ringing lobe "
            "or a PSF feature of that star rather than a separate source."
        )
    if "near_edge" in flags:
        penalty += 1
        caveats.append("Close to the image edge in at least one frame.")
    if "ambiguous_link" in flags:
        penalty += 1
        caveats.append("More than one detection could have continued this track: the association across epochs may be wrong.")
    if "registration_unchecked" in flags:
        caveats.append("Image registration could not be checked against Gaia stars in every frame (too few reference stars).")
    if median_snr < 8:
        penalty += 1
        caveats.append(f"Faint (median S/N {median_snr:.1f}).")
    reliability = "high" if penalty == 0 else "medium" if penalty <= 1 else "low"

    # --- indicators ---
    motion = 0.0
    if fit.pm_significance is not None and fit.n_epochs >= 2:
        motion = _clip01((fit.pm_significance - 3.0) / 7.0)
        if fit.n_epochs == 2:
            motion *= 0.5
        if fit.pm_significance >= 3.0:
            reasons.append(
                f"Moves {fit.pm_total_mas_yr / 1000.0:.2f}\"/yr (east {fit.pmra_mas_yr:.0f}, north {fit.pmdec_mas_yr:.0f} mas/yr, "
                f"±{fit.pmra_err_mas_yr:.0f}), {fit.pm_significance:.1f}σ from zero over {fit.baseline_yr:.2f} yr in {fit.n_epochs} epochs."
            )

    variability = 0.0
    if var.p_value is not None and var.dof >= 1:
        mlog = -math.log10(max(var.p_value, 1e-300))
        variability = _clip01((mlog - 2.0) / 6.0)
        if var.wavelength_mismatch:
            variability *= 0.3
        if mlog >= 2.0:
            txt = (f"Brightness changes by {100 * (var.amplitude_fraction or 0):.0f}% across {var.n_points} frames "
                   f"(p = {var.p_value:.1e} for constant flux after adding a {100 * var.sys_fraction:.0f}% systematic error).")
            if var.wavelength_mismatch:
                txt += f" The frames differ by {var.wavelength_span_um:.2f} µm in wavelength, so this may be the spectrum's shape, not a change in time."
            reasons.append(txt)

    unmatched = 0.0
    gaia_un, cw_un = gaia.status == "unmatched", catwise.status == "unmatched"
    if gaia_un and cw_un:
        unmatched = 1.0
        reasons.append("No Gaia DR3 or CatWISE2020 source inside the positional uncertainty. This alone does not make it new: faint, blended and artefact detections are also unmatched.")
    elif gaia_un and catwise.status in ("matched", "ambiguous"):
        unmatched = 0.3
        reasons.append("In CatWISE2020 but not in Gaia DR3: detected in the infrared, not in the optical (typical of very red or cold objects, also of faint galaxies).")
    elif cw_un and gaia.status in ("matched", "ambiguous"):
        unmatched = 0.1
    if gaia.status == "catalog_unavailable" or catwise.status == "catalog_unavailable":
        caveats.append("A catalogue query failed, so the catalogue comparison is incomplete.")

    for m, label in ((gaia, "Gaia DR3"), (catwise, "CatWISE2020")):
        if m.pm_difference_sigma is not None and m.status == "matched":
            if m.pm_difference_sigma <= 3.0:
                if fit.pm_significance and fit.pm_significance >= 3.0:
                    reasons.append(f"Measured motion agrees with {label} ({m.pm_difference_sigma:.1f}σ difference): a known moving object.")
            else:
                caveats.append(f"Measured motion differs from {label} by {m.pm_difference_sigma:.1f}σ.")
        if m.status == "ambiguous":
            caveats.append(f"{m.n_in_gate} {label} sources fit inside the positional uncertainty: the identification is not unique.")
        if m.n_within_psf > 1:
            caveats.append(f"{m.n_within_psf} {label} sources lie within one SPHEREx PSF width: the detection may be a blend.")

    components = {"motion": motion, "variability": variability, "no_catalogue_counterpart": unmatched}
    weights = {"motion": 0.45, "variability": 0.30, "no_catalogue_counterpart": 0.25}
    anomaly = sum(weights[k] * components[k] for k in components)

    known = gaia.status == "matched" or catwise.status == "matched"
    if reliability == "low":
        label = "uncertain"
    elif known:
        label = "catalogued"
    elif gaia.status == "catalog_unavailable" or catwise.status == "catalog_unavailable":
        # Not matched in the catalogue we could reach, and the other one
        # could not be asked: the comparison is incomplete, so no verdict.
        label = "unverified"
    else:
        label = "candidate"

    return {
        "label": label,
        "anomaly_score": anomaly,
        "score_components": components,
        "score_weights": weights,
        "reliability": reliability,
        "class_probability": None,
        "reasons": reasons,
        "caveats": caveats,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def analyze_series(
    frames: list[SeriesFrame],
    wcs: WCS,
    grid_scale_arcsec: float,
    native_scale_arcsec: float,
    centre_ra: float,
    centre_dec: float,
    gaia: Optional[list[CatalogSource]] = None,
    catwise: Optional[list[CatalogSource]] = None,
    cfg: Optional[DiscoveryConfig] = None,
    series: str = "",
) -> dict:
    cfg = cfg or DiscoveryConfig()
    order = np.argsort([f.time_mjd for f in frames])
    frames = [frames[i] for i in order]
    per_frame = [detect_sources_in_frame(f, i, wcs, grid_scale_arcsec, native_scale_arcsec, cfg) for i, f in enumerate(frames)]
    for dets_f in per_frame:
        for d in dets_f:
            for b in dets_f:
                if b is d or b.peak_snr < cfg.ring_flux_ratio * d.peak_snr:
                    continue
                sep = math.hypot(b.east_arcsec - d.east_arcsec, b.north_arcsec - d.north_arcsec)
                if cfg.ring_inner_arcsec <= sep <= cfg.ring_outer_arcsec:
                    d.near_bright_star = True
                    break
    # Registration is measured per frame against Gaia and removed BEFORE
    # linking: a shifted frame would otherwise break the associations (the
    # link gate is only a few times the positional error).
    registration = estimate_registration(
        [Track([d]) for dets in per_frame for d in dets], gaia, len(frames), centre_ra, centre_dec, cfg
    )
    all_checked = all(r["applied"] for r in registration) if registration else False
    for r in registration:
        if r["applied"]:
            for d in per_frame[r["frame_index"]]:
                d.east_arcsec -= r["offset_east_arcsec"]
                d.north_arcsec -= r["offset_north_arcsec"]
    tracks = link_detections(per_frame, cfg)

    h, w = frames[0].image.shape if frames else (0, 0)
    results = []
    for tr in tracks:
        dets = tr.detections
        east = [d.east_arcsec for d in dets]
        north = [d.north_arcsec for d in dets]
        fit = fit_motion([d.time_mjd for d in dets], east, north, [d.pos_err_arcsec for d in dets])

        # Parallax-aware fit, reported next to (never instead of) the linear
        # fit unless it passes the constraint test in app/astrometry.py.
        parallax = None
        if len(dets) >= 3:
            pfit = fit_motion_parallax([d.time_mjd for d in dets], east, north, [d.pos_err_arcsec for d in dets],
                                       centre_ra, centre_dec)
            ok_plx, why_plx = parallax_is_constrained(pfit)
            if pfit is not None:
                parallax = {**pfit.__dict__, "constrained": ok_plx, "reason": why_plx}

        ok = [d for d in dets if d.phot_status == "ok" and d.flux_mjy is not None]
        var = fit_variability([d.flux_mjy for d in ok], [d.flux_err_mjy for d in ok],
                              [frames[d.frame_index].wavelength_um for d in ok], cfg)

        flags = []
        if any(d.core_masked for d in dets):
            flags.append("core_masked")
        if dets and all(d.core_masked for d in dets):
            flags.append("core_masked_all_epochs")
        if any(d.near_bright_star for d in dets):
            flags.append("near_bright_star")
        if any(d.near_edge for d in dets):
            flags.append("near_edge")
        if tr.ambiguous_links:
            flags.append("ambiguous_link")
        if not all_checked:
            flags.append("registration_unchecked")
        if fit.n_epochs == 2:
            flags.append("two_epochs_only")
        if var.wavelength_mismatch:
            flags.append("wavelength_mismatch")

        m_gaia = match_to_catalog(fit.east0_arcsec, fit.north0_arcsec, max(fit.pos0_err_arcsec, cfg.pos_sys_arcsec),
                                  fit.t0_mjd, gaia, centre_ra, centre_dec, cfg, fit)
        m_cw = match_to_catalog(fit.east0_arcsec, fit.north0_arcsec, max(fit.pos0_err_arcsec, cfg.pos_sys_arcsec),
                                fit.t0_mjd, catwise, centre_ra, centre_dec, cfg, fit)
        median_snr = float(np.median([d.peak_snr for d in dets]))
        score = score_track(fit, var, m_gaia, m_cw, flags, median_snr)

        x0 = (w - 1) / 2.0 - fit.east0_arcsec / grid_scale_arcsec
        y0 = (h - 1) / 2.0 + fit.north0_arcsec / grid_scale_arcsec
        sky = wcs.pixel_to_world(x0, y0)
        results.append({
            "series": series,
            "ra_deg": float(sky.ra.deg),
            "dec_deg": float(sky.dec.deg),
            "epoch_mjd": fit.t0_mjd,
            "pos_err_arcsec": fit.pos0_err_arcsec,
            "detection_snr_median": median_snr,
            "detection_snr_max": float(max(d.peak_snr for d in dets)),
            "n_epochs": fit.n_epochs,
            "n_frames_in_series": len(frames),
            "motion": fit.__dict__,
            "parallax_fit": parallax,
            "variability": var.__dict__,
            "gaia": m_gaia.__dict__,
            "catwise": m_cw.__dict__,
            "flags": flags,
            **score,
            "detections": [
                {"frame_index": d.frame_index, "time_mjd": d.time_mjd, "x": d.x, "y": d.y,
                 # registration-corrected tangent-plane offsets from the grid centre (what the fits use)
                 "east_arcsec": d.east_arcsec, "north_arcsec": d.north_arcsec, "ra_deg": d.ra_deg,
                 "dec_deg": d.dec_deg, "pos_err_arcsec": d.pos_err_arcsec, "peak_snr": d.peak_snr,
                 "flux_mjy": d.flux_mjy, "flux_err_mjy": d.flux_err_mjy, "phot_status": d.phot_status,
                 "wavelength_um": frames[d.frame_index].wavelength_um}
                for d in dets
            ],
        })

    results.sort(key=lambda r: (-r["anomaly_score"], -r["detection_snr_median"]))
    for i, r in enumerate(results):
        r["rank"] = i + 1
    # Phase Three, Priority 2: calibrated ranking ALONGSIDE the fields above
    # (app/ranking.py). Adds calibration / calibrated_score / calibrated_rank;
    # changes nothing that already exists.
    from .ranking import calibrate_series
    calibrate_series(results)
    return {
        "series": series,
        "n_frames": len(frames),
        "frame_times_mjd": [f.time_mjd for f in frames],
        "n_detections_per_frame": [len(p) for p in per_frame],
        "registration": registration,
        "catalogs": {"gaia_dr3": None if gaia is None else len(gaia), "catwise2020": None if catwise is None else len(catwise)},
        "config": cfg.__dict__,
        "tracks": results,
    }
