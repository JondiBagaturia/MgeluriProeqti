#!/usr/bin/env python3
"""Validation of the five NASA targets on the cached REAL SPHEREx cutouts
(backend/tests/data/real, see scripts/cache_validation_cutouts.py).

Runs the production pipeline offline and writes docs/data/validation_report.json.
Catalogue queries (Gaia DR3, CatWISE2020) are live unless --no-catalogs.

Sections of the report:
  targets.<name>.frames         per detector-epoch frame: dates, wavelength, target aperture photometry
  targets.<name>.target_track   the tracked source at the target's position: measured vs catalogue motion
  field_stars                   flux and position repeatability of ordinary field stars (the empirical
                                systematic floors)
  wise0855_shift_stack          blind search, forced stack and controls
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
from astropy.time import Time
from astropy.wcs import WCS

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
warnings.filterwarnings("ignore")

from app.jsonsafe import clean  # noqa: E402
from app.offline import run_cached  # noqa: E402
from app.shiftstack import ShiftStackConfig, StackFrame, forced, search  # noqa: E402

TARGETS = ["barnard", "eps_ind_b", "wise0855", "rw_aur", "r_dra"]


def grid_wcs(meta: dict) -> WCS:
    g = meta["grid_wcs"]
    w = WCS(naxis=2)
    w.wcs.ctype = g["ctype"]; w.wcs.crval = g["crval"]; w.wcs.crpix = g["crpix"]; w.wcs.cdelt = g["cdelt"]
    return w


def sep_arcsec(ra1, dec1, ra2, dec2):
    return math.hypot((ra1 - ra2) * math.cos(math.radians(dec1)), dec1 - dec2) * 3600


def target_track(run, detector):
    """Track nearest to the target's predicted position at the track's own epoch."""
    if not run.analysis:
        return None
    series = next((s for s in run.analysis.get("series", []) if s["series"] == detector), None)
    if not series or not series["tracks"]:
        return None
    best = None
    for t in series["tracks"]:
        c = run.resolved.propagate_to(Time(t["epoch_mjd"], format="mjd"))
        d = sep_arcsec(t["ra_deg"], t["dec_deg"], c.ra.deg, c.dec.deg)
        if best is None or d < best[0]:
            best = (d, t)
    if best[0] > 12.0:
        return {"found": False, "nearest_track_arcsec": best[0]}
    t = best[1]
    m, v = t["motion"], t["variability"]
    return {
        "found": True, "offset_from_prediction_arcsec": best[0], "rank": t["rank"], "label": t["label"],
        "reliability": t["reliability"], "n_epochs": t["n_epochs"], "detection_snr_median": t["detection_snr_median"],
        "pm_east_mas_yr": m["pmra_mas_yr"], "pm_north_mas_yr": m["pmdec_mas_yr"], "pm_err_mas_yr": m["pmra_err_mas_yr"],
        "pm_significance": m["pm_significance"], "baseline_yr": m["baseline_yr"],
        "catalogue_pm_east_mas_yr": run.resolved.pmra, "catalogue_pm_north_mas_yr": run.resolved.pmdec,
        "variability_p_value": v["p_value"], "variability_amplitude_fraction": v["amplitude_fraction"],
        "wavelength_span_um": v["wavelength_span_um"], "gaia": t["gaia"]["status"], "catwise": t["catwise"]["status"],
        "flags": t["flags"], "reasons": t["reasons"], "caveats": t["caveats"],
    }


def field_star_stats(runs):
    """Repeatability of ordinary stars: everything tracked in >=3 epochs with
    clean photometry, excluding the targets themselves."""
    rows = []
    for name, run in runs.items():
        if not run.analysis:
            continue
        for s in run.analysis.get("series", []):
            for t in s["tracks"]:
                d = t["detections"]
                if t["n_epochs"] < 3 or any(x["phot_status"] != "ok" for x in d) or "core_masked" in t["flags"] or "near_edge" in t["flags"]:
                    continue
                c = run.resolved.propagate_to(Time(t["epoch_mjd"], format="mjd"))
                if sep_arcsec(t["ra_deg"], t["dec_deg"], c.ra.deg, c.dec.deg) < 15:
                    continue
                f = np.array([x["flux_mjy"] for x in d]); e = np.array([x["flux_err_mjy"] for x in d])
                if f.mean() <= 0:
                    continue
                ra = np.array([x["ra_deg"] for x in d]); dec = np.array([x["dec_deg"] for x in d])
                east = (ra - ra.mean()) * math.cos(math.radians(dec.mean())) * 3600
                north = (dec - dec.mean()) * 3600
                waves = [x["wavelength_um"] for x in d if x["wavelength_um"] is not None]
                rows.append({
                    "field": name, "series": s["series"], "snr": t["detection_snr_median"], "mean_flux_mjy": float(f.mean()),
                    "flux_scatter_fraction": float(f.std(ddof=1) / f.mean()),
                    "stat_error_fraction": float(np.median(e) / f.mean()),
                    "pos_rms_arcsec": float(math.sqrt((east.var(ddof=1) + north.var(ddof=1)) / 2)),
                    "wavelength_span_um": float(max(waves) - min(waves)) if len(waves) > 1 else None,
                    "gaia": t["gaia"]["status"],
                    "gaia_pm_mas_yr": (math.hypot(t["gaia"]["source"]["pmra_mas_yr"] or 0, t["gaia"]["source"]["pmdec_mas_yr"] or 0)
                                       if t["gaia"].get("source") else None),
                })
    out = {"n_stars": len(rows), "stars": rows, "summary": {}}
    for det in sorted({r["series"] for r in rows}):
        for label, sel in (("snr>=30", lambda r: r["snr"] >= 30), ("10<=snr<30", lambda r: 10 <= r["snr"] < 30)):
            sub = [r for r in rows if r["series"] == det and sel(r)]
            if not sub:
                continue
            fs = np.array([r["flux_scatter_fraction"] for r in sub]); ps = np.array([r["pos_rms_arcsec"] for r in sub])
            out["summary"][f"{det} {label}"] = {
                "n": len(sub), "flux_scatter_median": float(np.median(fs)), "flux_scatter_p84": float(np.percentile(fs, 84)),
                "stat_error_median": float(np.median([r["stat_error_fraction"] for r in sub])),
                "pos_rms_median_arcsec": float(np.median(ps)), "pos_rms_p84_arcsec": float(np.percentile(ps, 84)),
            }
    return out


def wise0855_shift_stack():
    out = {}
    cfg = ShiftStackConfig(n_controls=30)
    for dets in (["D5"], ["D6"], ["D5", "D6"]):
        run = run_cached("wise0855", mode="single", detectors=dets, analyze=False)
        exps = sorted(run.exposures, key=lambda p: p.record.time_mjd)
        frames = [StackFrame(image=p.image, variance=p.variance, time_mjd=p.record.time_mjd) for p in exps]
        meta = run.frames[0]
        scale = meta["grid_scale_arcsec"]
        w = grid_wcs(meta)
        t_ref = float(np.mean([f.time_mjd for f in frames]))
        c = run.resolved.propagate_to(Time(t_ref, format="mjd"))
        x, y = (float(v) for v in w.world_to_pixel(c))
        ve, vn = run.resolved.pmra / 1000.0, run.resolved.pmdec / 1000.0

        res_blind = search(frames, scale, cfg)
        for cand in res_blind["candidates"]:
            cand["offset_from_catalogue_prediction_arcsec"] = math.hypot(cand["x"] - x, cand["y"] - y) * scale
        res_forced = forced(frames, scale, x, y, ve, vn, t_ref=t_ref, cfg=cfg)

        # Where the object is NOT: its catalogue positions 5 and 11 years
        # earlier, stacked with the same motion (tests for a static source
        # sitting on the track, and for the motion model producing signal by itself).
        elsewhere = {}
        for years in (-11.0, -5.0, 3.0):
            cc = run.resolved.propagate_to(Time(t_ref + years * 365.25, format="mjd"))
            ex, ey = (float(v) for v in w.world_to_pixel(cc))
            if 8 < ex < frames[0].image.shape[1] - 8 and 8 < ey < frames[0].image.shape[0] - 8:
                elsewhere[f"{years:+.0f} yr"] = forced(frames, scale, ex, ey, ve, vn, t_ref=t_ref, cfg=cfg, n_control_positions=10)["snr_on_trajectory"]
            else:
                elsewhere[f"{years:+.0f} yr"] = None  # outside the cached 5' field

        # single-frame view for comparison
        per_frame = [forced([f], scale, x - ve * (f.time_mjd - t_ref) / 365.25 / scale, y + vn * (f.time_mjd - t_ref) / 365.25 / scale,
                            0.0, 0.0, cfg=cfg, n_control_positions=5)["snr_on_trajectory"] for f in frames]
        out["+".join(dets)] = {
            "n_exposures": len(frames), "predicted_xy": [x, y], "catalogue_pm_arcsec_yr": [ve, vn],
            "blind": res_blind, "forced": res_forced, "snr_at_catalogue_position_other_epochs": elsewhere,
            "single_exposure_snr_at_prediction": {"median": float(np.median(per_frame)), "max": float(np.max(per_frame))},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-catalogs", action="store_true")
    ap.add_argument("--out", default=str(REPO / "docs" / "data" / "validation_report.json"))
    args = ap.parse_args()

    report = {"note": "Computed from cached REAL SPHEREx cutouts with the production pipeline.", "targets": {}}
    runs = {}
    for name in TARGETS:
        run = run_cached(name, use_catalogs=not args.no_catalogs)
        runs[name] = run
        dets = run.manifest["detectors"]
        entry = {"query": run.manifest["query"], "retrieved_utc": run.manifest["retrieved_utc"], "n_exposures": len(run.exposures),
                 "catalog_errors": (run.analysis or {}).get("catalog_errors"), "frames": [], "target_track": {}}
        for m in run.frames:
            ph = m["photometry"]
            entry["frames"].append({"detector": m["detectors"][0], "mjd_min": m["time_mjd_min"], "mjd_max": m["time_mjd_max"],
                                    "n_exposures": m["n_exposures"], "wavelength_um": m["central_wavelength_um"],
                                    "status": ph["status"], "flux_mjy": ph["flux_mjy"], "flux_err_mjy": ph["flux_err_mjy"],
                                    "snr": ph["snr"], "n_invalid_in_aperture": ph["n_invalid_in_aperture"]})
        for d in dets:
            entry["target_track"][d] = target_track(run, d)
        report["targets"][name] = entry
        print(f"\n== {name} ({entry['n_exposures']} exposures)")
        for f in entry["frames"]:
            fl = "refused: " + f["status"] if f["flux_mjy"] is None else f"{f['flux_mjy']:.3g} +- {f['flux_err_mjy']:.2g} mJy"
            print(f"   {f['detector']} MJD {f['mjd_min']:.0f}-{f['mjd_max']:.0f} lam {f['wavelength_um'] or float('nan'):.3f}: {fl}")
        for d, t in entry["target_track"].items():
            if t and t.get("found"):
                pm = "no motion fit" if t["pm_east_mas_yr"] is None else f"PM ({t['pm_east_mas_yr']:.0f}, {t['pm_north_mas_yr']:.0f}) +- {t['pm_err_mas_yr']:.0f} mas/yr [{t['pm_significance']:.1f} sigma]"
                print(f"   track {d}: rank {t['rank']} {t['label']} epochs {t['n_epochs']} {pm}; catalogue ({t['catalogue_pm_east_mas_yr']:.0f}, {t['catalogue_pm_north_mas_yr']:.0f}); var p={t['variability_p_value']}; gaia {t['gaia']} catwise {t['catwise']}")
            else:
                print(f"   track {d}: not found by blind detection ({t})")

    report["field_stars"] = field_star_stats(runs)
    print("\n== field stars (repeatability)")
    for k, v in report["field_stars"]["summary"].items():
        print(f"   {k}: n={v['n']} flux scatter median {100 * v['flux_scatter_median']:.1f}% (p84 {100 * v['flux_scatter_p84']:.1f}%), stat {100 * v['stat_error_median']:.2f}%; position rms median {v['pos_rms_median_arcsec']:.2f}\" (p84 {v['pos_rms_p84_arcsec']:.2f}\")")

    report["wise0855_shift_stack"] = wise0855_shift_stack()
    print("\n== WISE 0855 shift-and-stack")
    for k, v in report["wise0855_shift_stack"].items():
        b, f = v["blind"], v["forced"]
        print(f"   {k}: {v['n_exposures']} exposures; single-exposure S/N at prediction median {v['single_exposure_snr_at_prediction']['median']:.1f} max {v['single_exposure_snr_at_prediction']['max']:.1f}")
        print(f"      forced: S/N {f['snr_on_trajectory']:.1f}; no motion {f['snr_same_position_no_motion']:.1f}; reversed {f['snr_same_position_reversed_motion']:.1f}; off-position p95 {f['off_position']['p95']:.1f} max {f['off_position']['max']:.1f} ({f['off_position']['n']} positions); other epochs {v['snr_at_catalogue_position_other_epochs']}")
        print(f"      blind: {b['n_velocities']} velocities step {b['velocity_step_arcsec_yr']:.1f}\"/yr, {b['seconds']['search']:.1f}s (+controls {b['seconds']['total']:.0f}s); false-alarm p {b['controls']['false_alarm_probability']}")
        for c in [c for c in b["candidates"] if c["moving"]] + [c for c in b["candidates"] if not c["moving"]][:2]:
            print(f"         S/N {c['snr']:.1f} static {c['snr_static']:.1f} v=({c['pm_east_arcsec_yr']:.1f},{c['pm_north_arcsec_yr']:.1f}) moving={c['moving']} offset from prediction {c['offset_from_catalogue_prediction_arcsec']:.1f}\"")


    Path(args.out).write_text(json.dumps(clean(report), indent=1))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
