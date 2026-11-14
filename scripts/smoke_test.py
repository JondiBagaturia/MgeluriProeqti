#!/usr/bin/env python3
"""Acceptance smoke test (see SPHEREx_tool_prompt.md's "Test targets"
section): run the full pipeline on all five named targets, save a blink
GIF and a difference image for each into examples/, and report timing.

Honesty note on timing: this hits live IRSA/SIMBAD for every run and the
backend intentionally has no disk cache (PLAN.md's "stateless processing
pipe" requirement), so there's no meaningful server-side cold/warm
distinction to measure here beyond what PLAN.md's Phase 0 numbers already
report. The *browser's* warm-cache path (IndexedDB) is a frontend concern,
validated separately with mocked network responses rather than by hitting
IRSA again for the same position twice in this script.

WISE 0855-0714 is resolved by its catalogue name (not the raw sexagesimal
string from the brief) so that proper-motion propagation actually runs -
"check its epoch and propagate" only means something if a name resolver
supplies the proper motion. Parsing that exact sexagesimal string is
covered separately by tests/test_coords.py::test_sexagesimal_wise0855.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import numpy as np
from astropy.time import Time
from PIL import Image

from app.coords import parse_position, CoordParseError  # noqa: E402
from app.irsa import find_images, fetch_cutouts, DiscoveryError  # noqa: E402
from app.pipeline import make_target_wcs, process_cutout, build_frames  # noqa: E402
from app.config import get_config  # noqa: E402

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

# (label, query, fov_arcmin, detectors) - single detector each, chosen for
# speed/politeness to IRSA; D6 (longest wavelength) throughout both matches
# the brief's hint for WISE 0855 and keeps every target directly comparable.
TARGETS = [
    ("Barnards_Star", "Barnard's star", 3.0, ["D6"]),
    ("Epsilon_Indi_B", "eps Ind B", 3.0, ["D6"]),
    ("WISE_0855-0714", "WISE J085510.83-071442.5", 3.0, ["D6"]),
    ("RW_Aurigae", "RW Aurigae", 3.0, ["D6"]),
    ("R_Draconis", "R Draconis", 3.0, ["D6"]),
]


def stretch_to_uint8(arr, pct=(1, 99), diverging=False):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    if diverging:
        vmax = np.percentile(np.abs(finite), pct[1]) or 1e-6
        normed = np.clip((arr + vmax) / (2 * vmax), 0, 1)
        fill = 0.5
    else:
        vmin, vmax = np.percentile(finite, pct)
        if vmax <= vmin:
            vmax = vmin + 1e-6
        normed = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
        fill = 0.0
    normed = np.where(np.isfinite(arr), normed, fill)
    return (normed * 255).astype(np.uint8)


def to_gif_frame(arr, upscale=8):
    img8 = stretch_to_uint8(arr)
    im = Image.fromarray(img8, mode="L").resize((img8.shape[1] * upscale, img8.shape[0] * upscale), Image.NEAREST)
    return im.convert("P")


def run_target(label, query, fov, detectors):
    report = {"label": label, "query": query, "fov_arcmin": fov, "detectors": detectors}
    cfg = get_config()
    t_start = time.time()

    try:
        resolved = parse_position(query)
    except CoordParseError as exc:
        report["error"] = f"could not resolve: {exc}"
        return report

    report["resolved_ra_deg"] = resolved.coord.ra.deg
    report["resolved_dec_deg"] = resolved.coord.dec.deg
    report["resolution_source"] = resolved.source
    report["has_proper_motion"] = resolved.has_proper_motion
    if resolved.has_proper_motion:
        report["pmra_mas_yr"] = resolved.pmra
        report["pmdec_mas_yr"] = resolved.pmdec

    t0 = time.time()
    try:
        records = find_images(resolved.coord, fov_arcmin=fov, detectors=detectors)
    except DiscoveryError as exc:
        report["error"] = f"discovery failed: {exc}"
        return report
    report["discover_s"] = round(time.time() - t0, 1)
    report["n_images_found"] = len(records)
    if not records:
        report["error"] = "no images found at this position/FOV/detector"
        return report

    if resolved.has_proper_motion:
        mean_mjd = float(np.mean([r.time_mjd for r in records]))
        centred_coord = resolved.propagate_to(Time(mean_mjd, format="mjd"))
    else:
        centred_coord = resolved.coord
    report["centred_ra_deg"] = centred_coord.ra.deg
    report["centred_dec_deg"] = centred_coord.dec.deg

    target_wcs, shape_out = make_target_wcs(centred_coord, fov, cfg.pixel_scale_arcsec)

    processed = []
    t_fetch_start = time.time()
    t_first_cutout_s = None
    for cutout in fetch_cutouts(records, centred_coord, fov_arcmin=fov):
        p = process_cutout(cutout, centred_coord, target_wcs, shape_out)
        processed.append(p)
        if t_first_cutout_s is None:
            t_first_cutout_s = round(time.time() - t_fetch_start, 1)
    report["n_images_processed"] = len(processed)
    report["time_to_first_cutout_s"] = t_first_cutout_s
    report["fetch_total_s"] = round(time.time() - t_fetch_start, 1)

    if not processed:
        report["error"] = "all cutout fetches failed after retries"
        return report

    processed.sort(key=lambda p: p.record.time_mjd)
    frames = build_frames(processed, mode="epoch")
    report["n_frames"] = len(frames)
    report["total_s"] = round(time.time() - t_start, 1)
    report["frame_dates_mjd"] = [round(f.time_mjd_mean, 1) for f in frames]

    EXAMPLES_DIR.mkdir(exist_ok=True)
    gif_frames = [to_gif_frame(f.image) for f in frames]
    if gif_frames:
        gif_path = EXAMPLES_DIR / f"{label}_blink.gif"
        gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=700, loop=0)
        report["gif"] = f"examples/{gif_path.name}"

    if len(frames) >= 2:
        diff = frames[-1].image - frames[0].image
        finite = diff[np.isfinite(diff)]
        report["difference_peak_abs"] = float(np.nanmax(np.abs(diff))) if finite.size else None
        img8 = stretch_to_uint8(diff, diverging=True)
        im = Image.fromarray(img8, mode="L").resize((img8.shape[1] * 8, img8.shape[0] * 8), Image.NEAREST)
        diff_path = EXAMPLES_DIR / f"{label}_difference.png"
        im.save(diff_path)
        report["difference_image"] = f"examples/{diff_path.name}"
        report["difference_epochs_mjd"] = [round(frames[0].time_mjd_mean, 1), round(frames[-1].time_mjd_mean, 1)]
    else:
        report["note"] = "only one epoch found - no difference image possible"

    return report


def main():
    reports = []
    for label, query, fov, detectors in TARGETS:
        print(f"=== {label} ({query!r}) ===", flush=True)
        t0 = time.time()
        report = run_target(label, query, fov, detectors)
        report["wall_clock_s"] = round(time.time() - t0, 1)
        reports.append(report)
        print(json.dumps(report, indent=2, default=str), flush=True)
        print(flush=True)

    EXAMPLES_DIR.mkdir(exist_ok=True)
    with open(EXAMPLES_DIR / "smoke_test_report.json", "w") as f:
        json.dump(reports, f, indent=2, default=str)
    print(f"Wrote {EXAMPLES_DIR / 'smoke_test_report.json'}")


if __name__ == "__main__":
    main()
