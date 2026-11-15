#!/usr/bin/env python3
"""Download and cache REAL SPHEREx cutouts for the five NASA validation
targets, so the regression tests can run the whole pipeline offline on real
observations.

What is stored (backend/tests/data/real/<target>/):
  * one gzip-compressed FITS file per exposure, holding exactly the HDUs the
    pipeline uses (IMAGE, FLAGS, VARIANCE, ZODI, WCS-WAVE; the 3.9 MB EPSF
    table is dropped, as the live pipeline does). Pixel values are untouched.
  * manifest.json: the query, the resolved astrometry (as returned by
    SIMBAD on the day of caching), the list of exposures with their IRSA
    metadata, and the date of retrieval.

These files are real observations, cut out by IRSA's cutout service. They
are NOT synthetic. Re-run this script to refresh them (e.g. after a new
data release); the regression tests' expected values may then need review.

Usage:  python scripts/cache_validation_cutouts.py [--targets barnard wise0855 ...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.time import Time  # noqa: E402

from app.coords import parse_position  # noqa: E402
from app.irsa import find_images, fetch_cutouts  # noqa: E402
from app.pipeline import group_indices, cap_group_indices  # noqa: E402

OUT = REPO / "backend" / "tests" / "data" / "real"

TARGETS = {
    "barnard": {"query": "Barnard's star", "detectors": ["D6", "D3"]},
    "eps_ind_b": {"query": "eps Ind B", "detectors": ["D6", "D3"]},
    "wise0855": {"query": "WISE J085510.83-071442.5", "detectors": ["D6", "D5"]},
    "rw_aur": {"query": "RW Aur", "detectors": ["D6", "D3"]},
    "r_dra": {"query": "R Dra", "detectors": ["D6", "D3"]},
}
FOV_ARCMIN = 5.0
MAX_PER_EPOCH = 6


def cache_target(slug: str, spec: dict) -> None:
    out = OUT / slug
    out.mkdir(parents=True, exist_ok=True)
    resolved = parse_position(spec["query"])
    records = find_images(resolved.search_position(), fov_arcmin=FOV_ARCMIN, detectors=spec["detectors"])
    if not records:
        print(f"[{slug}] no images found")
        return
    mean_mjd = float(np.mean([r.time_mjd for r in records]))
    centre = resolved.propagate_to(Time(mean_mjd, format="mjd")) if resolved.has_proper_motion else resolved.coord

    groups = [cap_group_indices(g, MAX_PER_EPOCH) for g in group_indices(records, mode="detector_epoch")]
    wanted = [records[i] for g in groups for i in g]
    print(f"[{slug}] {len(records)} images found, caching {len(wanted)} in {len(groups)} detector-epoch groups")

    entries = []
    for cutout in fetch_cutouts(wanted, centre, fov_arcmin=FOV_ARCMIN):
        r = cutout.record
        name = f"{r.detector}_{r.obs_id}.fits.gz"
        # The fetched primary HDU still points at the (closed) download
        # buffer; rebuild it from its header so the list can be written.
        hdul = fits.HDUList([fits.PrimaryHDU(header=cutout.hdul[0].header)] + [h.copy() for h in cutout.hdul[1:]])
        hdul.writeto(out / name, overwrite=True)
        entries.append({"file": name, "obs_id": r.obs_id, "detector": r.detector, "collection": r.collection,
                        "time_mjd": r.time_mjd})
        print(f"[{slug}]   {name}  ({len(entries)}/{len(wanted)})")
    entries.sort(key=lambda e: e["time_mjd"])

    manifest = {
        "note": "REAL SPHEREx observations (IRSA cutout service). Not synthetic.",
        "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "query": spec["query"],
        "fov_arcmin": FOV_ARCMIN,
        "detectors": spec["detectors"],
        "max_exposures_per_detector_epoch": MAX_PER_EPOCH,
        "n_images_available": len(records),
        "n_requested": len(wanted),
        "n_cached": len(entries),
        "cutout_centre_deg": [float(centre.ra.deg), float(centre.dec.deg)],
        "resolved": {
            "source": resolved.source, "name": resolved.name,
            "ra_deg": float(resolved.coord.ra.deg), "dec_deg": float(resolved.coord.dec.deg),
            "ref_epoch_mjd": float(resolved.ref_epoch.mjd),
            "pmra_mas_yr": resolved.pmra, "pmdec_mas_yr": resolved.pmdec, "parallax_mas": resolved.parallax_mas,
        },
        "exposures": entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[{slug}] done: {len(entries)}/{len(wanted)} cached")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=list(TARGETS))
    args = ap.parse_args()
    for slug in args.targets:
        try:
            cache_target(slug, TARGETS[slug])
        except Exception as exc:  # noqa: BLE001 - keep going with the other targets
            print(f"[{slug}] FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
