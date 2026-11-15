#!/usr/bin/env python3
"""Phase 1 CLI: given RA/Dec/FOV (or an object name), list matching SPHEREx
images, align and group them, and write frames + a metadata JSON to a local
directory.

This script is for testing only. The FastAPI server (Phase 2) runs the same
pipeline entirely in memory and writes nothing to disk.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from astropy.time import Time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.coords import parse_position, CoordParseError  # noqa: E402
from app.irsa import find_images, fetch_cutouts, DiscoveryError  # noqa: E402
from app.pipeline import make_target_wcs, process_cutout, build_frames  # noqa: E402
from app.config import get_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("position", help="RA/Dec (sexagesimal or decimal) or object name")
    parser.add_argument("--fov", type=float, default=None, help="Field of view, arcmin (default from config)")
    parser.add_argument("--detectors", nargs="+", default=None, help="Detector subset, e.g. D6 D2 (default from config)")
    parser.add_argument("--mode", choices=["epoch", "detector_epoch", "single"], default="detector_epoch")
    parser.add_argument("-o", "--outdir", default="./out", help="Output directory")
    args = parser.parse_args()

    cfg = get_config()
    fov = args.fov or cfg.fov.default_arcmin
    detectors = args.detectors or list(cfg.detectors_default)

    try:
        resolved = parse_position(args.position)
    except CoordParseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    name_part = f", name={resolved.name}" if resolved.name else ""
    print(
        f"Resolved {args.position!r} -> RA={resolved.coord.ra.deg:.5f} "
        f"Dec={resolved.coord.dec.deg:.5f} (source={resolved.source}{name_part})"
    )
    if resolved.has_proper_motion:
        print(
            f"  proper motion: pmra={resolved.pmra:.2f} pmdec={resolved.pmdec:.2f} mas/yr, "
            f"plx={resolved.parallax_mas} mas"
        )

    try:
        records = find_images(resolved.coord, fov_arcmin=fov, detectors=detectors)
    except DiscoveryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not records:
        print("No SPHEREx images found at this position/FOV/detector selection.")
        return 0

    print(f"Found {len(records)} images across detectors {sorted({r.detector for r in records})}")

    if resolved.has_proper_motion:
        mean_mjd = float(np.mean([r.time_mjd for r in records]))
        target_epoch = Time(mean_mjd, format="mjd")
        centred_coord = resolved.propagate_to(target_epoch)
        print(
            f"  propagated to mean epoch MJD {mean_mjd:.1f}: "
            f"RA={centred_coord.ra.deg:.5f} Dec={centred_coord.dec.deg:.5f}"
        )
    else:
        centred_coord = resolved.coord

    target_wcs, shape_out = make_target_wcs(centred_coord, fov_arcmin=fov, pixel_scale_arcsec=cfg.pixel_scale_arcsec)

    processed = []
    n_failed = 0
    for cutout in fetch_cutouts(records, centred_coord, fov_arcmin=fov):
        p = process_cutout(cutout, centred_coord, target_wcs, shape_out)
        processed.append(p)
        print(f"  processed {cutout.record.obs_id} ({cutout.record.detector}, MJD {cutout.record.time_mjd:.1f})")
    n_failed = len(records) - len(processed)
    if n_failed:
        print(f"  ({n_failed} image(s) failed to fetch after retries and were skipped)")

    if not processed:
        print("All cutout fetches failed; nothing to write.")
        return 1

    processed.sort(key=lambda p: p.record.time_mjd)  # fetch_cutouts completes out of order
    frames = build_frames(processed, mode=args.mode)
    print(f"Built {len(frames)} frame(s) using mode={args.mode}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "query": args.position,
        "resolved_ra_deg": centred_coord.ra.deg,
        "resolved_dec_deg": centred_coord.dec.deg,
        "fov_arcmin": fov,
        "detectors": detectors,
        "mode": args.mode,
        "n_images_found": len(records),
        "n_images_processed": len(processed),
        "n_frames": len(frames),
        "frames": [],
    }
    image_stack = []
    for f in frames:
        image_stack.append(f.image)
        metadata["frames"].append(
            {
                "group_index": f.group_index,
                "time_mjd_mean": f.time_mjd_mean,
                "time_mjd_min": f.time_mjd_min,
                "time_mjd_max": f.time_mjd_max,
                "detectors": f.detectors,
                "n_exposures": f.n_exposures,
                "central_wavelength_um": f.central_wavelength_um,
                "bandwidth_um": f.bandwidth_um,
                "contamination": f.contamination,
                "release_versions": f.release_versions,
            }
        )

    np.savez_compressed(outdir / "frames.npz", *image_stack)
    with open(outdir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"Wrote {len(frames)} frame(s) to {outdir}/frames.npz and {outdir}/metadata.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
