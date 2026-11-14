#!/usr/bin/env python3
"""Audit: for a position, fetch every matching exposure and report how
many have valid (unmasked) flux at the exact target pixel - and, for the
ones that don't, why: SPHEREx's own raw IMAGE value already NaN there
(e.g. genuine saturation), our own flag mask tripping it, or neither
(a reprojection/interpolation edge effect touching a masked neighbour).

This is the script that originally found the "Barnard's Star shows no
data at its own centre" bug (0 of 361 exposures, all 6 detectors, before
the flag-mask fix in app/pipeline.py's DEFAULT_BAD_FLAG_NAMES) and the
one to rerun any time you want to check whether that's still true, for
this target or any other.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import numpy as np
from astropy.time import Time
from astropy.wcs import WCS

from app.coords import parse_position  # noqa: E402
from app.irsa import find_images, fetch_cutouts  # noqa: E402
from app.pipeline import (  # noqa: E402
    make_target_wcs,
    process_cutout,
    read_flag_names,
    default_bad_bits,
)
from app.config import get_config  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <position or object name> [fov_arcmin] [detector ...]", file=sys.stderr)
        print(f"example: {sys.argv[0]} \"Barnard's star\" 3 D1 D2 D3 D4 D5 D6", file=sys.stderr)
        return 1

    position = sys.argv[1]
    fov_arcmin = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    detectors = sys.argv[3:] or None  # None = all 6

    resolved = parse_position(position)
    records = find_images(resolved.coord, fov_arcmin=fov_arcmin, detectors=detectors)
    print(f"{position!r}: found {len(records)} image(s) across detectors "
          f"{sorted({r.detector for r in records})}")

    if resolved.has_proper_motion:
        mean_mjd = float(np.mean([r.time_mjd for r in records]))
        centred_coord = resolved.propagate_to(Time(mean_mjd, format="mjd"))
    else:
        centred_coord = resolved.coord

    cfg = get_config()
    target_wcs, shape_out = make_target_wcs(centred_coord, fov_arcmin, cfg.pixel_scale_arcsec)
    cy, cx = shape_out[0] // 2, shape_out[1] // 2

    n_processed = 0
    n_valid = 0
    reason_counts: Counter = Counter()
    culprit_flags: Counter = Counter()
    by_detector_valid: Counter = Counter()
    by_detector_total: Counter = Counter()

    for cutout in fetch_cutouts(records, centred_coord, fov_arcmin=fov_arcmin):
        n_processed += 1
        det = cutout.record.detector
        by_detector_total[det] += 1

        p = process_cutout(cutout, centred_coord, target_wcs, shape_out)
        valid = bool(np.isfinite(p.image[cy, cx]))
        if valid:
            n_valid += 1
            by_detector_valid[det] += 1
            continue

        # Figure out why: check the RAW native pixel this exposure's own
        # WCS maps the target onto.
        hdul = cutout.hdul
        in_wcs = WCS(hdul["IMAGE"].header)
        x, y = in_wcs.world_to_pixel(centred_coord)
        xi, yi = int(round(float(x))), int(round(float(y)))
        h, w = hdul["IMAGE"].data.shape
        if not (0 <= xi < w and 0 <= yi < h):
            reason_counts["off_footprint"] += 1
            continue

        raw_val = hdul["IMAGE"].data[yi, xi]
        if not np.isfinite(raw_val):
            reason_counts["sphere_x_raw_nan (genuine saturation/defect)"] += 1
            continue

        flag_names = read_flag_names(hdul["FLAGS"].header) if "FLAGS" in hdul else {}
        bad_bits = default_bad_bits(flag_names)
        flag_val = int(hdul["FLAGS"].data[yi, xi]) if "FLAGS" in hdul else 0
        set_bad_bits = [b for b in bad_bits if flag_val & (1 << b)]
        if set_bad_bits:
            reason_counts["our_mask_tripped"] += 1
            for b in set_bad_bits:
                culprit_flags[flag_names.get(b, f"bit{b}")] += 1
        else:
            reason_counts["reprojection_edge_effect (native pixel was fine)"] += 1

    n_dropped = len(records) - n_processed
    pct = (n_valid / n_processed * 100) if n_processed else 0.0
    print(f"processed {n_processed} exposure(s) ({n_dropped} dropped by fetch failures)")
    print(f"valid (unmasked) flux at the exact target pixel: {n_valid}/{n_processed} ({pct:.1f}%)")

    print("\nby detector:")
    for det in sorted(by_detector_total):
        v, t = by_detector_valid[det], by_detector_total[det]
        print(f"  {det}: {v}/{t} ({v / t * 100:.1f}%)")

    if reason_counts:
        print("\nwhy the invalid ones are invalid:")
        for reason, count in reason_counts.most_common():
            print(f"  {reason}: {count}")
    if culprit_flags:
        print("\nflag bits responsible when our mask tripped:")
        for name, count in culprit_flags.most_common():
            print(f"  {name}: {count}")

    if n_valid == 0 and n_processed > 0:
        print("\n-> 100% missing: every single exposure masks its own centre pixel.")
    elif pct < 100:
        print(f"\n-> {100 - pct:.1f}% still missing per-exposure - see breakdown above. "
              "Combining across exposures still recovers a value as long as >0% are valid "
              "(inverse-variance combine only needs one contributor).")
    else:
        print("\n-> every exposure has valid flux at the centre.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
