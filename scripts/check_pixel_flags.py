#!/usr/bin/env python3
"""Diagnostic: fetch ONE cutout for a position and print the IMAGE value
and FLAGS bit names at the exact target pixel.

Written to explain an odd-looking result in the smoke test: R Draconis's
blink GIF shows a dark hole surrounded by a bright ring. Running this
confirmed why - the core pixel's `IMAGE` value is already `NaN` in the raw
file, with `FLAGS` bits `OVERFLOW` and `BLOOM` set: R Draconis saturates
and blooms SPHEREx's D6 detector, and SPHEREx's own Level-2 pipeline has
already blanked that pixel out. See README.md's acceptance-target table.

Deliberately fetches only one cutout (not a whole position's worth) - this
is for a quick, polite look at one file, not a replacement for the real
pipeline in cli/spherex_blink.py or scripts/smoke_test.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from astropy.wcs import WCS  # noqa: E402

from app.coords import parse_position  # noqa: E402
from app.irsa import find_images, fetch_cutouts  # noqa: E402
from app.pipeline import read_flag_names  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <position or object name> [fov_arcmin] [detector]", file=sys.stderr)
        return 1
    position = sys.argv[1]
    fov_arcmin = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    detector = sys.argv[3] if len(sys.argv) > 3 else "D6"

    resolved = parse_position(position)
    records = find_images(resolved.coord, fov_arcmin=fov_arcmin, detectors=[detector])
    print(f"Found {len(records)} images; checking only the first one.")
    if not records:
        return 0

    for cutout in fetch_cutouts(records[:1], resolved.coord, fov_arcmin=fov_arcmin):
        hdul = cutout.hdul
        wcs = WCS(hdul["IMAGE"].header)
        x, y = wcs.world_to_pixel(resolved.coord)
        xi, yi = int(round(float(x))), int(round(float(y)))
        image = hdul["IMAGE"].data
        flags = hdul["FLAGS"].data
        flag_names = read_flag_names(hdul["FLAGS"].header)

        print(f"{cutout.record.obs_id}: target pixel ({xi},{yi}) of shape {image.shape}")
        print(f"  IMAGE value: {image[yi, xi]}")
        flag_val = int(flags[yi, xi])
        set_bits = [b for b in range(32) if flag_val & (1 << b)]
        print(f"  FLAGS value: {flag_val}, bit names: {[flag_names.get(b, f'bit{b}') for b in set_bits]}")
        print("  3x3 IMAGE neighbourhood:")
        print(image[max(0, yi - 1): yi + 2, max(0, xi - 1): xi + 2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
