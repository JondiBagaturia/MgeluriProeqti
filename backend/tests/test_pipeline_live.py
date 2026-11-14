"""Live, network-dependent tests for the full discovery -> fetch -> process
-> combine pipeline. Marked `live` and skipped by default (see pytest.ini)
since every one of these hits real IRSA - run with `pytest -m live`.

These are deliberately small (tight FOV, one detector) to be reasonably
quick and polite to IRSA; scripts/smoke_test.py is the thorough version
that exercises all 5 acceptance targets and saves GIFs/difference images.
"""
import numpy as np
import pytest

from app.coords import parse_position
from app.irsa import find_images, fetch_cutouts
from app.pipeline import make_target_wcs, process_cutout, build_frames
from app.config import get_config


@pytest.mark.live
def test_live_pipeline_barnards_star_produces_sensible_frames():
    cfg = get_config()
    resolved = parse_position("269.452 4.693")  # Barnard's Star, decimal (no PM lookup needed for this check)
    fov_arcmin = 3.0

    records = find_images(resolved.coord, fov_arcmin=fov_arcmin, detectors=["D6"])
    assert len(records) > 0, "expected at least one SPHEREx D6 image at this well-covered position"

    target_wcs, shape_out = make_target_wcs(resolved.coord, fov_arcmin, cfg.pixel_scale_arcsec)
    processed = [
        process_cutout(cutout, resolved.coord, target_wcs, shape_out)
        for cutout in fetch_cutouts(records, resolved.coord, fov_arcmin=fov_arcmin)
    ]
    assert len(processed) > 0, "every cutout fetch failed - likely an IRSA outage, not a code bug"

    frames = build_frames(processed, mode="epoch")
    assert len(frames) >= 1
    for frame in frames:
        assert frame.image.shape == shape_out
        finite = frame.image[np.isfinite(frame.image)]
        # Expect mostly-valid pixels for a well-covered position at a small FOV
        # (Phase 0 measured 97-100% finite coverage for this exact target).
        assert finite.size / frame.image.size > 0.5
        # Flux values should be physically reasonable (MJy/sr), not NaN-poisoned
        # garbage or absurdly large from a units/scale bug.
        assert np.all(np.abs(finite) < 1e6)
