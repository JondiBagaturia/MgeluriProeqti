"""Per-pixel saturation mask sent with every frame (saturation_mask_b64),
used by the false-colour composite to refuse colours built from saturated
or non-linear data."""
import base64

import numpy as np
import pytest

from app.offline import run_cached, series_frames
from app.pipeline import SATURATION_FLAG_NAMES, saturation_on_grid


def _shape(meta):
    # the output grid is centred on the target: crpix = (n + 1) / 2 (FITS, 1-based)
    cx, cy = meta["grid_wcs"]["crpix"]
    return int(round(2 * cy - 1)), int(round(2 * cx - 1))


def _decode(meta):
    h, w = _shape(meta)
    bits = np.unpackbits(np.frombuffer(base64.b64decode(meta["saturation_mask_b64"]), np.uint8), bitorder="little")
    assert bits.size == 8 * ((h * w + 7) // 8)
    return bits[:h * w].astype(bool).reshape(h, w)


def test_saturation_on_grid_marks_the_2x2_footprint_only():
    names = {3: "OVERFLOW", 5: "HOT", 7: "BLOOM"}
    flags = np.zeros((10, 10), np.int32)
    flags[4, 6] = 1 << 3            # OVERFLOW
    flags[8, 1] = 1 << 5            # HOT: not a saturation flag
    # output pixel k sits at input (x, y) = (k + 0.5, 4.5): footprint x in {k, k+1}, y in {4, 5}
    xs = np.arange(10) + 0.5
    ys = np.full(10, 4.5)
    out = saturation_on_grid(flags, names, xs, ys, (1, 10))[0]
    assert out.tolist() == [k in (5, 6) for k in range(10)]
    assert "HOT" not in SATURATION_FLAG_NAMES


def test_no_saturation_bits_gives_empty_mask():
    flags = np.full((5, 5), 1 << 2, np.int32)
    # SUR_ERROR is a ramp-fit flag, mostly on empty sky: not saturation
    out = saturation_on_grid(flags, {2: "SUR_ERROR"}, np.arange(5.0), np.arange(5.0), (1, 5))
    assert not out.any()


@pytest.fixture(scope="module")
def barnard():
    return run_cached("barnard", detectors=["D3", "D6"], analyze=False)


def test_every_frame_carries_a_mask_of_the_grid_size(barnard):
    assert barnard.frames
    for m in barnard.frames:
        mask = _decode(m)
        assert mask.shape == _shape(m)
        assert int(mask.sum()) == m["n_saturation_flagged_pixels"]


def test_barnard_d3_core_is_saturation_flagged(barnard):
    """Barnard's Star saturates D3: the pixels at its predicted position
    must be in the mask in every D3 epoch (they still carry values, so
    without the mask the composite would colour them)."""
    for m in series_frames(barnard, "D3"):
        mask = _decode(m)
        x, y = int(round(m["target_pixel"]["x"])), int(round(m["target_pixel"]["y"]))
        assert mask[y - 1:y + 2, x - 1:x + 2].any(), m["time_mjd_mean"]
        assert m["n_saturation_flagged_pixels"] < 0.5 * mask.size
