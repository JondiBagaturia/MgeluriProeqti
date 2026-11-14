"""Encode a processed Frame into the compact wire format the browser gets.

Per the brief: send a quantised array + JSON metadata, not a rendered PNG —
contrast/stretch changes then never need a new network request, they're all
client-side math on the cached array.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .pipeline import Frame

# 65535 is reserved to mean "no data at this pixel" (masked or outside this
# frame's footprint) — real values are quantised into 0..65534.
INVALID_SENTINEL = 65535


@dataclass
class EncodedFrame:
    payload_b64: str
    width: int
    height: int
    scale: float
    offset: float
    metadata: dict


def quantize_uint16(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    """uint16 array + (scale, offset) such that
    original ≈ array * scale + offset  (NaN -> INVALID_SENTINEL).
    """
    valid = np.isfinite(image)
    if not np.any(valid):
        return np.full(image.shape, INVALID_SENTINEL, dtype=np.uint16), 1.0, 0.0

    vmin = float(np.nanmin(image))
    vmax = float(np.nanmax(image))
    if vmax <= vmin:
        vmax = vmin + 1.0
    scale = (vmax - vmin) / (INVALID_SENTINEL - 1)

    q = np.full(image.shape, INVALID_SENTINEL, dtype=np.uint16)
    q[valid] = np.clip(np.round((image[valid] - vmin) / scale), 0, INVALID_SENTINEL - 1).astype(np.uint16)
    return q, scale, vmin


def encode_frame(frame: Frame, extra_metadata: Optional[dict] = None) -> EncodedFrame:
    q, scale, offset = quantize_uint16(frame.image)
    payload = base64.b64encode(q.tobytes()).decode("ascii")
    meta = {
        "group_index": frame.group_index,
        "time_mjd_mean": frame.time_mjd_mean,
        "time_mjd_min": frame.time_mjd_min,
        "time_mjd_max": frame.time_mjd_max,
        "detectors": frame.detectors,
        "n_exposures": frame.n_exposures,
        "central_wavelength_um": frame.central_wavelength_um,
        "bandwidth_um": frame.bandwidth_um,
        "contamination": frame.contamination,
        "release_versions": frame.release_versions,
        "invalid_sentinel": INVALID_SENTINEL,
    }
    if extra_metadata:
        meta.update(extra_metadata)
    return EncodedFrame(
        payload_b64=payload,
        width=int(frame.image.shape[1]),
        height=int(frame.image.shape[0]),
        scale=scale,
        offset=offset,
        metadata=meta,
    )
