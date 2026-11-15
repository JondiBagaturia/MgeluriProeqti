"""Make Python objects strict-JSON safe.

docs/AUDIT.md C2/M1: Starlette's JSONResponse uses allow_nan=False, so a
single NaN (a saturated pixel, an off-detector wavelength) raised a 500.
Plain json.dumps instead writes the token `NaN`, which the browser's
JSON.parse rejects, and the frontend then silently dropped that frame. Missing
values are therefore always sent as null.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
