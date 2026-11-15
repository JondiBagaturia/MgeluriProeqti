"""Resampling of one exposure onto the common grid, with honest masks and
variances.

One function, three kernels, so that the choice can be benchmarked on equal
terms (scripts/benchmark_resampling.py, results in docs/PHASE2.md):

  * "nearest"   each output pixel takes the value of the nearest input pixel.
  * "bilinear"  2x2 input pixels, linear weights.
  * "lanczos3"  6x6 input pixels, Lanczos a=3 (has negative lobes).

Rules that hold for every kernel:
  * Surface brightness is interpolated (MJy/sr in, MJy/sr out), so
    integrated flux is preserved to the accuracy of the kernel.
  * Invalid input pixels (NaN) get weight zero and the remaining weights
    are renormalised. An output pixel is invalid if the valid weights carry
    less than `min_weight_fraction` of the kernel (default 0.5), or if the
    NEAREST input pixel itself is invalid. The second rule keeps a
    saturated core from being silently filled in from its neighbours.
  * Variance is propagated for the linear combination actually applied:
    var_out = sum(w_i^2 var_i) / (sum w_i)^2. This is the variance of ONE
    output pixel. Neighbouring output pixels still share input pixels, so
    sums over several output pixels (apertures) need the measured
    correlation factor from the benchmark.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from astropy.wcs import WCS

KERNELS = ("nearest", "bilinear", "lanczos3")


def _lanczos(x: np.ndarray, a: int = 3) -> np.ndarray:
    out = np.sinc(x) * np.sinc(x / a)
    out[np.abs(x) >= a] = 0.0
    return out


def input_pixel_coords(wcs_in: WCS, wcs_out: WCS, shape_out: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """For every output pixel centre, the (x, y) position in the input image."""
    ny, nx = shape_out
    yy, xx = np.mgrid[0:ny, 0:nx]
    world = wcs_out.pixel_to_world(xx.ravel(), yy.ravel())
    x_in, y_in = wcs_in.world_to_pixel(world)
    return np.asarray(x_in, float), np.asarray(y_in, float)


def resample(
    data: np.ndarray,
    variance: Optional[np.ndarray],
    wcs_in: WCS,
    wcs_out: WCS,
    shape_out: tuple[int, int],
    kernel: str = "bilinear",
    min_weight_fraction: float = 0.5,
    coords: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Returns (image_out, variance_out), each `shape_out`, NaN where invalid."""
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")
    x_in, y_in = coords if coords is not None else input_pixel_coords(wcs_in, wcs_out, shape_out)
    h, w = data.shape
    n = x_in.size

    xn = np.round(x_in).astype(np.int64)
    yn = np.round(y_in).astype(np.int64)
    inside = (xn >= 0) & (xn < w) & (yn >= 0) & (yn < h)
    valid_in = np.isfinite(data)
    if variance is not None:
        valid_in &= np.isfinite(variance)

    nearest_ok = np.zeros(n, dtype=bool)
    nearest_ok[inside] = valid_in[yn[inside], xn[inside]]

    out = np.full(n, np.nan, dtype=np.float64)
    var_out = np.full(n, np.nan, dtype=np.float64) if variance is not None else None

    if kernel == "nearest":
        out[nearest_ok] = data[yn[nearest_ok], xn[nearest_ok]]
        if variance is not None:
            var_out[nearest_ok] = variance[yn[nearest_ok], xn[nearest_ok]]
    else:
        if kernel == "bilinear":
            x0 = np.floor(x_in).astype(np.int64)
            y0 = np.floor(y_in).astype(np.int64)
            offsets = (0, 1)
            wfun = lambda d: np.clip(1.0 - np.abs(d), 0.0, None)  # noqa: E731
        else:
            x0 = np.floor(x_in).astype(np.int64) - 2
            y0 = np.floor(y_in).astype(np.int64) - 2
            offsets = tuple(range(6))
            wfun = _lanczos

        acc = np.zeros(n)
        wsum = np.zeros(n)
        wabs_valid = np.zeros(n)
        wabs_all = np.zeros(n)
        vacc = np.zeros(n) if variance is not None else None
        for dy in offsets:
            yi = y0 + dy
            wy = wfun(y_in - yi)
            for dx in offsets:
                xi = x0 + dx
                wgt = wy * wfun(x_in - xi)
                ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
                good = np.zeros(n, dtype=bool)
                good[ok] = valid_in[yi[ok], xi[ok]]
                wabs_all += np.abs(wgt)
                wg = np.where(good, wgt, 0.0)
                vals = np.zeros(n)
                vals[good] = data[yi[good], xi[good]]
                acc += wg * vals
                wsum += wg
                wabs_valid += np.abs(wg)
                if vacc is not None:
                    vv = np.zeros(n)
                    vv[good] = variance[yi[good], xi[good]]
                    vacc += wg ** 2 * vv
        with np.errstate(invalid="ignore", divide="ignore"):
            usable = nearest_ok & (wabs_valid >= min_weight_fraction * wabs_all) & (np.abs(wsum) > 1e-6)
            out[usable] = acc[usable] / wsum[usable]
            if var_out is not None:
                var_out[usable] = vacc[usable] / wsum[usable] ** 2

    img = out.reshape(shape_out).astype(np.float32)
    return img, (var_out.reshape(shape_out).astype(np.float32) if var_out is not None else None)
