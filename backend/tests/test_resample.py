"""Unit tests for app/resample.py. SYNTHETIC arrays only."""
import math

import numpy as np
import pytest
from astropy.wcs import WCS

from app.resample import KERNELS, resample


def _wcs(npix, scale_arcsec, dx=0.0, dy=0.0):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [150.0, 20.0]
    w.wcs.crpix = [npix / 2 + 0.5 + dx, npix / 2 + 0.5 + dy]
    w.wcs.cdelt = [-scale_arcsec / 3600, scale_arcsec / 3600]
    return w


def _gauss(n, x0, y0, sigma, flux=100.0):
    yy, xx = np.mgrid[0:n, 0:n]
    g = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
    return flux * g / g.sum()


@pytest.mark.parametrize("kernel", KERNELS)
def test_identity_when_grids_match(kernel):
    w = _wcs(21, 6.15)
    data = np.random.default_rng(0).normal(5, 1, (21, 21))
    out, var = resample(data, np.ones_like(data), w, w, (21, 21), kernel=kernel)
    np.testing.assert_allclose(out, data, atol=1e-5)
    np.testing.assert_allclose(var, 1.0, atol=1e-5)


@pytest.mark.parametrize("kernel", KERNELS)
def test_masked_pixel_stays_masked_and_does_not_spread(kernel):
    w_in, w_out = _wcs(21, 6.15), _wcs(41, 3.075)
    data = np.full((21, 21), 7.0)
    data[10, 10] = np.nan
    out, _ = resample(data, None, w_in, w_out, (41, 41), kernel=kernel)
    # output pixels whose nearest input pixel is the masked one are invalid ...
    assert np.isnan(out[20, 20])
    # ... but the hole does not grow beyond that input pixel's footprint (2x2 output pixels, +1 for edges)
    assert np.isnan(out).sum() <= 9
    # ... and everything valid is still exactly the constant (weights renormalised, no NaN leakage)
    np.testing.assert_allclose(out[np.isfinite(out)], 7.0, atol=1e-5)


def test_outside_input_footprint_is_invalid():
    out, _ = resample(np.ones((11, 11)), None, _wcs(11, 6.15), _wcs(61, 3.075), (61, 61), kernel="lanczos3")
    assert np.isnan(out[0, 0]) and np.isfinite(out[30, 30])


def test_bilinear_variance_of_half_pixel_shift():
    # Shift by half an input pixel in x: each output pixel = mean of two inputs -> variance halves.
    w_in, w_out = _wcs(21, 6.15), _wcs(21, 6.15, dx=0.5)
    out, var = resample(np.zeros((21, 21)), np.ones((21, 21)), w_in, w_out, (21, 21), kernel="bilinear")
    assert var[10, 10] == pytest.approx(0.5, abs=1e-5)


def test_propagated_variance_matches_measured_noise():
    rng = np.random.default_rng(1)
    w_in, w_out = _wcs(41, 6.15, dx=0.3, dy=-0.2), _wcs(61, 3.075)
    for kernel in ("bilinear", "lanczos3"):
        outs = []
        for _ in range(300):
            o, v = resample(rng.normal(0, 1, (41, 41)), np.ones((41, 41)), w_in, w_out, (61, 61), kernel=kernel)
            outs.append(o[20:40, 20:40])
        measured = np.var(np.array(outs), axis=0).mean()
        assert measured == pytest.approx(float(np.nanmean(v[20:40, 20:40])), rel=0.05), kernel


@pytest.mark.parametrize("kernel, tol_px", [("bilinear", 0.06), ("lanczos3", 0.06), ("nearest", 0.6)])
def test_subpixel_position_is_preserved(kernel, tol_px):
    """A star at a known sky position must land at the matching output position."""
    sigma_in = 9.0 / 2.3548 / 6.15
    w_in, w_out = _wcs(41, 6.15, dx=0.37, dy=-0.21), _wcs(81, 3.075)
    x_in, y_in = 21.3, 19.6
    sky = w_in.pixel_to_world(x_in, y_in)
    tx, ty = (float(v) for v in w_out.world_to_pixel(sky))
    # finely sampled Gaussian rendered at pixel centres (smooth enough for this check)
    out, _ = resample(_gauss(41, x_in, y_in, sigma_in), None, w_in, w_out, (81, 81), kernel=kernel)
    d = np.where(np.isfinite(out), out, 0.0)
    yy, xx = np.mgrid[0:81, 0:81]
    cx, cy = (xx * d).sum() / d.sum(), (yy * d).sum() / d.sum()
    assert math.hypot(cx - tx, cy - ty) < tol_px


@pytest.mark.parametrize("kernel", ["bilinear", "lanczos3"])
def test_total_flux_is_conserved(kernel):
    sigma_in = 9.0 / 2.3548 / 6.15
    w_in, w_out = _wcs(41, 6.15, dx=0.25, dy=0.4), _wcs(81, 3.075)
    out, _ = resample(_gauss(41, 20.2, 20.7, sigma_in, flux=100.0), None, w_in, w_out, (81, 81), kernel=kernel)
    # surface brightness in, surface brightness out: 4 output pixels per input pixel
    assert np.nansum(out) / 4.0 == pytest.approx(100.0, rel=0.01)
