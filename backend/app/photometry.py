"""Background-subtracted circular-aperture photometry on SPHEREx images.

Why this exists (docs/AUDIT.md C4): the light curve and spectrum used to
read ONE pixel of surface brightness (MJy/sr). With a ~9" PSF on 6.15"
native pixels that number depends strongly on where the star falls inside
the pixel. It also gave meaningless numbers for moving or saturated sources,
because the saturated core was hole-filled from its own wings. An aperture
sum is much less sensitive to where the star lands within a pixel, and it
comes with a propagated uncertainty.

Scientific conventions and limitations, stated explicitly:
  * Input surface brightness is MJy/sr (SPHEREx L2 IMAGE BUNIT), variance
    is (MJy/sr)^2. Flux density = sum(SB * pixel solid angle) -> reported in mJy.
  * Pixels whose centres lie within r_ap are included (no fractional-pixel
    weighting). Coarse at 6.15" native pixels, so treat the result as an
    *aperture flux*. NO aperture correction is applied: the absolute
    normalisation is low by the fraction of the PSF outside the aperture.
    Because the PSF changes with wavelength, comparisons across very
    different wavelengths carry this bias too.
  * Background = sigma-clipped median of an annulus; its uncertainty is
    propagated (std / sqrt(n_annulus) per pixel, times n_ap).
  * If ANY aperture pixel is invalid (NaN: masked, saturated, off-image,
    or hole-filled for display) the measurement is REFUSED (flux=None) with
    a status saying why, instead of returning a biased number.
  * On an oversampled grid (nearest-neighbour duplicates each native pixel
    over ~k grid pixels) neighbouring pixels are perfectly correlated; the
    naive variance sum underestimates the true variance by ~k, so callers
    pass `variance_inflation = (native_scale / grid_scale)**2`.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from astropy.stats import sigma_clipped_stats

ARCSEC_PER_RAD = 206264.80624709636

# Defaults tied to the measured SPHEREx PSF (header PSF_FWHM ~ 9" in D6).
DEFAULT_R_AP_ARCSEC = 9.0
DEFAULT_R_IN_ARCSEC = 18.0
DEFAULT_R_OUT_ARCSEC = 30.0


@dataclass
class ApertureResult:
    status: str                       # "ok" | "masked_in_aperture" | "off_image" | "no_background"
    x: float                          # 0-based pixel position measured at
    y: float
    flux_mjy: Optional[float] = None
    flux_err_mjy: Optional[float] = None
    snr: Optional[float] = None
    background_mjy_sr: Optional[float] = None
    n_aperture_pixels: int = 0
    n_invalid_in_aperture: int = 0
    r_ap_arcsec: float = DEFAULT_R_AP_ARCSEC

    def to_dict(self) -> dict:
        return asdict(self)


def aperture_photometry(
    image: np.ndarray,
    variance: Optional[np.ndarray],
    x: float,
    y: float,
    pixel_scale_arcsec: float,
    r_ap_arcsec: float = DEFAULT_R_AP_ARCSEC,
    r_in_arcsec: float = DEFAULT_R_IN_ARCSEC,
    r_out_arcsec: float = DEFAULT_R_OUT_ARCSEC,
    extra_invalid: Optional[np.ndarray] = None,
    variance_inflation: float = 1.0,
) -> ApertureResult:
    """Measure at 0-based pixel (x, y) = (column, row)."""
    h, w = image.shape
    if not (np.isfinite(x) and np.isfinite(y)) or not (-0.5 <= x < w - 0.5 and -0.5 <= y < h - 0.5):
        return ApertureResult(status="off_image", x=float(x), y=float(y), r_ap_arcsec=r_ap_arcsec)

    yy, xx = np.mgrid[0:h, 0:w]
    r_arcsec = np.hypot(xx - x, yy - y) * pixel_scale_arcsec
    in_ap = r_arcsec <= r_ap_arcsec
    in_ann = (r_arcsec >= r_in_arcsec) & (r_arcsec <= r_out_arcsec)

    invalid = ~np.isfinite(image)
    if variance is not None:
        invalid |= ~np.isfinite(variance)
    if extra_invalid is not None:
        invalid |= extra_invalid

    # An aperture that would extend past the image edge is just as biased
    # as one containing a masked pixel.
    r_ap_pix = r_ap_arcsec / pixel_scale_arcsec
    if x - r_ap_pix < -0.5 or y - r_ap_pix < -0.5 or x + r_ap_pix > w - 0.5 or y + r_ap_pix > h - 0.5:
        return ApertureResult(status="off_image", x=float(x), y=float(y), r_ap_arcsec=r_ap_arcsec,
                              n_aperture_pixels=int(in_ap.sum()))

    n_ap = int(in_ap.sum())
    n_bad = int((in_ap & invalid).sum())
    if n_ap == 0 or n_bad:
        return ApertureResult(status="masked_in_aperture", x=float(x), y=float(y), n_aperture_pixels=n_ap,
                              n_invalid_in_aperture=n_bad, r_ap_arcsec=r_ap_arcsec)

    ann_vals = image[in_ann & ~invalid]
    if ann_vals.size < 8:
        return ApertureResult(status="no_background", x=float(x), y=float(y), n_aperture_pixels=n_ap,
                              r_ap_arcsec=r_ap_arcsec)
    _, bkg, bkg_std = sigma_clipped_stats(ann_vals, sigma=3.0, maxiters=5)

    omega_sr = (pixel_scale_arcsec / ARCSEC_PER_RAD) ** 2
    to_mjy = omega_sr * 1e9  # MJy -> mJy
    net_sb_sum = float(np.sum(image[in_ap] - bkg))
    flux = net_sb_sum * to_mjy

    # Background error: the annulus pixels are also correlated on an
    # oversampled grid, so the effective number of independent samples is
    # n_ann / inflation.
    n_ann_eff = max(ann_vals.size / max(variance_inflation, 1.0), 1.0)
    var_bkg_total = (n_ap ** 2) * (bkg_std ** 2) / n_ann_eff
    if variance is not None:
        var_src = float(np.sum(variance[in_ap])) * variance_inflation
    else:
        var_src = n_ap * bkg_std ** 2 * variance_inflation
    err = float(np.sqrt(var_src + var_bkg_total)) * to_mjy

    return ApertureResult(
        status="ok",
        x=float(x),
        y=float(y),
        flux_mjy=flux,
        flux_err_mjy=err,
        snr=flux / err if err > 0 else None,
        background_mjy_sr=float(bkg),
        n_aperture_pixels=n_ap,
        r_ap_arcsec=r_ap_arcsec,
    )
