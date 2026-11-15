"""Synthetic point-source injection into REAL SPHEREx cutouts (validation only).

Sources are added to the IMAGE extension of each native cutout, i.e. before
registration, resampling, detection and photometry, so an injected source goes
through exactly the same pipeline as a real one (app/offline.py:
run_cached(..., cutout_transform=...)).

Model of an injected source in one exposure:
  * position: (ra0, dec0) at ref_mjd, moved by a linear proper motion to the
    exposure's mid-time (header MJD-AVG);
  * PSF: circular Gaussian with the exposure's own header PSF_FWHM, integrated
    over each native pixel (the real ePSF is not Gaussian; documented);
  * flux: total flux density in mJy, optionally multiplied by a per-epoch
    factor (variable sources);
  * units: MJy/sr per pixel = flux / (pixel solid angle * 1e9) * PSF fraction.
Pixels that are blanked (NaN) in the real image stay blanked. The VARIANCE
plane is not changed, so the injected source's own photon noise is not added:
negligible for faint injections, a small underestimate of errors for the
brightest (documented in docs/RANKING.md).

The injected cutouts are never written to disk and never replace the cached
data; the real fixtures are read-only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import astropy.units as u
from scipy.special import erf

ARCSEC_PER_RAD = 206264.80624709636


@dataclass
class InjectedSource:
    ra_deg: float
    dec_deg: float
    flux_mjy: float
    ref_mjd: float
    pm_east_mas_yr: float = 0.0
    pm_north_mas_yr: float = 0.0
    # epoch boundaries (MJD) and a flux factor per epoch, for variable sources
    epoch_edges_mjd: Sequence[float] = field(default_factory=tuple)
    epoch_factors: Sequence[float] = field(default_factory=tuple)
    kind: str = "stationary"
    label: str = ""
    # optional per-detector total flux (mJy), overriding flux_mjy: a source with a "colour"
    flux_by_detector: Optional[dict] = None

    def position_at(self, mjd: float) -> tuple[float, float]:
        dt = (mjd - self.ref_mjd) / 365.25
        dec = self.dec_deg + self.pm_north_mas_yr * dt / 3.6e6
        ra = self.ra_deg + self.pm_east_mas_yr * dt / 3.6e6 / math.cos(math.radians(0.5 * (dec + self.dec_deg)))
        return ra, dec

    def flux_at(self, mjd: float, detector: Optional[str] = None) -> float:
        base = self.flux_mjy
        if self.flux_by_detector and detector in self.flux_by_detector:
            base = float(self.flux_by_detector[detector])
        if not self.epoch_factors:
            return base
        idx = int(np.searchsorted(np.asarray(self.epoch_edges_mjd), mjd))
        return base * float(self.epoch_factors[min(idx, len(self.epoch_factors) - 1)])


def render(shape, x0: float, y0: float, fwhm_pix: float) -> np.ndarray:
    """Pixel-integrated circular Gaussian, normalised to unit total."""
    s = fwhm_pix / 2.3548 * math.sqrt(2.0)
    ix = np.arange(shape[1]); iy = np.arange(shape[0])
    gx = 0.5 * (erf((ix + 0.5 - x0) / s) - erf((ix - 0.5 - x0) / s))
    gy = 0.5 * (erf((iy + 0.5 - y0) / s) - erf((iy - 0.5 - y0) / s))
    return np.outer(gy, gx)


def inject(hdul, sources: Sequence[InjectedSource], default_fwhm_arcsec: float = 9.0) -> list[dict]:
    """Add `sources` to hdul["IMAGE"] in place. Returns what was injected where."""
    header = hdul["IMAGE"].header
    img = hdul["IMAGE"].data
    mjd = float(header.get("MJD-AVG", header.get("MJD-OBS", 0.0)))
    wcs = WCS(header)
    scale = float(np.mean(proj_plane_pixel_scales(wcs)) * 3600.0)
    omega = (scale / ARCSEC_PER_RAD) ** 2
    fwhm = float(header.get("PSF_FWHM", default_fwhm_arcsec)) / scale
    data = img.astype(np.float64)
    log = []
    for s in sources:
        ra, dec = s.position_at(mjd)
        x, y = (float(v) for v in wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg)))
        if not (-5 < x < data.shape[1] + 5 and -5 < y < data.shape[0] + 5):
            log.append({"label": s.label, "x": x, "y": y, "injected": False})
            continue
        det = f"D{int(header['DETECTOR'])}" if "DETECTOR" in header else None
        f = s.flux_at(mjd, det)
        add = render(data.shape, x, y, fwhm) * f / (omega * 1e9)
        finite = np.isfinite(data)
        data[finite] += add[finite]
        log.append({"label": s.label, "x": x, "y": y, "flux_mjy": f, "mjd": mjd, "injected": True})
    hdul["IMAGE"].data = data.astype(img.dtype)
    return log


def make_transform(sources: Sequence[InjectedSource]):
    """A cutout_transform for app/offline.run_cached."""
    def transform(cutout):
        inject(cutout.hdul, sources)
        return cutout
    return transform
