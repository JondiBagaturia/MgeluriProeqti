"""Phase 4 (bonus): click a pixel, get flux vs wavelength at that exact
position from every matching cutout.

This is forced photometry at a single native pixel — not PSF/aperture
photometry (the brief's stretch goal if time permits; it isn't done here).
It deliberately does NOT reproject onto the common grid used by the blink
viewer: each cutout's own wavelength solution is per-pixel (the "linear
variable filter" physics in PLAN.md), so reading the native pixel directly
keeps the correct wavelength for each exposure. Proper motion is applied
per-exposure, at that exposure's exact epoch, before picking the pixel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from astropy.wcs import WCS
from astropy.time import Time

from .coords import ResolvedPosition
from .irsa import find_images, fetch_cutouts, DiscoveryError
from .pipeline import read_flag_names, default_bad_bits, central_wavelength


@dataclass
class SpectrumPoint:
    time_mjd: float
    detector: str
    wavelength_um: Optional[float]
    bandwidth_um: Optional[float]
    flux: Optional[float]  # None if the target pixel fell outside this cutout or was masked
    masked: bool


def compute_spectrum(resolved: ResolvedPosition, fov_arcmin: float, detectors: Optional[list[str]]) -> list[SpectrumPoint]:
    records = find_images(resolved.coord, fov_arcmin=fov_arcmin, detectors=detectors)
    points: list[SpectrumPoint] = []

    for cutout in fetch_cutouts(records, resolved.coord, fov_arcmin=fov_arcmin):
        hdul = cutout.hdul
        header = hdul["IMAGE"].header
        wcs = WCS(header)

        if resolved.has_proper_motion:
            coord_here = resolved.propagate_to(Time(cutout.record.time_mjd, format="mjd"))
        else:
            coord_here = resolved.coord

        x, y = wcs.world_to_pixel(coord_here)
        xi, yi = int(round(float(x))), int(round(float(y)))

        image = hdul["IMAGE"].data
        h, w = image.shape
        if not (0 <= xi < w and 0 <= yi < h):
            continue  # proper motion (or a small FOV) moved it off this cutout

        zodi = hdul["ZODI"].data[yi, xi] if "ZODI" in hdul else 0.0
        flux = float(image[yi, xi] - zodi)

        masked = False
        if "FLAGS" in hdul:
            flag_names = read_flag_names(hdul["FLAGS"].header)
            bad_bits = default_bad_bits(flag_names)
            bad_value = 0
            for bit in bad_bits:
                bad_value |= 1 << bit
            masked = bool(int(hdul["FLAGS"].data[yi, xi]) & bad_value)

        wavelength_um, bandwidth_um = central_wavelength(hdul, coord_here)

        points.append(
            SpectrumPoint(
                time_mjd=cutout.record.time_mjd,
                detector=cutout.record.detector,
                wavelength_um=wavelength_um,
                bandwidth_um=bandwidth_um,
                flux=None if masked else flux,
                masked=masked,
            )
        )

    points.sort(key=lambda p: (p.wavelength_um if p.wavelength_um is not None else 0.0))
    return points
