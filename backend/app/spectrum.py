"""Per-exposure forced aperture photometry at one sky position: F(lambda, t).

Every exposure is measured on its OWN native pixels and its OWN wavelength
solution (the linear-variable filter puts a different wavelength at each
detector position, so reprojecting first would mix wavelengths). Proper
motion, when known, is applied at each exposure's own epoch before measuring.

What the output is, and is not (docs/AUDIT.md H3):
  * Aperture flux density in mJy with a 1-sigma error (see app/photometry.py
    for conventions: no aperture correction, refuse-on-masked).
  * Every point carries its observation time. Points from different epochs
    are NOT a single simultaneous spectrum. SPHEREx does not observe all
    wavelengths at once, so for variable sources the client must bin by
    epoch before reading a spectral shape.
  * Downloads are capped (config `spectrum.max_exposures`) and spread evenly
    in time within each detector.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.time import Time

from .config import get_config
from .coords import ResolvedPosition
from .irsa import find_images, fetch_cutouts
from .photometry import aperture_photometry
from .pipeline import read_flag_names, default_bad_bits, build_mask, central_wavelength, cap_group_indices


@dataclass
class SpectrumPoint:
    time_mjd: float
    detector: str
    collection: str
    wavelength_um: Optional[float]
    bandwidth_um: Optional[float]
    flux_mjy: Optional[float]       # None when the measurement was refused (see status)
    flux_err_mjy: Optional[float]
    snr: Optional[float]
    status: str                     # photometry status, e.g. "ok", "masked_in_aperture", "off_image"
    flags_at_target: list[str]      # flag names set on the target's nearest native pixel (any bit)
    ra_deg: float                   # where it was measured (PM-propagated if applicable)
    dec_deg: float


def _cap_records(records, max_total: int):
    """Spread at most `max_total` exposures evenly in time across detectors."""
    if not max_total or len(records) <= max_total:
        return records
    by_det: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_det[r.detector].append(i)
    per_det = max(1, math.ceil(max_total / len(by_det)))
    keep: list[int] = []
    for idxs in by_det.values():
        keep.extend(cap_group_indices(idxs, per_det))
    return [records[i] for i in sorted(keep)]


def compute_spectrum(
    resolved: ResolvedPosition,
    fov_arcmin: float,
    detectors: Optional[list[str]],
    max_exposures: Optional[int] = None,
    records_override: Optional[list] = None,
    cutout_source=None,
) -> list[SpectrumPoint]:
    cfg = get_config()
    cap = cfg.spectrum_max_exposures if max_exposures is None else max_exposures

    # Search (and cut out) around the position at the mean epoch, not J2000:
    # for Barnard's Star those differ by ~4.5 arcmin.
    if records_override is not None:   # app/localdata.py (downloaded data)
        records = [r for r in records_override if not detectors or r.detector in detectors]
    else:
        records = find_images(resolved.search_position(), fov_arcmin=fov_arcmin, detectors=detectors)
    if records and resolved.has_proper_motion:
        mean_mjd = float(np.mean([r.time_mjd for r in records]))
        centre = resolved.propagate_to(Time(mean_mjd, format="mjd"))
    else:
        centre = resolved.coord
    records = _cap_records(records, cap)

    points: list[SpectrumPoint] = []
    for cutout in (cutout_source or fetch_cutouts)(records, centre, fov_arcmin=fov_arcmin):
        hdul = cutout.hdul
        header = hdul["IMAGE"].header
        wcs = WCS(header)

        coord_here = (
            resolved.propagate_to(Time(cutout.record.time_mjd, format="mjd"))
            if resolved.has_proper_motion
            else resolved.coord
        )
        x, y = (float(v) for v in wcs.world_to_pixel(coord_here))

        image = hdul["IMAGE"].data.astype(np.float64)
        zodi = hdul["ZODI"].data.astype(np.float64) if "ZODI" in hdul else 0.0
        variance = hdul["VARIANCE"].data.astype(np.float64) if "VARIANCE" in hdul else None
        sb = image - zodi

        flags_at_target: list[str] = []
        bad = None
        if "FLAGS" in hdul:
            flags = hdul["FLAGS"].data
            names = read_flag_names(hdul["FLAGS"].header)
            bad = build_mask(flags, default_bad_bits(names))
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < flags.shape[0] and 0 <= xi < flags.shape[1]:
                v = int(flags[yi, xi])
                flags_at_target = sorted(n for b, n in names.items() if (v >> b) & 1)

        scale_arcsec = float(np.mean(proj_plane_pixel_scales(wcs)) * 3600.0)
        phot = aperture_photometry(sb, variance, x, y, scale_arcsec, extra_invalid=bad)
        wavelength_um, bandwidth_um = central_wavelength(hdul, coord_here)

        points.append(
            SpectrumPoint(
                time_mjd=cutout.record.time_mjd,
                detector=cutout.record.detector,
                collection=cutout.record.collection,
                wavelength_um=wavelength_um,
                bandwidth_um=bandwidth_um,
                flux_mjy=phot.flux_mjy,
                flux_err_mjy=phot.flux_err_mjy,
                snr=phot.snr,
                status=phot.status,
                flags_at_target=flags_at_target,
                ra_deg=float(coord_here.ra.deg),
                dec_deg=float(coord_here.dec.deg),
            )
        )

    points.sort(key=lambda p: (p.wavelength_um if p.wavelength_um is not None else 0.0))
    return points
