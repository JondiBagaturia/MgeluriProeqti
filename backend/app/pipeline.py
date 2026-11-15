"""The actual image-processing pipeline: zodi-subtract -> mask -> reproject
onto one common grid -> group -> inverse-variance combine.

Flag bit names are read from each file's own FLAGS header (MSKNnnnn cards)
rather than hard-coded — PLAN.md Phase 0 found the live bit assignments
don't exactly match the brief's quick-reference table, so hard-coding would
silently mask (or fail to mask) the wrong bits on some files.

Reprojection is our own nearest-neighbour (reproject_nearest below), not
the `reproject` package's `reproject_interp` - a deliberate, evidence-based
exception to PLAN.md's "use reproject, don't reimplement it" architecture
note. See reproject_nearest's docstring for the live-confirmed bug this
replaced (its own nearest-neighbor mode zeroed the footprint at some
exposures' pixel-boundary coordinates even though the actual nearest
native pixel was clean).
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from astropy.wcs import WCS, FITSFixedWarning
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy import ndimage

from .irsa import Cutout, ImageRecord
from .resample import resample, input_pixel_coords
from .config import get_config

# Every SPHEREx file has this CTYPE/SIP mismatch (brief: "Sky WCS is
# TAN-SIP in the IMAGE header", but CTYPE itself just says RA---TAN /
# DEC--TAN). astropy correctly applies the SIP coefficients anyway; the
# warning is expected on every single file, not a sign something's wrong.
# astropy routes this through both the `warnings` module *and* its own
# "astropy" logger (the latter is what actually prints it under uvicorn/
# plain scripts), so both have to be silenced to stop the per-file spam.
warnings.filterwarnings("ignore", category=FITSFixedWarning, message=".*SIP.*")
logging.getLogger("astropy").setLevel(logging.ERROR)

# Only these four are treated as genuinely bad by default - matches a
# teammate's independent "planet-x" renderer exactly
# (phase1/stack.py: `BAD = sum(1 << b for b in (6, 9, 10, 11))`,
# i.e. NONFUNC/MISSING_DATA/HOT/COLD by name, read dynamically here rather
# than by their hard-coded bit numbers per PLAN.md's finding that bit
# assignments vary file to file).
#
# Everything else - TRANSIENT, OVERFLOW, SUR_ERROR, BLOOM, GHOST*, STREAK,
# PERSIST, CROSSTALK, DICHROIC, PHANMISS, NONLINEAR, OUTLIER, SNOWBALL,
# HALO, SATELLITE_HALO - is left unmasked by default, same as their
# script. The original, narrower safe-list here (just TRANSIENT/HALO) was
# a first attempt that still left a real-bug-report hole: Barnard's Star
# (not even an especially bright target) showed "no data" at its own
# brightest pixel in every exposure, every detector, because ~20 "bad"
# bits were all disqualifying and a point source's peak is statistically
# likely to trip at least one. Their teammate's wider list fixes it fully.
# Genuine saturation (R Draconis: OVERFLOW+BLOOM) still shows as NaN
# regardless of this list, because SPHEREx's own Level-2 pipeline already
# blanks the raw IMAGE value there - independent of any flag we mask on.
DEFAULT_BAD_FLAG_NAMES = {"NONFUNC", "MISSING_DATA", "HOT", "COLD"}


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

def read_flag_names(flags_header) -> dict[int, str]:
    """{bit_number: name} straight from this file's own header."""
    names = {}
    for key in flags_header:
        if key.startswith("MSKN"):
            try:
                bit = int(key[4:])
            except ValueError:
                continue
            names[bit] = str(flags_header[key]).strip()
    return names


def default_bad_bits(flag_names: dict[int, str]) -> set[int]:
    return {bit for bit, name in flag_names.items() if name in DEFAULT_BAD_FLAG_NAMES}


def build_mask(flags_data: np.ndarray, bad_bits: set[int]) -> np.ndarray:
    """True where a pixel carries any bit in `bad_bits` and should be excluded."""
    if not bad_bits:
        return np.zeros(flags_data.shape, dtype=bool)
    bad_mask_value = 0
    for bit in bad_bits:
        bad_mask_value |= 1 << bit
    return (flags_data.astype(np.int64) & bad_mask_value) != 0


def flag_contamination_summary(flags_data: np.ndarray, flag_names: dict[int, str]) -> dict[str, int]:
    """Pixel count per flag name present in this cutout (Phase 3 wants this
    surfaced to users so they don't mistake an artifact for a discovery)."""
    summary = {}
    for bit, name in flag_names.items():
        count = int(np.count_nonzero((flags_data.astype(np.int64) >> bit) & 1))
        if count:
            summary[name] = count
    return summary


# ---------------------------------------------------------------------------
# Common output grid
# ---------------------------------------------------------------------------

def make_target_wcs(coord: SkyCoord, fov_arcmin: float, pixel_scale_arcsec: float):
    """A simple TAN grid centred on `coord`, sized to cover `fov_arcmin`.

    npix is forced odd. An even-sized axis has no single centre pixel -
    its geometric centre sits exactly on the boundary between two pixels
    (confirmed live: for npix=30, the target landed at pixel coordinate
    (14.5, 14.5), not on any integer pixel). Every piece of this app that
    reads "the target pixel" (the crosshair, the pixel readout, the light
    curve, the spectrum feature, and this very script's diagnostics) uses
    shape[0]//2 as that pixel - which is only exactly correct when npix is
    odd and crpix lands on an integer. Odd npix changes the FOV actually
    delivered by at most one pixel's width (~6 arcsec) - negligible next
    to getting "the target pixel" unambiguously right.
    """
    npix = max(int(np.ceil(fov_arcmin * 60.0 / pixel_scale_arcsec)), 8)
    if npix % 2 == 0:
        npix += 1
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [coord.ra.deg, coord.dec.deg]
    wcs.wcs.crpix = [npix / 2.0 + 0.5, npix / 2.0 + 0.5]
    scale_deg = pixel_scale_arcsec / 3600.0
    wcs.wcs.cdelt = [-scale_deg, scale_deg]
    wcs.wcs.cunit = ["deg", "deg"]
    return wcs, (npix, npix)


# ---------------------------------------------------------------------------
# Per-cutout processing
# ---------------------------------------------------------------------------

@dataclass
class ProcessedCutout:
    record: ImageRecord
    image: np.ndarray       # reprojected, zodi-subtracted, masked (NaN) -> shape_out
    variance: np.ndarray    # reprojected, masked (NaN) -> shape_out
    flag_names: dict[int, str]
    contamination: dict[str, int]
    central_wavelength_um: Optional[float] = None
    bandwidth_um: Optional[float] = None
    # Output-grid pixels whose 2x2 interpolation footprint in the native
    # cutout contains a saturation flag (SATURATION_FLAG_NAMES).
    # These pixels still carry a value (only NONFUNC/MISSING_DATA/HOT/COLD
    # are masked), so display code that must not trust them (the
    # false-colour composite) needs to know where they are.
    saturation: Optional[np.ndarray] = None


# Flags that mean "the value is there but saturated or non-linear".
# SUR_ERROR is deliberately NOT included: on the cached real cutouts
# (Barnard, RW Aur, WISE 0855) 21-70 % of SUR_ERROR pixels are at sky level
# (pixel S/N < 5), i.e. it marks ramp-fit problems such as cosmic-ray hits,
# not saturation; OVERFLOW and BLOOM pixels are 97-100 % bright (S/N > 5).
SATURATION_FLAG_NAMES = ("OVERFLOW", "BLOOM", "NONLINEAR")


def saturation_on_grid(flags: np.ndarray, flag_names: dict[int, str], x_in: np.ndarray, y_in: np.ndarray,
                       shape_out: tuple[int, int]) -> np.ndarray:
    """Boolean map on the output grid: True where any native pixel of the
    2x2 neighbourhood around the output pixel's input position carries a
    saturation-related flag. Conservative on purpose: the resampled value
    there is built partly from saturated or non-linear data."""
    bits = [b for b, n in flag_names.items() if n in SATURATION_FLAG_NAMES]
    out = np.zeros(shape_out[0] * shape_out[1], dtype=bool)
    if not bits:
        return out.reshape(shape_out)
    f = flags.astype(np.int64)
    sat = np.zeros(f.shape, dtype=bool)
    for b in bits:
        sat |= ((f >> b) & 1).astype(bool)
    if not sat.any():
        return out.reshape(shape_out)
    h, w = sat.shape
    x0 = np.floor(x_in).astype(np.int64)
    y0 = np.floor(y_in).astype(np.int64)
    for dy in (0, 1):
        for dx in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            out[ok] |= sat[yi[ok], xi[ok]]
    return out.reshape(shape_out)


def reproject_nearest(data_in: np.ndarray, wcs_in: WCS, wcs_out: WCS, shape_out: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour reprojection, computed directly rather than via
    `reproject.reproject_interp(..., order="nearest-neighbor")`.

    Why not just use reproject's own nearest-neighbor mode: it was tried
    first and still left a real discrepancy, confirmed live
    (scripts/check_center_pixel_coverage.py +
    tmp_phase0/debug_nn_residual.py): for several exposures, the output
    pixel's nearest native pixel was clean and unmasked, but reproject's
    *own* returned footprint was 0 at exactly that output pixel anyway -
    its internal validity check is more conservative than "is the nearest
    native pixel actually there", right at certain pixel-boundary
    coordinates (e.g. input coords like (13.79, 13.56) - just inside a
    pixel, not past any real edge).

    This computes exactly what we actually want: for every output pixel,
    find its sky position, map that to the nearest native pixel via the
    *same* astropy WCS (SIP distortion handled identically either way),
    and take that pixel's value - NaN and all, if it's masked. No
    separate footprint heuristic to disagree with.
    """
    ny, nx = shape_out
    yy, xx = np.mgrid[0:ny, 0:nx]
    world = wcs_out.pixel_to_world(xx.ravel(), yy.ravel())
    x_in, y_in = wcs_in.world_to_pixel(world)
    xi = np.round(x_in).astype(np.int64)
    yi = np.round(y_in).astype(np.int64)

    h, w = data_in.shape
    in_bounds = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)

    out = np.full(ny * nx, np.nan, dtype=np.float32)
    out[in_bounds] = data_in[yi[in_bounds], xi[in_bounds]]
    return out.reshape(shape_out)


def central_wavelength(hdul, coord: SkyCoord) -> tuple[Optional[float], Optional[float]]:
    """Wavelength + bandwidth at `coord`'s pixel, per the brief's documented
    pattern: a *separate* WCS keyed 'W' on the same header, SIP disabled."""
    try:
        header = hdul["IMAGE"].header
        spatial_wcs = WCS(header)
        x, y = spatial_wcs.world_to_pixel(coord)
        spectral_wcs = WCS(header, fobj=hdul, key="W")
        spectral_wcs.sip = None
        wavelength, bandwidth = spectral_wcs.pixel_to_world(x, y)
        return float(wavelength.to(u.micrometer).value), float(bandwidth.to(u.micrometer).value)
    except Exception:  # noqa: BLE001 - optional metadata, never fatal
        return None, None


def process_cutout(
    cutout: Cutout,
    target_coord: SkyCoord,
    target_wcs: WCS,
    shape_out: tuple[int, int],
    bad_bit_overrides: Optional[set[int]] = None,
) -> ProcessedCutout:
    hdul = cutout.hdul
    image = hdul["IMAGE"].data.astype(np.float32)
    zodi = hdul["ZODI"].data.astype(np.float32) if "ZODI" in hdul else np.zeros_like(image)
    variance = hdul["VARIANCE"].data.astype(np.float32) if "VARIANCE" in hdul else np.ones_like(image)
    flags = hdul["FLAGS"].data if "FLAGS" in hdul else np.zeros(image.shape, dtype=np.int32)

    corrected = image - zodi  # subtract zodiacal light model, per the brief

    flag_names = read_flag_names(hdul["FLAGS"].header) if "FLAGS" in hdul else {}
    bad_bits = bad_bit_overrides if bad_bit_overrides is not None else default_bad_bits(flag_names)
    mask = build_mask(flags, bad_bits)

    corrected = np.where(mask, np.nan, corrected)
    variance_masked = np.where(mask, np.nan, variance).astype(np.float32)

    in_wcs = WCS(hdul["IMAGE"].header)
    # Our own nearest-neighbour (see reproject_nearest's docstring for why
    # not reproject.reproject_interp's own nearest-neighbor mode): bilinear
    # blends each output pixel from up to 4 native neighbours, so one
    # masked (NaN) neighbour NaNs out an output pixel whose own nearest
    # native pixel was perfectly clean, and reproject's nearest-neighbor
    # mode still disagreed with the actual nearest pixel at some exposures'
    # exact pixel-boundary coordinates (its own footprint said invalid
    # when the nearest native pixel plainly wasn't). Confirmed live
    # (scripts/check_center_pixel_coverage.py): both effects were real,
    # non-trivial contributors to Barnard's Star's "missing centre" bug.
    #
    # 2026-11 (docs/PHASE2.md): the kernel is now configurable and defaults
    # to Lanczos-3, chosen on a synthetic benchmark (lowest photometric
    # scatter, 7x smaller false residuals in difference images than
    # nearest-neighbour). app/resample.py keeps the property that mattered
    # above: an output pixel whose nearest native pixel is masked stays
    # masked, and masked neighbours get zero weight instead of spreading NaN.
    kernel = get_config().resample_kernel
    coords = input_pixel_coords(in_wcs, target_wcs, shape_out)
    reproj_image, reproj_var = resample(
        corrected, variance_masked, in_wcs, target_wcs, shape_out, kernel=kernel, coords=coords
    )

    wavelength_um, bandwidth_um = central_wavelength(hdul, target_coord)
    saturation = saturation_on_grid(flags, flag_names, coords[0], coords[1], shape_out)

    return ProcessedCutout(
        record=cutout.record,
        image=reproj_image.astype(np.float32),
        variance=reproj_var.astype(np.float32),
        flag_names=flag_names,
        contamination=flag_contamination_summary(flags, flag_names),
        central_wavelength_um=wavelength_um,
        bandwidth_um=bandwidth_um,
        saturation=saturation,
    )


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def assign_epochs(records: list[ImageRecord], gap_days: float = 45.0) -> list[int]:
    """0-based survey-epoch index per record (records assumed time-sorted).

    The brief says the sky is covered once per ~6 months with many exposures
    over a few days per pass; rather than hard-coding a 6-month bin width
    (which would misassign exposures near a bin edge), we cluster on gaps:
    a new epoch starts whenever consecutive exposures are more than
    `gap_days` apart, which is far longer than any single pass takes.
    """
    epochs = []
    current = 0
    prev_time = None
    for r in records:
        if prev_time is not None and (r.time_mjd - prev_time) > gap_days:
            current += 1
        epochs.append(current)
        prev_time = r.time_mjd
    return epochs


def group_indices(records: list[ImageRecord], mode: str, gap_days: float = 45.0) -> list[list[int]]:
    """Indices into `records`, bucketed per `mode`, groups chronologically ordered.

    mode: "epoch" (coadd everything in a survey pass, incl. across
    detectors — the default, best S/N), "detector_epoch" (separate per
    detector per pass), or "single" (no combination at all).
    """
    epochs = assign_epochs(records, gap_days)
    if mode == "epoch":
        key_fn = lambda i: (epochs[i],)
    elif mode == "detector_epoch":
        key_fn = lambda i: (epochs[i], records[i].detector)
    elif mode == "single":
        key_fn = lambda i: (i,)
    else:
        raise ValueError(f"Unknown grouping mode: {mode!r} (expected epoch|detector_epoch|single)")

    groups: dict = {}
    for i in range(len(records)):
        groups.setdefault(key_fn(i), []).append(i)
    ordered_keys = sorted(groups.keys(), key=lambda k: min(records[i].time_mjd for i in groups[k]))
    return [groups[k] for k in ordered_keys]


def cap_group_indices(idxs: list[int], max_count: Optional[int]) -> list[int]:
    """Subsample one group's (time-ordered) indices down to at most
    `max_count`, evenly spaced across the group rather than just the first
    N - that keeps the full time baseline (first and last exposure are
    always kept) instead of silently biasing toward one end of the epoch.
    `max_count` of None or <=0 means no cap (current behaviour, unchanged).

    This exists to cut download count+bytes *before* fetching - that's
    where ~97% of request time goes (see app/stream.py's timing report),
    not in reprojecting or combining - so capping here, rather than after
    fetch, is what actually buys speed.
    """
    if not max_count or len(idxs) <= max_count:
        return idxs
    if max_count == 1:
        return [idxs[len(idxs) // 2]]
    step = (len(idxs) - 1) / (max_count - 1)
    picked = sorted({idxs[round(k * step)] for k in range(max_count)})
    return picked


# ---------------------------------------------------------------------------
# Combination
# ---------------------------------------------------------------------------

def combine_inverse_variance(
    images: list[np.ndarray], variances: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel-wise inverse-variance weighted mean, honouring per-pixel NaNs
    (masked-out or outside that exposure's footprint). Returns
    (combined_image, combined_variance, n_contributing), each shape_out;
    NaN where nothing contributed at that pixel.
    """
    stack_img = np.stack(images)
    stack_var = np.stack(variances)
    valid = np.isfinite(stack_img) & np.isfinite(stack_var) & (stack_var > 0)

    safe_var = np.where(valid, stack_var, 1.0)
    weights = np.where(valid, 1.0 / safe_var, 0.0)
    weighted_sum = np.sum(np.where(valid, stack_img * weights, 0.0), axis=0)
    weight_sum = np.sum(weights, axis=0)
    n_contrib = np.sum(valid, axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        combined = np.where(weight_sum > 0, weighted_sum / weight_sum, np.nan)
        combined_var = np.where(weight_sum > 0, 1.0 / weight_sum, np.nan)
    return combined.astype(np.float32), combined_var.astype(np.float32), n_contrib.astype(np.int16)


def fill_small_gaps(image: np.ndarray, max_hole_pixels: int = 6) -> tuple[np.ndarray, int]:
    """Fill small, isolated NaN regions with the mean of their immediate
    valid neighbours - for display, after every exposure has already been
    combined.

    Real bug this exists for: a live check (all 6 detectors, Barnard's
    Star - not an especially bright target) found SPHEREx's own automated
    flagging marks the exact peak pixel of a point source in 100% of
    exposures, every detector, every epoch, with some flag or other. With
    DEFAULT_BAD_FLAG_NAMES now matched to the narrower "planet-x" list
    (see that constant's comment), this is now rare - but genuine
    saturation (OVERFLOW+BLOOM, e.g. R Draconis) still blanks the raw
    IMAGE value itself, independent of any mask, so a small hole is still
    possible. Filling only small holes (<= max_hole_pixels, after
    combining across every available exposure) is the conservative middle
    ground: a large masked region (a real bad column, a saturated
    extended source, a satellite streak) is left alone - only a small,
    bounded gap like this one gets a locally-interpolated value. Returns
    (filled_image, n_pixels_filled) so callers can report this honestly
    rather than silently manufacturing data.
    """
    nan_mask = ~np.isfinite(image)
    if not nan_mask.any():
        return image, 0

    labeled, n_features = ndimage.label(nan_mask)
    out = image.copy()
    n_filled = 0
    for label_id in range(1, n_features + 1):
        hole = labeled == label_id
        size = int(hole.sum())
        if size > max_hole_pixels:
            continue  # a real gap, not a one-off flag collision - leave it as NaN
        ring = ndimage.binary_dilation(hole, iterations=1) & ~hole
        neighbour_vals = image[ring]
        neighbour_vals = neighbour_vals[np.isfinite(neighbour_vals)]
        if neighbour_vals.size == 0:
            continue  # no valid neighbours either - can't interpolate, leave as NaN
        out[hole] = float(np.mean(neighbour_vals))
        n_filled += size
    return out, n_filled


@dataclass
class Frame:
    """One blink-viewer frame: possibly several exposures combined."""

    group_index: int
    time_mjd_mean: float
    time_mjd_min: float
    time_mjd_max: float
    detectors: list[str]
    n_exposures: int
    image: np.ndarray
    variance: np.ndarray
    n_contributing: np.ndarray
    contamination: dict[str, int] = field(default_factory=dict)
    central_wavelength_um: Optional[float] = None
    bandwidth_um: Optional[float] = None
    release_versions: list[str] = field(default_factory=list)


def build_frames(
    processed: list[ProcessedCutout],
    mode: str,
    gap_days: float = 45.0,
) -> list[Frame]:
    """Group already-reprojected ProcessedCutouts and combine each group."""
    records = [p.record for p in processed]
    groups = group_indices(records, mode=mode, gap_days=gap_days)

    frames = []
    for gi, idxs in enumerate(groups):
        members = [processed[i] for i in idxs]
        image, variance, n_contrib = combine_inverse_variance(
            [m.image for m in members], [m.variance for m in members]
        )
        contamination: dict[str, int] = {}
        for m in members:
            for name, count in m.contamination.items():
                contamination[name] = contamination.get(name, 0) + count

        waves = [m.central_wavelength_um for m in members if m.central_wavelength_um is not None]
        bws = [m.bandwidth_um for m in members if m.bandwidth_um is not None]
        times = [m.record.time_mjd for m in members]
        frames.append(
            Frame(
                group_index=gi,
                time_mjd_mean=float(np.mean(times)),
                time_mjd_min=float(np.min(times)),
                time_mjd_max=float(np.max(times)),
                detectors=sorted({m.record.detector for m in members}),
                n_exposures=len(members),
                image=image,
                variance=variance,
                n_contributing=n_contrib,
                contamination=contamination,
                central_wavelength_um=float(np.mean(waves)) if waves else None,
                bandwidth_um=float(np.mean(bws)) if bws else None,
                release_versions=sorted({m.record.collection for m in members}),
            )
        )
    return frames
