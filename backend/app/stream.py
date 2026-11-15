"""Incremental frame streaming.

Grouping only depends on each image's metadata (time, detector), which we
already have right after discovery — before any pixels are fetched. So we
compute the groups up front, then as cutouts complete in *arrival* order
(not time order — ten workers finish in whatever order IRSA answers them),
bucket each into its group and emit (combine + encode) that frame the
instant its group is fully assembled. A fast early epoch therefore reaches
the browser without waiting on a slow later one.
"""
from __future__ import annotations

import logging
import base64
import time
from dataclasses import dataclass
from typing import Iterator, Optional, Union

import numpy as np
from astropy.time import Time

from .coords import ResolvedPosition
from .irsa import find_images, fetch_cutouts, DiscoveryError, FetchFailure
from .pipeline import (
    make_target_wcs,
    process_cutout,
    group_indices,
    cap_group_indices,
    combine_inverse_variance,
    fill_small_gaps,
    Frame,
    ProcessedCutout,
)
from .encode import encode_frame, EncodedFrame
from .config import get_config
from .photometry import aperture_photometry
from .discovery import SeriesFrame, analyze_series, DiscoveryConfig
from .catalogs import gaia_sources, catwise_sources, CatalogError

log = logging.getLogger(__name__)


@dataclass
class DownloadProgress:
    """How far the cutout downloads of one request have got. Frames only
    appear once a whole group is in, which can take minutes - this is what
    lets the browser show "41 of 70 images" in the meantime."""

    n_downloaded: int
    n_failed: int
    n_total: int


def stream_frames(
    resolved: ResolvedPosition,
    fov_arcmin: float,
    detectors: Optional[list[str]],
    mode: str,
    resolve_seconds: float = 0.0,
    max_exposures_per_group: Optional[int] = None,
    analyze: bool = True,
    records_override: Optional[list] = None,
    cutout_source=None,
    exposure_sink: Optional[list] = None,
    shiftstack: bool = False,
    spectrotemporal: bool = False,
    report_progress: bool = False,
    data_source: str = "irsa_live",
    local_field=None,
    source_note: str = "",
) -> Iterator[object]:
    """Yields EncodedFrame objects as groups complete, then (unless
    `analyze` is False) one final dict {"type": "analysis", ...} with the
    detection / tracking / catalogue / ranking results for every detector
    series, computed from the very frames just streamed (no second fetch).

    With `report_progress=True`, DownloadProgress items are yielded in
    between: one right after discovery (0 of N) and one per finished
    download."""
    t_start = time.time()
    stats = _TimingStats(resolve_s=resolve_seconds)

    t0 = time.time()
    # `records_override` / `cutout_source` let the offline regression tests and
    # scripts/validate_targets.py run this exact code path on cached real
    # cutouts instead of live IRSA. `exposure_sink`, if given, receives every
    # processed exposure (for per-exposure analyses such as shift-and-stack).
    if records_override is not None:
        records = [r for r in records_override if not detectors or r.detector in detectors]
    else:
        records = find_images(resolved.search_position(), fov_arcmin=fov_arcmin, detectors=detectors)
    stats.discover_s = time.time() - t0
    stats.n_images_found = len(records)
    if not records:
        stats.log_report(t_start, n_fetched=0, n_frames_emitted=0)
        return

    if resolved.has_proper_motion:
        mean_mjd = float(np.mean([r.time_mjd for r in records]))
        centred_coord = resolved.propagate_to(Time(mean_mjd, format="mjd"))
    else:
        centred_coord = resolved.coord

    cfg = get_config()
    target_wcs, shape_out = make_target_wcs(centred_coord, fov_arcmin, cfg.pixel_scale_arcsec)

    grid = _GridContext(
        resolved=resolved,
        wcs=target_wcs,
        shape=shape_out,
        centre_ra=float(centred_coord.ra.deg),
        centre_dec=float(centred_coord.dec.deg),
        grid_scale_arcsec=cfg.pixel_scale_arcsec,
        native_scale_arcsec=cfg.native_pixel_scale_arcsec,
    )
    # where the pixels come from (app/localdata.py); every frame and the analysis say so
    grid.data_source = data_source
    grid.local_field = local_field
    grid.source_note = source_note

    groups_full = group_indices(records, mode=mode)
    # Cap *before* fetching, not after - the timing report showed ~97% of
    # request time is the fetch itself, so trimming here (not just at
    # combine time) is what actually buys speed. Evenly-spaced subsample
    # keeps the full time baseline (first/last exposure always kept)
    # rather than silently biasing toward one end of the epoch.
    groups = [cap_group_indices(idxs, max_exposures_per_group) for idxs in groups_full]

    record_to_group = {}
    needed = {}
    needed_indices: set[int] = set()
    for gi, idxs in enumerate(groups):
        needed[gi] = set(idxs)
        needed_indices.update(idxs)
        for i in idxs:
            record_to_group[id(records[i])] = gi
    index_of_record = {id(records[i]): i for i in needed_indices}

    records_to_fetch = [records[i] for i in sorted(needed_indices)]
    stats.n_images_capped_out = len(records) - len(records_to_fetch)

    pending: dict[int, dict[int, ProcessedCutout]] = {gi: {} for gi in range(len(groups))}
    emitted: set[int] = set()
    n_fetched = 0
    n_failed = 0
    n_frames_emitted = 0
    n_to_fetch = len(records_to_fetch)

    if report_progress:
        yield DownloadProgress(0, 0, n_to_fetch)

    t_fetch_start = time.time()
    fetch = cutout_source or fetch_cutouts
    for r in records_to_fetch:
        grid.requested_by_detector[r.detector] = grid.requested_by_detector.get(r.detector, 0) + 1
    # Only the live fetcher can report a download it gave up on; a test's
    # cutout_source keeps its plain (records, coord, fov_arcmin) signature.
    fetch_kwargs = {"yield_failures": True} if (report_progress and cutout_source is None) else {}
    for cutout in fetch(records_to_fetch, centred_coord, fov_arcmin=fov_arcmin, **fetch_kwargs):
        if isinstance(cutout, FetchFailure):
            n_failed += 1
            yield DownloadProgress(n_fetched, n_failed, n_to_fetch)
            continue
        n_fetched += 1
        if report_progress:
            yield DownloadProgress(n_fetched, n_failed, n_to_fetch)
        grid.fetched_by_detector[cutout.record.detector] = grid.fetched_by_detector.get(cutout.record.detector, 0) + 1
        if spectrotemporal:
            # native pixels, untouched: spectro-temporal photometry is done on these
            grid.native_cutouts.append(cutout)
        stats.record_fetch(cutout)

        t0 = time.time()
        p = process_cutout(cutout, centred_coord, target_wcs, shape_out)
        stats.reproject_s += time.time() - t0
        if exposure_sink is not None:
            exposure_sink.append(p)
        if shiftstack:
            grid.exposures.append(p)

        ridx = index_of_record[id(cutout.record)]
        gi = record_to_group[id(cutout.record)]
        pending[gi][ridx] = p

        if gi not in emitted and set(pending[gi].keys()) >= needed[gi]:
            emitted.add(gi)
            members = [pending[gi][i] for i in sorted(pending[gi].keys())]
            n_frames_emitted += 1
            yield _combine_and_encode(gi, len(groups), members, stats, grid=grid)
    stats.fetch_wall_s = time.time() - t_fetch_start

    # Groups that never fully completed (some fetch(es) failed after
    # retries) are still emitted, best-effort, once the stream ends — the
    # brief wants "if it's not detectable, say so" rather than silently
    # dropping a partial epoch.
    for gi in range(len(groups)):
        if gi not in emitted and pending[gi]:
            members = [pending[gi][i] for i in sorted(pending[gi].keys())]
            n_frames_emitted += 1
            yield _combine_and_encode(gi, len(groups), members, stats, partial=True, grid=grid)

    stats.log_report(t_start, n_fetched, n_frames_emitted)

    # Images exist here but not one cutout could be downloaded (IRSA busy):
    # report that as a failure, so the browser doesn't mistake an empty
    # stream for "no SPHEREx images at this position".
    if n_frames_emitted == 0:
        raise DiscoveryError(
            f"found {len(records)} image(s) but every cutout download failed (IRSA busy) - try again shortly"
        )

    if analyze and grid.collected:
        t0 = time.time()
        try:
            analysis = run_analysis(grid, fov_arcmin, mode)
            if shiftstack:
                analysis["shift_stack"] = run_shift_stack(grid)
            if spectrotemporal:
                # Isolated: a failure here must never discard the frames analysis above.
                try:
                    analysis["spectrotemporal"] = run_spectrotemporal(grid, analysis)
                except Exception as exc:  # noqa: BLE001
                    log.exception("spectro-temporal analysis failed")
                    analysis["spectrotemporal"] = {"error": f"{type(exc).__name__}: {exc}", "sources": []}
        except Exception as exc:  # noqa: BLE001 - analysis must never take the frames down with it
            log.exception("analysis failed")
            analysis = {"type": "analysis", "error": f"{type(exc).__name__}: {exc}", "series": []}
        analysis["seconds"] = time.time() - t0
        yield analysis


class _GridContext:
    """Everything needed to locate the target on the common grid at any
    epoch, and to tell the client exactly how that grid maps to the sky."""

    def __init__(self, resolved, wcs, shape, centre_ra, centre_dec, grid_scale_arcsec, native_scale_arcsec):
        self.resolved = resolved
        self.wcs = wcs
        self.shape = shape
        self.centre_ra = centre_ra
        self.centre_dec = centre_dec
        self.grid_scale_arcsec = grid_scale_arcsec
        self.native_scale_arcsec = native_scale_arcsec
        # where the pixels come from (app/localdata.py); set by stream_frames
        self.data_source = "irsa_live"
        self.local_field = None
        self.source_note = ""
        # (series key, SeriesFrame) for every emitted frame, kept for the
        # end-of-stream analysis. Grids are small (a 5' field is 99x99).
        self.collected: list[tuple[str, SeriesFrame]] = []
        # Individual processed exposures, kept only when a shift-and-stack
        # search was requested (each is one small grid image + variance).
        self.exposures: list = []
        # Native (not resampled) cutouts, kept only for the spectro-temporal analysis.
        self.native_cutouts: list = []
        self.requested_by_detector: dict = {}
        self.fetched_by_detector: dict = {}

    def wcs_metadata(self) -> dict:
        # FITS conventions: crpix is 1-based. cdelt in degrees. Array row 0 is
        # the SOUTH edge (cdelt2 > 0) and column 0 the EAST edge (cdelt1 < 0),
        # so a north-up/east-left display must draw row h-1 at the top.
        w = self.wcs.wcs
        return {
            "ctype": [str(c) for c in w.ctype],
            "crval": [float(v) for v in w.crval],
            "crpix": [float(v) for v in w.crpix],
            "cdelt": [float(v) for v in w.cdelt],
        }

    def target_pixel(self, mjd: float) -> tuple[float, float, bool]:
        """0-based (x, y) of the target at `mjd`; third value says whether
        proper motion was applied (False = fixed position)."""
        if self.resolved.has_proper_motion:
            coord = self.resolved.propagate_to(Time(mjd, format="mjd"))
            moved = True
        else:
            coord = self.resolved.coord
            moved = False
        x, y = self.wcs.world_to_pixel(coord)
        return float(x), float(y), moved


class _TimingStats:
    """Accumulates where the time went for one /api/frames request, so a
    slow run can be diagnosed from the numbers instead of guessed at.
    fetch_cutouts runs cutouts across a thread pool, so `fetch_wall_s`
    (elapsed clock time) and the sum of individual fetch times can differ
    a lot - that gap *is* the parallelism actually achieved."""

    def __init__(self, resolve_s: float = 0.0):
        self.resolve_s = resolve_s
        self.discover_s = 0.0
        self.n_images_found = 0
        self.n_images_capped_out = 0
        self.fetch_wall_s = 0.0
        self.fetch_bytes_total = 0
        self.fetch_seconds_sum = 0.0  # sum of individual successful fetches
        self.slowest_fetch_s = 0.0
        self.slowest_fetch_id = ""
        self.reproject_s = 0.0
        self.combine_s = 0.0
        self.encode_s = 0.0

    def record_fetch(self, cutout) -> None:
        self.fetch_bytes_total += cutout.downloaded_bytes
        self.fetch_seconds_sum += cutout.fetch_seconds
        if cutout.fetch_seconds > self.slowest_fetch_s:
            self.slowest_fetch_s = cutout.fetch_seconds
            self.slowest_fetch_id = cutout.record.obs_id

    def log_report(self, t_start: float, n_fetched: int, n_frames_emitted: int) -> None:
        total_s = time.time() - t_start
        mb = self.fetch_bytes_total / (1024 * 1024)
        effective_parallelism = self.fetch_seconds_sum / self.fetch_wall_s if self.fetch_wall_s > 0 else 0.0
        log.info(
            "request timing report:\n"
            "  resolve (SIMBAD/Sesame): %.2fs\n"
            "  discover (SIA query):    %.2fs -> %d image(s) found, %d skipped by quality cap\n"
            "  fetch cutouts:            %.1fs wall-clock, %d ok, %.1f MB total, "
            "sum-of-individual=%.1fs (~%.1fx effective parallelism), slowest=%s %.1fs\n"
            "  reproject (align):        %.2fs total (%.3fs/image avg)\n"
            "  combine (stack):          %.2fs total\n"
            "  encode (quantize+b64):    %.2fs total\n"
            "  TOTAL:                    %.1fs, %d frame(s) emitted",
            self.resolve_s,
            self.discover_s,
            self.n_images_found,
            self.n_images_capped_out,
            self.fetch_wall_s,
            n_fetched,
            mb,
            self.fetch_seconds_sum,
            effective_parallelism,
            self.slowest_fetch_id or "-",
            self.slowest_fetch_s,
            self.reproject_s,
            (self.reproject_s / n_fetched) if n_fetched else 0.0,
            self.combine_s,
            self.encode_s,
            total_s,
            n_frames_emitted,
        )


def _combine_and_encode(
    group_index: int,
    n_groups: int,
    members: list[ProcessedCutout],
    stats: "_TimingStats",
    partial: bool = False,
    grid: Optional["_GridContext"] = None,
) -> EncodedFrame:
    t0 = time.time()
    image, variance, n_contrib = combine_inverse_variance(
        [m.image for m in members], [m.variance for m in members]
    )
    times = [m.record.time_mjd for m in members]

    # Measure BEFORE any display hole-filling: a saturated/masked core must
    # make the measurement refuse, not be measured off interpolated wings
    # (docs/AUDIT.md C4).
    target_meta = None
    phot_meta = None
    if grid is not None:
        tx, ty, moved = grid.target_pixel(float(np.mean(times)))
        inflation = get_config().aperture_variance_inflation
        phot = aperture_photometry(
            image, variance, tx, ty, grid.grid_scale_arcsec, variance_inflation=inflation
        )
        target_meta = {"x": tx, "y": ty, "proper_motion_applied": moved}
        phot_meta = phot.to_dict()
        waves_here = [m.central_wavelength_um for m in members if m.central_wavelength_um is not None]
        grid.collected.append((
            "+".join(sorted({m.record.detector for m in members})),
            SeriesFrame(image=image, variance=variance, time_mjd=float(np.mean(times)),
                        wavelength_um=float(np.mean(waves_here)) if waves_here else None),
        ))

    # Fill small, isolated NaN holes FOR DISPLAY ONLY (see fill_small_gaps's
    # docstring). The filled pixel indices are sent to the client so the
    # readout can say "interpolated", and no measurement uses them.
    before_nan = ~np.isfinite(image)
    image, n_gap_filled = fill_small_gaps(image)
    filled_idx = np.flatnonzero(before_nan & np.isfinite(image))
    stats.combine_s += time.time() - t0

    # Pixels built partly from saturated / non-linear native pixels in at
    # least one combined exposure (pipeline.saturation_on_grid). Sent as a
    # bit-packed mask so the client can refuse to colour them.
    sat_mask = np.zeros(image.shape, dtype=bool)
    for m in members:
        if m.saturation is not None and m.saturation.shape == image.shape:
            sat_mask |= m.saturation

    contamination: dict[str, int] = {}
    for m in members:
        for name, count in m.contamination.items():
            contamination[name] = contamination.get(name, 0) + count
    waves = [m.central_wavelength_um for m in members if m.central_wavelength_um is not None]
    bws = [m.bandwidth_um for m in members if m.bandwidth_um is not None]

    frame = Frame(
        group_index=group_index,
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

    t0 = time.time()
    extra = {
        "n_groups": n_groups,
        "partial": partial,
        "n_gap_filled_pixels": n_gap_filled,
        # Capped so a pathological frame can't bloat the stream; the count above is exact.
        "gap_filled_indices": filled_idx[:2000].tolist(),
        "photometry": phot_meta,
        "target_pixel": target_meta,
        # Row-major over the grid, 1 bit per pixel, little-endian bit order
        # (numpy.packbits(..., bitorder="little")).
        "saturation_mask_b64": base64.b64encode(np.packbits(sat_mask.ravel(), bitorder="little").tobytes()).decode("ascii"),
        "n_saturation_flagged_pixels": int(sat_mask.sum()),
    }
    if grid is not None:
        extra["data_source"] = getattr(grid, "data_source", "irsa_live")
        if getattr(grid, "source_note", ""):
            extra["data_source_note"] = grid.source_note
        extra["grid_wcs"] = grid.wcs_metadata()
        extra["grid_centre_deg"] = [grid.centre_ra, grid.centre_dec]
        extra["grid_scale_arcsec"] = grid.grid_scale_arcsec
        extra["native_scale_arcsec"] = grid.native_scale_arcsec
    encoded = encode_frame(frame, extra_metadata=extra)
    stats.encode_s += time.time() - t0
    return encoded


def run_analysis(grid: "_GridContext", fov_arcmin: float, mode: str) -> dict:
    """Detect, track, cross-match and rank the sources of every detector
    series collected during this request."""
    # One cone per catalogue covers the whole square field (half-diagonal + margin).
    radius_deg = (fov_arcmin / 60.0) * 0.5 * 1.5
    catalog_errors = []
    catalog_source = {}

    def _query(fn, name):
        """Live cone search, or the locally downloaded rows (app/localdata.py):
        directly when the images are local too, as a fallback otherwise."""
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        from . import localdata

        field = getattr(grid, "local_field", None)
        if field is None and localdata.fallback_allowed():
            field = localdata.field_for(SkyCoord(grid.centre_ra * u.deg, grid.centre_dec * u.deg))
        use_local_first = field is not None and (getattr(grid, "data_source", "") == localdata.LOCAL
                                                 or localdata.data_mode() == "cached")
        if not use_local_first:
            try:
                rows = fn(grid.centre_ra, grid.centre_dec, radius_deg)
                catalog_source[name] = localdata.LIVE
                return rows
            except CatalogError as exc:
                if field is None or not localdata.fallback_allowed():
                    catalog_errors.append(str(exc))
                    return None
                live_error = str(exc)
        else:
            live_error = None
        try:
            with localdata.local_catalogues(field):
                rows = fn(grid.centre_ra, grid.centre_dec, radius_deg)
            catalog_source[name] = localdata.LOCAL
            if live_error:
                catalog_errors.append(f"{live_error} (using the {name} rows downloaded for '{field.name}' on {field.retrieved} instead)")
            return rows
        except CatalogError as exc:
            catalog_errors.append(f"{live_error + '; ' if live_error else ''}local {name} rows: {exc}")
            return None

    gaia = _query(gaia_sources, "gaia_dr3")
    catwise = _query(catwise_sources, "catwise2020")

    by_series: dict[str, list[SeriesFrame]] = {}
    for key, frame in grid.collected:
        by_series.setdefault(key, []).append(frame)

    series_out = []
    for key in sorted(by_series):
        res = analyze_series(
            by_series[key], grid.wcs, grid.grid_scale_arcsec, grid.native_scale_arcsec,
            grid.centre_ra, grid.centre_dec, gaia=gaia, catwise=catwise, series=key,
            cfg=DiscoveryConfig(aperture_variance_inflation=get_config().aperture_variance_inflation),
        )
        if "+" in key:
            res["warning"] = (
                "This series stacks several detectors, so its frames mix very different wavelengths. "
                "Brightness changes here are not evidence of variability. Use per-detector grouping."
            )
        series_out.append(res)

    return {
        "type": "analysis",
        "grouping": mode,
        "catalog_errors": catalog_errors,
        "catalog_source": catalog_source,
        "data_source": getattr(grid, "data_source", "irsa_live"),
        "data_source_note": getattr(grid, "source_note", ""),
        "statement": (
            "Automated measurements for human inspection. 'candidate' and 'unmatched' are not discovery claims; "
            "no object class is estimated."
        ),
        "series": series_out,
    }


SHIFT_STACK_CONTROLS = 20


def _encode_map(arr: np.ndarray) -> dict:
    from .encode import quantize_uint16
    import base64

    q, scale, offset = quantize_uint16(np.asarray(arr, dtype=np.float32))
    return {"width": int(arr.shape[1]), "height": int(arr.shape[0]), "scale": scale, "offset": offset,
            "payload_b64": base64.b64encode(q.tobytes()).decode("ascii")}


def run_shift_stack(grid: "_GridContext") -> dict:
    """Trial-motion shift-and-stack per detector, on the single exposures
    fetched for this request (app/shiftstack.py). For each detector:

      * blind: search over position and velocity, with SHIFT_STACK_CONTROLS
        time-shuffled control searches;
      * forced: if the target has a known proper motion, the stack along
        that motion at its predicted position, with its controls;
      * one S/N image (in units of the stack's own measured noise), stacked
        at the velocity of the strongest moving candidate, else at the
        target's catalogue motion, else at zero motion.

    A search, not a confirmation: see `assessment` and docs/PHASE2.md.
    """
    from .shiftstack import ShiftStackConfig, StackFrame, forced, prepare, search, stack_at

    out = {
        "statement": (
            "Shift-and-stack is a search technique. A candidate that passes its controls is worth inspecting; "
            "it is not an independent confirmation of a detection or a discovery."
        ),
        "n_controls": SHIFT_STACK_CONTROLS,
        "series": [],
    }
    by_det: dict[str, list] = {}
    for p in grid.exposures:
        by_det.setdefault(p.record.detector, []).append(p)

    for det in sorted(by_det):
        exps = sorted(by_det[det], key=lambda p: p.record.time_mjd)
        entry = {"series": det, "n_exposures": len(exps)}
        if len(exps) < 3:
            entry["assessment"] = "not_run"
            entry["assessment_text"] = f"Only {len(exps)} exposure(s): too few for a shift-and-stack search."
            out["series"].append(entry)
            continue
        frames = [StackFrame(image=p.image, variance=p.variance, time_mjd=p.record.time_mjd) for p in exps]
        cfg = ShiftStackConfig(n_controls=SHIFT_STACK_CONTROLS)
        res = search(frames, grid.grid_scale_arcsec, cfg)
        t_ref = res["epoch_ref_mjd"]
        for c in res["candidates"]:
            sky = grid.wcs.pixel_to_world(c["x"], c["y"])
            c["ra_deg"], c["dec_deg"] = float(sky.ra.deg), float(sky.dec.deg)
        entry["blind"] = res

        target_v = None
        if grid.resolved.has_proper_motion:
            tx, ty, _ = grid.target_pixel(t_ref)
            target_v = (grid.resolved.pmra / 1000.0, grid.resolved.pmdec / 1000.0)
            f = forced(frames, grid.grid_scale_arcsec, tx, ty, *target_v, t_ref=t_ref, cfg=cfg)
            f["predicted_xy"] = [tx, ty]
            f["predicted_pm_arcsec_yr"] = list(target_v)
            entry["forced"] = f

        moving = [c for c in res["candidates"] if c["moving"]]
        p_fa = res["controls"]["false_alarm_probability"]
        n_c = res["controls"]["n"]
        floor = 1.0 / (n_c + 1) if n_c else None
        if not moving:
            entry["assessment"] = "no_moving_candidate"
            entry["assessment_text"] = (
                f"No peak passed the moving-source criteria (S/N >= {cfg.detect_snr:.0f}, motion >= "
                f"{res['min_distinguishable_speed_arcsec_yr']:.1f}\"/yr, S/N^2 gain over zero motion >= {cfg.min_delta_chi2})."
            )
        elif p_fa is not None and floor is not None and p_fa <= floor + 1e-12:
            entry["assessment"] = "passes_controls"
            entry["assessment_text"] = (
                f"None of the {n_c} time-shuffled control searches produced a moving peak as strong as the best "
                f"real one. Empirical false-alarm probability <= {floor:.3f} (the smallest value {n_c} controls "
                "can establish, not zero). Tentative until confirmed by other means."
            )
        else:
            entry["assessment"] = "tentative"
            entry["assessment_text"] = (
                f"A moving peak was found, but {('some' if p_fa is None else round(p_fa * (n_c + 1)) - 1)} of the "
                f"{n_c} time-shuffled controls produced one at least as strong (false-alarm probability "
                f"{'n/a' if p_fa is None else f'{p_fa:.2f}'}). Not distinguishable from chance."
            )

        if moving:
            best = max(moving, key=lambda c: c["snr"])
            v_disp, why = (best["pm_east_arcsec_yr"], best["pm_north_arcsec_yr"]), "strongest moving candidate"
        elif target_v is not None:
            v_disp, why = target_v, "target's catalogue motion"
        else:
            v_disp, why = (0.0, 0.0), "zero motion (ordinary stack)"
        snr_map, _ = stack_at(prepare(frames, grid.grid_scale_arcsec, cfg), v_disp[0], v_disp[1], t_ref=t_ref)
        entry["image"] = {
            **_encode_map(snr_map),
            "units": "S/N (matched filter, in units of this stack's measured noise)",
            "pm_east_arcsec_yr": float(v_disp[0]),
            "pm_north_arcsec_yr": float(v_disp[1]),
            "velocity_source": why,
            "epoch_mjd": float(t_ref),
            "note": ("Every exposure is shifted so that a source moving with this velocity stays at its position "
                     "at the reference epoch. Sources with that motion add up into a point; static stars are "
                     "smeared along the opposite direction. Same grid and orientation as the frames (north up, "
                     "east left)."),
        }
        out["series"].append(entry)
    return out


MAX_SPECTRO_SOURCES = 20


def _sep_arcsec(ra1, dec1, ra2, dec2):
    import math
    dra = ((ra1 - ra2 + 180.0) % 360.0 - 180.0) * math.cos(math.radians(0.5 * (dec1 + dec2)))
    return math.hypot(dra, dec1 - dec2) * 3600.0


def run_spectrotemporal(grid: "_GridContext", analysis: dict) -> dict:
    """F(lambda, t) for the target and the tracked sources, on the native
    cutouts fetched for this request (app/spectrotemporal.py)."""
    from .spectrotemporal import (analyse_source, assign_measurement_epochs, measure_exposure,
                                  reference_distribution, source_positions_at)

    cutouts = sorted(grid.native_cutouts, key=lambda c: c.record.time_mjd)
    retrieval_failed = {d: grid.requested_by_detector.get(d, 0) - grid.fetched_by_detector.get(d, 0)
                        for d in grid.requested_by_detector}

    # ---- sources: the target, then tracked sources (deduplicated across detector series)
    sources = []
    if grid.resolved.has_proper_motion:
        def target_pos(mjd):
            c = grid.resolved.propagate_to(Time(mjd, format="mjd"))
            return float(c.ra.deg), float(c.dec.deg)
        motion_note = "catalogue proper motion applied per exposure"
    else:
        def target_pos(mjd):
            return float(grid.resolved.coord.ra.deg), float(grid.resolved.coord.dec.deg)
        motion_note = "fixed position"
    sources.append({"id": "target", "label": grid.resolved.name or "Target", "kind": "target",
                    "position": target_pos, "motion": motion_note, "linked_tracks": []})

    tracks = []
    for series in analysis.get("series", []):
        for t in series.get("tracks", []):
            if t["n_epochs"] >= 2 and t["detection_snr_median"] >= 5:
                tracks.append((series["series"], t))
    tracks.sort(key=lambda st: st[1]["rank"])
    t_ref_c = grid.resolved.propagate_to(Time(float(np.mean([c.record.time_mjd for c in cutouts])), format="mjd")) \
        if cutouts else grid.resolved.coord
    refs = []
    for series_key, t in tracks:
        if any(_sep_arcsec(t["ra_deg"], t["dec_deg"], s.get("ra", 1e9), s.get("dec", 1e9)) < 4.0 for s in sources[1:]):
            for s in sources[1:]:
                if _sep_arcsec(t["ra_deg"], t["dec_deg"], s["ra"], s["dec"]) < 4.0:
                    s["linked_tracks"].append({"series": series_key, "rank": t["rank"], "candidate_id": t.get("candidate_id")})
            continue
        if _sep_arcsec(t["ra_deg"], t["dec_deg"], float(t_ref_c.ra.deg), float(t_ref_c.dec.deg)) < 6.0:
            sources[0]["linked_tracks"].append({"series": series_key, "rank": t["rank"], "candidate_id": t.get("candidate_id")})
            continue
        m = t["motion"]
        moving = m.get("pm_significance") is not None and m["pm_significance"] >= 3
        pe, pn = (m["pmra_mas_yr"], m["pmdec_mas_yr"]) if moving else (None, None)
        ra0, dec0, ep = t["ra_deg"], t["dec_deg"], t["epoch_mjd"]
        src = {"id": f"{series_key}#{t['rank']}", "label": f"{series_key} #{t['rank']} ({t['label']})", "kind": "track",
               "ra": ra0, "dec": dec0,
               "position": (lambda mjd, ra0=ra0, dec0=dec0, ep=ep, pe=pe, pn=pn: source_positions_at(ra0, dec0, ep, pe, pn, mjd)),
               "motion": "measured proper motion applied" if moving else "fixed position (no significant motion)",
               "linked_tracks": [{"series": series_key, "rank": t["rank"], "candidate_id": t.get("candidate_id")}]}
        is_ref = (not moving and t["detection_snr_median"] >= 30 and t["reliability"] == "high"
                  and not {"near_bright_star", "core_masked", "near_edge", "ambiguous_link"} & set(t["flags"]))
        if is_ref:
            refs.append(src)
        if len(sources) < MAX_SPECTRO_SOURCES + 1:
            sources.append(src)

    def measure(src):
        meas = []
        for c in cutouts:
            t_mid = c.hdul["IMAGE"].header.get("MJD-AVG", c.record.time_mjd)
            ra, dec = src["position"](float(t_mid))
            meas.append(measure_exposure(c, src["id"], ra, dec, fits_ref=c.record.obs_id))
        epochs = assign_measurement_epochs(meas)
        return meas, epochs

    ref_results = []
    for src in refs:
        meas, epochs = measure(src)
        ref_results.append(analyse_source(meas, epochs))
    reference = reference_distribution(ref_results)

    out_sources = []
    for src in sources:
        meas, epochs = measure(src)
        res = analyse_source(meas, epochs, reference=reference)
        out_sources.append({"id": src["id"], "label": src["label"], "kind": src["kind"], "motion": src["motion"],
                            "is_reference_star": src in refs, "linked_tracks": src["linked_tracks"], **res})

    ref_summary = {k: {"n": v["n"], "median": float(np.median(v["values"])),
                       "p05": float(np.percentile(v["values"], 5)), "p95": float(np.percentile(v["values"], 95)),
                       "values": v["values"]} for k, v in reference.items()}
    # fit:<det>: reference stars' reduced chi^2 of the shape model (p95 = adequacy limit);
    # other keys: reference stars' p-values of each metric
    return {
        "statement": ("Measured fluxes per exposure (one wavelength each) and interpretable change statistics. "
                      "Exposures at different times are not a simultaneous spectrum; only dichroic detector pairs "
                      "(D1/D4, D2/D5, D3/D6) are observed at the same instant. p-values describe consistency with "
                      "'no change' given the stated errors, not the probability that a source is astrophysically variable."),
        "n_exposures": len(cutouts),
        "retrieval_failed_by_detector": retrieval_failed,
        "reference": {"n_stars": len(refs), "metrics": ref_summary,
                      "selection": "stationary (PM < 3 sigma), detection S/N >= 30, reliability high, no edge/blend/ringing/masked flags, >= 2 epochs"},
        "sources": out_sources,
    }
