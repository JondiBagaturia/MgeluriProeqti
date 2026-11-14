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
import time
from typing import Iterator, Optional

import numpy as np
from astropy.time import Time

from .coords import ResolvedPosition
from .irsa import find_images, fetch_cutouts, DiscoveryError
from .pipeline import (
    make_target_wcs,
    process_cutout,
    group_indices,
    cap_group_indices,
    combine_inverse_variance,
    Frame,
    ProcessedCutout,
)
from .encode import encode_frame, EncodedFrame
from .config import get_config

log = logging.getLogger(__name__)


def stream_frames(
    resolved: ResolvedPosition,
    fov_arcmin: float,
    detectors: Optional[list[str]],
    mode: str,
    resolve_seconds: float = 0.0,
    max_exposures_per_group: Optional[int] = None,
) -> Iterator[EncodedFrame]:
    t_start = time.time()
    stats = _TimingStats(resolve_s=resolve_seconds)

    t0 = time.time()
    records = find_images(resolved.coord, fov_arcmin=fov_arcmin, detectors=detectors)
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
    n_frames_emitted = 0

    t_fetch_start = time.time()
    for cutout in fetch_cutouts(records_to_fetch, centred_coord, fov_arcmin=fov_arcmin):
        n_fetched += 1
        stats.record_fetch(cutout)

        t0 = time.time()
        p = process_cutout(cutout, centred_coord, target_wcs, shape_out)
        stats.reproject_s += time.time() - t0

        ridx = index_of_record[id(cutout.record)]
        gi = record_to_group[id(cutout.record)]
        pending[gi][ridx] = p

        if gi not in emitted and set(pending[gi].keys()) >= needed[gi]:
            emitted.add(gi)
            members = [pending[gi][i] for i in sorted(pending[gi].keys())]
            n_frames_emitted += 1
            yield _combine_and_encode(gi, len(groups), members, stats)
    stats.fetch_wall_s = time.time() - t_fetch_start

    # Groups that never fully completed (some fetch(es) failed after
    # retries) are still emitted, best-effort, once the stream ends — the
    # brief wants "if it's not detectable, say so" rather than silently
    # dropping a partial epoch.
    for gi in range(len(groups)):
        if gi not in emitted and pending[gi]:
            members = [pending[gi][i] for i in sorted(pending[gi].keys())]
            n_frames_emitted += 1
            yield _combine_and_encode(gi, len(groups), members, stats, partial=True)

    stats.log_report(t_start, n_fetched, n_frames_emitted)


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
) -> EncodedFrame:
    t0 = time.time()
    image, variance, n_contrib = combine_inverse_variance(
        [m.image for m in members], [m.variance for m in members]
    )
    stats.combine_s += time.time() - t0

    contamination: dict[str, int] = {}
    for m in members:
        for name, count in m.contamination.items():
            contamination[name] = contamination.get(name, 0) + count
    waves = [m.central_wavelength_um for m in members if m.central_wavelength_um is not None]
    bws = [m.bandwidth_um for m in members if m.bandwidth_um is not None]
    times = [m.record.time_mjd for m in members]

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
    encoded = encode_frame(frame, extra_metadata={"n_groups": n_groups, "partial": partial})
    stats.encode_s += time.time() - t0
    return encoded
