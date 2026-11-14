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

from typing import Iterator, Optional

import numpy as np
from astropy.time import Time

from .coords import ResolvedPosition
from .irsa import find_images, fetch_cutouts, DiscoveryError
from .pipeline import (
    make_target_wcs,
    process_cutout,
    group_indices,
    combine_inverse_variance,
    Frame,
    ProcessedCutout,
)
from .encode import encode_frame, EncodedFrame
from .config import get_config


def stream_frames(
    resolved: ResolvedPosition,
    fov_arcmin: float,
    detectors: Optional[list[str]],
    mode: str,
) -> Iterator[EncodedFrame]:
    records = find_images(resolved.coord, fov_arcmin=fov_arcmin, detectors=detectors)
    if not records:
        return

    if resolved.has_proper_motion:
        mean_mjd = float(np.mean([r.time_mjd for r in records]))
        centred_coord = resolved.propagate_to(Time(mean_mjd, format="mjd"))
    else:
        centred_coord = resolved.coord

    cfg = get_config()
    target_wcs, shape_out = make_target_wcs(centred_coord, fov_arcmin, cfg.pixel_scale_arcsec)

    groups = group_indices(records, mode=mode)
    record_to_group = {}
    needed = {}
    for gi, idxs in enumerate(groups):
        needed[gi] = set(idxs)
        for i in idxs:
            record_to_group[id(records[i])] = gi
    index_of_record = {id(r): i for i, r in enumerate(records)}

    pending: dict[int, dict[int, ProcessedCutout]] = {gi: {} for gi in range(len(groups))}
    emitted: set[int] = set()

    for cutout in fetch_cutouts(records, centred_coord, fov_arcmin=fov_arcmin):
        p = process_cutout(cutout, centred_coord, target_wcs, shape_out)
        ridx = index_of_record[id(cutout.record)]
        gi = record_to_group[id(cutout.record)]
        pending[gi][ridx] = p

        if gi not in emitted and set(pending[gi].keys()) >= needed[gi]:
            emitted.add(gi)
            members = [pending[gi][i] for i in sorted(pending[gi].keys())]
            yield _combine_and_encode(gi, len(groups), members)

    # Groups that never fully completed (some fetch(es) failed after
    # retries) are still emitted, best-effort, once the stream ends — the
    # brief wants "if it's not detectable, say so" rather than silently
    # dropping a partial epoch.
    for gi in range(len(groups)):
        if gi not in emitted and pending[gi]:
            members = [pending[gi][i] for i in sorted(pending[gi].keys())]
            yield _combine_and_encode(gi, len(groups), members, partial=True)


def _combine_and_encode(
    group_index: int, n_groups: int, members: list[ProcessedCutout], partial: bool = False
) -> EncodedFrame:
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
    return encode_frame(frame, extra_metadata={"n_groups": n_groups, "partial": partial})
