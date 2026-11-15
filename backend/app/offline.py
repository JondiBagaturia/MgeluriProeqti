"""Run the real pipeline on cutouts cached on disk instead of live IRSA.

The cache (backend/tests/data/real/<target>/, written by
scripts/cache_validation_cutouts.py) holds REAL SPHEREx observations. This
module only replaces the two network calls (image discovery and cutout
download); every processing step is the production code in app/stream.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time

from .coords import ResolvedPosition
from .encode import EncodedFrame
from .irsa import Cutout, ImageRecord
from .pipeline import ProcessedCutout
from .stream import stream_frames

REAL_DATA_DIR = Path(__file__).resolve().parents[1] / "tests" / "data" / "real"


@dataclass
class CachedRun:
    manifest: dict
    resolved: ResolvedPosition
    frames: list[dict] = field(default_factory=list)        # frame metadata, time-ordered within series
    analysis: Optional[dict] = None
    exposures: list[ProcessedCutout] = field(default_factory=list)


def load_manifest(target: str, root: Path = REAL_DATA_DIR) -> dict:
    return json.loads((root / target / "manifest.json").read_text())


def resolved_from_manifest(manifest: dict) -> ResolvedPosition:
    r = manifest["resolved"]
    return ResolvedPosition(
        coord=SkyCoord(r["ra_deg"] * u.deg, r["dec_deg"] * u.deg, frame="icrs"),
        ref_epoch=Time(r["ref_epoch_mjd"], format="mjd"),
        pmra=r["pmra_mas_yr"], pmdec=r["pmdec_mas_yr"], parallax_mas=r["parallax_mas"],
        name=r["name"], source=r["source"],
    )


def run_cached(
    target: str,
    mode: str = "detector_epoch",
    detectors: Optional[list[str]] = None,
    analyze: bool = True,
    root: Path = REAL_DATA_DIR,
    use_catalogs: bool = False,
    shiftstack: bool = False,
    spectrotemporal: bool = False,
) -> CachedRun:
    """`use_catalogs=False` (the default, for offline tests) skips the Gaia /
    CatWISE queries: every track then reports catalog_unavailable."""
    manifest = load_manifest(target, root)
    resolved = resolved_from_manifest(manifest)
    by_obs = {}
    records = []
    for e in manifest["exposures"]:
        rec = ImageRecord(access_url="cached://" + e["file"], collection=e["collection"], detector=e["detector"],
                          time_mjd=e["time_mjd"], obs_id=e["obs_id"])
        records.append(rec)
        by_obs[id(rec)] = root / target / e["file"]
    records.sort(key=lambda r: r.time_mjd)

    def source(recs, coord, fov_arcmin):
        for rec in recs:
            with fits.open(by_obs[id(rec)], memmap=False) as hdul:
                loaded = fits.HDUList([h.copy() for h in hdul])
            yield Cutout(record=rec, hdul=loaded)

    run = CachedRun(manifest=manifest, resolved=resolved)
    import app.stream as stream_mod

    saved = (stream_mod.gaia_sources, stream_mod.catwise_sources)
    if not use_catalogs:
        def _unavailable(*a, **k):
            raise stream_mod.CatalogError("catalogue queries disabled for offline run")
        stream_mod.gaia_sources = stream_mod.catwise_sources = _unavailable
    try:
        for item in stream_frames(resolved, manifest["fov_arcmin"], detectors, mode, analyze=analyze,
                                  records_override=records, cutout_source=source, exposure_sink=run.exposures,
                                  shiftstack=shiftstack, spectrotemporal=spectrotemporal):
            if isinstance(item, dict):
                run.analysis = item
            elif isinstance(item, EncodedFrame):
                run.frames.append(item.metadata)
    finally:
        stream_mod.gaia_sources, stream_mod.catwise_sources = saved
    run.frames.sort(key=lambda m: ("+".join(m["detectors"]), m["time_mjd_mean"]))
    if run.analysis and "spectrotemporal" in run.analysis:
        # Exposures that could not be downloaded when the cache was built.
        run.analysis["spectrotemporal"]["retrieval_failed_at_caching"] = (
            manifest.get("n_requested", 0) - manifest.get("n_cached", 0))
    return run


def series_frames(run: CachedRun, detector: str) -> list[dict]:
    return [m for m in run.frames if m["detectors"] == [detector]]
