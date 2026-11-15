"""Run the real pipeline on cutouts cached on disk instead of live IRSA.

The cache (backend/tests/data/real/<target>/, written by
scripts/cache_validation_cutouts.py) holds REAL SPHEREx observations. This
module only replaces the two network calls (image discovery and cutout
download); every processing step is the production code in app/stream.py.
"""
from __future__ import annotations

import json
import math
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


def cached_cone(target: str, table: str, ra: float, dec: float, radius_deg: float, root: Path = REAL_DATA_DIR) -> str:
    """Cone search against the cached catalogue rows (CSV text, same format as IRSA /SCS)."""
    import io
    from astropy.table import Table
    from .catalogs import CatalogError

    path = root / target / "catalogs" / f"{table}.csv"
    if not path.exists():
        raise CatalogError(f"no cached {table} rows for {target} (run scripts/cache_catalogs.py)")
    meta = json.loads(path.with_suffix(".json").read_text())
    cra, cdec = meta["centre_deg"]
    if math.hypot((ra - cra) * math.cos(math.radians(dec)), dec - cdec) + radius_deg > meta["radius_deg"] + 1e-6:
        raise CatalogError(f"requested cone exceeds the cached {table} cone for {target}")
    t = Table.read(path, format="ascii.csv")
    rc, dc = ("ra", "dec") if "ra" in t.colnames else ("ra_pm", "dec_pm")
    if len(t):
        sep = np.hypot((np.array(t[rc], float) - ra) * math.cos(math.radians(dec)), np.array(t[dc], float) - dec)
        t = t[sep <= radius_deg]
    buf = io.StringIO()
    t.write(buf, format="ascii.csv")
    return buf.getvalue()


def run_cached(
    target: str,
    mode: str = "detector_epoch",
    detectors: Optional[list[str]] = None,
    analyze: bool = True,
    root: Path = REAL_DATA_DIR,
    use_catalogs=False,
    shiftstack: bool = False,
    spectrotemporal: bool = False,
    cutout_transform=None,
) -> CachedRun:
    """`use_catalogs`: False (default) = catalogues unavailable; "cached" = the
    REAL catalogue rows cached next to the cutouts (scripts/cache_catalogs.py),
    no network; True = live IRSA queries.
    `use_catalogs=False` (the default, for offline tests) skips the Gaia /
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
            c = Cutout(record=rec, hdul=loaded)
            # validation only (app/injection.py): modifies the in-memory copy, never the cached file
            yield cutout_transform(c) if cutout_transform is not None else c

    run = CachedRun(manifest=manifest, resolved=resolved)
    import app.stream as stream_mod

    import app.catalogs as catalogs_mod

    saved = (stream_mod.gaia_sources, stream_mod.catwise_sources)
    saved_override = catalogs_mod.CONE_OVERRIDE
    if use_catalogs == "cached":
        catalogs_mod.CONE_OVERRIDE = lambda table, ra, dec, radius: cached_cone(target, table, ra, dec, radius, root)
    elif not use_catalogs:
        def _unavailable(*a, **k):
            raise stream_mod.CatalogError("catalogue queries disabled for offline run")
        stream_mod.gaia_sources = stream_mod.catwise_sources = _unavailable
    from .localdata import no_fallback

    try:
        with no_fallback():
            for item in stream_frames(resolved, manifest["fov_arcmin"], detectors, mode, analyze=analyze,
                                      records_override=records, cutout_source=source, exposure_sink=run.exposures,
                                      shiftstack=shiftstack, spectrotemporal=spectrotemporal):
                if isinstance(item, dict):
                    run.analysis = item
                elif isinstance(item, EncodedFrame):
                    run.frames.append(item.metadata)
    finally:
        stream_mod.gaia_sources, stream_mod.catwise_sources = saved
        catalogs_mod.CONE_OVERRIDE = saved_override
    run.frames.sort(key=lambda m: ("+".join(m["detectors"]), m["time_mjd_mean"]))
    if run.analysis and "spectrotemporal" in run.analysis:
        # Exposures that could not be downloaded when the cache was built.
        run.analysis["spectrotemporal"]["retrieval_failed_at_caching"] = (
            manifest.get("n_requested", 0) - manifest.get("n_cached", 0))
    return run


def series_frames(run: CachedRun, detector: str) -> list[dict]:
    return [m for m in run.frames if m["detectors"] == [detector]]
