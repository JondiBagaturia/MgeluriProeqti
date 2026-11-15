"""FastAPI backend: a stateless processing pipe. No disk, no database (see
PLAN.md). Every endpoint either answers from a fast metadata-only IRSA call
or streams processed pixels straight from memory and discards them.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .coords import parse_position, parse_coordinates_only, CoordParseError, ResolvedPosition
from .irsa import find_images, discover_collections, DiscoveryError
from .stream import stream_frames, DownloadProgress
from .spectrum import compute_spectrum
from .cache import TTLDedupeCache
from .config import get_config
from .jsonsafe import clean
from . import store
from . import localdata
from pydantic import BaseModel
from dataclasses import asdict
from functools import lru_cache
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="SPHEREx Blink API", version="0.1.0")

# Public, read-only GET endpoints over public data, no credentials/cookies
# used — open CORS is deliberate so the static frontend can be hosted
# anywhere (a CDN, file://, a different domain than the API).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# The only cache in this service: a few-minutes TTL, in-memory, de-dupes
# identical concurrent/near-concurrent requests. Not a results store.
_dedupe = TTLDedupeCache(ttl_seconds=180.0)


def _parse_detectors(detectors: Optional[str]) -> Optional[list[str]]:
    if not detectors:
        return None
    return [d.strip().upper() for d in detectors.split(",") if d.strip()]


@lru_cache(maxsize=512)
def _parse_cached(position: str, pm_lookup: bool) -> ResolvedPosition:
    # Name resolution hits SIMBAD (seconds); every endpoint re-resolves the
    # same string, so memoise successful lookups for the process lifetime.
    mode = localdata.data_mode()
    if mode == "cached":
        # no network: a downloaded field's saved SIMBAD result, else coordinates
        local = localdata.resolve_name(position)
        if local is not None:
            return local
        try:
            return parse_coordinates_only(position)
        except CoordParseError as exc:
            raise CoordParseError(f"Offline mode: {position!r} is not coordinates and not a downloaded field. "
                                  f"{localdata.available_text()}") from exc
    try:
        return parse_position(position, pm_lookup=pm_lookup)
    except CoordParseError as exc:
        local = localdata.resolve_name(position) if mode == "auto" else None
        if local is None:
            raise
        local.note = f"Name resolvers unavailable ({str(exc)[:100]}...). " + (local.note or "")
        return local


def _resolve_or_400(position: str, pm_lookup: bool = True) -> ResolvedPosition:
    try:
        return _parse_cached(position.strip(), pm_lookup)
    except CoordParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validate_fov(fov: Optional[float]) -> float:
    cfg = get_config()
    value = fov if fov is not None else cfg.fov.default_arcmin
    if not (cfg.fov.min_arcmin <= value <= cfg.fov.max_arcmin):
        raise HTTPException(
            status_code=400,
            detail=f"fov must be between {cfg.fov.min_arcmin} and {cfg.fov.max_arcmin} arcmin",
        )
    return value


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/local-data")
def local_data():
    """Data mode (SPHEREX_DATA_MODE: live / auto / cached) and the fields
    downloaded for offline use (app/localdata.py)."""
    return {"mode": localdata.data_mode(), "root": str(localdata.data_root()),
            "fields": [f.describe() for f in localdata.fields()]}


@app.get("/api/releases")
async def releases():
    """Currently-discovered SPHEREx collections (see app/irsa.py's runtime
    discovery). The frontend folds this into its IndexedDB cache key so a
    new data release invalidates cached frames automatically, without the
    browser ever hard-coding a release name either."""
    if localdata.data_mode() == "cached":
        # offline: the releases the downloaded exposures came from (no network)
        collections = sorted({e["collection"] for f in localdata.fields() for e in f.manifest["exposures"]})
        return JSONResponse({"collections": collections, "data_mode": "cached"})
    collections = await run_in_threadpool(discover_collections)
    return JSONResponse({"collections": collections}, headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/resolve")
async def resolve(
    position: str = Query(..., description="RA/Dec (sexagesimal or decimal) or object name"),
    pm_lookup: bool = Query(True, description="For typed coordinates: look for a CatWISE2020 fast mover whose track passes through them"),
):
    """Fast, cutout-free: resolve a query to RA/Dec (+ proper motion if a
    named object has one). The frontend calls this as the user types/submits,
    before deciding on FOV/detectors."""
    resolved = await run_in_threadpool(_resolve_or_400, position, pm_lookup)
    return {
        "ra_deg": resolved.coord.ra.deg,
        "dec_deg": resolved.coord.dec.deg,
        "source": resolved.source,
        "name": resolved.name,
        "note": resolved.note,
        "ref_epoch_mjd": float(resolved.ref_epoch.mjd),
        "has_proper_motion": resolved.has_proper_motion,
        "pmra_mas_yr": resolved.pmra,
        "pmdec_mas_yr": resolved.pmdec,
        "parallax_mas": resolved.parallax_mas,
    }


@app.get("/api/images")
async def images(
    position: str = Query(...),
    pm_lookup: bool = Query(True, description="For typed coordinates: look for a CatWISE2020 fast mover whose track passes through them"),
    fov: Optional[float] = Query(None, description="arcmin"),
    detectors: Optional[str] = Query(None, description="Comma-separated, e.g. D6,D2"),
):
    """Metadata-only list of matching images (no pixels fetched) — this is
    the fast call the UI uses to show "N images found" before streaming."""
    resolved = await run_in_threadpool(_resolve_or_400, position, pm_lookup)
    fov_arcmin = _validate_fov(fov)
    det_list = _parse_detectors(detectors)

    cache_key = f"images:{resolved.coord.ra.deg:.6f}:{resolved.coord.dec.deg:.6f}:{fov_arcmin}:{det_list}"

    async def compute():
        try:
            choice = await run_in_threadpool(localdata.choose_source, resolved, fov_arcmin, det_list)
        except (DiscoveryError, localdata.NoLocalData) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        records = choice.records or []
        return [
            {"obs_id": r.obs_id, "collection": r.collection, "detector": r.detector, "time_mjd": r.time_mjd}
            for r in records
        ]

    result = await _dedupe.get_or_compute(cache_key, compute)
    return JSONResponse(result, headers={"Cache-Control": "public, max-age=300"})


class _ClosingStreamingResponse(StreamingResponse):
    """A StreamingResponse over a plain generator that closes the generator
    the moment the response ends - finished, or abandoned because the client
    went away (Stop pressed in the page, tab closed).

    Starlette stops iterating on a disconnect but leaves the generator
    suspended until it is garbage collected, which in practice was only after
    every image had been downloaded anyway. Closing it is what lets
    app/irsa.fetch_cutouts drop the downloads it has not started."""

    def __init__(self, content, **kwargs):
        super().__init__(content, **kwargs)
        self._generator = content

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Off the event loop: closing waits for the downloads already in
            # flight. A no-op when the stream simply finished.
            threading.Thread(target=_close_when_idle, args=(self._generator,), daemon=True).start()


def _close_when_idle(gen) -> None:
    """Close a plain generator from a helper thread. If another thread is
    inside next() on it right now, wait until that call returns: a generator
    cannot be closed while it is executing."""
    while True:
        try:
            gen.close()
            return
        except ValueError:  # "generator already executing"
            time.sleep(0.2)


@app.get("/api/frames")
async def frames(
    position: str = Query(...),
    pm_lookup: bool = Query(True, description="For typed coordinates: look for a CatWISE2020 fast mover whose track passes through them"),
    fov: Optional[float] = Query(None, description="arcmin"),
    detectors: Optional[str] = Query(None, description="Comma-separated, e.g. D6,D2"),
    # Default is per-detector: stacking different detectors mixes very
    # different wavelengths into one frame (docs/AUDIT.md C1). "epoch"
    # (cross-detector stack) remains available as a display aid only.
    group: str = Query("detector_epoch", pattern="^(epoch|detector_epoch|single)$"),
    quality: Optional[str] = Query(None, pattern="^(low|mid|high)$", description="Max exposures/group: low=fast, high=full quality"),
    shiftstack: bool = Query(False, description="Also run the trial-motion shift-and-stack search on the fetched exposures (slower: ~20 control searches per detector)"),
    spectrotemporal: bool = Query(False, description="Also measure F(lambda, t) per exposure on native pixels for the target and tracked sources (app/spectrotemporal.py)"),
    progress: bool = Query(False, description="Also send download-progress lines between the frames"),
):
    """NDJSON stream: one processed, combined frame per line, as soon as
    each is ready. Each line carries width/height/scale/offset + a base64
    uint16 payload + metadata — the browser does all stretching client-side.

    `quality` trades S/N for speed by capping exposures fetched per group
    (see app/pipeline.py's cap_group_indices) - applied before fetching,
    since that's ~97% of request time, not after.

    With `progress=1` the stream also carries lines of the form
    {"progress": {"downloaded": 41, "failed": 1, "total": 70}} - one after
    discovery and one per finished cutout download - so a client can show
    how far along it is before the first frame is complete. Off by default,
    so clients that expect every line to be a frame keep working."""
    t0 = time.time()
    resolved = await run_in_threadpool(_resolve_or_400, position, pm_lookup)
    resolve_seconds = time.time() - t0
    fov_arcmin = _validate_fov(fov)
    det_list = _parse_detectors(detectors)
    max_exposures = get_config().quality.max_exposures(quality)

    def lines_for(choice, sent):
        """NDJSON lines of one data source (app/localdata.py): live IRSA or the
        locally downloaded copy. `sent` counts frames already streamed."""
        if choice.kind == localdata.LOCAL:
            yield (json.dumps({"type": "data_source", "data_source": choice.kind, "note": choice.note,
                               "fov_arcmin": choice.fov_arcmin, "field": choice.field.describe()}) + "\n").encode("utf-8")
        for encoded in stream_frames(resolved, choice.fov_arcmin, det_list, group, resolve_seconds, max_exposures,
                                     shiftstack=shiftstack, spectrotemporal=spectrotemporal,
                                     records_override=choice.records, cutout_source=choice.cutout_source,
                                     data_source=choice.kind, local_field=choice.field, source_note=choice.note,
                                     report_progress=progress):
            if isinstance(encoded, DownloadProgress):
                line = {
                    "progress": {
                        "downloaded": encoded.n_downloaded,
                        "failed": encoded.n_failed,
                        "total": encoded.n_total,
                    }
                }
                yield (json.dumps(line) + "\n").encode("utf-8")
                continue
            if isinstance(encoded, dict):  # the end-of-stream analysis record
                _store_candidates(encoded, position)
                yield (json.dumps(clean(encoded), allow_nan=False) + "\n").encode("utf-8")
                continue
            line = {
                "width": encoded.width,
                "height": encoded.height,
                "scale": encoded.scale,
                "offset": encoded.offset,
                "payload_b64": encoded.payload_b64,
                **encoded.metadata,
            }
            sent["frames"] += 1
            yield (json.dumps(clean(line), allow_nan=False) + "\n").encode("utf-8")

    def generate():
        try:
            # live IRSA, or the locally downloaded data (app/localdata.py, SPHEREX_DATA_MODE)
            choice = localdata.choose_source(resolved, fov_arcmin, det_list)
            sent = {"frames": 0}
            try:
                yield from lines_for(choice, sent)
            except DiscoveryError as exc:
                # IRSA found the images but could not deliver any of them (busy):
                # in auto mode, use the downloaded copy if this field has one.
                if choice.kind != localdata.LIVE or sent["frames"] or not localdata.fallback_allowed():
                    raise
                try:
                    local = localdata.local_choice(resolved.search_position(), fov_arcmin, det_list,
                                                   f"IRSA could not deliver the images ({localdata._short(exc)}).")
                except localdata.NoLocalData:
                    raise exc
                yield from lines_for(local, sent)
        except (DiscoveryError, localdata.NoLocalData) as exc:
            yield (json.dumps({"error": str(exc)}) + "\n").encode("utf-8")
        except Exception as exc:  # noqa: BLE001 - a mid-stream crash must reach the user, not just cut the stream
            log.exception("frame stream failed")
            yield (json.dumps({"error": f"Processing failed: {type(exc).__name__}: {exc}"}) + "\n").encode("utf-8")

    return _ClosingStreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        # no-store: the stream can carry a transient failure (a catalogue 502, a
        # partial epoch). The browser keeps good results in IndexedDB itself.
        headers={"Cache-Control": "no-store"},
    )


MAX_STORED_PER_SERIES = 25


def _store_candidates(analysis: dict, field_query: str) -> None:
    """Give the top-ranked tracks of each series a persistent id so
    volunteers can classify them. A database problem is reported in the
    analysis record, never raised."""
    try:
        for series in analysis.get("series", []):
            for track in series.get("tracks", [])[:MAX_STORED_PER_SERIES]:
                track["candidate_id"] = store.upsert_candidate(track, field_query)
    except Exception as exc:  # noqa: BLE001
        log.exception("could not store candidates")
        analysis["store_error"] = f"{type(exc).__name__}: {exc}"


class ClassificationIn(BaseModel):
    label: str
    voter: str


@app.get("/api/candidates")
def candidates(
    limit: int = Query(50, ge=1, le=200),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    label: Optional[str] = Query(None, pattern="^(candidate|catalogued|uncertain|unverified)$"),
):
    """Stored tracked sources, highest anomaly score first, with volunteer agreement."""
    return JSONResponse(clean({"labels": store.LABEL_TEXT, "candidates": store.list_candidates(limit, min_score, label)}))


@app.get("/api/candidates/{candidate_id}")
def candidate(candidate_id: str, voter: Optional[str] = Query(None)):
    found = store.get_candidate(candidate_id, voter)
    if found is None:
        raise HTTPException(status_code=404, detail="Unknown candidate")
    return JSONResponse(clean(found))


@app.post("/api/candidates/{candidate_id}/classifications")
def classify(candidate_id: str, body: ClassificationIn):
    """Record one volunteer's label. One vote per voter token; voting again replaces it."""
    try:
        agreement = store.classify(candidate_id, body.voter, body.label)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown candidate") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(clean({"your_label": body.label, "classifications": agreement}))


@app.get("/api/spectrum")
async def spectrum(
    position: str = Query(..., description="RA/Dec or object name - the exact clicked pixel, not necessarily the search target"),
    pm_lookup: bool = Query(True, description="For typed coordinates: look for a CatWISE2020 fast mover whose track passes through them"),
    fov: Optional[float] = Query(None, description="arcmin"),
    detectors: Optional[str] = Query(None, description="Comma-separated; omit for all 6 (the point of a spectrum)"),
):
    """Phase 4 (bonus): forced photometry at one position across every
    matching cutout, native-pixel and native-wavelength per exposure (see
    app/spectrum.py) - not reprojected, so proper motion is applied at each
    exposure's own epoch rather than a shared mean epoch like /api/frames."""
    resolved = await run_in_threadpool(_resolve_or_400, position, pm_lookup)
    fov_arcmin = _validate_fov(fov)
    det_list = _parse_detectors(detectors)
    try:
        choice = await run_in_threadpool(localdata.choose_source, resolved, fov_arcmin, det_list)
        points = await run_in_threadpool(compute_spectrum, resolved, fov_arcmin, det_list, None,
                                         choice.records if choice.kind == localdata.LOCAL else None,
                                         choice.cutout_source)
    except (DiscoveryError, localdata.NoLocalData) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return JSONResponse(
        clean(
            {
                "measured_at": {
                    "source": resolved.source,
                    "name": resolved.name,
                    "proper_motion_applied_per_exposure": resolved.has_proper_motion,
                },
                "units": {"flux": "mJy (aperture flux, no aperture correction)", "wavelength": "um"},
                "caveat": (
                    "Points come from different observation times. SPHEREx does not observe all "
                    "wavelengths simultaneously, so for a variable source compare points only "
                    "within one epoch."
                ),
                "points": [asdict(p) for p in points],
                "data_source": choice.kind,
                "data_source_note": choice.note,
            }
        ),
        headers={"Cache-Control": "public, max-age=300"},
    )


# Serve the static frontend from this same process, so "one uvicorn command"
# really does run the whole tool (per the brief's ops requirement). Mounted
# last so it never shadows the /api/* routes above — Starlette matches
# routes in registration order.
_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
