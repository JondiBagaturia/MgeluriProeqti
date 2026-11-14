"""FastAPI backend: a stateless processing pipe. No disk, no database (see
PLAN.md). Every endpoint either answers from a fast metadata-only IRSA call
or streams processed pixels straight from memory and discards them.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .coords import parse_position, CoordParseError, ResolvedPosition
from .irsa import find_images, discover_collections, DiscoveryError
from .stream import stream_frames
from .spectrum import compute_spectrum
from .cache import TTLDedupeCache
from .config import get_config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="SPHEREx Blink API", version="0.1.0")

# Public, read-only GET endpoints over public data, no credentials/cookies
# used — open CORS is deliberate so the static frontend can be hosted
# anywhere (a CDN, file://, a different domain than the API).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# The only cache in this service: a few-minutes TTL, in-memory, de-dupes
# identical concurrent/near-concurrent requests. Not a results store.
_dedupe = TTLDedupeCache(ttl_seconds=180.0)


def _parse_detectors(detectors: Optional[str]) -> Optional[list[str]]:
    if not detectors:
        return None
    return [d.strip().upper() for d in detectors.split(",") if d.strip()]


def _resolve_or_400(position: str) -> ResolvedPosition:
    try:
        return parse_position(position)
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


@app.get("/api/releases")
async def releases():
    """Currently-discovered SPHEREx collections (see app/irsa.py's runtime
    discovery). The frontend folds this into its IndexedDB cache key so a
    new data release invalidates cached frames automatically, without the
    browser ever hard-coding a release name either."""
    collections = await run_in_threadpool(discover_collections)
    return JSONResponse({"collections": collections}, headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/resolve")
async def resolve(position: str = Query(..., description="RA/Dec (sexagesimal or decimal) or object name")):
    """Fast, cutout-free: resolve a query to RA/Dec (+ proper motion if a
    named object has one). The frontend calls this as the user types/submits,
    before deciding on FOV/detectors."""
    resolved = await run_in_threadpool(_resolve_or_400, position)
    return {
        "ra_deg": resolved.coord.ra.deg,
        "dec_deg": resolved.coord.dec.deg,
        "source": resolved.source,
        "name": resolved.name,
        "has_proper_motion": resolved.has_proper_motion,
        "pmra_mas_yr": resolved.pmra,
        "pmdec_mas_yr": resolved.pmdec,
        "parallax_mas": resolved.parallax_mas,
    }


@app.get("/api/images")
async def images(
    position: str = Query(...),
    fov: Optional[float] = Query(None, description="arcmin"),
    detectors: Optional[str] = Query(None, description="Comma-separated, e.g. D6,D2"),
):
    """Metadata-only list of matching images (no pixels fetched) — this is
    the fast call the UI uses to show "N images found" before streaming."""
    resolved = await run_in_threadpool(_resolve_or_400, position)
    fov_arcmin = _validate_fov(fov)
    det_list = _parse_detectors(detectors)

    cache_key = f"images:{resolved.coord.ra.deg:.6f}:{resolved.coord.dec.deg:.6f}:{fov_arcmin}:{det_list}"

    async def compute():
        try:
            records = await run_in_threadpool(find_images, resolved.coord, fov_arcmin, det_list)
        except DiscoveryError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return [
            {"obs_id": r.obs_id, "collection": r.collection, "detector": r.detector, "time_mjd": r.time_mjd}
            for r in records
        ]

    result = await _dedupe.get_or_compute(cache_key, compute)
    return JSONResponse(result, headers={"Cache-Control": "public, max-age=300"})


@app.get("/api/frames")
async def frames(
    position: str = Query(...),
    fov: Optional[float] = Query(None, description="arcmin"),
    detectors: Optional[str] = Query(None, description="Comma-separated, e.g. D6,D2"),
    group: str = Query("epoch", pattern="^(epoch|detector_epoch|single)$"),
    quality: Optional[str] = Query(None, pattern="^(low|mid|high)$", description="Max exposures/group: low=fast, high=full quality"),
):
    """NDJSON stream: one processed, combined frame per line, as soon as
    each is ready. Each line carries width/height/scale/offset + a base64
    uint16 payload + metadata — the browser does all stretching client-side.

    `quality` trades S/N for speed by capping exposures fetched per group
    (see app/pipeline.py's cap_group_indices) - applied before fetching,
    since that's ~97% of request time, not after."""
    t0 = time.time()
    resolved = await run_in_threadpool(_resolve_or_400, position)
    resolve_seconds = time.time() - t0
    fov_arcmin = _validate_fov(fov)
    det_list = _parse_detectors(detectors)
    max_exposures = get_config().quality.max_exposures(quality)

    def generate():
        try:
            for encoded in stream_frames(resolved, fov_arcmin, det_list, group, resolve_seconds, max_exposures):
                line = {
                    "width": encoded.width,
                    "height": encoded.height,
                    "scale": encoded.scale,
                    "offset": encoded.offset,
                    "payload_b64": encoded.payload_b64,
                    **encoded.metadata,
                }
                yield (json.dumps(line) + "\n").encode("utf-8")
        except DiscoveryError as exc:
            yield (json.dumps({"error": str(exc)}) + "\n").encode("utf-8")

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "public, max-age=300"},
    )


@app.get("/api/spectrum")
async def spectrum(
    position: str = Query(..., description="RA/Dec or object name - the exact clicked pixel, not necessarily the search target"),
    fov: Optional[float] = Query(None, description="arcmin"),
    detectors: Optional[str] = Query(None, description="Comma-separated; omit for all 6 (the point of a spectrum)"),
):
    """Phase 4 (bonus): forced photometry at one position across every
    matching cutout, native-pixel and native-wavelength per exposure (see
    app/spectrum.py) - not reprojected, so proper motion is applied at each
    exposure's own epoch rather than a shared mean epoch like /api/frames."""
    resolved = await run_in_threadpool(_resolve_or_400, position)
    fov_arcmin = _validate_fov(fov)
    det_list = _parse_detectors(detectors)
    try:
        points = await run_in_threadpool(compute_spectrum, resolved, fov_arcmin, det_list)
    except DiscoveryError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return JSONResponse(
        [
            {
                "time_mjd": p.time_mjd,
                "detector": p.detector,
                "wavelength_um": p.wavelength_um,
                "bandwidth_um": p.bandwidth_um,
                "flux": p.flux,
                "masked": p.masked,
            }
            for p in points
        ],
        headers={"Cache-Control": "public, max-age=300"},
    )


# Serve the static frontend from this same process, so "one uvicorn command"
# really does run the whole tool (per the brief's ops requirement). Mounted
# last so it never shadows the /api/* routes above — Starlette matches
# routes in registration order.
_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
