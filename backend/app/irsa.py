"""Discover SPHEREx images covering a position, and fetch cutouts.

PLAN.md Phase 0 decisions baked in here:
  - SIA is the primary discovery call (measured 3-6x faster than TAP).
  - Collections are discovered at runtime, never hard-coded to one release.
  - Cutout fetches run through a bounded ThreadPoolExecutor with retry +
    backoff that explicitly treats HTTP 503 ("Server busy, retry later.") as
    retryable, because it showed up on real, unloaded single requests.
  - The EPSF extension is dropped immediately after fetch (it was ~97.7% of
    a cutout's bytes and doesn't shrink with the requested FOV) and never
    passed further down the pipeline.
"""
from __future__ import annotations

import io
import time
import random
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from astroquery.ipac.irsa import Irsa

from .config import get_config

log = logging.getLogger(__name__)

# Per-request concurrency is bounded by ThreadPoolExecutor(max_workers)
# below, but that's per *request* - if several users hit the backend at
# once, each gets their own pool. This semaphore is process-wide, so the
# brief's "plus a global cap across users" actually holds regardless of
# how many concurrent requests are in flight.
_global_fetch_semaphore = threading.BoundedSemaphore(get_config().fetch.global_concurrency_cap)

# Extensions we actually use. EPSF is deliberately excluded (see module
# docstring); WCS-WAVE is tiny (~720B) and kept for the per-pixel wavelength
# lookup needed by Phase 4 spectra.
KEPT_EXTENSIONS = ("IMAGE", "FLAGS", "VARIANCE", "ZODI", "WCS-WAVE")


class DiscoveryError(Exception):
    pass


@dataclass
class ImageRecord:
    """One SPHEREx exposure's metadata, before any pixels are fetched."""

    access_url: str          # SIA access_url (no ?center=&size= yet)
    collection: str           # e.g. "spherex_dr1"
    detector: str             # "D1".."D6"
    time_mjd: float            # t_min, start of exposure
    obs_id: str


def discover_collections() -> list[str]:
    """SIA collections matching the configured prefix, minus excluded
    pre-aggregated products. Falls back to the configured static list only
    if the discovery call itself raises."""
    cfg = get_config()
    try:
        rows = Irsa.list_collections(servicetype="SIA")
        names = [str(r["collection"]) for r in rows]
    except Exception as exc:  # noqa: BLE001
        log.warning("SIA collection discovery failed (%s); using fallback list", exc)
        return list(cfg.irsa.fallback_collections)

    prefix = cfg.irsa.sia_collection_prefix
    matches = [n for n in names if n.startswith(prefix)]
    matches = [n for n in matches if not any(n.endswith(suf) for suf in cfg.irsa.exclude_suffixes)]
    if not matches:
        log.warning("No live collections matched prefix %r; using fallback list", prefix)
        return list(cfg.irsa.fallback_collections)
    return matches


def find_images(coord: SkyCoord, fov_arcmin: float, detectors: Optional[list[str]] = None) -> list[ImageRecord]:
    """SIA query for every SPHEREx image covering `coord` within `fov_arcmin`.

    Optionally restricted to a subset of detectors (D1..D6) client-side,
    since SIA's `band` filter is wavelength-based, not detector-ID-based,
    and we already know the D->wavelength mapping (config/collections.yaml).
    """
    collections = discover_collections()
    radius = (fov_arcmin / 2.0) * u.arcmin
    try:
        rows = Irsa.query_sia(pos=(coord, radius), collection=collections)
    except Exception as exc:  # noqa: BLE001
        raise DiscoveryError(f"SIA query failed: {exc}") from exc

    records = []
    for row in rows:
        band = str(row["energy_bandpassname"])  # "SPHEREx-D6"
        det = band.replace("SPHEREx-", "") if band.startswith("SPHEREx-") else band
        if detectors and det not in detectors:
            continue
        records.append(
            ImageRecord(
                access_url=str(row["access_url"]),
                collection=str(row["obs_collection"]),
                detector=det,
                time_mjd=float(row["t_min"]),
                obs_id=str(row["obs_id"]),
            )
        )
    records.sort(key=lambda r: r.time_mjd)
    return records


@dataclass
class Cutout:
    record: ImageRecord
    hdul: fits.HDUList  # only KEPT_EXTENSIONS present


def _cutout_url(record: ImageRecord, coord: SkyCoord, fov_arcmin: float) -> str:
    size_deg = fov_arcmin / 60.0
    return f"{record.access_url}?center={coord.ra.deg},{coord.dec.deg}d&size={size_deg}"


class _RetryableHTTPStatus(Exception):
    """Internal marker: an HTTP status we've configured as transient
    (503/502/504 - "server busy", not "your request is wrong"). Kept
    separate from `requests.HTTPError` so a genuinely bad request (a 4xx
    like the 422 IRSA sometimes returns for a malformed/out-of-range
    cutout request) fails fast instead of burning 4 retries + backoff on
    something that was never going to succeed."""


def _fetch_one(record: ImageRecord, coord: SkyCoord, fov_arcmin: float, session: requests.Session) -> Cutout:
    cfg = get_config().fetch
    url = _cutout_url(record, coord, fov_arcmin)
    last_exc: Optional[Exception] = None

    for attempt in range(cfg.max_retries + 1):
        try:
            with _global_fetch_semaphore:
                resp = session.get(url, timeout=(cfg.connect_timeout_s, cfg.read_timeout_s))
            if resp.status_code in cfg.retry_status_codes:
                raise _RetryableHTTPStatus(f"{resp.status_code} {resp.text[:100]}")
            resp.raise_for_status()  # non-retryable 4xx/5xx -> plain HTTPError, not retried below
            full = fits.open(io.BytesIO(resp.content))
            # Keep only the extensions we use; drop EPSF (and anything else
            # unexpected) before this ever leaves memory for the pipeline.
            trimmed = fits.HDUList(
                [full[0]] + [full[ext].copy() for ext in KEPT_EXTENSIONS if ext in full]
            )
            full.close()
            return Cutout(record=record, hdul=trimmed)
        except (
            requests.Timeout,
            requests.ConnectionError,
            ConnectionResetError,
            _RetryableHTTPStatus,
        ) as exc:
            last_exc = exc
            if attempt < cfg.max_retries:
                sleep_s = cfg.backoff_base_s * (2 ** attempt) + random.uniform(0, 0.5)
                log.debug("cutout fetch retry %d/%d for %s: %s", attempt + 1, cfg.max_retries, url, exc)
                time.sleep(sleep_s)
            continue
        except requests.HTTPError as exc:
            # A real client/server error (e.g. 422) - retrying with the same
            # URL won't change the outcome, so fail fast instead of wasting
            # the full retry budget.
            raise DiscoveryError(f"Cutout fetch permanently failed (not retrying): {url} ({exc})") from exc
    raise DiscoveryError(f"Cutout fetch failed after {cfg.max_retries} retries: {url} ({last_exc})")


def fetch_cutouts(records: list[ImageRecord], coord: SkyCoord, fov_arcmin: float):
    """Fetch cutouts for `records` with bounded concurrency, yielding each
    Cutout as soon as it's ready (order not guaranteed) so callers can start
    streaming/combining before the whole batch finishes.
    """
    cfg = get_config().fetch
    session = requests.Session()
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = {
            pool.submit(_fetch_one, record, coord, fov_arcmin, session): record
            for record in records
        }
        for future in as_completed(futures):
            record = futures[future]
            try:
                yield future.result()
            except Exception as exc:  # noqa: BLE001
                log.warning("Skipping %s after fetch failure: %s", record.obs_id, exc)
                continue
