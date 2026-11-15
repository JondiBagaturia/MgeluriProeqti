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
import math
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


# Pauses between attempts of one SIA query (so 3 attempts in ~6s at most).
_SIA_RETRY_SLEEPS_S = (2, 4)

# astroquery / pyvo send the search requests without a read timeout. When
# IRSA accepts the connection and then never answers (seen 2026-11-15, in
# between its 502/503 spells), the call blocks for good and the page sits on
# "Fetching from IRSA..." forever. These bound it: one attempt, and the whole
# search including its retries. A normal answer takes 3-25 s.
_SIA_ATTEMPT_TIMEOUT_S = 40.0
_SIA_TOTAL_BUDGET_S = 75.0
_COLLECTIONS_TIMEOUT_S = 15.0


def _call_with_timeout(fn, timeout_s: float, what: str):
    """Run `fn()` in a helper thread and give up waiting after `timeout_s`.

    The helper thread cannot be interrupted: if it is stuck on a dead
    connection it stays behind (daemon, idle) until the socket closes. That
    is the price of not hanging the request it was serving."""
    box: dict = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - handed to the caller below
            box["error"] = exc

    worker = threading.Thread(target=run, daemon=True, name=f"irsa-{what}")
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise TimeoutError(f"IRSA did not answer the {what} within {timeout_s:.0f} s")
    if "error" in box:
        raise box["error"]
    return box["value"]


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
        rows = _call_with_timeout(
            lambda: Irsa.list_collections(servicetype="SIA"), _COLLECTIONS_TIMEOUT_S, "collection list"
        )
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
    # IRSA's search front end drops out intermittently ("502 Proxy Error",
    # "Unable to access the capabilities endpoint") and often answers again
    # seconds later, so a failed query gets a few quick retries here. An
    # outage that outlasts them is reported to the browser, which keeps
    # retrying on a slower, visible schedule (frontend/app.js).
    rows = None
    deadline = time.monotonic() + _SIA_TOTAL_BUDGET_S
    for attempt in range(len(_SIA_RETRY_SLEEPS_S) + 1):
        try:
            rows = _call_with_timeout(
                lambda: Irsa.query_sia(pos=(coord, radius), collection=collections),
                max(5.0, min(_SIA_ATTEMPT_TIMEOUT_S, deadline - time.monotonic())),
                "image search",
            )
            break
        except Exception as exc:  # noqa: BLE001
            out_of_time = time.monotonic() + _SIA_RETRY_SLEEPS_S[min(attempt, len(_SIA_RETRY_SLEEPS_S) - 1)] + 5.0 > deadline
            if attempt == len(_SIA_RETRY_SLEEPS_S) or out_of_time:
                raise DiscoveryError(f"SIA query failed: {exc}") from exc
            log.warning("SIA query failed (attempt %d), retrying in %ds: %s", attempt + 1, _SIA_RETRY_SLEEPS_S[attempt], exc)
            time.sleep(_SIA_RETRY_SLEEPS_S[attempt])

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
    downloaded_bytes: int = 0  # full response size, incl. the EPSF table we then strip
    fetch_seconds: float = 0.0  # this cutout's successful attempt only, not retries/backoff


# IRSA cuts a square of the requested size out of the *detector's* pixel
# grid, and the detector sits at whatever angle the spacecraft had on the
# sky for that exposure (27 deg from north for eps Ind B, for instance). A
# rotated square of side `fov` leaves the corners of our north-up `fov`
# grid empty - up to 17% of it. Asking for the diagonal instead (sqrt(2),
# plus a little slack for the pixel grid) covers the whole grid at any
# angle. It costs little: nearly all of a cutout's bytes are the EPSF
# table, which does not grow with the requested size.
_CUTOUT_PAD = math.sqrt(2) * 1.03


def _cutout_url(record: ImageRecord, coord: SkyCoord, fov_arcmin: float) -> str:
    size_deg = fov_arcmin * _CUTOUT_PAD / 60.0
    return f"{record.access_url}?center={coord.ra.deg},{coord.dec.deg}d&size={size_deg}"


class _RetryableHTTPStatus(Exception):
    """Internal marker: an HTTP status we've configured as transient
    (503/502/504 - "server busy", not "your request is wrong"). Kept
    separate from `requests.HTTPError` so a genuinely bad request (a 4xx
    like the 422 IRSA sometimes returns for a malformed/out-of-range
    cutout request) fails fast instead of burning 4 retries + backoff on
    something that was never going to succeed."""


def _fetch_one(
    record: ImageRecord,
    coord: SkyCoord,
    fov_arcmin: float,
    session: requests.Session,
    cancelled: Optional[threading.Event] = None,
) -> Cutout:
    cfg = get_config().fetch
    url = _cutout_url(record, coord, fov_arcmin)
    last_exc: Optional[Exception] = None

    for attempt in range(cfg.max_retries + 1):
        # Set by fetch_cutouts when its caller stopped listening (the user
        # pressed Stop, or closed the page): don't start or retry a download
        # nobody will use.
        if cancelled is not None and cancelled.is_set():
            raise DiscoveryError(f"Cutout fetch cancelled: {url}")
        try:
            t0 = time.time()
            with _global_fetch_semaphore:
                resp = session.get(url, timeout=(cfg.connect_timeout_s, cfg.read_timeout_s))
            if resp.status_code in cfg.retry_status_codes:
                raise _RetryableHTTPStatus(f"{resp.status_code} {resp.text[:100]}")
            resp.raise_for_status()  # non-retryable 4xx/5xx -> plain HTTPError, not retried below
            n_bytes = len(resp.content)
            full = fits.open(io.BytesIO(resp.content))
            # Keep only the extensions we use; drop EPSF (and anything else
            # unexpected) before this ever leaves memory for the pipeline.
            trimmed = fits.HDUList(
                [full[0]] + [full[ext].copy() for ext in KEPT_EXTENSIONS if ext in full]
            )
            full.close()
            return Cutout(record=record, hdul=trimmed, downloaded_bytes=n_bytes, fetch_seconds=time.time() - t0)
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
                if cancelled is not None:
                    cancelled.wait(sleep_s)  # a sleep that a cancel cuts short
                else:
                    time.sleep(sleep_s)
            continue
        except requests.HTTPError as exc:
            # A real client/server error (e.g. 422) - retrying with the same
            # URL won't change the outcome, so fail fast instead of wasting
            # the full retry budget.
            raise DiscoveryError(f"Cutout fetch permanently failed (not retrying): {url} ({exc})") from exc
    raise DiscoveryError(f"Cutout fetch failed after {cfg.max_retries} retries: {url} ({last_exc})")


@dataclass
class FetchFailure:
    """A cutout that could not be downloaded (only yielded on request, see
    fetch_cutouts)."""

    record: ImageRecord
    error: str


def fetch_cutouts(records: list[ImageRecord], coord: SkyCoord, fov_arcmin: float, yield_failures: bool = False):
    """Fetch cutouts for `records` with bounded concurrency, yielding each
    Cutout as soon as it's ready (order not guaranteed) so callers can start
    streaming/combining before the whole batch finishes.

    A download that fails after its retries is skipped. With
    `yield_failures=True` a FetchFailure is yielded in its place, so a
    caller counting progress can tell "still downloading" from "gave up".

    If the caller stops iterating early (the browser cancelled the request
    or went away, which closes this generator), downloads that have not
    started are dropped and running ones stop retrying - IRSA is not asked
    for images nobody is waiting for.
    """
    cfg = get_config().fetch
    session = requests.Session()
    cancelled = threading.Event()
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = {
            pool.submit(_fetch_one, record, coord, fov_arcmin, session, cancelled): record
            for record in records
        }
        try:
            for future in as_completed(futures):
                record = futures[future]
                try:
                    yield future.result()
                except Exception as exc:  # noqa: BLE001
                    log.warning("Skipping %s after fetch failure: %s", record.obs_id, exc)
                    if yield_failures:
                        yield FetchFailure(record=record, error=str(exc))
                    continue
        finally:
            # Normal end: everything is done and this is a no-op. Early end
            # (generator closed): cancel what has not started, and flag the
            # rest so they give up at their next retry.
            cancelled.set()
            for future in futures:
                future.cancel()
