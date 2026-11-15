"""Locally downloaded REAL SPHEREx data as an offline / fallback data source.

When IRSA (image search, cutouts, Gaia/CatWISE cone search) or SIMBAD is
unavailable, the app can run its unchanged pipeline on data already on disk:
the real cutouts and real catalogue rows cached under backend/tests/data/real
(scripts/cache_validation_cutouts.py, scripts/cache_catalogs.py). Nothing here is
synthetic; every result says where its data came from.

Mode (environment variable SPHEREX_DATA_MODE):
  live    IRSA and SIMBAD only (the previous behaviour)
  auto    default: try IRSA/SIMBAD first; if a request fails and the position
          is one of the downloaded fields, use the local copy instead
  cached  never touch the network: downloaded fields only

SPHEREX_LOCAL_DATA overrides the data directory.

Limits of a local field: only its cached detectors, its field of view (5'
for the bundled fields: a larger request is reduced to it), the exposures
cached per detector per epoch, and the catalogue rows within the cached cone.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time

log = logging.getLogger(__name__)

MODES = ("live", "auto", "cached")
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "tests" / "data" / "real"
# How close a requested position must be to a downloaded field: its cached
# cutout centre, or the target's catalogue position (a high-proper-motion
# star's J2000 position can be several arcmin from where it is now).
MATCH_RADIUS_ARCMIN = 2.0

LIVE = "irsa_live"
LOCAL = "local_cache"


def data_mode() -> str:
    mode = os.environ.get("SPHEREX_DATA_MODE", "auto").strip().lower()
    if mode not in MODES:
        log.warning("SPHEREX_DATA_MODE=%r is not one of %s; using 'auto'", mode, MODES)
        return "auto"
    return mode


def data_root() -> Path:
    return Path(os.environ.get("SPHEREX_LOCAL_DATA") or DEFAULT_ROOT)


@dataclass(frozen=True)
class LocalField:
    name: str
    path: Path
    manifest: dict

    @property
    def fov_arcmin(self) -> float:
        return float(self.manifest["fov_arcmin"])

    @property
    def detectors(self) -> list[str]:
        return sorted({e["detector"] for e in self.manifest["exposures"]})

    @property
    def centre(self) -> SkyCoord:
        ra, dec = self.manifest["cutout_centre_deg"]
        return SkyCoord(ra * u.deg, dec * u.deg)

    @property
    def target(self) -> SkyCoord:
        r = self.manifest["resolved"]
        return SkyCoord(r["ra_deg"] * u.deg, r["dec_deg"] * u.deg)

    @property
    def retrieved(self) -> str:
        return str(self.manifest.get("retrieved_utc", "?"))[:10]

    def has_catalogues(self) -> bool:
        return (self.path / "catalogs").is_dir()

    def describe(self) -> dict:
        return {"name": self.name, "query": self.manifest.get("query"), "detectors": self.detectors,
                "fov_arcmin": self.fov_arcmin, "retrieved_utc": self.manifest.get("retrieved_utc"),
                "n_exposures": len(self.manifest["exposures"]), "catalogues": self.has_catalogues()}


@lru_cache(maxsize=4)
def _fields(root: str) -> tuple[LocalField, ...]:
    out = []
    p = Path(root)
    if not p.is_dir():
        return ()
    for d in sorted(p.iterdir()):
        mf = d / "manifest.json"
        if mf.is_file():
            try:
                out.append(LocalField(name=d.name, path=d, manifest=json.loads(mf.read_text())))
            except (OSError, ValueError) as exc:
                log.warning("skipping local field %s: %s", d, exc)
    return tuple(out)


def fields() -> tuple[LocalField, ...]:
    return _fields(str(data_root()))


def field_for(coord: SkyCoord) -> Optional[LocalField]:
    best = None
    for f in fields():
        sep = min(coord.separation(f.centre).arcmin, coord.separation(f.target).arcmin)
        if sep <= MATCH_RADIUS_ARCMIN and (best is None or sep < best[0]):
            best = (sep, f)
    return best[1] if best else None


def available_text() -> str:
    fs = fields()
    if not fs:
        return f"No downloaded fields found in {data_root()}."
    return "Downloaded fields: " + "; ".join(f"{f.manifest.get('query', f.name)} ({'+'.join(f.detectors)}, {f.fov_arcmin:g}')" for f in fs) + "."


# ---------------------------------------------------------------- names
def _norm(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"^(name|v\*|\*)\s+", "", t)
    t = t.replace("'s", "s").replace("’s", "s")
    return re.sub(r"[^a-z0-9+\-.]", "", t)


def resolve_name(text: str):
    """ResolvedPosition for a downloaded field whose query or SIMBAD name
    matches `text`, from its manifest (no network); None otherwise."""
    from .coords import ResolvedPosition

    key = _norm(text)
    if not key:
        return None
    for f in fields():
        r = f.manifest["resolved"]
        names = {_norm(f.manifest.get("query", "")), _norm(r.get("name") or ""), _norm(f.name.replace("_", " "))}
        if key in names:
            return ResolvedPosition(
                coord=SkyCoord(r["ra_deg"] * u.deg, r["dec_deg"] * u.deg),
                ref_epoch=Time(r.get("ref_epoch_mjd", 51544.5), format="mjd"),
                pmra=r.get("pmra_mas_yr"), pmdec=r.get("pmdec_mas_yr"), parallax_mas=r.get("parallax_mas"),
                name=r.get("name") or f.manifest.get("query"),
                source="local_cache",
                note=f"Resolved from the downloaded field '{f.name}' (SIMBAD result saved {f.retrieved}); no network used.",
            )
    return None


# ---------------------------------------------------------------- images
def records(field: LocalField, detectors: Optional[list[str]] = None) -> list:
    from .irsa import ImageRecord

    recs = [ImageRecord(access_url="local://" + str(field.path / e["file"]), collection=e["collection"],
                        detector=e["detector"], time_mjd=e["time_mjd"], obs_id=e["obs_id"])
            for e in field.manifest["exposures"] if not detectors or e["detector"] in detectors]
    return sorted(recs, key=lambda r: r.time_mjd)


def cutout_source(recs, coord, fov_arcmin) -> Iterator:
    """Drop-in for irsa.fetch_cutouts on local:// records (read-only copies)."""
    from .irsa import Cutout

    for rec in recs:
        path = rec.access_url[len("local://"):]
        with fits.open(path, memmap=False) as hdul:
            yield Cutout(record=rec, hdul=fits.HDUList([h.copy() for h in hdul]))


# ---------------------------------------------------------------- catalogues
_CONE_FIELD: contextvars.ContextVar[Optional[LocalField]] = contextvars.ContextVar("local_cone_field", default=None)


@contextlib.contextmanager
def local_catalogues(field: LocalField):
    """Within this block, catalogs._cone answers from `field`'s cached rows."""
    token = _CONE_FIELD.set(field)
    try:
        yield
    finally:
        _CONE_FIELD.reset(token)


def active_cone_field() -> Optional[LocalField]:
    return _CONE_FIELD.get()


_NO_FALLBACK: contextvars.ContextVar[bool] = contextvars.ContextVar("local_no_fallback", default=False)


@contextlib.contextmanager
def no_fallback():
    """Disable the automatic local fallback (offline regression runs that
    simulate an unavailable catalogue must see it as unavailable)."""
    token = _NO_FALLBACK.set(True)
    try:
        yield
    finally:
        _NO_FALLBACK.reset(token)


def fallback_allowed() -> bool:
    return data_mode() != "live" and not _NO_FALLBACK.get()


def cone(field: LocalField, table: str, ra: float, dec: float, radius_deg: float) -> str:
    from .offline import cached_cone

    return cached_cone(field.name, table, ra, dec, radius_deg, root=field.path.parent)


# ---------------------------------------------------------------- source choice
@dataclass
class SourceChoice:
    kind: str                         # LIVE or LOCAL
    records: Optional[list] = None    # None: let stream_frames query IRSA itself
    cutout_source: object = None
    fov_arcmin: Optional[float] = None
    field: Optional[LocalField] = None
    note: str = ""


class NoLocalData(Exception):
    pass


def local_choice(coord: SkyCoord, fov_arcmin: float, detectors: Optional[list[str]], why: str) -> SourceChoice:
    f = field_for(coord)
    if f is None:
        raise NoLocalData(f"{why} There is no downloaded data for this position. {available_text()}")
    missing = [d for d in (detectors or []) if d not in f.detectors]
    recs = records(f, detectors)
    if not recs:
        raise NoLocalData(f"{why} The downloaded field '{f.name}' has only {', '.join(f.detectors)}, "
                          f"not {', '.join(detectors or [])}.")
    fov = min(fov_arcmin, f.fov_arcmin)
    parts = [why, f"Showing REAL SPHEREx data downloaded earlier ({f.retrieved}) for '{f.manifest.get('query', f.name)}': "
             f"{len(recs)} cached exposure(s), detectors {', '.join(sorted({r.detector for r in recs}))}."]
    if fov < fov_arcmin:
        parts.append(f"Field of view reduced from {fov_arcmin:g}' to the downloaded {fov:g}'.")
    if missing:
        parts.append(f"Not downloaded: {', '.join(missing)}.")
    parts.append("Exposures per epoch are capped as when the data were downloaded, so S/N can be lower than live.")
    return SourceChoice(kind=LOCAL, records=recs, cutout_source=cutout_source, fov_arcmin=fov, field=f,
                        note=" ".join(p for p in parts if p))


def choose_source(resolved, fov_arcmin: float, detectors: Optional[list[str]]) -> SourceChoice:
    """Decide where the images of one /api/frames request come from."""
    from .irsa import DiscoveryError, find_images

    mode = data_mode()
    coord = resolved.search_position()
    if mode == "cached":
        return local_choice(coord, fov_arcmin, detectors, "Offline mode (SPHEREX_DATA_MODE=cached).")
    try:
        recs = find_images(coord, fov_arcmin=fov_arcmin, detectors=detectors)
        return SourceChoice(kind=LIVE, records=recs, fov_arcmin=fov_arcmin)
    except DiscoveryError as exc:
        if mode == "live":
            raise
        try:
            return local_choice(coord, fov_arcmin, detectors, f"IRSA image search is unavailable ({_short(exc)}).")
        except NoLocalData as nd:
            raise DiscoveryError(f"{exc} — {nd}") from exc


def _short(exc: Exception, n: int = 120) -> str:
    """First line of an IRSA error, plus the HTTP status if it is buried further down."""
    full = str(exc)
    s = full.split("\n")[0].rstrip(" :")
    m = re.search(r"\b(5\d\d)\b[^\n]{0,40}?(Proxy Error|Server Error|Bad Gateway|Service Unavailable|Gateway Timeout)", full)
    if m and m.group(1) not in s:
        s = f"{s}: HTTP {m.group(1)} {m.group(2)}"
    return s if len(s) <= n else s[:n] + "…"
