"""External catalogues: Gaia DR3 and CatWISE2020, via IRSA's cone-search service.

Why SCS and not TAP: measured 2026-11-15, an ADQL cone on catwise_2020
through IRSA TAP did not return within 60-150 s, while the same cone
through /SCS returned in 1.5-5 s (Gaia: 1-3 s).

Scientific purpose of each catalogue (do not add one without a purpose):
  * Gaia DR3: precise optical astrometry (position at J2016.0, proper motion,
    parallax). Used to identify known stars, to check our image registration
    against stationary stars, and to validate measured proper motions.
  * CatWISE2020: infrared (W1/W2) positions at MJD 57170 with motion
    estimates. Covers cold brown dwarfs that Gaia cannot see (WISE 0855),
    and gives typed coordinates an epoch (docs/AUDIT.md H1).

Known catalogue pitfalls handled here (all observed live):
  * CatWISE lists the same fast mover more than once (WISE 0855: 2 rows).
  * CatWISE contains spurious high-motion "fragments" of bright stars
    (an 18th-mag row with 10"/yr next to Barnard's Star). Its motions for
    very fast or blended objects can be wrong (eps Ind B).
  * Gaia rows may have no proper motion (2-parameter solutions).
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np
import requests
from astropy.table import Table

log = logging.getLogger(__name__)

SCS_URL = "https://irsa.ipac.caltech.edu/SCS"
GAIA_REF_MJD = 57388.5       # J2016.0
CATWISE_REF_MJD = 57170.0    # ra_pm / dec_pm epoch (2015.405)
DAYS_PER_YEAR = 365.25


class CatalogError(Exception):
    pass


@dataclass
class CatalogSource:
    catalog: str                    # "gaia_dr3" | "catwise2020"
    source_id: str
    ra_deg: float                   # at ref_mjd
    dec_deg: float
    ref_mjd: float
    pos_err_arcsec: float           # 1-sigma, per coordinate (max of the two)
    pmra_mas_yr: Optional[float] = None    # mu_alpha * cos(dec)
    pmdec_mas_yr: Optional[float] = None
    pm_err_mas_yr: Optional[float] = None  # max of the two components
    parallax_mas: Optional[float] = None
    mags: dict = field(default_factory=dict)

    @property
    def has_pm(self) -> bool:
        return self.pmra_mas_yr is not None and self.pmdec_mas_yr is not None

    def position_at(self, mjd: float) -> tuple[float, float, float]:
        """(ra_deg, dec_deg, sigma_arcsec) at `mjd`, linear motion. The
        linear tangent-plane step is accurate to <0.01" for the largest
        offsets used here (~2')."""
        dt = (mjd - self.ref_mjd) / DAYS_PER_YEAR
        ra, dec = self.ra_deg, self.dec_deg
        sigma = self.pos_err_arcsec
        if self.has_pm:
            dec_new = dec + self.pmdec_mas_yr * dt / 3.6e6
            ra = ra + self.pmra_mas_yr * dt / 3.6e6 / math.cos(math.radians(0.5 * (dec + dec_new)))
            dec = dec_new
            if self.pm_err_mas_yr:
                sigma = math.hypot(sigma, self.pm_err_mas_yr * abs(dt) / 1000.0)
        return ra % 360.0, dec, sigma

    def pm_total_mas_yr(self) -> float:
        return math.hypot(self.pmra_mas_yr, self.pmdec_mas_yr) if self.has_pm else 0.0

    def to_dict(self) -> dict:
        return {
            "catalog": self.catalog,
            "source_id": self.source_id,
            "ra_deg": self.ra_deg,
            "dec_deg": self.dec_deg,
            "ref_mjd": self.ref_mjd,
            "pos_err_arcsec": self.pos_err_arcsec,
            "pmra_mas_yr": self.pmra_mas_yr,
            "pmdec_mas_yr": self.pmdec_mas_yr,
            "pm_err_mas_yr": self.pm_err_mas_yr,
            "parallax_mas": self.parallax_mas,
            "mags": self.mags,
        }


def _num(row, col) -> Optional[float]:
    if col not in row.colnames:
        return None
    v = row[col]
    if v is None or np.ma.is_masked(v):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


@lru_cache(maxsize=128)
def _cone_csv(table: str, ra_r: float, dec_r: float, radius_r: float) -> str:
    # IRSA occasionally answers 502/503 (seen live); two retries with a short
    # pause. Failures are not cached, so the next request tries again.
    last: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.get(
                SCS_URL,
                params={"table": table, "RA": ra_r, "DEC": dec_r, "SR": radius_r, "format": "csv"},
                timeout=(10, 45),
            )
            if resp.status_code in (502, 503, 504) and attempt < 2:
                time.sleep(2.0 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            last = exc
            if attempt < 2:
                time.sleep(2.0 * (attempt + 1))
    raise CatalogError(f"{table} cone search failed: {last}")


# Offline hook (app/offline.py): when set, called as override(table, ra, dec,
# radius_deg) and returns cached cone-search CSV text (REAL catalogue rows
# cached by scripts/cache_catalogs.py) instead of querying IRSA. None = live.
CONE_OVERRIDE = None

# Columns the pipeline reads; the cache keeps only these.
NEEDED_COLUMNS = {
    "gaia_dr3_source": ["source_id", "ra", "dec", "ra_error", "dec_error", "pmra", "pmdec", "pmra_error",
                        "pmdec_error", "parallax", "phot_g_mean_mag"],
    "catwise_2020": ["source_name", "ra_pm", "dec_pm", "sigra_pm", "sigdec_pm", "pmra", "pmdec", "sigpmra",
                     "sigpmdec", "w1mpro", "w2mpro"],
}


def _cone(table: str, ra: float, dec: float, radius_deg: float) -> Table:
    from . import localdata

    local = localdata.active_cone_field()
    if CONE_OVERRIDE is not None:
        text = CONE_OVERRIDE(table, ra, dec, radius_deg)
    elif local is not None:
        # app/localdata.py: REAL rows downloaded earlier for this field
        text = localdata.cone(local, table, ra, dec, radius_deg)
    else:
        text = _cone_csv(table, round(ra, 5), round(dec, 5), round(radius_deg, 5))
    if not text.strip():
        return Table()
    try:
        return Table.read(text, format="ascii.csv")
    except Exception as exc:  # noqa: BLE001
        raise CatalogError(f"{table} cone search returned unparseable data: {exc}") from exc


def gaia_sources(ra: float, dec: float, radius_deg: float) -> list[CatalogSource]:
    out = []
    for row in _cone("gaia_dr3_source", ra, dec, radius_deg):
        r, d = _num(row, "ra"), _num(row, "dec")
        if r is None or d is None:
            continue
        # Gaia errors are in mas.
        pos_err = max(_num(row, "ra_error") or 0.0, _num(row, "dec_error") or 0.0) / 1000.0
        pmra, pmdec = _num(row, "pmra"), _num(row, "pmdec")
        pm_err = None
        if pmra is not None and pmdec is not None:
            pm_err = max(_num(row, "pmra_error") or 0.0, _num(row, "pmdec_error") or 0.0)
        else:
            pmra = pmdec = None
        out.append(
            CatalogSource(
                catalog="gaia_dr3",
                source_id=str(row["source_id"]),
                ra_deg=r, dec_deg=d, ref_mjd=GAIA_REF_MJD,
                pos_err_arcsec=pos_err,
                pmra_mas_yr=pmra, pmdec_mas_yr=pmdec, pm_err_mas_yr=pm_err,
                parallax_mas=_num(row, "parallax"),
                mags={k: v for k, v in {"G": _num(row, "phot_g_mean_mag")}.items() if v is not None},
            )
        )
    return out


def catwise_sources(ra: float, dec: float, radius_deg: float) -> list[CatalogSource]:
    out = []
    for row in _cone("catwise_2020", ra, dec, radius_deg):
        # ra_pm/dec_pm: position at MJD 57170 from the fit that includes motion.
        r, d = _num(row, "ra_pm"), _num(row, "dec_pm")
        if r is None or d is None:
            continue
        pos_err = max(_num(row, "sigra_pm") or 0.0, _num(row, "sigdec_pm") or 0.0)
        pmra, pmdec = _num(row, "pmra"), _num(row, "pmdec")  # arcsec/yr, RA term includes cos(dec)
        pm_err = None
        if pmra is not None and pmdec is not None:
            pm_err = 1000.0 * max(_num(row, "sigpmra") or 0.0, _num(row, "sigpmdec") or 0.0)
            pmra, pmdec = 1000.0 * pmra, 1000.0 * pmdec
        else:
            pmra = pmdec = None
        mags = {k: v for k, v in {"W1": _num(row, "w1mpro"), "W2": _num(row, "w2mpro")}.items() if v is not None}
        out.append(
            CatalogSource(
                catalog="catwise2020",
                source_id=str(row["source_name"]),
                ra_deg=r, dec_deg=d, ref_mjd=CATWISE_REF_MJD,
                pos_err_arcsec=pos_err,
                pmra_mas_yr=pmra, pmdec_mas_yr=pmdec, pm_err_mas_yr=pm_err,
                parallax_mas=None,
                mags=mags,
            )
        )
    return out


def tangent_offset_arcsec(ra: float, dec: float, ra0: float, dec0: float) -> tuple[float, float]:
    """(east, north) offset of (ra, dec) from (ra0, dec0), small-angle."""
    dra = ((ra - ra0 + 180.0) % 360.0) - 180.0
    return dra * math.cos(math.radians(0.5 * (dec + dec0))) * 3600.0, (dec - dec0) * 3600.0


# ---------------------------------------------------------------------------
# Giving typed coordinates an epoch (docs/AUDIT.md H1)
# ---------------------------------------------------------------------------

@dataclass
class TrackMatch:
    source: CatalogSource
    matched_epoch_year: float        # when the catalogued track passes the typed position
    miss_arcsec: float               # closest approach distance
    n_merged_rows: int               # duplicate catalogue rows merged into this object


def _year(mjd: float) -> float:
    return 2000.0 + (mjd - 51544.5) / DAYS_PER_YEAR


def match_typed_position_to_track(
    ra: float,
    dec: float,
    sources: list[CatalogSource],
    epoch_range_mjd: tuple[float, float] = (55197.0, 59215.0),   # 2010.0 .. 2021.0 (WISE/NEOWISE era)
    min_pm_mas_yr: float = 500.0,
    min_pm_snr: float = 5.0,
    max_miss_arcsec: float = 3.0,
) -> tuple[Optional[TrackMatch], str]:
    """Find the catalogued fast mover whose track passes through a typed position.

    Typed coordinates carry no epoch. A published position of a fast mover is
    where it WAS when measured, so today's position can be arcminutes away
    (WISE 0855: ~2'). A catalogue source qualifies only if (a) its motion is
    large (>= 0.5"/yr: slower objects move less than about one SPHEREx pixel
    between any 2010-2021 epoch and the survey, so treating them as fixed
    is equivalent) and significant, and (b) its straight-line track passes within
    `max_miss_arcsec` (+3 sigma) of the typed position at some epoch inside
    `epoch_range_mjd`. Duplicate rows of one object are merged. If more than
    one distinct object qualifies the result is ambiguous and nothing is
    applied.

    Returns (match_or_None, human_readable_reason).
    """
    t_lo, t_hi = epoch_range_mjd
    hits: list[TrackMatch] = []
    for s in sources:
        if not s.has_pm or not s.pm_err_mas_yr:
            continue
        pm = s.pm_total_mas_yr()
        if pm < min_pm_mas_yr or pm / s.pm_err_mas_yr < min_pm_snr:
            continue
        # Offset of the typed point from the source's reference position (arcsec),
        # then the time of closest approach along the linear track.
        ox, oy = tangent_offset_arcsec(ra, dec, s.ra_deg, s.dec_deg)
        vx, vy = s.pmra_mas_yr / 1000.0, s.pmdec_mas_yr / 1000.0  # arcsec/yr
        dt = (ox * vx + oy * vy) / (vx * vx + vy * vy)             # years from ref epoch
        mjd = min(max(s.ref_mjd + dt * DAYS_PER_YEAR, t_lo), t_hi)
        dt = (mjd - s.ref_mjd) / DAYS_PER_YEAR
        miss = math.hypot(ox - vx * dt, oy - vy * dt)
        sigma = math.hypot(s.pos_err_arcsec, s.pm_err_mas_yr * abs(dt) / 1000.0)
        if miss <= max_miss_arcsec + 3.0 * sigma:
            hits.append(TrackMatch(source=s, matched_epoch_year=_year(mjd), miss_arcsec=miss, n_merged_rows=1))

    if not hits:
        return None, "no catalogued fast mover passes through the typed position"

    # Merge duplicate rows of one object: same reference position within 5"
    # and motion within 1.5"/yr.
    groups: list[list[TrackMatch]] = []
    for h in hits:
        for g in groups:
            a, b = g[0].source, h.source
            dx, dy = tangent_offset_arcsec(b.ra_deg, b.dec_deg, a.ra_deg, a.dec_deg)
            if math.hypot(dx, dy) < 5.0 and math.hypot(a.pmra_mas_yr - b.pmra_mas_yr, a.pmdec_mas_yr - b.pmdec_mas_yr) < 1500.0:
                g.append(h)
                break
        else:
            groups.append([h])

    if len(groups) > 1:
        names = ", ".join(g[0].source.source_id for g in groups)
        return None, f"ambiguous: {len(groups)} different catalogued movers pass through the typed position ({names})"

    group = groups[0]
    best = min(group, key=lambda h: h.source.pm_err_mas_yr)
    if len(group) > 1:
        # CatWISE splits very fast movers into several rows (WISE 0855: 3).
        # Combine them with inverse-variance weights rather than pick one.
        w_pm = np.array([1.0 / h.source.pm_err_mas_yr ** 2 for h in group])
        w_pos = np.array([1.0 / max(h.source.pos_err_arcsec, 1e-3) ** 2 for h in group])
        b = best.source
        merged = CatalogSource(
            catalog=b.catalog,
            source_id=b.source_id,
            ra_deg=float(np.average([h.source.ra_deg for h in group], weights=w_pos)),
            dec_deg=float(np.average([h.source.dec_deg for h in group], weights=w_pos)),
            ref_mjd=b.ref_mjd,
            pos_err_arcsec=float(1.0 / math.sqrt(w_pos.sum())),
            pmra_mas_yr=float(np.average([h.source.pmra_mas_yr for h in group], weights=w_pm)),
            pmdec_mas_yr=float(np.average([h.source.pmdec_mas_yr for h in group], weights=w_pm)),
            pm_err_mas_yr=float(1.0 / math.sqrt(w_pm.sum())),
            mags=b.mags,
        )
        best = TrackMatch(source=merged, matched_epoch_year=best.matched_epoch_year, miss_arcsec=best.miss_arcsec,
                          n_merged_rows=len(group))
    merged_note = f" ({best.n_merged_rows} catalogue rows of this object combined)" if best.n_merged_rows > 1 else ""
    return best, (
        f"{best.source.catalog} {best.source.source_id} passes {best.miss_arcsec:.1f}\" from the typed position "
        f"at epoch {best.matched_epoch_year:.1f}{merged_note}"
    )
