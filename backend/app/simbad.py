"""SIMBAD name resolution for proper motion / parallax.

PLAN.md Phase 0 finding #7: astroquery's `Simbad.query_object`/`query_tap`
hard-fail against the default host (`simbad.cds.unistra.fr` returned 404 on
every path we tried, including its capabilities endpoint), and `conf.server`
does not redirect astroquery's internal TAP URL in the installed version. So
we talk to SIMBAD's TAP service directly with `pyvo`, trying each configured
mirror in turn. `astropy.coordinates.SkyCoord.from_name` (Sesame, a separate
resolver) is unaffected and still used as the position-only fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pyvo

from .config import get_config


class SimbadResolutionError(Exception):
    """Raised when no configured SIMBAD mirror could resolve a name."""


@dataclass
class SimbadAstrometry:
    main_id: str
    ra_deg: float
    dec_deg: float
    pmra: Optional[float]   # mas/yr, already includes cos(dec) per SIMBAD's pmra convention
    pmdec: Optional[float]  # mas/yr
    plx_mas: Optional[float]


def _escape_adql_literal(s: str) -> str:
    """ADQL string literals use '' to escape a literal single quote."""
    return s.replace("'", "''")


def _query_one_mirror(tap_url: str, name: str) -> Optional[SimbadAstrometry]:
    tap = pyvo.dal.TAPService(tap_url)
    escaped = _escape_adql_literal(name)

    # Try an exact main_id match first (fast, indexed).
    q_basic = f"""
        SELECT main_id, ra, dec, pmra, pmdec, plx_value
        FROM basic
        WHERE main_id = '{escaped}'
    """
    rows = tap.search(q_basic).to_table()
    if len(rows) == 0:
        # Fall back to any identifier SIMBAD knows this object by
        # (catalogue designations are rarely the main_id).
        q_ident = f"""
            SELECT b.main_id, b.ra, b.dec, b.pmra, b.pmdec, b.plx_value
            FROM ident AS i JOIN basic AS b ON i.oidref = b.oid
            WHERE i.id = '{escaped}'
        """
        rows = tap.search(q_ident).to_table()

    if len(rows) == 0:
        return None

    row = rows[0]

    def _val(col):
        v = row[col]
        # astropy masked columns: masked entries mean SIMBAD has no value.
        if v is None or (hasattr(v, "mask") and v.mask):
            return None
        try:
            if v != v:  # NaN
                return None
        except (TypeError, ValueError):
            pass
        return float(v)

    return SimbadAstrometry(
        main_id=str(row["main_id"]),
        ra_deg=float(row["ra"]),
        dec_deg=float(row["dec"]),
        pmra=_val("pmra"),
        pmdec=_val("pmdec"),
        plx_mas=_val("plx_value"),
    )


def resolve_name_astrometry(name: str) -> SimbadAstrometry:
    """Resolve `name` to position + proper motion/parallax via SIMBAD.

    Tries each mirror in config/collections.yaml in order. Raises
    SimbadResolutionError if every mirror fails or none know the name.
    """
    cfg = get_config()
    last_error: Optional[Exception] = None
    for tap_url in cfg.simbad.tap_mirrors:
        try:
            result = _query_one_mirror(tap_url, name)
            if result is not None:
                return result
            last_error = SimbadResolutionError(f"{tap_url}: name not found")
        except Exception as exc:  # noqa: BLE001 - we deliberately try the next mirror
            last_error = exc
            continue
    raise SimbadResolutionError(
        f"Could not resolve {name!r} astrometry via any SIMBAD mirror "
        f"({[m for m in cfg.simbad.tap_mirrors]}): {last_error}"
    )
