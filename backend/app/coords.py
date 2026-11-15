"""Parse a user's position query and, for named objects, propagate proper
motion to any requested epoch.

Accepts:
  - sexagesimal: "08h55m10.831s -07d14m42.53s" (also -7d, +4d, etc.)
  - decimal degrees: "269.452 4.693" or "269.452, 4.693"
  - an object name: resolved via SIMBAD (astrometry + PM/parallax when
    available) with Sesame (`SkyCoord.from_name`) as a position-only fallback
    if SIMBAD can't find it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from .simbad import resolve_name_astrometry, SimbadResolutionError

# SPHEREx survey start, used as a sane "now-ish" default epoch when none is
# requested (e.g. for a quick preview position before epochs are known).
SURVEY_START = Time("2025-05-01")
# Nominal epoch inside the SPHEREx survey, used only to place the archive
# SEARCH cone. A high-PM star's J2000 position can be minutes of arc from
# where SPHEREx saw it (Barnard's Star: 4.5'), see docs/AUDIT.md C3.
SEARCH_REFERENCE_EPOCH = Time("2026-01-01")


class CoordParseError(ValueError):
    """The input string is neither parseable coordinates nor a resolvable name."""


_SEXAGESIMAL_RE = re.compile(
    r"""^\s*
        (?P<ra>\d{1,2}h\s*\d{1,2}m\s*\d{1,2}(?:\.\d+)?s)
        \s+
        (?P<dec>[+-]?\d{1,2}d\s*\d{1,2}m\s*\d{1,2}(?:\.\d+)?s)
        \s*$""",
    re.VERBOSE | re.IGNORECASE,
)
# "08:55:10.83 -07:14:42.5" or "08 55 10.83 -07 14 42.5" (RA in hours).
_SEXAGESIMAL_SEP_RE = re.compile(
    r"""^\s*
        (?P<ra>\d{1,2}[:\s]\d{1,2}[:\s]\d{1,2}(?:\.\d+)?)
        \s*[,\s]\s*
        (?P<dec>[+-]?\d{1,2}[:\s]\d{1,2}[:\s]\d{1,2}(?:\.\d+)?)
        \s*$""",
    re.VERBOSE,
)
_DECIMAL_RE = re.compile(
    r"^\s*(?P<ra>[+-]?\d+(?:\.\d+)?)\s*[,\s]\s*(?P<dec>[+-]?\d+(?:\.\d+)?)\s*$"
)


@dataclass
class ResolvedPosition:
    """A sky position plus everything needed to move it through time."""

    coord: SkyCoord            # position at ref_epoch (ICRS)
    ref_epoch: Time
    pmra: Optional[float] = None    # mas/yr, SIMBAD's pmra already includes cos(dec)
    pmdec: Optional[float] = None   # mas/yr
    parallax_mas: Optional[float] = None
    name: Optional[str] = None      # SIMBAD main_id, if resolved by name
    source: str = "coords"          # "coords" | "simbad" | "sesame" | "catwise"
    note: Optional[str] = None      # what was (or was not) done to give typed coordinates an epoch

    @property
    def has_proper_motion(self) -> bool:
        return self.pmra is not None and self.pmdec is not None

    def search_position(self) -> SkyCoord:
        """Where to centre an archive search: the position at a nominal
        in-survey epoch (identical to `coord` when no PM is known)."""
        return self.propagate_to(SEARCH_REFERENCE_EPOCH)

    def propagate_to(self, epoch: Time) -> SkyCoord:
        """Position at `epoch`, applying proper motion if we have it.

        Distance (from parallax) barely matters at century timescales for a
        sky-plane shift and is often unknown for faint targets, so it's
        included when available and omitted otherwise rather than treated
        as a hard requirement (verified against astropy directly: the
        difference is ~1e-9 arcsec for Barnard's Star, see phase0/).
        """
        if not self.has_proper_motion:
            return self.coord

        kwargs = dict(
            ra=self.coord.ra,
            dec=self.coord.dec,
            pm_ra_cosdec=self.pmra * u.mas / u.yr,
            pm_dec=self.pmdec * u.mas / u.yr,
            obstime=self.ref_epoch,
            frame="icrs",
        )
        if self.parallax_mas and self.parallax_mas > 0:
            kwargs["distance"] = (1000.0 / self.parallax_mas) * u.pc
        c = SkyCoord(**kwargs)
        return c.apply_space_motion(new_obstime=epoch)


def _try_sexagesimal(text: str) -> Optional[SkyCoord]:
    m = _SEXAGESIMAL_RE.match(text)
    if not m:
        return None
    return SkyCoord(m.group("ra"), m.group("dec"), frame="icrs")


def _try_sexagesimal_separated(text: str) -> Optional[SkyCoord]:
    m = _SEXAGESIMAL_SEP_RE.match(text)
    if not m:
        return None
    ra = re.sub(r"[:\s]+", ":", m.group("ra"))
    dec = re.sub(r"[:\s]+", ":", m.group("dec"))
    try:
        return SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame="icrs")
    except ValueError as exc:
        raise CoordParseError(f"Invalid sexagesimal coordinates {text!r}: {exc}") from exc


def _try_decimal(text: str) -> Optional[SkyCoord]:
    m = _DECIMAL_RE.match(text)
    if not m:
        return None
    ra = float(m.group("ra"))
    dec = float(m.group("dec"))
    if not (0.0 <= ra <= 360.0) or not (-90.0 <= dec <= 90.0):
        raise CoordParseError(
            f"Decimal RA/Dec out of range: ra={ra}, dec={dec} "
            "(expected 0<=ra<=360, -90<=dec<=90)"
        )
    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")


def _try_name(text: str) -> ResolvedPosition:
    try:
        astrom = resolve_name_astrometry(text)
        coord = SkyCoord(ra=astrom.ra_deg * u.deg, dec=astrom.dec_deg * u.deg, frame="icrs")
        return ResolvedPosition(
            coord=coord,
            ref_epoch=Time("J2000"),  # SIMBAD basic.ra/dec are J2000-equinox ICRS positions
            pmra=astrom.pmra,
            pmdec=astrom.pmdec,
            parallax_mas=astrom.plx_mas,
            name=astrom.main_id,
            source="simbad",
        )
    except SimbadResolutionError:
        pass  # fall through to Sesame below

    try:
        coord = SkyCoord.from_name(text)
    except Exception as exc:  # noqa: BLE001
        raise CoordParseError(
            f"{text!r} is not valid sexagesimal/decimal coordinates, and no "
            f"name resolver (SIMBAD, Sesame) could find it: {exc}"
        ) from exc
    return ResolvedPosition(coord=coord, ref_epoch=Time("J2000"), name=text, source="sesame")


# Search radius for the CatWISE track lookup around typed coordinates. A
# 10"/yr mover is <= 55" from its CatWISE (2015.4) position anywhere in the
# 2010-2021 window the lookup accepts.
PM_LOOKUP_RADIUS_DEG = 1.5 / 60.0


def _apply_pm_lookup(coord: SkyCoord) -> ResolvedPosition:
    """Typed coordinates have no epoch (docs/AUDIT.md H1). Ask CatWISE2020
    whether a catalogued fast mover's track passes through them; if exactly
    one does, adopt its position and motion. Always records what happened
    in `.note`, including failures."""
    from .catalogs import catwise_sources, match_typed_position_to_track, CatalogError

    fixed = ResolvedPosition(coord=coord, ref_epoch=Time("J2000"), source="coords")
    try:
        sources = catwise_sources(coord.ra.deg, coord.dec.deg, PM_LOOKUP_RADIUS_DEG)
    except CatalogError as exc:
        fixed.note = f"CatWISE lookup failed ({exc}); coordinates treated as fixed."
        return fixed
    match, reason = match_typed_position_to_track(coord.ra.deg, coord.dec.deg, sources)
    if match is None:
        fixed.note = f"CatWISE2020: {reason}; coordinates treated as fixed."
        return fixed
    src = match.source
    return ResolvedPosition(
        coord=SkyCoord(ra=src.ra_deg * u.deg, dec=src.dec_deg * u.deg, frame="icrs"),
        ref_epoch=Time(src.ref_mjd, format="mjd"),
        pmra=src.pmra_mas_yr,
        pmdec=src.pmdec_mas_yr,
        parallax_mas=None,
        name=f"CatWISE {src.source_id}",
        source="catwise",
        note=(
            f"Typed coordinates matched to a moving object: {reason}. Using its CatWISE2020 motion "
            f"({src.pmra_mas_yr:.0f}, {src.pmdec_mas_yr:.0f}) mas/yr to predict where it is in each SPHEREx frame. "
            "CatWISE motions of very fast or blended objects can be inaccurate; resolve the object by name for "
            "better astrometry."
        ),
    )


def parse_coordinates_only(text: str) -> ResolvedPosition:
    """Typed coordinates only: no name resolution, no network (offline mode)."""
    text = text.strip()
    for parser in (_try_sexagesimal, _try_sexagesimal_separated, _try_decimal):
        coord = parser(text)
        if coord is not None:
            return ResolvedPosition(coord=coord, ref_epoch=Time("J2000"), source="coords")
    raise CoordParseError(f"{text!r} is not sexagesimal or decimal coordinates.")


def parse_position(text: str, pm_lookup: bool = False) -> ResolvedPosition:
    """Parse `text` as coordinates, or resolve it as an object name.

    `pm_lookup=True` additionally asks CatWISE2020 whether typed coordinates
    lie on a catalogued fast mover's track (a network call; off by default
    so the parser stays usable offline).

    Raises CoordParseError if nothing works.
    """
    text = text.strip()
    if not text:
        raise CoordParseError("Empty position query.")

    for parser in (_try_sexagesimal, _try_sexagesimal_separated, _try_decimal):
        coord = parser(text)
        if coord is not None:
            if pm_lookup:
                return _apply_pm_lookup(coord)
            return ResolvedPosition(coord=coord, ref_epoch=Time("J2000"), source="coords")

    return _try_name(text)
