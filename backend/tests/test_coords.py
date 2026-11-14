"""Unit tests for coordinate parsing (Phase 1 requirement).

Pure parsing tests run offline. Name resolution against SIMBAD/Sesame is
marked `live` and skipped by default (see pytest.ini) — run with
`pytest -m live` when you want to hit the network.
"""
import pytest
from astropy.time import Time

from app.coords import parse_position, CoordParseError, ResolvedPosition


def test_sexagesimal_wise0855():
    # Exact string from SPHEREx_tool_prompt.md's test target #3.
    pos = parse_position("08h55m10.831s -7d14m42.53s")
    assert pos.source == "coords"
    assert pos.coord.ra.deg == pytest.approx(133.7951, abs=1e-3)
    assert pos.coord.dec.deg == pytest.approx(-7.2451, abs=1e-3)


def test_sexagesimal_with_plus_sign():
    pos = parse_position("12h00m00.00s +45d30m00.00s")
    assert pos.coord.ra.deg == pytest.approx(180.0, abs=1e-6)
    assert pos.coord.dec.deg == pytest.approx(45.5, abs=1e-6)


def test_decimal_space_separated():
    pos = parse_position("269.452 4.693")
    assert pos.source == "coords"
    assert pos.coord.ra.deg == pytest.approx(269.452)
    assert pos.coord.dec.deg == pytest.approx(4.693)


def test_decimal_comma_separated():
    pos = parse_position("269.452, 4.693")
    assert pos.coord.ra.deg == pytest.approx(269.452)
    assert pos.coord.dec.deg == pytest.approx(4.693)


def test_decimal_negative_dec():
    pos = parse_position("133.819 -7.247")
    assert pos.coord.dec.deg == pytest.approx(-7.247)


def test_decimal_out_of_range_raises():
    with pytest.raises(CoordParseError):
        parse_position("400.0 4.693")
    with pytest.raises(CoordParseError):
        parse_position("269.452 95.0")


def test_empty_raises():
    with pytest.raises(CoordParseError):
        parse_position("   ")


def test_no_proper_motion_propagate_is_identity():
    pos = parse_position("269.452 4.693")
    assert not pos.has_proper_motion
    propagated = pos.propagate_to(Time("2030-01-01"))
    assert propagated.ra.deg == pytest.approx(pos.coord.ra.deg)
    assert propagated.dec.deg == pytest.approx(pos.coord.dec.deg)


def test_proper_motion_propagation_matches_known_shift():
    """Barnard's Star moves ~4.5' between J2000 and ~2026 (verified live in
    phase0/; pmdec dominates). This test hard-codes its real SIMBAD values
    so it doesn't need the network, but still checks the physics."""
    pos = ResolvedPosition(
        coord=__import__("astropy.coordinates", fromlist=["SkyCoord"]).SkyCoord(
            ra=269.45208, dec=4.69336, unit="deg", frame="icrs"
        ),
        ref_epoch=Time("J2000"),
        pmra=-798.71,
        pmdec=10328.12,
        parallax_mas=546.98,
        name="NAME Barnard's star",
        source="simbad",
    )
    assert pos.has_proper_motion
    moved = pos.propagate_to(Time("2026-01-01"))
    shift_arcmin = pos.coord.separation(moved).arcmin
    assert 4.0 < shift_arcmin < 5.0


@pytest.mark.live
@pytest.mark.parametrize(
    "name",
    [
        "Barnard's star",
        "eps Ind B",
        "WISE J085510.83-071442.5",
        "RW Aurigae",
        "R Draconis",
    ],
)
def test_live_name_resolution(name):
    pos = parse_position(name)
    assert pos.source in ("simbad", "sesame")
    assert -90 <= pos.coord.dec.deg <= 90
