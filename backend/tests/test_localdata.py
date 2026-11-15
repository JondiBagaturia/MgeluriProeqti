"""Locally downloaded data as an offline / fallback source (app/localdata.py).
No network: IRSA and SIMBAD outages are simulated."""
import asyncio
import json

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord

from app import localdata, main
from app.coords import CoordParseError
from app.irsa import DiscoveryError


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    main._parse_cached.cache_clear()
    yield
    main._parse_cached.cache_clear()


def _down(*a, **k):
    raise DiscoveryError("SIA query failed: 502 Proxy Error (simulated)")


def test_fields_are_the_downloaded_targets():
    names = {f.name for f in localdata.fields()}
    assert {"barnard", "eps_ind_b", "rw_aur", "wise0855", "r_dra"} <= names
    bar = next(f for f in localdata.fields() if f.name == "barnard")
    assert bar.detectors == ["D3", "D6"] and bar.fov_arcmin == 5.0 and bar.has_catalogues()


@pytest.mark.parametrize("text", ["Barnard's star", "barnards star", "NAME Barnard's star", "BARNARD'S STAR"])
def test_resolve_name_from_manifest(text):
    r = localdata.resolve_name(text)
    assert r is not None and r.source == "local_cache"
    assert r.has_proper_motion and abs(r.pmdec - 10362.394) < 1e-6
    assert abs(r.coord.dec.deg - 4.6934) < 1e-3


def test_unknown_name_is_not_resolved_locally():
    assert localdata.resolve_name("Betelgeuse") is None


def test_field_for_matches_target_and_rejects_elsewhere():
    assert localdata.field_for(SkyCoord(269.452 * u.deg, 4.6934 * u.deg)).name == "barnard"       # J2000 position
    assert localdata.field_for(SkyCoord(269.4462 * u.deg, 4.7688 * u.deg)).name == "barnard"      # where it is now
    assert localdata.field_for(SkyCoord(10 * u.deg, 10 * u.deg)) is None


def test_cached_mode_never_queries_irsa(monkeypatch):
    monkeypatch.setenv("SPHEREX_DATA_MODE", "cached")
    monkeypatch.setattr("app.irsa.find_images", lambda *a, **k: pytest.fail("network used in cached mode"))
    r = main._parse_cached("RW Aur", True)
    ch = localdata.choose_source(r, 8.0, ["D3", "D6", "D1"])
    assert ch.kind == localdata.LOCAL and ch.field.name == "rw_aur"
    assert ch.fov_arcmin == 5.0 and "reduced from 8' to the downloaded 5'" in ch.note
    assert "Not downloaded: D1" in ch.note
    assert {rec.detector for rec in ch.records} == {"D3", "D6"}


def test_cached_mode_elsewhere_lists_what_is_available(monkeypatch):
    monkeypatch.setenv("SPHEREX_DATA_MODE", "cached")
    r = main._parse_cached("10.0 10.0", True)
    with pytest.raises(localdata.NoLocalData, match="Downloaded fields:"):
        localdata.choose_source(r, 5.0, None)
    with pytest.raises(CoordParseError, match="Offline mode"):
        main._parse_cached("Betelgeuse", True)


def test_auto_mode_falls_back_only_when_irsa_fails(monkeypatch):
    monkeypatch.setenv("SPHEREX_DATA_MODE", "auto")
    r = localdata.resolve_name("eps Ind B")
    monkeypatch.setattr("app.irsa.find_images", lambda *a, **k: ["live-record"])
    assert localdata.choose_source(r, 5.0, None).kind == localdata.LIVE
    monkeypatch.setattr("app.irsa.find_images", _down)
    ch = localdata.choose_source(r, 5.0, None)
    assert ch.kind == localdata.LOCAL and "IRSA image search is unavailable" in ch.note


def test_live_mode_never_falls_back(monkeypatch):
    monkeypatch.setenv("SPHEREX_DATA_MODE", "live")
    monkeypatch.setattr("app.irsa.find_images", _down)
    with pytest.raises(DiscoveryError):
        localdata.choose_source(localdata.resolve_name("eps Ind B"), 5.0, None)


def test_auto_mode_name_fallback_when_simbad_is_down(monkeypatch):
    monkeypatch.setenv("SPHEREX_DATA_MODE", "auto")

    def no_resolver(text, pm_lookup=False):
        raise CoordParseError("no name resolver could find it (simulated outage)")

    monkeypatch.setattr(main, "parse_position", no_resolver)
    r = main._parse_cached("WISE J085510.83-071442.5", True)
    assert r.source == "local_cache" and "Name resolvers unavailable" in r.note


def _stream(**params):
    async def run():
        resp = await main.frames(**params)
        chunks = [c async for c in resp.body_iterator]
        return [json.loads(line) for c in chunks for line in (c.decode() if isinstance(c, bytes) else c).splitlines() if line]
    return asyncio.run(run())


def test_frames_endpoint_offline_end_to_end(monkeypatch):
    """IRSA images AND catalogues down, auto mode: the whole request is served
    from the local copy and every part of the answer says so."""
    monkeypatch.setenv("SPHEREX_DATA_MODE", "auto")
    monkeypatch.setattr("app.irsa.find_images", _down)

    def cone_down(*a, **k):
        from app.catalogs import CatalogError
        raise CatalogError("gaia_dr3_source cone search failed: 502 (simulated)")

    monkeypatch.setattr("app.catalogs._cone_csv", cone_down)
    lines = _stream(position="WISE J085510.83-071442.5", pm_lookup=False, fov=5.0, detectors="D5,D6",
                    group="detector_epoch", quality="low", shiftstack=False, spectrotemporal=False, progress=False)
    assert not [x for x in lines if "error" in x], lines[:2]
    assert lines[0]["type"] == "data_source" and lines[0]["field"]["name"] == "wise0855"
    frames = [x for x in lines if "payload_b64" in x]
    assert frames and all(f["data_source"] == "local_cache" for f in frames)
    analysis = next(x for x in lines if x.get("type") == "analysis")
    assert analysis["data_source"] == "local_cache"
    # images are local, so the catalogues come straight from the local rows (no live attempt)
    assert analysis["catalog_source"] == {"gaia_dr3": "local_cache", "catwise2020": "local_cache"}
    assert any(t.get("gaia", {}).get("status") == "matched" for s in analysis["series"] for t in s["tracks"])


def test_catalogue_fallback_with_live_images(monkeypatch):
    """Images live, catalogue service down: the analysis falls back to the
    downloaded rows for that field and records the outage."""
    import numpy as np
    from app.stream import _GridContext, run_analysis
    from app.pipeline import make_target_wcs

    monkeypatch.setenv("SPHEREX_DATA_MODE", "auto")

    def cone_down(*a, **k):
        from app.catalogs import CatalogError
        raise CatalogError("cone search failed: 502 (simulated)")

    monkeypatch.setattr("app.catalogs._cone_csv", cone_down)
    f = next(x for x in localdata.fields() if x.name == "eps_ind_b")
    c = f.centre
    wcs, shape = make_target_wcs(c, 5.0, 3.075)
    grid = _GridContext(resolved=None, wcs=wcs, shape=shape, centre_ra=c.ra.deg, centre_dec=c.dec.deg,
                        grid_scale_arcsec=3.075, native_scale_arcsec=6.15)
    out = run_analysis(grid, 5.0, "detector_epoch")
    assert out["catalog_source"] == {"gaia_dr3": "local_cache", "catwise2020": "local_cache"}
    assert len(out["catalog_errors"]) == 2 and all("instead" in e for e in out["catalog_errors"])


def test_releases_offline_uses_downloaded_collections(monkeypatch):
    monkeypatch.setenv("SPHEREX_DATA_MODE", "cached")
    monkeypatch.setattr(main, "discover_collections", lambda: pytest.fail("network used in cached mode"))
    body = json.loads(asyncio.run(main.releases()).body)
    assert body["collections"] and body["data_mode"] == "cached"


def test_all_downloads_failing_falls_back_to_local_copy(monkeypatch):
    """IRSA finds the images but cannot deliver any (the stream's 'every cutout
    download failed' error): auto mode then serves the downloaded copy."""
    monkeypatch.setenv("SPHEREX_DATA_MODE", "auto")
    field = next(f for f in localdata.fields() if f.name == "eps_ind_b")
    monkeypatch.setattr("app.irsa.find_images", lambda *a, **k: localdata.records(field, ["D3", "D6"]))

    def nothing_arrives(records, coord, fov_arcmin, **kw):
        return iter(())

    monkeypatch.setattr("app.stream.fetch_cutouts", nothing_arrives)
    monkeypatch.setattr("app.catalogs._cone_csv", lambda *a, **k: (_ for _ in ()).throw(__import__("app.catalogs", fromlist=["CatalogError"]).CatalogError("down")))
    lines = _stream(position="eps Ind B", pm_lookup=False, fov=5.0, detectors="D3,D6", group="detector_epoch",
                    quality="low", shiftstack=False, spectrotemporal=False, progress=False)
    assert not [x for x in lines if "error" in x], [x for x in lines if "error" in x]
    assert lines[0]["type"] == "data_source" and "IRSA could not deliver the images" in lines[0]["note"]
    assert [x for x in lines if "payload_b64" in x]
