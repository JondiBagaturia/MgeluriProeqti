"""Calibrated ranking (app/ranking.py) and source injection (app/injection.py).

Unit tests use SYNTHETIC track records with known statistics. The injection
tests add SYNTHETIC sources to REAL cached SPHEREx cutouts (in memory only).
"""
import copy
import json
import math

import numpy as np
import pytest
from scipy.stats import chi2

from app.ranking import LAMBDA_MAX, PEER_MIN, calibrate_series, calibrate_track

RNG = np.random.default_rng(7)


def T(snr=20.0, n_epochs=4, pm_sig=1.0, p_var=0.5, flags=(), reliability="high", gaia="matched",
      catwise="matched", flux=5.0, C=0.0, M=0.0, V=0.0, wl_mismatch=False):
    return {
        "series": "D6", "detection_snr_median": snr, "n_epochs": n_epochs, "flags": list(flags),
        "reliability": reliability, "motion": {"pm_significance": pm_sig},
        "variability": {"p_value": p_var, "wavelength_mismatch": wl_mismatch},
        "score_components": {"motion": M, "variability": V, "no_catalogue_counterpart": C},
        "anomaly_score": 0.45 * M + 0.30 * V + 0.25 * C, "rank": 1,
        "gaia": {"status": gaia}, "catwise": {"status": catwise},
        "detections": [{"phot_status": "ok", "flux_mjy": flux}] * max(n_epochs, 1),
    }


def null_peers(n, snr=20.0, inflate=1.0, n_epochs=4):
    """Peers whose PM significance^2 ~ inflate * chi^2(2) and variability ~ inflate * chi^2(1)."""
    out = []
    for _ in range(n):
        x2 = inflate * RNG.chisquare(2)
        x1 = inflate * RNG.chisquare(1)
        out.append(T(snr=snr, n_epochs=n_epochs, pm_sig=math.sqrt(x2), p_var=float(chi2.sf(x1, 1))))
    return out


def test_raw_phase_two_fields_are_never_modified():
    tracks = null_peers(12) + [T(pm_sig=6.0, M=0.43, C=1.0, gaia="unmatched", catwise="unmatched", flux=0.1)]
    before = copy.deepcopy(tracks)
    calibrate_series(tracks)
    for b, a in zip(before, tracks):
        for k, v in b.items():
            assert a[k] == v
        assert {"calibration", "calibrated_score", "calibrated_rank"} <= set(a)
    json.dumps(tracks)   # plain JSON types


def test_peer_inflation_deflates_noisy_bins_and_leaves_clean_bins_alone():
    src = T(pm_sig=6.0)
    noisy = null_peers(40, inflate=3.0) + [src]
    clean = null_peers(40, inflate=1.0) + [src]
    cn = calibrate_track(src, noisy)["motion_calibration"]
    cc = calibrate_track(src, clean)["motion_calibration"]
    assert cn["inflation"] == pytest.approx(3.0, rel=0.35)
    assert cc["inflation"] == pytest.approx(1.0, abs=0.35)
    assert cn["p_calibrated"] > 10 * cc["p_calibrated"]
    assert cc["peer_level"] == "snr_and_coverage_bin" and cc["n_peers"] == 40


def test_inflation_estimate_survives_many_anomalous_peers():
    """Regression for the injection finding: 60% genuinely moving peers must not
    make the field look 'noisy' and blind the pipeline to motion (a median and a
    lower quartile both failed this)."""
    movers = [T(pm_sig=float(s)) for s in RNG.uniform(6, 15, 24)]
    src = T(pm_sig=8.0)
    cal = calibrate_track(src, null_peers(16) + movers + [src])["motion_calibration"]
    assert cal["inflation"] < 1.6


def test_inflation_is_capped_and_reported():
    src = T(pm_sig=20.0)
    cal = calibrate_track(src, null_peers(20, inflate=50.0) + [src])["motion_calibration"]
    assert cal["inflation"] == LAMBDA_MAX and cal["inflation_capped"] is True


def test_peer_fallback_levels_and_empirical_bound():
    src = T(snr=60.0, n_epochs=3, pm_sig=5.0)
    few_same_bin = null_peers(2, snr=60.0, n_epochs=3)
    other_coverage = null_peers(4, snr=60.0, n_epochs=4)
    other_snr = null_peers(10, snr=15.0)
    cal = calibrate_track(src, few_same_bin + other_coverage + [src])["motion_calibration"]
    assert cal["peer_level"] == "snr_bin" and cal["n_peers"] == 6
    cal = calibrate_track(src, few_same_bin + other_snr + [src])["motion_calibration"]
    assert cal["peer_level"] == "all_peers" and cal["n_peers"] == 12
    cal = calibrate_track(src, null_peers(PEER_MIN - 1) + [src])["motion_calibration"]
    assert cal["peer_level"] == "analytic" and cal["inflation"] == 1.0 and cal["empirical_rarity_bound"] is None
    # bound: k peers at least as extreme out of n -> (k + 1) / (n + 1), never zero
    peers = [T(pm_sig=s) for s in (0.5, 0.8, 1.0, 1.2, 9.0, 0.3, 0.7, 1.1)]
    cal = calibrate_track(src, peers + [src])["motion_calibration"]
    assert cal["n_peers_at_least_as_extreme"] == 1 and cal["empirical_rarity_bound"] == pytest.approx(2 / 9)


def test_unclean_tracks_are_not_peers_and_source_is_not_its_own_peer():
    src = T(pm_sig=5.0)
    unclean = [T(flags=["near_bright_star"]), T(reliability="medium"), T(flags=["ambiguous_link"]), T(n_epochs=1)]
    cal = calibrate_track(src, unclean + null_peers(3) + [src])["motion_calibration"]
    assert cal["peer_level"] == "analytic"


def _field(n_bright_matched=8, faint_matched=0):
    """Catalogue-matched sources from 1 to 50 mJy; completeness limit near the faintest matched ones."""
    out = [T(flux=f, gaia="matched", catwise="matched") for f in np.geomspace(2, 50, n_bright_matched)]
    out += [T(flux=f, gaia="matched", catwise="matched") for f in np.geomspace(1, 2, faint_matched)]
    return out


def test_faint_unmatched_source_is_below_completeness_and_loses_its_boost():
    faint = T(flux=0.3, C=1.0, gaia="unmatched", catwise="unmatched")
    cal = calibrate_track(faint, _field() + [faint])
    assert cal["completeness_regime"] == "below_completeness"
    assert cal["C_norm"] == 0.0 and cal["completeness"]["gaia_dr3"]["completeness_flux_mjy"] == pytest.approx(2.0, rel=0.01)
    assert any("below the local completeness limit" in s for s in cal["suppression_reasons"])


def test_bright_unmatched_source_keeps_its_boost_and_cannot_suppress_itself():
    bright = T(flux=30.0, C=1.0, gaia="unmatched", catwise="unmatched")
    cal = calibrate_track(bright, _field() + [bright])
    assert cal["completeness_regime"] == "above_completeness" and cal["C_norm"] == 1.0


def test_completeness_unknown_and_catalogue_without_coverage():
    src = T(flux=10.0, C=1.0, gaia="unmatched", catwise="unmatched")
    cal = calibrate_track(src, [T(flux=5.0)] * 3 + [src])                       # too few tracks
    assert cal["completeness_regime"] == "completeness_unknown" and cal["C_norm"] == 0.5
    nocov = [T(flux=f, gaia="matched", catwise="unmatched") for f in (3, 5, 8, 12, 20, 30)]
    src = T(flux=10.0, C=0.1, gaia="matched", catwise="unmatched")
    cal = calibrate_track(src, nocov + [src])                                   # CatWISE matches nothing here
    assert cal["completeness"]["catwise2020"]["regime"] == "catalogue_incomplete_here" and cal["C_norm"] == 0.0


def test_unavailable_catalogue_is_not_unmatched():
    src = T(flux=10.0, C=0.0, gaia="catalog_unavailable", catwise="catalog_unavailable")
    cal = calibrate_track(src, _field() + [src])
    assert cal["completeness_regime"] == "unavailable" and cal["C_norm"] == 0.0


@pytest.mark.parametrize("kw, cap", [
    (dict(flags=["near_bright_star"]), 0.05), (dict(reliability="low"), 0.05),
    (dict(flags=["core_masked_all_epochs"]), 0.05), (dict(flags=["ambiguous_link"]), 0.10),
    (dict(n_epochs=2, flags=["two_epochs_only"]), 0.30), (dict(n_epochs=1), 0.0),
])
def test_quality_caps(kw, cap):
    src = T(pm_sig=30.0, p_var=1e-30, C=1.0, gaia="unmatched", catwise="unmatched", flux=40.0, **kw)
    tracks = _field() + null_peers(10) + [src]
    cal = calibrate_track(src, tracks)
    assert cal["calibrated_score"] <= cap + 1e-12 and cal["quality_cap"] == cap
    assert cal["suppression_reasons"]


def test_calibrated_rank_orders_by_calibrated_score():
    tracks = null_peers(10) + [T(pm_sig=12.0), T(pm_sig=12.0, flags=["near_bright_star"]), T(p_var=1e-12)]
    calibrate_series(tracks)
    by_rank = sorted(tracks, key=lambda t: t["calibrated_rank"])
    scores = [t["calibrated_score"] for t in by_rank]
    assert scores == sorted(scores, reverse=True)
    assert by_rank[0]["motion"]["pm_significance"] == 12.0 and "near_bright_star" not in by_rank[0]["flags"]


# ------------------------------------------------------------------ injection into REAL cutouts

from app.offline import REAL_DATA_DIR  # noqa: E402

real = pytest.mark.skipif(not (REAL_DATA_DIR / "wise0855" / "manifest.json").exists(), reason="real-data cache not present")


@real
def test_injected_flux_is_recovered_by_native_aperture_photometry():
    from astropy.io import fits
    from astropy.wcs import WCS
    from photutils.aperture import CircularAperture
    from app.injection import InjectedSource, inject, render
    from app.irsa import Cutout, ImageRecord
    from app.spectrotemporal import measure_exposure

    m = json.loads((REAL_DATA_DIR / "wise0855" / "manifest.json").read_text())
    e = m["exposures"][5]
    with fits.open(REAL_DATA_DIR / "wise0855" / e["file"]) as h:
        base = fits.HDUList([x.copy() for x in h])
        inj = fits.HDUList([x.copy() for x in h])
    hdr = base["IMAGE"].header
    w = WCS(hdr)
    ny, nx = base["IMAGE"].data.shape
    sky = w.pixel_to_world(nx / 2 + 7.3, ny / 2 - 6.6)          # blank-ish spot away from the target
    src = InjectedSource(ra_deg=float(sky.ra.deg), dec_deg=float(sky.dec.deg), flux_mjy=20.0, ref_mjd=e["time_mjd"])
    log = inject(inj, [src])
    assert log[0]["injected"]
    rec = ImageRecord(access_url="", collection=e["collection"], detector=e["detector"], time_mjd=e["time_mjd"], obs_id=e["obs_id"])
    a = measure_exposure(Cutout(rec, base), "s", src.ra_deg, src.dec_deg)
    b = measure_exposure(Cutout(rec, inj), "s", src.ra_deg, src.dec_deg)
    scale = 6.15
    expected = 20.0 * float(np.sum(CircularAperture((log[0]["x"], log[0]["y"]), r=9.0 / scale).to_mask(method="exact").to_image(
        base["IMAGE"].data.shape) * render(base["IMAGE"].data.shape, log[0]["x"], log[0]["y"], hdr["PSF_FWHM"] / scale)))
    assert b.flux_mjy - a.flux_mjy == pytest.approx(expected, rel=0.03)
    # the cached file on disk is untouched
    with fits.open(REAL_DATA_DIR / "wise0855" / e["file"]) as h:
        assert np.array_equal(h["IMAGE"].data, base["IMAGE"].data, equal_nan=True)


@real
def test_end_to_end_injection_mover_ranked_stationary_not():
    from app.injection import InjectedSource, make_transform
    from app.offline import run_cached

    m = json.loads((REAL_DATA_DIR / "wise0855" / "manifest.json").read_text())
    ra0, dec0 = m["cutout_centre_deg"]
    ref = float(np.mean([e["time_mjd"] for e in m["exposures"]]))
    def pos(e, n):
        return ra0 + e / 3600 / math.cos(math.radians(dec0)), dec0 + n / 3600
    mover = InjectedSource(*pos(-60, 80), flux_mjy=3.0, ref_mjd=ref, pm_east_mas_yr=4000, pm_north_mas_yr=-3000)
    still = InjectedSource(*pos(80, -60), flux_mjy=3.0, ref_mjd=ref)
    run = run_cached("wise0855", detectors=["D5"], use_catalogs="cached", cutout_transform=make_transform([mover, still]))
    tracks = run.analysis["series"][0]["tracks"]

    def find(src):
        best = None
        for t in tracks:
            ra, dec = src.position_at(t["epoch_mjd"])
            d = math.hypot((t["ra_deg"] - ra) * math.cos(math.radians(dec)), t["dec_deg"] - dec) * 3600
            if d < 5 and (best is None or d < best[0]):
                best = (d, t)
        return best[1] if best else None
    tm, ts = find(mover), find(still)
    assert tm is not None and ts is not None and tm["n_epochs"] == ts["n_epochs"] == 3
    assert tm["calibration"]["M_norm"] > 0.3 and tm["motion"]["pm_significance"] > 6
    assert ts["calibration"]["M_norm"] == 0.0 and ts["calibration"]["V_norm"] == 0.0
    assert tm["calibrated_rank"] < ts["calibrated_rank"]
    for t in (tm, ts):                  # injected sources are absent from the (real) catalogues
        assert t["gaia"]["status"] == "unmatched"
