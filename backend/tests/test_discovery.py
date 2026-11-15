"""Tests for detection, tracking, motion/variability fits, catalogue matching
and ranking.

ALL IMAGES AND CATALOGUES IN THIS FILE ARE SYNTHETIC, built in-test with
known injected motion and brightness. None of it is a SPHEREx observation
or a real catalogue row.
"""
import math

import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from app.catalogs import CatalogSource, match_typed_position_to_track, CATWISE_REF_MJD, GAIA_REF_MJD
from app.discovery import (
    DiscoveryConfig, SeriesFrame, analyze_series, fit_motion, fit_variability, link_detections, Detection,
)
from app.pipeline import make_target_wcs

SCALE = 3.075
NATIVE = 6.15
RA0, DEC0 = 100.0, 20.0
T = [61000.0, 61180.0, 61365.0, 61545.0]   # ~1.5 yr baseline, 4 epochs


def _scene(stars, t_index, shape, noise=0.02, seed=0):
    """stars: list of dict(east, north, pm_e, pm_n ["/yr], flux(t_index)->pixel-sum, )"""
    rng = np.random.default_rng(seed + t_index)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    img = 5.0 + rng.normal(0, noise, shape)
    dt = (T[t_index] - T[0]) / 365.25
    sigma = (9.0 / SCALE) / 2.3548
    for s in stars:
        e = s["east"] + s.get("pm_e", 0.0) * dt
        n = s["north"] + s.get("pm_n", 0.0) * dt
        x = (w - 1) / 2 - e / SCALE
        y = (h - 1) / 2 + n / SCALE
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        amp = s["flux"](t_index) if callable(s["flux"]) else s["flux"]
        img += amp * g / g.sum()
    return img


def _frames(stars, noise=0.02, wavelengths=None):
    wcs, shape = make_target_wcs(SkyCoord(RA0, DEC0, unit="deg"), 5, SCALE)
    frames = [
        SeriesFrame(image=_scene(stars, i, shape, noise), variance=np.full(shape, noise ** 2), time_mjd=T[i],
                    wavelength_um=(wavelengths[i] if wavelengths else 4.7))
        for i in range(len(T))
    ]
    return frames, wcs


def _nearest(tracks, east, north):
    def off(t):
        e = (t["ra_deg"] - RA0) * math.cos(math.radians(DEC0)) * 3600
        n = (t["dec_deg"] - DEC0) * 3600
        return math.hypot(e - east, n - north)
    return min(tracks, key=off)


STARS = [
    {"east": 40.0, "north": 30.0, "flux": 60.0},                                 # stationary
    {"east": -50.0, "north": -40.0, "flux": 80.0},                               # stationary
    {"east": 60.0, "north": -55.0, "flux": 50.0},                                # stationary
    {"east": -20.0, "north": 45.0, "flux": 70.0, "pm_e": -0.8, "pm_n": 10.4},    # Barnard-like mover
    {"east": -70.0, "north": 10.0, "flux": lambda i: [40.0, 80.0, 40.0, 90.0][i]},  # strong variable
]


def test_fit_motion_recovers_injected_velocity_and_error():
    t = np.array(T)
    tau = (t - t[0]) / 365.25
    fit = fit_motion(t, 1.0 - 0.8 * tau, 2.0 + 10.4 * tau, np.full(4, 0.5))
    assert fit.pmra_mas_yr == pytest.approx(-800, abs=1e-6)
    assert fit.pmdec_mas_yr == pytest.approx(10400, abs=1e-6)
    # analytic slope error: sigma / sqrt(sum (tau - mean)^2)
    expected = 0.5 / math.sqrt(np.sum((tau - tau.mean()) ** 2)) * 1000
    assert fit.pmra_err_mas_yr == pytest.approx(expected, rel=1e-6)
    assert fit.dof == 4


def test_fit_motion_error_calibrated_on_noise():
    """Over many noisy realisations of a stationary source, PM/err must scatter ~N(0,1)."""
    rng = np.random.default_rng(1)
    z = []
    for _ in range(500):
        f = fit_motion(T, rng.normal(0, 1.0, 4), rng.normal(0, 1.0, 4), np.full(4, 1.0))
        z.append(f.pmra_mas_yr / (1000.0 / math.sqrt(np.sum(((np.array(T) - np.mean(T)) / 365.25) ** 2))))
    assert np.std(z) == pytest.approx(1.0, rel=0.1)


def test_variability_respects_systematic_floor_and_flags_wavelength():
    cfg = DiscoveryConfig()
    # 5% scatter with tiny statistical errors: below the 6% floor -> not significant
    quiet = fit_variability([100, 105, 97, 102], [0.5] * 4, [4.7] * 4, cfg)
    assert quiet.p_value > 0.05
    loud = fit_variability([100, 200, 100, 220], [0.5] * 4, [4.7] * 4, cfg)
    assert loud.p_value < 1e-6
    mixed = fit_variability([100, 200, 100, 220], [0.5] * 4, [4.45, 4.7, 4.9, 4.6], cfg)
    assert mixed.wavelength_mismatch and mixed.wavelength_span_um == pytest.approx(0.45)


def test_linking_keeps_mover_and_stationary_apart():
    def det(fi, e, n):
        return Detection(frame_index=fi, time_mjd=T[fi], x=0, y=0, east_arcsec=e, north_arcsec=n, pos_err_arcsec=1.0,
                         peak_snr=50, flux_mjy=1.0, flux_err_mjy=0.1, phot_status="ok", core_masked=False, near_edge=False)
    per_frame = [[det(i, 0.0, 0.0), det(i, 8.0, 5.0 * i)] for i in range(4)]   # stationary + 5"/epoch mover, 8" apart at start
    tracks = link_detections(per_frame, DiscoveryConfig())
    assert sorted(len(t.detections) for t in tracks) == [4, 4]
    stat = min(tracks, key=lambda t: abs(t.detections[-1].north_arcsec))
    assert all(d.north_arcsec == 0.0 for d in stat.detections)


def test_analyze_series_recovers_mover_variable_and_stationary():
    frames, wcs = _frames(STARS)
    res = analyze_series(frames, wcs, SCALE, NATIVE, RA0, DEC0, gaia=[], catwise=[], series="D6")
    tracks = res["tracks"]
    assert res["n_detections_per_frame"] == [5, 5, 5, 5]
    assert len(tracks) == 5 and all(t["n_epochs"] == 4 for t in tracks)

    dt_mid = (np.mean(T) - T[0]) / 365.25
    mover = _nearest(tracks, -20.0 - 0.8 * dt_mid, 45.0 + 10.4 * dt_mid)
    m = mover["motion"]
    assert m["pmdec_mas_yr"] == pytest.approx(10400, abs=3 * m["pmdec_err_mas_yr"])
    assert m["pmra_mas_yr"] == pytest.approx(-800, abs=3 * m["pmra_err_mas_yr"])
    assert m["pm_significance"] > 5
    assert mover["rank"] == 1
    assert any("Moves" in r for r in mover["reasons"])

    stat = _nearest(tracks, 40.0, 30.0)
    assert stat["motion"]["pm_significance"] < 3
    assert stat["score_components"]["motion"] == 0.0

    variable = _nearest(tracks, -70.0, 10.0)
    assert variable["variability"]["p_value"] < 1e-4
    assert variable["score_components"]["variability"] > 0.3
    assert stat["variability"]["p_value"] > 0.01

    for t in tracks:
        # the four quantities are separate fields and class probability is never invented
        assert {"detection_snr_median", "anomaly_score", "reliability", "class_probability"} <= set(t)
        assert t["class_probability"] is None
        assert 0.0 <= t["anomaly_score"] <= 1.0


def test_unmatched_is_flagged_but_never_called_a_discovery():
    frames, wcs = _frames(STARS)
    res = analyze_series(frames, wcs, SCALE, NATIVE, RA0, DEC0, gaia=[], catwise=[], series="D6")
    for t in res["tracks"]:
        assert t["gaia"]["status"] == "unmatched"
        text = " ".join(t["reasons"] + t["caveats"] + [t["label"]]).lower()
        assert "discover" not in text or "does not make it new" in text
        assert t["label"] in ("candidate", "uncertain")


def _gaia_at(east, north, pm_e=0.0, pm_n=0.0, sid="g"):
    """Synthetic Gaia-like row whose position at T[0] is (east, north)."""
    dt = (GAIA_REF_MJD - T[0]) / 365.25
    e, n = east + pm_e * dt, north + pm_n * dt
    return CatalogSource(catalog="gaia_dr3", source_id=sid, ra_deg=RA0 + e / 3600 / math.cos(math.radians(DEC0)),
                         dec_deg=DEC0 + n / 3600, ref_mjd=GAIA_REF_MJD, pos_err_arcsec=0.001,
                         pmra_mas_yr=pm_e * 1000, pmdec_mas_yr=pm_n * 1000, pm_err_mas_yr=0.05, parallax_mas=5.0,
                         mags={"G": 12.0})


def test_catalogue_match_propagates_proper_motion_to_the_observation_epoch():
    frames, wcs = _frames(STARS)
    gaia = [_gaia_at(40, 30, sid="s1"), _gaia_at(-50, -40, sid="s2"), _gaia_at(60, -55, sid="s3"),
            _gaia_at(-20, 45, -0.8, 10.4, sid="mover")]
    res = analyze_series(frames, wcs, SCALE, NATIVE, RA0, DEC0, gaia=gaia, catwise=[], series="D6")
    dt_mid = (np.mean(T) - T[0]) / 365.25
    mover = _nearest(res["tracks"], -20.0 - 0.8 * dt_mid, 45.0 + 10.4 * dt_mid)
    # The Gaia row's J2016 position is ~100" from where the star is in these
    # frames; it can only match if its motion is propagated.
    assert mover["gaia"]["status"] == "matched"
    assert mover["gaia"]["source"]["source_id"] == "mover"
    assert mover["gaia"]["separation_arcsec"] < 1.5
    assert mover["gaia"]["pm_difference_sigma"] < 3
    assert mover["label"] == "catalogued"
    assert any("known moving object" in r for r in mover["reasons"])
    variable = _nearest(res["tracks"], -70.0, 10.0)
    assert variable["gaia"]["status"] == "unmatched"
    assert all(r["applied"] and r["n_reference_stars"] >= 3 for r in res["registration"])


def test_registration_shift_is_measured_and_removed():
    """Shift one whole frame by 4" east: without the Gaia-based correction every
    star would show a spurious motion."""
    frames, wcs = _frames(STARS[:3] + [{"east": 0.0, "north": 0.0, "flux": 70.0}, {"east": 20.0, "north": -70.0, "flux": 65.0}])
    shift_px = 4.0 / SCALE
    from scipy.ndimage import shift as nd_shift
    frames[2].image = nd_shift(frames[2].image, (0, -shift_px), order=3, mode="nearest")  # -x = east
    gaia = [_gaia_at(40, 30, sid="s1"), _gaia_at(-50, -40, sid="s2"), _gaia_at(60, -55, sid="s3"),
            _gaia_at(0, 0, sid="s4"), _gaia_at(20, -70, sid="s5")]
    res = analyze_series(frames, wcs, SCALE, NATIVE, RA0, DEC0, gaia=gaia, catwise=[], series="D6")
    assert res["registration"][2]["offset_east_arcsec"] == pytest.approx(4.0, abs=0.5)
    assert abs(res["registration"][0]["offset_east_arcsec"]) < 0.5
    assert all(t["motion"]["pm_significance"] < 3 for t in res["tracks"])


def test_single_epoch_detection_is_low_reliability_uncertain():
    stars = STARS[:3] + [{"east": -20.0, "north": -70.0, "flux": lambda i: 60.0 if i == 1 else 0.0}]
    frames, wcs = _frames(stars)
    res = analyze_series(frames, wcs, SCALE, NATIVE, RA0, DEC0, gaia=[], catwise=[], series="D6")
    one = [t for t in res["tracks"] if t["n_epochs"] == 1]
    assert len(one) == 1
    assert one[0]["reliability"] == "low" and one[0]["label"] == "uncertain"
    assert one[0]["motion"]["pmra_mas_yr"] is None


def test_catalog_unavailable_is_not_reported_as_unmatched():
    frames, wcs = _frames(STARS[:2])
    res = analyze_series(frames, wcs, SCALE, NATIVE, RA0, DEC0, gaia=None, catwise=None, series="D6")
    for t in res["tracks"]:
        assert t["gaia"]["status"] == "catalog_unavailable"
        assert t["score_components"]["no_catalogue_counterpart"] == 0.0
        assert t["label"] == "unverified"


# ------------------------------------------------------------ typed-coordinate epoch lookup

def _catwise(sid, ra, dec, pm_e, pm_n, pm_err=150.0, pos_err=0.15):
    return CatalogSource(catalog="catwise2020", source_id=sid, ra_deg=ra, dec_deg=dec, ref_mjd=CATWISE_REF_MJD,
                         pos_err_arcsec=pos_err, pmra_mas_yr=pm_e, pmdec_mas_yr=pm_n, pm_err_mas_yr=pm_err, mags={"W2": 15.8})


def test_typed_position_matches_track_and_merges_duplicate_rows():
    # Synthetic rows shaped like the three CatWISE fragments of one fast mover.
    rows = [
        _catwise("frag_a", 133.78392, -7.24434, -8021, 1036, 141),
        _catwise("frag_b", 133.78398, -7.24420, -7965, 806, 265),
        _catwise("frag_c", 133.78356, -7.24405, -7920, 494, 235, pos_err=0.47),
        _catwise("slow_at_typed", 133.79513, -7.24515, -250, 60, 40),      # below the motion threshold
        _catwise("noisy", 133.80, -7.23, -420, -9, 261),                    # not significant
    ]
    match, reason = match_typed_position_to_track(133.79513, -7.24515, rows)
    assert match is not None and match.n_merged_rows == 3
    assert 2009.5 < match.matched_epoch_year < 2011.5
    assert -8021 < match.source.pmra_mas_yr < -7920
    assert "3 catalogue rows" in reason


def test_typed_position_two_distinct_movers_is_ambiguous_and_not_applied():
    rows = [_catwise("a", 133.78392, -7.24434, -8021, 1036), _catwise("b", 133.79513, -7.25000, 0, 3000, 100)]
    match, reason = match_typed_position_to_track(133.79513, -7.24515, rows)
    assert match is None and "ambiguous" in reason


def test_typed_position_no_mover():
    match, reason = match_typed_position_to_track(10.0, 10.0, [_catwise("far", 10.01, 10.01, 30, -20, 10)])
    assert match is None and "no catalogued fast mover" in reason


def test_faint_detection_next_to_a_much_brighter_star_is_flagged_not_removed():
    """SYNTHETIC: a faint star 28" from one ~40x brighter (inside the
    Lanczos ringing annulus) keeps its track but is marked unreliable; the
    same faint star far from anything bright is not flagged.
    (At 20" the faint star is not detected at all: it merges into the bright
    star's 5-sigma segment and the deblender, contrast 0.05, does not split
    off a component with 1% of the flux. Known limitation, docs/PHASE2.md.)"""
    stars = [{"east": 0.0, "north": 0.0, "flux": 2500.0},
             {"east": 28.0, "north": 0.0, "flux": 60.0},
             {"east": -60.0, "north": -60.0, "flux": 60.0}]
    frames, wcs = _frames(stars, noise=0.2)  # at very high S/N the 5-sigma isophotes merge (see docstring)
    res = analyze_series(frames, wcs, SCALE, NATIVE, RA0, DEC0, gaia=[], catwise=[], series="D6")
    near = _nearest(res["tracks"], 28.0, 0.0)
    far = _nearest(res["tracks"], -60.0, -60.0)
    assert near["n_epochs"] == 4 and "near_bright_star" in near["flags"]
    assert near["reliability"] == "low" and near["label"] == "uncertain"
    assert any("ringing" in c for c in near["caveats"])
    assert "near_bright_star" not in far["flags"]
    bright = _nearest(res["tracks"], 0.0, 0.0)
    assert "near_bright_star" not in bright["flags"]
