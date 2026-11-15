"""Spectro-temporal regression tests on REAL SPHEREx observations of the five
NASA targets (cached cutouts in tests/data/real, no network). These guard the
scientific behaviour: what may and may not be concluded from the data."""
import json

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u

from app.jsonsafe import clean
from app.offline import REAL_DATA_DIR, run_cached
from app.pipeline import central_wavelength

pytestmark = pytest.mark.skipif(not (REAL_DATA_DIR / "barnard" / "manifest.json").exists(),
                                reason="real-data cache not present")

_RUNS = {}


def target(name):
    if name not in _RUNS:
        _RUNS[name] = run_cached(name, spectrotemporal=True).analysis["spectrotemporal"]
    st = _RUNS[name]
    return st, next(s for s in st["sources"] if s["kind"] == "target")


def metric(src, name, det=None, data="all_usable"):
    for m in src["metrics"]:
        if m["metric"] == name and m["data"] == data and (det is None or m.get("detector") == det):
            return m
    raise KeyError((name, det, data))


def test_payload_is_strict_json_and_keeps_raw_measurements():
    st, t = target("wise0855")
    json.dumps(clean(st), allow_nan=False)
    assert len(t["measurements"]) == st["n_exposures"] == 33
    for m in t["measurements"]:
        assert m["time_source"] == "MJD-AVG" and m["method"] == "forced_aperture"
        assert m["state"] in ("valid", "valid_flagged", "below_threshold", "saturated", "masked", "off_image", "no_background")


def test_wavelength_metadata_matches_the_files():
    """Each measurement's wavelength is the WCS-WAVE value at the measured position."""
    from astropy.io import fits
    st, t = target("rw_aur")
    manifest = json.loads((REAL_DATA_DIR / "rw_aur" / "manifest.json").read_text())
    files = {e["obs_id"] + e["detector"]: e["file"] for e in manifest["exposures"]}
    checked = 0
    for m in t["measurements"][:8]:
        with fits.open(REAL_DATA_DIR / "rw_aur" / files[m["obs_id"] + m["detector"]]) as h:
            lam, bw = central_wavelength(h, SkyCoord(m["ra_deg"] * u.deg, m["dec_deg"] * u.deg))
        assert m["wavelength_um"] == pytest.approx(lam, abs=1e-6) and m["bandwidth_um"] == pytest.approx(bw, abs=1e-6)
        lo, hi = {"D3": (1.60, 2.42), "D6": (4.40, 5.02)}[m["detector"]]
        assert lo <= m["wavelength_um"] <= hi
        checked += 1
    assert checked == 8


def test_dichroic_pairs_are_simultaneous():
    st, t = target("rw_aur")
    by_obs = {}
    for m in t["measurements"]:
        by_obs.setdefault(m["obs_id"], []).append(m)
    pairs = [v for v in by_obs.values() if len(v) == 2]
    assert t["simultaneous_pairs"] == len(pairs) >= 15
    for a, b in pairs:
        assert {a["detector"], b["detector"]} == {"D3", "D6"}
        assert abs(a["time_mjd"] - b["time_mjd"]) < 1e-6


def test_r_dra_refuses_every_spectral_and_variability_conclusion():
    st, t = target("r_dra")
    assert set(t["state_counts"]) <= {"saturated", "off_image", "masked"}
    assert t["state_counts"].get("saturated", 0) >= 25
    assert all(f is None for f in t["fits"].values())
    assert all(m["assessment"] == "inconclusive" for m in t["metrics"])
    assert all(m["flux_mjy"] is None for m in t["measurements"])
    assert st["retrieval_failed_at_caching"] == 7


def test_barnard_saturated_in_d3_and_flagged_in_d6_never_claimed_variable():
    st, t = target("barnard")
    d3 = [m for m in t["measurements"] if m["detector"] == "D3"]
    assert all(m["state"] in ("saturated", "off_image") for m in d3)
    assert t["fits"]["D3"] is None
    d6 = [m for m in t["measurements"] if m["detector"] == "D6" and m["flux_mjy"] is not None]
    assert d6 and all(m["state"] == "valid_flagged" for m in d6)
    # a quiet star: no metric may claim a significant change
    assert all(m["assessment"] != "significant_change" for m in t["metrics"])
    # its epoch amplitudes agree to within the 7 % flagged-data control
    a = np.array(t["fits"]["D6"]["amplitudes_mjy"])
    assert (a.max() - a.min()) / a.mean() < 0.08


def test_wise0855_faint_source_handled_honestly():
    st, t = target("wise0855")
    assert t["state_counts"].get("below_threshold", 0) >= 3        # upper limits, not fake detections
    for m in t["measurements"]:
        if m["state"] == "below_threshold":
            assert m["upper_limit_mjy"] == pytest.approx(max(m["flux_mjy"], 0) + 3 * m["flux_err_mjy"])
    for d in ("D5", "D6"):
        assert t["fits"][d]["model_adequate"] is True
        assert metric(t, "grey_variability", d)["assessment"] == "no_significant_change"
    a5 = t["fits"]["D5"]["amplitudes_mjy"]
    assert all(0.3 < a < 0.9 for a in a5)                              # ~0.5 mJy, consistent with Phase 2


def test_eps_ind_b_structured_spectrum_is_inconclusive_not_variable():
    st, t = target("eps_ind_b")
    for d in ("D3", "D6"):
        assert t["fits"][d]["model_adequate"] is False
        assert metric(t, "grey_variability", d)["assessment"] == "inconclusive"
    assert all(m["assessment"] != "significant_change" for m in t["metrics"])


def test_rw_aur_dimming_is_chromatic_but_conditional():
    st, t = target("rw_aur")
    assert t["state_counts"]["valid_flagged"] >= 30
    p6, p3 = metric(t, "matched_wavelength_pairs", "D6"), metric(t, "matched_wavelength_pairs", "D3")
    assert p6["median_ratio_later_over_earlier"] < 0.8          # D6 fades between epochs
    assert abs(p3["median_ratio_later_over_earlier"] - 1) < 0.1  # D3 does not, typically
    for m in (p6, p3):
        assert m["assessment"] == "conditional"                 # flagged data: never an unconditional claim
        assert "OVERFLOW" in m["assessment_text"]
    a6 = t["fits"]["D6"]["amplitudes_mjy"]
    assert a6[0] > a6[1] > a6[2] and a6[2] / a6[0] < 0.6
    # without flagged data nothing can be tested, and the report says so
    assert all(m["status"] == "insufficient_data" for m in t["metrics"] if m["data"] == "unflagged_only")


def test_reference_stars_come_from_the_same_field_and_are_used():
    st, _ = target("rw_aur")
    assert st["reference"]["n_stars"] >= 5
    assert "fit:D6" in st["reference"]["metrics"]
