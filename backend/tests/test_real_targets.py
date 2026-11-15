"""Regression tests on REAL SPHEREx observations of the five NASA targets.

Data: backend/tests/data/real/<target>/ (cached IRSA cutouts, retrieved
2026-11-15 by scripts/cache_validation_cutouts.py). NOT synthetic.

The production pipeline runs offline on them (app/offline.py). Catalogue
queries are disabled, so nothing here needs the network; the expected
catalogue motions are the SIMBAD values stored in each manifest.

The tolerances are deliberately loose (several sigma of the measured
errors): the tests exist to catch regressions, not to certify the numbers.
If the cache is refreshed with a new data release, review them.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from app.offline import REAL_DATA_DIR, run_cached, series_frames  # noqa: E402
from validate_targets import field_star_stats, target_track  # noqa: E402

pytestmark = pytest.mark.skipif(not (REAL_DATA_DIR / "barnard" / "manifest.json").exists(),
                                reason="real-data cache not present")

_RUNS = {}


def run(name):
    if name not in _RUNS:
        _RUNS[name] = run_cached(name)
    return _RUNS[name]


def fluxes(name, det):
    return [m["photometry"] for m in series_frames(run(name), det)]


def test_barnard_proper_motion_from_spherex_alone():
    t = target_track(run("barnard"), "D6")
    assert t["found"] and t["n_epochs"] == 3
    # catalogue: (-802, +10362) mas/yr
    assert t["pm_north_mas_yr"] == pytest.approx(10362, abs=3 * t["pm_err_mas_yr"])
    assert t["pm_east_mas_yr"] == pytest.approx(-802, abs=3 * t["pm_err_mas_yr"])
    assert t["pm_significance"] > 8
    assert t["rank"] == 1


def test_barnard_excess_scatter_is_its_parallax():
    """The linear fit's chi^2 is high (~9 on 2 dof) and its motion error is
    rescaled up to ~1.2"/yr. Physical cause: the 0.547" parallax (Gaia DR3)
    is not in the linear model and shifts the star by ~1" east-west between
    these epochs, more than the 0.4" positional floor. Removing the KNOWN
    parallax must bring chi^2 down to what the floor predicts (95% point of
    chi^2 with 2 dof is 5.99) and the error back to the floor-limited value.
    This is a check of the error model, not a parallax measurement."""
    import numpy as np
    from app.astrometry import parallax_factors
    from app.discovery import fit_motion

    r = run("barnard")
    t = target_track(r, "D6")
    series = next(s for s in r.analysis["series"] if s["series"] == "D6")
    track = next(x for x in series["tracks"] if x["rank"] == t["rank"])
    d = track["detections"]
    tm = [x["time_mjd"] for x in d]
    e = np.array([x["east_arcsec"] for x in d])
    n = np.array([x["north_arcsec"] for x in d])
    sig = [x["pos_err_arcsec"] for x in d]
    linear = fit_motion(tm, e, n, sig)
    assert linear.chi2 > 5.99 and linear.errors_rescaled
    fe, fn = parallax_factors(tm, track["ra_deg"], track["dec_deg"])
    plx = r.manifest["resolved"]["parallax_mas"] / 1000.0
    corrected = fit_motion(tm, e - plx * fe, n - plx * fn, sig)
    assert corrected.chi2 < 5.99
    assert corrected.pmra_err_mas_yr < 700
    # and the parallax-aware fit is NOT reported as a measurement (3 epochs)
    assert track["parallax_fit"] is not None and track["parallax_fit"]["constrained"] is False


def test_barnard_is_saturated_in_d3_and_says_so():
    assert all(p["status"] == "masked_in_aperture" and p["flux_mjy"] is None for p in fluxes("barnard", "D3"))
    f = np.array([p["flux_mjy"] for p in fluxes("barnard", "D6")])
    assert len(f) == 3 and f.std() / f.mean() < 0.08     # a quiet star stays quiet (was ~15% before Lanczos)


def test_eps_ind_b_detected_and_moving_as_catalogued():
    for det in ("D6", "D3"):
        t = target_track(run("eps_ind_b"), det)
        assert t["found"] and t["n_epochs"] == 4
        # catalogue: (+3982, -2467) mas/yr
        assert t["pm_east_mas_yr"] == pytest.approx(3982, abs=3 * t["pm_err_mas_yr"])
        assert t["pm_north_mas_yr"] == pytest.approx(-2467, abs=3 * t["pm_err_mas_yr"])
        assert t["pm_significance"] > 5
    f = np.array([p["flux_mjy"] for p in fluxes("eps_ind_b", "D6")])
    assert 12 < f.mean() < 25 and f.std() / f.mean() < 0.10


def test_wise0855_measured_at_the_predicted_position_in_every_d5_epoch():
    ph = fluxes("wise0855", "D5")
    assert len(ph) == 3
    for p in ph:
        assert p["status"] == "ok" and p["snr"] > 5
        assert 0.2 < p["flux_mjy"] < 0.8
    t = target_track(run("wise0855"), "D5")
    assert t["found"], "expected blind detection of WISE 0855 in D5"
    assert t["pm_east_mas_yr"] < -3000        # moving west, fast (catalogue: -8124)
    assert t["gaia"] == "catalog_unavailable"  # offline run: must not pretend it is unmatched


def test_rw_aur_fades_by_nearly_half_at_fixed_wavelength():
    frames = series_frames(run("rw_aur"), "D6")
    waves = [m["central_wavelength_um"] for m in frames]
    assert max(waves) - min(waves) < 0.02                      # same wavelength in all three epochs
    f = [m["photometry"]["flux_mjy"] for m in frames]
    assert f[0] > f[1] > f[2]
    assert f[0] / f[2] == pytest.approx(1.94, abs=0.25)
    t = target_track(run("rw_aur"), "D6")
    assert t["variability_p_value"] < 1e-6
    assert "wavelength_mismatch" not in t["flags"]
    assert any("Brightness changes" in r for r in t["reasons"])
    assert t["pm_significance"] < 3                              # variable, not moving


def test_r_dra_is_saturated_everywhere_and_never_given_a_flux():
    for det in ("D3", "D6"):
        ph = fluxes("r_dra", det)
        assert len(ph) == 3
        assert all(p["status"] == "masked_in_aperture" and p["flux_mjy"] is None for p in ph)
        # ... and its own track is never offered as a reliable candidate
        t = target_track(run("r_dra"), det)
        assert t["found"] and "core_masked_all_epochs" in t["flags"]
        assert t["reliability"] == "low" and t["label"] == "uncertain"
        assert t["variability_p_value"] is None   # no flux, so no variability claim


def test_field_star_repeatability_supports_the_error_floors():
    stats = field_star_stats({n: run(n) for n in ("barnard", "eps_ind_b", "wise0855", "rw_aur", "r_dra")})
    bright = [s for s in stats["stars"] if s["snr"] >= 30]
    assert len(bright) >= 25
    pos = np.array([s["pos_rms_arcsec"] for s in bright])
    flux = np.array([s["flux_scatter_fraction"] for s in bright])
    # floors in app/discovery.py: 0.4" and 6%
    assert np.median(pos) < 0.4 and np.percentile(pos, 84) < 0.5
    assert np.median(flux) < 0.06


def test_wise0855_blind_shift_and_stack_finds_one_mover_at_the_right_place_and_velocity():
    """Blind in position and velocity. Uses the D5+D6 single exposures (33 real frames)."""
    import math
    import time as _time

    from astropy.time import Time
    from app.shiftstack import ShiftStackConfig, StackFrame, search
    from validate_targets import grid_wcs

    r = run_cached("wise0855", mode="single", analyze=False)
    exps = sorted(r.exposures, key=lambda p: p.record.time_mjd)
    frames = [StackFrame(image=p.image, variance=p.variance, time_mjd=p.record.time_mjd) for p in exps]
    scale = r.frames[0]["grid_scale_arcsec"]
    t0 = _time.perf_counter()
    res = search(frames, scale, ShiftStackConfig(n_controls=10))
    elapsed = _time.perf_counter() - t0

    moving = [c for c in res["candidates"] if c["moving"]]
    assert len(moving) == 1, moving
    m = moving[0]
    t_ref = res["epoch_ref_mjd"]
    pred = r.resolved.propagate_to(Time(t_ref, format="mjd"))
    px, py = (float(v) for v in grid_wcs(r.frames[0]).world_to_pixel(pred))
    assert math.hypot(m["x"] - px, m["y"] - py) * scale < 6.0           # within one native pixel of the prediction
    # catalogue: (-8.12, +0.67) "/yr
    assert m["pm_east_arcsec_yr"] == pytest.approx(-8.12, abs=2.5)
    assert m["pm_north_arcsec_yr"] == pytest.approx(0.67, abs=2.5)
    assert m["snr"] > 10 and m["delta_chi2"] > 11.8
    # none of the time-shuffled controls produced a moving peak this strong
    assert res["controls"]["false_alarm_probability"] == pytest.approx(1 / 11)
    # bright static stars in the field are found and not called moving
    assert sum(1 for c in res["candidates"] if c["snr"] > 20 and not c["moving"]) >= 3
    assert elapsed < 120


def test_rw_aur_field_ringing_lobe_is_flagged():
    """REAL data: in the RW Aur field (D3), Lanczos-3 leaves a faint detection
    (peak S/N 7-9, ~2% of its neighbour) ~19" from a field star of S/N ~360,
    in all three epochs. It is absent with bilinear resampling and sits at the
    ringing lobe measured on synthetic stars. It must be kept but flagged and
    marked low reliability, never offered as a reliable candidate."""
    import math

    series = next(s for s in run("rw_aur").analysis["series"] if s["series"] == "D3")
    lobe = [t for t in series["tracks"] if abs(t["ra_deg"] - 76.92983) < 3e-4 and abs(t["dec_deg"] - 30.40419) < 3e-4]
    assert len(lobe) == 1
    t = lobe[0]
    assert t["n_epochs"] == 3 and max(d["peak_snr"] for d in t["detections"]) < 15
    assert "near_bright_star" in t["flags"]
    assert t["reliability"] == "low" and t["label"] == "uncertain"
    assert any("ringing" in c for c in t["caveats"])
    # every flagged track in the field is faint relative to its bright neighbour
    for x in series["tracks"]:
        if "near_bright_star" in x["flags"]:
            assert x["detection_snr_median"] < 50
