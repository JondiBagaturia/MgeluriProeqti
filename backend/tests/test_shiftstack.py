"""Shift-and-stack tests.

ALL DATA HERE IS SYNTHETIC: Gaussian point sources with injected motion on
Gaussian noise. Nothing in this file is a SPHEREx observation.
"""
import math

import numpy as np
import pytest

from app.shiftstack import ShiftStackConfig, StackFrame, forced, prepare, search, stack_at

SCALE = 3.075
SIGMA_PX = 9.0 / 2.3548 / SCALE
SHAPE = (81, 81)
# 4 visits x 4 exposures over one year
TIMES = np.array([61000 + 122 * e + 0.3 * k for e in range(4) for k in range(4)], float)
T_REF = float(TIMES.mean())
MF_NORM = math.sqrt(4 * math.pi) * SIGMA_PX   # flux for matched-filter S/N = 1 at unit pixel noise


def _frames(sources, seed=0, noise=1.0, drop=()):
    """sources: list of (x_ref, y_ref, pm_east, pm_north ["/yr], flux)"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    frames = []
    for i, t in enumerate(TIMES):
        img = rng.normal(0, noise, SHAPE) + 3.0
        dt = (t - T_REF) / 365.25
        for (x0, y0, ve, vn, flux) in sources:
            x = x0 - ve * dt / SCALE
            y = y0 + vn * dt / SCALE
            g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * SIGMA_PX ** 2))
            img += flux * g / (2 * math.pi * SIGMA_PX ** 2)
        if i in drop:
            img[:] = np.nan
        frames.append(StackFrame(image=img, variance=np.full(SHAPE, noise ** 2), time_mjd=float(t)))
    return frames


MOVER = (30.0, 45.0, -8.0, 1.0, 2.2 * MF_NORM)     # S/N ~2.2 per frame -> ~8.8 stacked
STATIC = (58.0, 25.0, 0.0, 0.0, 12.0 * MF_NORM)    # bright stationary star
CFG = ShiftStackConfig(n_controls=12)


def test_single_frame_does_not_detect_but_true_trajectory_stack_does():
    frames = _frames([MOVER], seed=1)
    prep = prepare(frames, SCALE, CFG)
    one = prepare(frames[:1], SCALE, CFG)
    s1, _ = stack_at(one, 0, 0)
    assert s1[45 - 3:45 + 4, 30 - 6:30 + 7].max() < 5.0            # invisible in one frame
    s_true, _ = stack_at(prep, -8.0, 1.0, t_ref=T_REF)
    s_static, _ = stack_at(prep, 0.0, 0.0, t_ref=T_REF)
    assert s_true[45, 30] > 6.5
    # The right motion beats no motion by the criterion the search uses
    # (S/N^2 gain >= 11.8). Note how modest the gain is: 8" of motion under a
    # 9" PSF only raises S/N from ~7.4 to ~8.4.
    assert s_true[45, 30] ** 2 - s_static[42:49, 27:34].max() ** 2 > CFG.min_delta_chi2


def test_blind_search_recovers_velocity_and_position_and_beats_controls():
    res = search(_frames([MOVER, STATIC], seed=2), SCALE, CFG)
    moving = [c for c in res["candidates"] if c["moving"]]
    assert moving, res["candidates"]
    m = max(moving, key=lambda c: c["snr"])
    assert math.hypot(m["x"] - 30, m["y"] - 45) <= 2.0
    # A source at S/N ~9 under a 9" PSF over one year constrains its
    # motion to about 1.2"/yr per axis; allow 2.5 sigma.
    assert m["pm_east_arcsec_yr"] == pytest.approx(-8.0, abs=3.0)
    assert m["pm_north_arcsec_yr"] == pytest.approx(1.0, abs=3.0)
    assert res["controls"]["n"] == 12
    assert res["controls"]["false_alarm_probability"] == pytest.approx(1 / 13)   # no control as strong
    # the bright static star is found, and NOT called moving
    static = min(res["candidates"], key=lambda c: math.hypot(c["x"] - 58, c["y"] - 25))
    assert math.hypot(static["x"] - 58, static["y"] - 25) <= 2.0
    assert static["snr"] > 20 and not static["moving"]


def test_pure_noise_gives_no_convincing_mover():
    res = search(_frames([], seed=3), SCALE, CFG)
    p = res["controls"]["false_alarm_probability"]
    # Either nothing passes the thresholds, or what passes is no better than the shuffled controls.
    assert p is None or p > 0.15


def test_static_only_field_has_no_moving_candidate_above_controls():
    res = search(_frames([STATIC], seed=4), SCALE, CFG)
    star = min(res["candidates"], key=lambda c: math.hypot(c["x"] - 58, c["y"] - 25))
    assert not star["moving"]
    p = res["controls"]["false_alarm_probability"]
    assert p is None or p > 0.15


def test_missing_frames_and_unequal_noise_are_weighted_not_fatal():
    frames = _frames([MOVER], seed=5, drop=(2, 9))
    for f in frames[:4]:                       # first visit 3x noisier
        f.image = f.image + np.random.default_rng(9).normal(0, 2.8, SHAPE)
        f.variance = np.full(SHAPE, 1.0 + 2.8 ** 2)
    prep = prepare(frames, SCALE, CFG)
    s, _ = stack_at(prep, -8.0, 1.0, t_ref=T_REF)
    assert np.isfinite(s).all()
    assert s[44:47, 29:32].max() > 5.0


def test_forced_stack_separates_true_motion_from_controls():
    frames = _frames([MOVER], seed=6)
    res = forced(frames, SCALE, 30.0, 45.0, -8.0, 1.0, t_ref=T_REF, cfg=CFG)
    assert res["snr_on_trajectory"] > 6.5
    assert res["snr_on_trajectory"] > res["snr_same_position_reversed_motion"] + 2
    # Directions 90-270 degrees away are clearly worse. (30 degrees away is
    # NOT distinguishable: the end points differ by only ~4", half a PSF.)
    assert res["snr_on_trajectory"] ** 2 - max(res["snr_same_position_rotated_motion"][2:9]) ** 2 > CFG.min_delta_chi2
    assert res["off_position"]["fraction_at_least_as_high"] == 0.0
    # flux estimate is unbiased within its error (S/N ~ 9 -> ~11%)
    assert res["flux_sum_image_units"] == pytest.approx(MOVER[4], rel=0.35)


def test_forced_on_empty_sky_is_unremarkable():
    res = forced(_frames([], seed=7), SCALE, 30.0, 45.0, -8.0, 1.0, t_ref=T_REF, cfg=CFG)
    assert res["snr_on_trajectory"] < 4.0
    assert res["off_position"]["fraction_at_least_as_high"] > 0.01


def test_faint_mover_is_reported_even_when_bright_static_stars_outnumber_the_list():
    """SYNTHETIC: twelve bright static stars + one faint mover; max_candidates=10."""
    statics = [(15 + 10 * (k % 6), 12 + 12 * (k // 6), 0.0, 0.0, (20 + k) * MF_NORM) for k in range(12)]
    res = search(_frames(statics + [(30.0, 60.0, -8.0, 1.0, 2.4 * MF_NORM)], seed=11), SCALE,
                 ShiftStackConfig(n_controls=0, max_candidates=10))
    assert any(c["moving"] and math.hypot(c["x"] - 30, c["y"] - 60) <= 2 for c in res["candidates"])
    assert len(res["candidates"]) <= 11
