"""Parallax-aware fit. Positions are SYNTHETIC, generated from the model
with known parameters; the Earth ephemeris is astropy's."""
import math

import numpy as np
import pytest
from astropy.coordinates import CartesianRepresentation, SkyCoord, get_body_barycentric
from astropy.time import Time
import astropy.units as u

from app.astrometry import expected_parallax_error_mas, fit_motion_parallax, parallax_factors, parallax_is_constrained

RA, DEC = 269.45, 4.74          # Barnard's Star's direction
PLX, MU_E, MU_N = 0.547, -0.80, 10.36   # arcsec, arcsec/yr


def _positions(times, noise=0.0, seed=0):
    t = np.asarray(times, float)
    f_e, f_n = parallax_factors(t, RA, DEC)
    tau = (t - t.mean()) / 365.25
    rng = np.random.default_rng(seed)
    return (MU_E * tau + PLX * f_e + rng.normal(0, noise, len(t)), MU_N * tau + PLX * f_n + rng.normal(0, noise, len(t)))


def test_parallax_factors_have_the_right_size_and_follow_the_sun():
    t = np.linspace(61000, 61365, 73)
    f_e, f_n = parallax_factors(t, RA, DEC)
    amp = np.hypot(f_e, f_n)
    assert 0.95 < amp.max() < 1.03            # ~1 au seen from a star near the ecliptic plane... up to Earth's eccentricity
    # The apparent parallactic shift points towards the Sun's position on the sky.
    i = int(np.argmax(amp))
    # direction of the barycentre (~the Sun) as seen from the Earth
    earth = get_body_barycentric("earth", Time(t[i], format="mjd", scale="tdb"))
    sun = SkyCoord(CartesianRepresentation(-earth.xyz), frame="icrs")
    star = SkyCoord(RA * u.deg, DEC * u.deg)
    pa_sun = star.position_angle(sun).rad      # east of north
    pa_shift = math.atan2(f_e[i], f_n[i])
    assert abs(math.remainder(pa_sun - pa_shift, 2 * math.pi)) < math.radians(5)


def test_recovers_injected_parallax_and_motion_without_noise():
    times = 61000 + np.array([0, 91, 183, 274, 365, 456.0])
    e, n = _positions(times)
    fit = fit_motion_parallax(times, e, n, np.full(6, 0.05), RA, DEC)
    assert fit.parallax_mas == pytest.approx(547, abs=0.5)
    assert fit.pmra_mas_yr == pytest.approx(-800, abs=1)
    assert fit.pmdec_mas_yr == pytest.approx(10360, abs=1)
    assert parallax_is_constrained(fit)[0]


def test_parallax_error_is_calibrated():
    times = 61000 + np.array([0, 91, 183, 274, 365, 456.0])
    z = []
    for seed in range(300):
        e, n = _positions(times, noise=0.2, seed=seed)
        fit = fit_motion_parallax(times, e, n, np.full(6, 0.2), RA, DEC)
        z.append((fit.parallax_mas - 547) / expected_parallax_error_mas(times, 0.2, RA, DEC))
    assert np.std(z) == pytest.approx(1.0, rel=0.12)
    assert abs(np.mean(z)) < 0.2


def test_three_epochs_are_never_reported_as_constrained():
    times = np.array([61000, 61183, 61365.0])
    e, n = _positions(times)
    fit = fit_motion_parallax(times, e, n, np.full(3, 0.01), RA, DEC)
    ok, why = parallax_is_constrained(fit)
    assert not ok and "epochs" in why


def test_noisy_positions_are_not_reported_as_constrained():
    times = 61000 + np.array([0, 183, 365, 548.0])
    e, n = _positions(times, noise=1.0, seed=3)
    fit = fit_motion_parallax(times, e, n, np.full(4, 1.0), RA, DEC)
    ok, why = parallax_is_constrained(fit)
    assert not ok and ("sigma" in why or "negative" in why)


def test_degenerate_sampling_is_detected():
    # Four epochs bunched within three weeks: parallax looks like linear motion.
    times = np.array([61000, 61007, 61014, 61021.0])
    e, n = _positions(times)
    fit = fit_motion_parallax(times, e, n, np.full(4, 0.01), RA, DEC)
    assert fit.factor_independence < 0.2
    assert not parallax_is_constrained(fit)[0]
