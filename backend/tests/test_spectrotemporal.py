"""Spectro-temporal measurements and metrics (app/spectrotemporal.py).

SYNTHETIC data with known answers (built in-test) for the mathematics; the
real-data regression tests are in test_real_spectrotemporal.py.
"""
import math

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from scipy.special import erf

from app.irsa import Cutout, ImageRecord
from app.spectrotemporal import (
    FLAGGED_CONTROL_RANGE, Measurement, _assess, analyse_source, assign_measurement_epochs, fit_detector,
    measure_exposure, metric_colour_change, metric_grey_variability, metric_matched_pairs, metric_slope_change,
    reference_distribution,
)

SCALE = 6.15
N = 41
RA0, DEC0 = 150.0, 20.0


# ---------------------------------------------------------------- synthetic exposures

def _cutout(flux_mjy=50.0, sky=2.0, noise=0.0, mjd=61000.0, psf_fwhm=9.0, x0=20.3, y0=19.7, seed=0,
            nan_at=None, flag_at=None, flag_name="OVERFLOW", obs_id="syn", det="D6"):
    """One synthetic SPHEREx-like exposure (native 6.15" pixels, MJy/sr), with a
    Gaussian star integrated over pixels whose TOTAL flux is flux_mjy."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]; w.wcs.crval = [RA0, DEC0]
    w.wcs.crpix = [N / 2 + 0.5, N / 2 + 0.5]; w.wcs.cdelt = [-SCALE / 3600, SCALE / 3600]
    hdr = w.to_header()
    hdr["BUNIT"] = "MJy / sr"; hdr["PSF_FWHM"] = psf_fwhm; hdr["MJD-AVG"] = mjd + 0.0007
    omega = (SCALE / 206264.80624709636) ** 2
    s = psf_fwhm / 2.3548 / SCALE * math.sqrt(2)
    idx = np.arange(N)
    gx = 0.5 * (erf((idx + 0.5 - x0) / s) - erf((idx - 0.5 - x0) / s))
    gy = 0.5 * (erf((idx + 0.5 - y0) / s) - erf((idx - 0.5 - y0) / s))
    star_sb = flux_mjy / (omega * 1e9) * np.outer(gy, gx)        # MJy/sr per pixel
    rng = np.random.default_rng(seed)
    img = sky + star_sb + rng.normal(0, noise, (N, N)) if noise else sky + star_sb
    var = np.full((N, N), max(noise, 1e-6) ** 2)
    flags = np.zeros((N, N), np.int32)
    fh = fits.Header(); fh["MSKN0001"] = "OVERFLOW"; fh["MSKN0010"] = "HOT"; fh["MSKN0026"] = "BLOOM"
    bit = {"OVERFLOW": 1, "HOT": 10, "BLOOM": 26}
    if nan_at:
        for (yy, xx) in nan_at:
            img[yy, xx] = np.nan
    if flag_at:
        for (yy, xx) in flag_at:
            flags[yy, xx] |= 1 << bit[flag_name]
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(img.astype(np.float32), hdr, name="IMAGE"),
                         fits.ImageHDU(flags, fh, name="FLAGS"), fits.ImageHDU(var.astype(np.float32), name="VARIANCE"),
                         fits.ImageHDU(np.zeros((N, N), np.float32), name="ZODI")])
    rec = ImageRecord(access_url="synthetic://", collection="synthetic", detector=det, time_mjd=mjd, obs_id=obs_id)
    return Cutout(record=rec, hdul=hdul), w, star_sb, x0, y0


def _expected_aperture(star_sb, x0, y0, r_arcsec=9.0):
    from photutils.aperture import CircularAperture
    wmap = CircularAperture((x0, y0), r=r_arcsec / SCALE).to_mask(method="exact").to_image(star_sb.shape)
    return float(np.sum(wmap * star_sb)) * (SCALE / 206264.80624709636) ** 2 * 1e9


def _measure(c, w, x0, y0, **kw):
    sky = w.pixel_to_world(x0, y0)
    return measure_exposure(c, "syn", float(sky.ra.deg), float(sky.dec.deg), **kw)


def test_flux_unit_conversion_and_exact_aperture_recover_known_flux():
    c, w, star, x0, y0 = _cutout(flux_mjy=50.0)
    m = _measure(c, w, x0, y0)
    assert m.state == "valid"
    assert m.flux_mjy == pytest.approx(_expected_aperture(star, x0, y0), rel=1e-6)   # MJy/sr x sr -> mJy, sky removed
    assert 0.85 * 50 < m.flux_mjy < 50                                               # aperture loses the wings
    # Gaussian aperture correction is reported, never applied
    assert m.aperture_correction == pytest.approx(1 / (1 - math.exp(-81 / (2 * (9 / 2.3548) ** 2))))
    assert m.time_mjd == pytest.approx(61000.0007) and m.time_source == "MJD-AVG"
    assert m.wavelength_um is None          # no WCS-WAVE in this synthetic file: not invented


def test_flux_error_is_calibrated_on_noise():
    z = []
    for seed in range(150):
        c, w, star, x0, y0 = _cutout(flux_mjy=5.0, noise=0.5, seed=seed)
        m = _measure(c, w, x0, y0)
        z.append((m.flux_mjy - _expected_aperture(star, x0, y0)) / m.flux_err_mjy)
    assert np.std(z) == pytest.approx(1.0, rel=0.15)
    assert abs(np.mean(z)) < 0.25


def test_blanked_pixel_elsewhere_does_not_poison_the_sum():
    """Regression: a NaN outside the aperture used to turn the flux into NaN (0 * NaN)."""
    c, w, star, x0, y0 = _cutout(nan_at=[(2, 2)])
    m = _measure(c, w, x0, y0)
    assert m.state == "valid" and math.isfinite(m.flux_mjy)


@pytest.mark.parametrize("kw, state", [
    (dict(nan_at=[(20, 20)], flag_at=[(20, 20)], flag_name="OVERFLOW"), "saturated"),
    (dict(nan_at=[(20, 20)]), "masked"),
    (dict(flag_at=[(20, 20)], flag_name="HOT"), "masked"),
    (dict(flag_at=[(20, 21)], flag_name="OVERFLOW"), "valid_flagged"),
    (dict(flag_at=[(20, 21)], flag_name="BLOOM"), "valid_flagged"),
])
def test_measurement_states_are_distinct(kw, state):
    c, w, star, x0, y0 = _cutout(**kw)
    m = _measure(c, w, x0, y0)
    assert m.state == state
    if state in ("saturated", "masked"):
        assert m.flux_mjy is None and m.n_invalid_pixels >= 1     # no fabricated value
    if state == "valid_flagged":
        assert m.flux_mjy is not None and "OVERFLOW" in m.flags_in_aperture or "BLOOM" in m.flags_in_aperture


def test_below_threshold_gives_upper_limit_and_off_image():
    c, w, star, x0, y0 = _cutout(flux_mjy=0.2, noise=0.5, seed=4)
    m = _measure(c, w, x0, y0)
    assert m.state == "below_threshold"
    assert m.upper_limit_mjy == pytest.approx(max(m.flux_mjy, 0) + 3 * m.flux_err_mjy)
    c, w, star, _, _ = _cutout()
    assert _measure(c, w, 0.5, 20.0).state == "off_image"


def test_epochs_follow_time_gaps():
    ms = [Measurement(source_id="s", obs_id=f"o{i}", detector="D6", collection="c", time_mjd=t, time_source="MJD-AVG",
                      epoch_index=None, ra_deg=0, dec_deg=0, wavelength_um=4.5, bandwidth_um=0.04, flux_mjy=1.0,
                      flux_err_mjy=0.1, snr=10, upper_limit_mjy=None, state="valid", method="forced_aperture")
          for i, t in enumerate([61000, 61003, 61010, 61180, 61190, 61370])]
    ep = assign_measurement_epochs(ms)
    assert [m.epoch_index for m in ms] == [0, 0, 0, 1, 1, 2]
    assert len(ep) == 3 and ep[1]["mjd_min"] == 61180


# ---------------------------------------------------------------- spectro-temporal model

def _spectrum(lam):
    """A structured 'brown-dwarf-like' spectrum: rising continuum with a deep band at 4.65 um."""
    return 10.0 * (1 + 0.8 * (lam - 4.4)) * (1 - 0.6 * np.exp(-((lam - 4.65) / 0.08) ** 2))


def _meas(det="D6", amps=(1.0, 1.0, 1.0), tilts=(0.0, 0.0, 0.0), lam_range=(4.42, 5.0), n_per=6, noise=0.01,
          seed=0, spectrum=_spectrum, state="valid", lam0=4.7):
    """Synthetic measurements: each epoch samples DIFFERENT random wavelengths (as SPHEREx does)."""
    rng = np.random.default_rng(seed)
    out = []
    for e, (a, t) in enumerate(zip(amps, tilts)):
        lam = np.sort(rng.uniform(*lam_range, n_per))
        for k, l in enumerate(lam):
            true = a * spectrum(l) * (1 + t * (l - lam0))
            f = true * (1 + rng.normal(0, noise))
            out.append(Measurement(source_id="s", obs_id=f"e{e}k{k}", detector=det, collection="c",
                                   time_mjd=61000 + 182 * e + k * 0.5, time_source="MJD-AVG", epoch_index=e,
                                   ra_deg=0, dec_deg=0, wavelength_um=float(l), bandwidth_um=0.037,
                                   flux_mjy=float(f), flux_err_mjy=float(noise * true), snr=1 / noise,
                                   upper_limit_mjy=None, state=state, method="forced_aperture"))
    return out


def _smooth(lam):
    """A spectrum the shared-shape model can follow (gentle slope and curvature)."""
    return 10.0 * (1 + 0.8 * (lam - 4.4)) * (1 + 0.3 * (lam - 4.7) ** 2)


EPOCHS3 = [{"index": i, "mjd_min": 0, "mjd_max": 0, "mjd_mean": 0} for i in range(3)]


def test_structured_spectrum_never_produces_a_variability_claim():
    """The failure mode found on real data: a constant source with spectral
    structure, sampled at different wavelengths in each epoch, looked variable.
    The raw model p-value cannot protect against this (the degeneracy is
    real); the adequacy guard must. 300 constant sources x 3 noise levels."""
    claims = 0
    inconclusive = 0
    for noise in (0.01, 0.02, 0.05):
        for seed in range(100):
            res = analyse_source(_meas(seed=seed, n_per=8, noise=noise), EPOCHS3)
            for m in res["metrics"]:
                if m["metric"] in ("grey_variability", "slope_change", "matched_wavelength_pairs"):
                    claims += m["assessment"] == "significant_change"
                    inconclusive += m["assessment"] == "inconclusive"
    assert claims == 0
    assert inconclusive > 0      # the guard is what is doing the work


def test_smooth_constant_source_is_not_variable():
    ps = [metric_grey_variability(_meas(seed=s, n_per=8, spectrum=_smooth, noise=0.02), "D6")["p_value"] for s in range(60)]
    assert np.mean(np.array(ps) < 0.01) <= 0.05


def test_grey_variability_detected_and_amplitudes_recovered():
    m = metric_grey_variability(_meas(amps=(1.0, 0.7, 1.0), n_per=8, seed=2, spectrum=_smooth), "D6", sys_floor=0.0)
    assert m["status"] == "ok" and m["dof"] == 2 and m["p_value"] < 1e-6
    a = np.array(m["amplitudes_mjy"])
    assert a[1] / a[0] == pytest.approx(0.7, abs=0.04) and a[2] / a[0] == pytest.approx(1.0, abs=0.04)
    assert m["fractional_range"] == pytest.approx(0.3 / 0.9, abs=0.05)


def test_grey_p_values_are_uniform_for_a_constant_source():
    ps = np.array([metric_grey_variability(_meas(seed=s, n_per=8, spectrum=lambda l: 10 + 0 * l), "D6",
                                           sys_floor=0.0)["p_value"] for s in range(200)])
    assert np.mean(ps < 0.05) == pytest.approx(0.05, abs=0.04)
    assert np.mean(ps < 0.5) == pytest.approx(0.5, abs=0.12)


def test_slope_change_detected_only_when_injected():
    m0 = metric_slope_change(_meas(n_per=8, seed=5, spectrum=_smooth), "D6", sys_floor=0.0)
    m1 = metric_slope_change(_meas(tilts=(0.0, 0.0, 1.2), n_per=8, seed=5, spectrum=_smooth), "D6", sys_floor=0.0)
    assert m0["p_value"] > 0.01
    assert m1["p_value"] < 1e-4 and m1["tilts_per_um"][2] == pytest.approx(1.2, abs=0.3) and m1["tilts_per_um"][0] == 0.0


def test_colour_change_chromatic_vs_grey():
    def both(a6, a3, seed):
        d6 = _meas("D6", amps=a6, n_per=8, seed=seed, spectrum=_smooth)
        d3 = _meas("D3", amps=a3, n_per=8, seed=seed + 100, lam_range=(1.63, 2.41), lam0=2.0,
                   spectrum=lambda l: 30 * (1 - 0.3 * (l - 1.6)))
        return fit_detector(d6, "D6", sys_floor=0.0), fit_detector(d3, "D3", sys_floor=0.0)
    grey = metric_colour_change(*both((1, 0.6, 1), (1, 0.6, 1), 7), simultaneous_fraction=1.0)
    chrom = metric_colour_change(*both((1, 0.6, 1), (1, 1, 1), 7), simultaneous_fraction=1.0)
    assert grey["p_value"] > 0.01
    assert chrom["p_value"] < 1e-6 and chrom["max_change"] == pytest.approx(abs(math.log(0.6)), abs=0.08)


def test_matched_pairs_model_free():
    ms = _meas(amps=(1.0, 0.8, 1.0), n_per=10, seed=9, spectrum=_smooth)
    m = metric_matched_pairs(ms, "D6", sys_floor=0.0, tolerance_fraction=0.25)
    assert m["status"] == "ok"
    for p in m["pairs"]:
        assert abs(p["wavelengths_um"][0] - p["wavelengths_um"][1]) <= 0.25 * 0.037 + 1e-12
        assert p["epochs"][0] != p["epochs"][1]


def test_insufficient_data_is_reported_not_guessed():
    one_epoch = _meas(amps=(1.0,), tilts=(0.0,))
    assert metric_grey_variability(one_epoch, "D6")["status"] == "insufficient_data"
    sat = _meas(state="saturated")
    for m in sat:
        m.flux_mjy = None
    res = analyse_source(sat, [{"index": i, "mjd_min": 0, "mjd_max": 0, "mjd_mean": 0} for i in range(3)])
    assert res["fits"]["D6"] is None
    assert all(m["assessment"] == "inconclusive" for m in res["metrics"])
    assert res["state_counts"] == {"saturated": 18}


# ---------------------------------------------------------------- interpretation rules

def test_assessment_rules():
    base = {"status": "ok", "p_value": 1e-8, "sys_floor": 0.025}
    assert _assess(dict(base), None)["assessment"] == "significant_change"
    # mostly flagged data -> conditional, with the control comparison
    m = _assess(dict(base, flagged_fraction=1.0, fractional_range=0.5), None)
    assert m["assessment"] == "conditional" and "Barnard" in m["assessment_text"]
    # model does not describe the spectrum -> inconclusive, never a p-value claim
    m = _assess(dict(base, model_reduced_chi2=9.0), None, adequacy_limit=2.0)
    assert m["assessment"] == "inconclusive"
    # field stars show the same -> not unusual
    ref = {"n": 10, "values": [1e-9] + [0.5] * 9}
    assert _assess(dict(base), ref)["assessment"] == "not_unusual"
    # none of n reference stars: the bound 1/(n+1) is stated, not "zero"
    m = _assess(dict(base), {"n": 5, "values": [0.5] * 5})
    assert m["reference_rarity_upper_bound"] == pytest.approx(1 / 6) and "1 in 6" in m["assessment_text"]
    for m in (_assess(dict(base), None), _assess(dict(base, flagged_fraction=1.0), None)):
        assert "probability that" not in m["assessment_text"].lower()
    assert FLAGGED_CONTROL_RANGE == pytest.approx(0.07)


def test_reference_distribution_collects_p_values_and_fit_quality():
    stars = [analyse_source(_meas(seed=s, n_per=8, spectrum=_smooth), EPOCHS3) for s in range(6)]
    ref = reference_distribution(stars)
    assert ref["grey:D6"]["n"] == 6 and all(0 <= v <= 1 for v in ref["grey:D6"]["values"])
    assert ref["fit:D6"]["n"] == 6 and "p84" in ref["fit:D6"]
