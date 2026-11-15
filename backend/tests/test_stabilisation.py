"""Regression tests for the docs/AUDIT.md stabilisation fixes.

ALL DATA IN THIS FILE IS SYNTHETIC (Gaussian stars on flat noise backgrounds
built in-test). None of it is a SPHEREx observation. These tests check the
arithmetic and the bookkeeping, not astrophysical results.
"""
import json
import math

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from app.coords import parse_position, ResolvedPosition
from app.irsa import ImageRecord
from app.jsonsafe import clean
from app.photometry import aperture_photometry, ARCSEC_PER_RAD
from app.pipeline import make_target_wcs, ProcessedCutout
from app.spectrum import _cap_records
from app.stream import _GridContext, _TimingStats, _combine_and_encode

SCALE = 3.075


def _gaussian_star(shape, x0, y0, total_sb_sum, fwhm_pix, bkg=10.0, noise=0.0, seed=0):
    """Synthetic image in MJy/sr whose background-subtracted pixel SUM is total_sb_sum."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    sigma = fwhm_pix / 2.3548
    g = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
    g *= total_sb_sum / g.sum()
    rng = np.random.default_rng(seed)
    return (bkg + g + rng.normal(0, noise, shape)).astype(np.float64)


# ---------------------------------------------------------------- photometry

def test_photometry_recovers_synthetic_flux_with_large_aperture():
    total_sb = 500.0
    img = _gaussian_star((61, 61), 30, 30, total_sb, fwhm_pix=3.0)
    var = np.full(img.shape, 0.01)
    # Aperture big enough to contain ~all the Gaussian -> no aperture loss.
    res = aperture_photometry(img, var, 30, 30, SCALE, r_ap_arcsec=20, r_in_arcsec=40, r_out_arcsec=80)
    assert res.status == "ok"
    expected_mjy = total_sb * (SCALE / ARCSEC_PER_RAD) ** 2 * 1e9
    assert res.flux_mjy == pytest.approx(expected_mjy, rel=1e-3)
    assert res.background_mjy_sr == pytest.approx(10.0, abs=1e-6)
    assert res.flux_err_mjy > 0


def test_photometry_error_matches_scatter_of_noise_realisations():
    """Propagated error should agree with the empirical scatter over noise draws (within ~25%)."""
    fluxes, errs = [], []
    for seed in range(200):
        img = _gaussian_star((61, 61), 30, 30, 200.0, fwhm_pix=3.0, noise=0.5, seed=seed)
        var = np.full(img.shape, 0.25)
        r = aperture_photometry(img, var, 30, 30, SCALE)
        fluxes.append(r.flux_mjy)
        errs.append(r.flux_err_mjy)
    assert np.std(fluxes) == pytest.approx(np.mean(errs), rel=0.25)


def test_photometry_refuses_when_core_is_masked():
    """Saturated core (NaN) must give NO number, not a wing-interpolated one."""
    img = _gaussian_star((61, 61), 30, 30, 500.0, fwhm_pix=3.0)
    img[29:32, 29:32] = np.nan
    res = aperture_photometry(img, np.ones_like(img), 30, 30, SCALE)
    assert res.status == "masked_in_aperture"
    assert res.flux_mjy is None
    assert res.n_invalid_in_aperture == 9


def test_photometry_refuses_display_filled_pixels():
    img = _gaussian_star((61, 61), 30, 30, 500.0, fwhm_pix=3.0)
    filled = np.zeros(img.shape, bool)
    filled[30, 30] = True
    res = aperture_photometry(img, np.ones_like(img), 30, 30, SCALE, extra_invalid=filled)
    assert res.status == "masked_in_aperture"


@pytest.mark.parametrize("x, y", [(-5.0, 30.0), (30.0, 70.0), (1.0, 30.0)])
def test_photometry_off_image_or_truncated(x, y):
    img = np.ones((61, 61))
    res = aperture_photometry(img, img, x, y, SCALE)
    assert res.status == "off_image"
    assert res.flux_mjy is None


def test_variance_inflation_scales_error_by_sqrt():
    img = _gaussian_star((61, 61), 30, 30, 200.0, fwhm_pix=3.0, noise=0.1)
    var = np.full(img.shape, 0.01)
    a = aperture_photometry(img, var, 30, 30, SCALE, variance_inflation=1.0)
    b = aperture_photometry(img, var, 30, 30, SCALE, variance_inflation=4.0)
    assert b.flux_mjy == pytest.approx(a.flux_mjy)
    assert b.flux_err_mjy == pytest.approx(2 * a.flux_err_mjy, rel=0.05)


# ---------------------------------------------------------------- JSON safety

def test_clean_makes_nan_strict_json():
    obj = {"a": float("nan"), "b": [np.float32(np.inf), np.int64(3), np.float64(1.5)], "c": np.bool_(True)}
    out = json.dumps(clean(obj), allow_nan=False)
    assert json.loads(out) == {"a": None, "b": [None, 3, 1.5], "c": True}


# ---------------------------------------------------------------- coordinates

@pytest.mark.parametrize("text", ["08:55:10.831 -07:14:42.53", "08 55 10.831 -07 14 42.53", "08:55:10.831, -07:14:42.53"])
def test_separated_sexagesimal_parsed_as_coordinates_not_name(text):
    pos = parse_position(text)
    assert pos.source == "coords"
    assert pos.coord.ra.deg == pytest.approx(133.795129, abs=1e-5)
    assert pos.coord.dec.deg == pytest.approx(-7.245147, abs=1e-5)


def test_search_position_uses_survey_epoch_for_high_pm():
    # Barnard's Star SIMBAD astrometry (J2000 position + PM): the search
    # cone must be at the survey-era position, ~4.5' from J2000.
    pos = ResolvedPosition(
        coord=SkyCoord(269.45208, 4.69336, unit="deg"), ref_epoch=Time("J2000"),
        pmra=-801.551, pmdec=10362.394, parallax_mas=546.98, source="simbad",
    )
    sep = pos.coord.separation(pos.search_position()).arcsec
    assert 260 < sep < 285


# ---------------------------------------------------------------- grid orientation

def test_grid_orientation_row0_is_south_and_col0_is_east():
    """The frontend relies on this to draw north-up/east-left (AUDIT C5)."""
    wcs, shape = make_target_wcs(SkyCoord(100, 20, unit="deg"), 5, SCALE)
    c = shape[0] // 2
    centre = wcs.pixel_to_world(c, c)
    assert centre.ra.deg == pytest.approx(100) and centre.dec.deg == pytest.approx(20)
    assert wcs.pixel_to_world(c, 0).dec.deg < 20          # row 0 = south
    assert wcs.pixel_to_world(0, c).ra.deg > 100          # col 0 = east


def test_grid_target_pixel_follows_proper_motion():
    pos = ResolvedPosition(
        coord=SkyCoord(269.45208, 4.69336, unit="deg"), ref_epoch=Time("J2000"),
        pmra=-801.551, pmdec=10362.394, parallax_mas=546.98, source="simbad",
    )
    centre = pos.propagate_to(Time(61100, format="mjd"))
    wcs, shape = make_target_wcs(centre, 3, SCALE)
    grid = _GridContext(pos, wcs, shape, centre.ra.deg, centre.dec.deg, SCALE, 6.15)
    x0, y0, moved = grid.target_pixel(61100)
    x1, y1, _ = grid.target_pixel(61100 + 365.25)
    assert moved
    assert x0 == pytest.approx(shape[1] // 2, abs=1e-6) and y0 == pytest.approx(shape[0] // 2, abs=1e-6)
    # +10.36"/yr in Dec = +3.37 grid px north (rows increase northward)
    assert (y1 - y0) == pytest.approx(10.362 / SCALE, rel=0.02)


# ---------------------------------------------------------------- stream encoding

def _record(t, det="D6"):
    return ImageRecord(access_url="synthetic://", collection="synthetic", detector=det, time_mjd=t, obs_id=f"syn{t}")


def test_combine_and_encode_measures_before_gap_fill_and_is_strict_json():
    pos = ResolvedPosition(coord=SkyCoord(100, 20, unit="deg"), ref_epoch=Time("J2000"), source="coords")
    wcs, shape = make_target_wcs(pos.coord, 3, SCALE)
    grid = _GridContext(pos, wcs, shape, 100.0, 20.0, SCALE, 6.15)
    c = shape[0] // 2
    img = _gaussian_star(shape, c, c, 300.0, fwhm_pix=3.0).astype(np.float32)
    img[c - 1:c + 1, c - 1:c + 1] = np.nan  # a 4-px "saturated core" -> fill_small_gaps fills it
    var = np.full(shape, 0.01, np.float32)
    var[np.isnan(img)] = np.nan
    members = [ProcessedCutout(record=_record(61000.0), image=img, variance=var, flag_names={}, contamination={},
                               central_wavelength_um=float("nan"))]
    enc = _combine_and_encode(0, 1, members, _TimingStats(), grid=grid)
    meta = enc.metadata
    assert meta["n_gap_filled_pixels"] == 4
    assert len(meta["gap_filled_indices"]) == 4
    assert meta["photometry"]["status"] == "masked_in_aperture"  # measured BEFORE filling
    assert meta["photometry"]["flux_mjy"] is None
    assert meta["grid_wcs"]["cdelt"][1] > 0
    line = json.dumps(clean({**meta, "scale": enc.scale}), allow_nan=False)  # must not raise
    assert '"central_wavelength_um": null' in line


def test_combine_and_encode_clean_star_measures_ok():
    pos = ResolvedPosition(coord=SkyCoord(100, 20, unit="deg"), ref_epoch=Time("J2000"), source="coords")
    wcs, shape = make_target_wcs(pos.coord, 3, SCALE)
    grid = _GridContext(pos, wcs, shape, 100.0, 20.0, SCALE, 6.15)
    c = shape[0] // 2
    members = [
        ProcessedCutout(record=_record(61000.0 + i), image=_gaussian_star(shape, c, c, 300.0, 3.0, noise=0.05, seed=i).astype(np.float32),
                        variance=np.full(shape, 0.0025, np.float32), flag_names={}, contamination={})
        for i in range(3)
    ]
    meta = _combine_and_encode(0, 1, members, _TimingStats(), grid=grid).metadata
    assert meta["photometry"]["status"] == "ok"
    assert meta["photometry"]["snr"] > 10
    assert meta["target_pixel"] == {"x": pytest.approx(c), "y": pytest.approx(c), "proper_motion_applied": False}


# ---------------------------------------------------------------- spectrum cap

def test_spectrum_cap_spreads_across_detectors_and_time():
    recs = [_record(60000 + i, det=f"D{1 + i % 6}") for i in range(300)]
    capped = _cap_records(recs, 60)
    assert len(capped) <= 60
    assert {r.detector for r in capped} == {f"D{i}" for i in range(1, 7)}
    for d in ("D1", "D6"):
        times = [r.time_mjd for r in capped if r.detector == d]
        all_t = [r.time_mjd for r in recs if r.detector == d]
        assert min(times) == min(all_t) and max(times) == max(all_t)
