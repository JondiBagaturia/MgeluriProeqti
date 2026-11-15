#!/usr/bin/env python3
"""Benchmark resampling methods for SPHEREx-like data. SYNTHETIC DATA ONLY.

Simulates what the pipeline does to one epoch: N dithered exposures of a
point source on 6.15" native pixels (Gaussian PSF, FWHM 9", integrated
exactly over each pixel; random sub-pixel phase and field rotation per
exposure) are resampled onto the 3.075" common grid and averaged.

For each method it measures, over many random realisations:
  centroid   RMS error of the coadd centroid for a bright, noise-free star
             (the systematic floor a method adds) - two estimators:
             intensity-weighted first moment (what app/discovery.py uses)
             and a 2D Gaussian fit.
  photometry mean and fractional scatter of the r=9" aperture flux relative
             to the true total flux (noise-free: pure resampling effect).
  noise      (true variance of an aperture sum) / (sum of per-pixel
             variances in the aperture), from pure-noise exposures. 1 means
             pixels are independent; >1 is the correction an aperture error
             needs.
  difference two independent coadds of the same star, subtracted: largest
             |residual| as a fraction of the star's peak, i.e. the fake
             signal a stationary star leaves in a difference image.
  runtime    seconds per exposure resampled.

Methods:
  nearest, bilinear, lanczos3   app/resample.py
  adaptive                      reproject.reproject_adaptive (flux-conserving, Hann kernel)
  exact                         reproject.reproject_exact (flux-conserving, exact pixel overlap)
  native                        no resampling at all: centroid and aperture
                                flux measured on each exposure's own pixels
                                and averaged ("measure, then combine").
                                Not applicable to noise correlation or
                                difference images.

Usage: python scripts/benchmark_resampling.py [--trials 200] [--nexp 5] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from astropy.wcs import WCS
from scipy.special import erf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
from app.resample import resample  # noqa: E402

warnings.filterwarnings("ignore")

NATIVE = 6.15
GRID = 3.075
FWHM = 9.0
SIGMA_PIX = FWHM / 2.3548 / NATIVE
RA0, DEC0 = 150.0, 20.0
N_IN = 41
N_OUT = 61
R_AP = 9.0


def out_wcs() -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.crpix = [N_OUT / 2 + 0.5, N_OUT / 2 + 0.5]
    w.wcs.cdelt = [-GRID / 3600, GRID / 3600]
    return w


def exposure_wcs(rng) -> WCS:
    theta = rng.uniform(0, 2 * math.pi)
    s = NATIVE / 3600
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.crpix = [N_IN / 2 + 0.5 + rng.uniform(-0.5, 0.5), N_IN / 2 + 0.5 + rng.uniform(-0.5, 0.5)]
    w.wcs.cd = [[-s * math.cos(theta), s * math.sin(theta)], [s * math.sin(theta), s * math.cos(theta)]]
    return w


def render_star(w: WCS, ra: float, dec: float, flux: float) -> np.ndarray:
    """Circular Gaussian integrated exactly over each native pixel. Returns flux per pixel."""
    x0, y0 = (float(v) for v in w.all_world2pix(ra, dec, 0))
    idx = np.arange(N_IN)
    a = math.sqrt(2) * SIGMA_PIX
    gx = 0.5 * (erf((idx + 0.5 - x0) / a) - erf((idx - 0.5 - x0) / a))
    gy = 0.5 * (erf((idx + 0.5 - y0) / a) - erf((idx - 0.5 - y0) / a))
    return flux * np.outer(gy, gx), (x0, y0)


def resample_one(method: str, img: np.ndarray, w_in: WCS, w_out: WCS) -> np.ndarray:
    if method in ("nearest", "bilinear", "lanczos3"):
        return resample(img, None, w_in, w_out, (N_OUT, N_OUT), kernel=method)[0].astype(float)
    if method == "adaptive":
        from reproject import reproject_adaptive
        out, _ = reproject_adaptive((img, w_in), w_out, shape_out=(N_OUT, N_OUT), conserve_flux=False, kernel="hann")
        return out
    if method == "exact":
        from reproject import reproject_exact
        out, _ = reproject_exact((img, w_in), w_out, shape_out=(N_OUT, N_OUT))
        return out
    raise ValueError(method)


def coadd(method, images, wcss, w_out):
    return np.nanmean([resample_one(method, im, w, w_out) for im, w in zip(images, wcss)], axis=0)


def com_centroid(img: np.ndarray) -> tuple[float, float]:
    d = np.where(np.isfinite(img), img, 0.0)
    m = d > 0.05 * d.max()          # the bright footprint, like a detection segment
    yy, xx = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    s = d[m].sum()
    return float((xx[m] * d[m]).sum() / s), float((yy[m] * d[m]).sum() / s)


def gauss_centroid(img: np.ndarray) -> tuple[float, float]:
    from photutils.centroids import centroid_2dg
    d = np.where(np.isfinite(img), img, 0.0)
    py, px = np.unravel_index(np.argmax(d), d.shape)
    r = 7
    cut = d[max(py - r, 0):py + r + 1, max(px - r, 0):px + r + 1]
    cx, cy = centroid_2dg(cut)
    return float(cx + max(px - r, 0)), float(cy + max(py - r, 0))


def aperture_sum(img, x, y, scale, r_arcsec=R_AP):
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    m = np.hypot(xx - x, yy - y) * scale <= r_arcsec
    return float(np.nansum(img[m])), m


def run(trials: int, nexp: int, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    w_out = out_wcs()
    methods = ["nearest", "bilinear", "lanczos3", "adaptive", "exact"]
    acc = {m: {"com": [], "g2d": [], "phot": [], "diff": [], "noise_ap": [], "noise_pix": [], "time": []} for m in methods}
    acc["native"] = {"com": [], "g2d": [], "phot": []}
    flux_total = 1000.0
    area_ratio = (GRID / NATIVE) ** 2      # output pixels are 1/4 of a native pixel

    for _ in range(trials):
        # true position: anywhere within +-1 output pixel of the grid centre
        tx = (N_OUT - 1) / 2 + rng.uniform(-1, 1)
        ty = (N_OUT - 1) / 2 + rng.uniform(-1, 1)
        ra, dec = (float(v) for v in w_out.all_pix2world(tx, ty, 0))

        def make_epoch():
            ws = [exposure_wcs(rng) for _ in range(nexp)]
            rendered = [render_star(w, ra, dec, flux_total) for w in ws]
            return ws, [r[0] for r in rendered], [r[1] for r in rendered]

        ws_a, imgs_a, pos_a = make_epoch()
        ws_b, imgs_b, _ = make_epoch()
        noise = [rng.normal(0, 1.0, (N_IN, N_IN)) for _ in range(nexp)]

        # --- native: measure each exposure on its own pixels, then average ---
        cx, cy, ph = [], [], []
        for w, im, (x0, y0) in zip(ws_a, imgs_a, pos_a):
            mx, my = com_centroid(im)
            sky = w.all_pix2world(mx, my, 0)
            ox, oy = w_out.all_world2pix(sky[0], sky[1], 0)
            cx.append(float(ox)); cy.append(float(oy))
            s, _ = aperture_sum(im, x0, y0, NATIVE)   # forced at the known position, as the spectrum code does
            ph.append(s / flux_total)
        acc["native"]["com"].append(math.hypot(np.mean(cx) - tx, np.mean(cy) - ty) * GRID)
        gx, gy = [], []
        for w, im in zip(ws_a, imgs_a):
            mx, my = gauss_centroid(im)
            sky = w.all_pix2world(mx, my, 0)
            ox, oy = w_out.all_world2pix(sky[0], sky[1], 0)
            gx.append(float(ox)); gy.append(float(oy))
        acc["native"]["g2d"].append(math.hypot(np.mean(gx) - tx, np.mean(gy) - ty) * GRID)
        acc["native"]["phot"].append(float(np.mean(ph)))

        for m in methods:
            t0 = time.perf_counter()
            a = coadd(m, imgs_a, ws_a, w_out)
            acc[m]["time"].append((time.perf_counter() - t0) / nexp)
            b = coadd(m, imgs_b, ws_b, w_out)
            x, y = com_centroid(a)
            acc[m]["com"].append(math.hypot(x - tx, y - ty) * GRID)
            x2, y2 = gauss_centroid(a)
            acc[m]["g2d"].append(math.hypot(x2 - tx, y2 - ty) * GRID)
            # image holds flux per NATIVE pixel (surface brightness); an output pixel covers area_ratio of one
            s, mask = aperture_sum(a, tx, ty, GRID)
            acc[m]["phot"].append(s * area_ratio / flux_total)
            acc[m]["diff"].append(float(np.nanmax(np.abs(a - b)) / np.nanmax(a)))
            n = coadd(m, noise, ws_a, w_out)
            acc[m]["noise_ap"].append(float(np.nansum(n[mask])))
            acc[m]["noise_pix"].append(n[N_OUT // 2 - 8:N_OUT // 2 + 9, N_OUT // 2 - 8:N_OUT // 2 + 9].ravel())

    yy, xx = np.mgrid[0:N_OUT, 0:N_OUT]
    n_ap = int((np.hypot(xx - (N_OUT - 1) / 2, yy - (N_OUT - 1) / 2) * GRID <= R_AP).sum())
    expected_frac = 1 - math.exp(-(R_AP ** 2) / (2 * (FWHM / 2.3548) ** 2))
    out = {"settings": {"trials": trials, "exposures_per_epoch": nexp, "native_arcsec": NATIVE, "grid_arcsec": GRID,
                        "psf_fwhm_arcsec": FWHM, "aperture_arcsec": R_AP,
                        "ideal_aperture_fraction_unpixelised": expected_frac}, "methods": {}}
    for m, d in acc.items():
        row = {
            "centroid_rms_arcsec_moment": float(np.sqrt(np.mean(np.square(d["com"])))),
            "centroid_rms_arcsec_gaussfit": float(np.sqrt(np.mean(np.square(d["g2d"])))),
            "aperture_fraction_mean": float(np.mean(d["phot"])),
            "aperture_fraction_scatter_percent": float(100 * np.std(d["phot"]) / np.mean(d["phot"])),
        }
        if "diff" in d:
            pix_var = float(np.var(np.concatenate(d["noise_pix"])))
            row.update({
                "difference_peak_residual_fraction_median": float(np.median(d["diff"])),
                "difference_peak_residual_fraction_p95": float(np.percentile(d["diff"], 95)),
                "pixel_noise_variance_vs_input": pix_var,      # input variance per exposure = 1; ideal coadd = 1/nexp
                "aperture_variance_inflation": float(np.var(d["noise_ap"]) / (n_ap * pix_var)),
                "seconds_per_exposure": float(np.median(d["time"])),
            })
        out["methods"][m] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--nexp", type=int, nargs="*", default=[1, 5])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    results = []
    for nexp in args.nexp:
        res = run(args.trials, nexp)
        results.append(res)
        print(f"\n=== {nexp} exposure(s) per epoch, {args.trials} trials (SYNTHETIC) ===")
        print(f"{'method':10s} {'cen(mom)\"':>10s} {'cen(fit)\"':>10s} {'ap.frac':>8s} {'scat %':>7s} {'diff med':>9s} {'diff p95':>9s} {'var infl':>9s} {'pix var':>8s} {'s/exp':>7s}")
        for m, r in res["methods"].items():
            extra = (f"{r['difference_peak_residual_fraction_median']:9.3f} {r['difference_peak_residual_fraction_p95']:9.3f} "
                     f"{r['aperture_variance_inflation']:9.2f} {r['pixel_noise_variance_vs_input']:8.3f} {r['seconds_per_exposure']:7.4f}") if "seconds_per_exposure" in r else ""
            print(f"{m:10s} {r['centroid_rms_arcsec_moment']:10.3f} {r['centroid_rms_arcsec_gaussfit']:10.3f} "
                  f"{r['aperture_fraction_mean']:8.3f} {r['aperture_fraction_scatter_percent']:7.2f} {extra}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
