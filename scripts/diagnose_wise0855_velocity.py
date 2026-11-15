#!/usr/bin/env python3
"""Diagnostic: S/N of WISE 0855 as a function of trial velocity, on the
cached REAL SPHEREx exposures (D5 + D6), on a fine 0.25"/yr grid with each
velocity plane normalised by its own measured noise (as in the coarse search).

Purpose: check how well shift-and-stack constrains the motion, independently
of the production search's coarse-to-fine refinement scheme. Reports the
peak, the Delta(S/N^2) = 1 (one-parameter 1-sigma) and 2.3 (joint 68%)
ranges, and Delta(S/N^2) at the catalogue motion. Results in docs/PHASE2.md.
Runtime ~2-4 minutes.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
from astropy.time import Time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO / "scripts"))
warnings.filterwarnings("ignore")

from app.offline import run_cached  # noqa: E402
from app.shiftstack import ShiftStackConfig, StackFrame, _raw_snr, prepare  # noqa: E402
from validate_targets import grid_wcs  # noqa: E402


def normalise(raw):
    v = raw[raw != 0]
    med = np.median(v)
    return (raw - med) / (1.4826 * np.median(np.abs(v - med)))


def main(detectors=("D5", "D6")):
    run = run_cached("wise0855", mode="single", detectors=list(detectors), analyze=False)
    exps = sorted(run.exposures, key=lambda p: p.record.time_mjd)
    frames = [StackFrame(p.image, p.variance, p.record.time_mjd) for p in exps]
    scale = run.frames[0]["grid_scale_arcsec"]
    prep = prepare(frames, scale, ShiftStackConfig())
    t_ref = float(np.mean(prep.times))
    pred = run.resolved.propagate_to(Time(t_ref, format="mjd"))
    x, y = (float(v) for v in grid_wcs(run.frames[0]).world_to_pixel(pred))
    xi, yi = int(round(x)), int(round(y))
    box = (slice(yi - 1, yi + 2), slice(xi - 1, xi + 2))

    ve = np.arange(-13.0, -2.99, 0.25)
    vn = np.arange(-4.0, 4.01, 0.25)
    S = np.zeros((ve.size, vn.size))
    for i, a in enumerate(ve):
        for j, b in enumerate(vn):
            S[i, j] = normalise(_raw_snr(prep, a, b, prep.times, t_ref))[box].max()
    k = np.unravel_index(np.argmax(S), S.shape)
    d = S[k] ** 2 - S ** 2
    cat = (run.resolved.pmra / 1000.0, run.resolved.pmdec / 1000.0)
    ic, jc = np.argmin(abs(ve - cat[0])), np.argmin(abs(vn - cat[1]))
    print(f"{'+'.join(detectors)}: {len(frames)} exposures, baseline {(prep.times.max() - prep.times.min()) / 365.25:.2f} yr")
    print(f"peak S/N {S[k]:.2f} at v = ({ve[k[0]]:.2f}, {vn[k[1]]:.2f}) \"/yr; catalogue ({cat[0]:.2f}, {cat[1]:.2f})")
    for lvl, name in ((1.0, "Delta=1 (1-param 1 sigma)"), (2.3, "Delta=2.3 (joint 68%)")):
        ii, jj = np.nonzero(d <= lvl)
        print(f"{name}: east {ve[ii].min():.2f} .. {ve[ii].max():.2f}, north {vn[jj].min():.2f} .. {vn[jj].max():.2f}")
    print(f"S/N at catalogue motion {S[ic, jc]:.2f}, Delta(S/N^2) from peak {d[ic, jc]:.1f}")
    print(f"S/N at zero motion {normalise(_raw_snr(prep, 0, 0, prep.times, t_ref))[box].max():.2f}")


if __name__ == "__main__":
    main()
