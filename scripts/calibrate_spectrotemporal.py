#!/usr/bin/env python3
"""Calibrate the per-exposure systematic floor of the spectro-temporal model
on REAL field stars in the cached fields (backend/tests/data/real).

Reference stars: stationary (PM < 3 sigma), detection S/N >= 30, reliability
high, no edge / blend / ringing / masked-core flags, >= 2 epochs; only
measurements in state "valid" (no saturation-related flags).

Method: fit each reference star's detector data with the shared-shape model
(no floor), take the fractional residuals (dof-corrected), and compare their
robust scatter with the median statistical error, in bins of per-exposure S/N.
  implied floor = sqrt(robust_scatter^2 - median_stat_error^2)
The floor is only measurable where it dominates the statistical error, so it
is taken from exposures with S/N >= 50 (pooled over detectors; there are too
few bright reference exposures to calibrate detectors separately).

Also printed: the median-reduced-chi^2 calibration per detector (the first
method tried). It is dominated by faint exposures and gave floors that are
too small for bright sources; kept for comparison.

Results are copied into SYS_FLOOR in backend/app/spectrotemporal.py and
docs/PHASE3.md.
"""
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
warnings.filterwarnings("ignore")

from app.offline import run_cached  # noqa: E402
from app.spectrotemporal import Measurement, _detector_data, _fit_model, _hat, _setup, fit_detector  # noqa: E402

TARGETS = ("barnard", "eps_ind_b", "wise0855", "rw_aur", "r_dra")


def collect():
    stars = []
    for tgt in TARGETS:
        st = run_cached(tgt, spectrotemporal=True).analysis["spectrotemporal"]
        for src in st["sources"]:
            if src["is_reference_star"]:
                meas = [Measurement(**m) for m in src["measurements"] if m["state"] != "valid_flagged"]
                stars.append((tgt, src["id"], meas))
    return stars


def residuals(stars):
    rows = []
    for _, _, meas in stars:
        for det in ("D1", "D2", "D3", "D4", "D5", "D6"):
            d = _detector_data(meas, det)
            if d is None:
                continue
            n_ep, knots, i0 = _setup(d, 0.0, extra_params=max(len(d["epochs"]) - 1, 0))
            f = _fit_model(d["lam"], d["F"], d["err"], d["ei"], n_ep, knots, i0, 0.0)
            if f is None or f["dof"] < 3:
                continue
            a, c, _ = f["unpack"](f["p"])
            model = a[d["ei"]] * (_hat(d["lam"], knots) @ c)
            corr = np.sqrt(d["F"].size / f["dof"])
            for F, e, mo in zip(d["F"], d["err"], model):
                rows.append((det, F / e, (F - mo) / mo * corr, e / mo))
    return rows


def main():
    stars = collect()
    print(f"{len(stars)} reference stars in {len({s[0] for s in stars})} fields")
    rows = residuals(stars)
    snr = np.array([r[1] for r in rows]); res = np.array([r[2] for r in rows]); stat = np.array([r[3] for r in rows])
    print(f"{len(rows)} reference exposures")
    for lo, hi in ((3, 20), (20, 50), (50, 100), (100, 200), (200, np.inf)):
        m = (snr >= lo) & (snr < hi)
        if m.sum() < 5:
            continue
        tot = 1.4826 * np.median(np.abs(res[m] - np.median(res[m])))
        floor = np.sqrt(max(tot ** 2 - np.median(stat[m]) ** 2, 0))
        print(f"  S/N {lo}-{hi}: n={m.sum()}  scatter {100 * tot:.2f}%  median stat {100 * np.median(stat[m]):.2f}%  -> floor {100 * floor:.2f}%")
    m = snr >= 50
    tot = 1.4826 * np.median(np.abs(res[m] - np.median(res[m])))
    floor = np.sqrt(max(tot ** 2 - np.median(stat[m]) ** 2, 0))
    print(f"ADOPTED (S/N >= 50, n={m.sum()}): floor = {100 * floor:.2f}%")

    print("\nFor comparison, median-reduced-chi^2 calibration per detector:")
    for det in ("D3", "D5", "D6"):
        def red(fsys):
            out = []
            for _, _, meas in stars:
                f = fit_detector(meas, det, sys_floor=fsys)
                if f is not None and f.dof >= 2:
                    out.append(f.chi2 / f.dof)
            return np.array(out)
        r0 = red(0.0)
        if r0.size < 3:
            continue
        lo, hi = 0.0, 0.5
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if np.median(red(mid)) > 1 else (lo, mid)
        print(f"  {det}: n={r0.size}: {100 * 0.5 * (lo + hi):.2f}%")


if __name__ == "__main__":
    main()
