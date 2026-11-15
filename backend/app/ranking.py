"""Calibrated anomaly ranking (Phase Three, Priority 2).

Adds, ALONGSIDE the Phase Two fields (anomaly_score, score_components,
reliability, rank: all unchanged), a calibrated score that accounts for
brightness, epoch coverage, catalogue completeness and measurement quality:

    calibrated_score = min(0.45 M_norm + 0.30 V_norm + 0.25 C_norm, quality cap)

Same weights and the same mapping of significance to [0, 1] as Phase Two, so
the two scores are directly comparable; only the INPUTS are calibrated.

M_norm, V_norm: peer calibration ("genomic-control" style)
  Under the error model, the motion statistic x_M = (PM significance)^2 follows
  chi^2 with 2 dof, and the variability statistic x_V = chi2.isf(p_var, 1)
  follows chi^2 with 1 dof. If measurements are noisier than modelled (faint
  sources near the 5-sigma limit, few epochs), real stationary constant
  sources show inflated x. The inflation is measured on PEERS: clean tracks in
  the same series (reliability high; no ringing, masked-core, ambiguous-link or
  edge flags), in the same S/N bin and epoch-coverage bin:
      lambda = clip(median(peer x, iteratively clipped at the null's 99.7%
               point after deflation) / truncated null median, 1, 4)
  so that genuine movers/variables among the peers do not count as "noise"
  x_cal = x / lambda; p_cal = chi2.sf(x_cal, dof). If a bin has fewer than
  PEER_MIN peers it falls back to the S/N bin over all coverages, then to all
  peers of the series, then to the analytic error model (lambda = 1), and the
  level used is reported. The source itself is never its own peer.
  Also reported: the empirical rarity among the same peers, k peers at least as
  extreme out of n, as the bound (k + 1) / (n + 1), never as zero.

C_norm: catalogue-completeness gating
  Being absent from a catalogue is only anomalous if the source is bright
  enough that the catalogue should contain it. Per series and per catalogue,
  the local completeness flux F_c is estimated from the clean tracks with a
  measured flux and a catalogue answer: the lowest flux such that at least 90%
  of the (at least 5) tracks at or above it are matched. It is computed
  leave-one-out, so an unmatched source cannot push the limit above itself.
  For each catalogue in which a track is unmatched, a weight is applied:
      above F_c                 -> 1.0  ("above_completeness": a missing counterpart is meaningful)
      below F_c                 -> 0.0  ("below_completeness": expected to be missing)
      F_c or the flux unknown   -> 0.5  ("completeness_unknown")
      catalogue matches none of >= 5 clean tracks -> 0.0 ("catalogue_incomplete_here";
                                   e.g. CatWISE2020 has no rows within 4' of R Dra)
  C_norm = raw C x (smallest weight among the catalogues it is unmatched in).

Quality caps (calibrated score only; reasons are listed)
  reliability low, near_bright_star (Lanczos ringing zone), core masked in
  every epoch -> cap 0.05; ambiguous_link -> cap 0.10; two_epochs_only ->
  cap 0.30; seen in < 2 epochs -> 0.

None of these is a probability; calibrated_score is a ranking device like
anomaly_score. See docs/RANKING.md for the validation by injection.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.stats import chi2 as chi2_dist

SNR_BINS = ((0.0, 10.0), (10.0, 30.0), (30.0, 100.0), (100.0, math.inf))
PEER_MIN = 5
# Inflation estimate: iteratively clipped median. Peers whose deflated statistic
# is beyond the 99.7% point of the null are treated as genuine movers/variables
# and excluded; lambda is re-estimated from the median of the rest, corrected for
# the truncation, until it converges. A genuine noise inflation scales every peer
# and is kept; strong anomalous peers drop out. (A plain median, and then a lower
# quartile, were tried first: with many anomalous peers, e.g. injected movers,
# both inflated lambda and blinded the pipeline to motion; docs/RANKING.md.)
CLIP_P = 0.003
LAMBDA_MAX = 4.0


def _truncated_median(dof: int) -> float:
    """Median of a chi^2(dof) variable truncated at its (1 - CLIP_P) quantile."""
    return float(chi2_dist.ppf(0.5 * (1 - CLIP_P), dof))


def robust_inflation(x: np.ndarray, dof: int, iterations: int = 10) -> tuple[float, float, int]:
    """(lambda clipped to [1, LAMBDA_MAX], raw lambda, number of peers kept)."""
    cut = float(chi2_dist.isf(CLIP_P, dof))
    # start from the lower quartile: it scales with a genuine inflation (so a
    # very noisy field is not clipped away entirely at lambda = 1) and is only
    # mildly affected by anomalous peers, which the clipping then removes
    lam = max(1.0, float(np.quantile(x, 0.25)) / float(chi2_dist.ppf(0.25, dof)))
    keep = np.ones(x.size, bool)
    for _ in range(iterations):
        keep = x / lam <= cut
        if keep.sum() < 3:
            break
        new = float(np.median(x[keep])) / _truncated_median(dof)
        new = max(new, 1.0)
        if abs(new - lam) < 1e-3:
            lam = new
            break
        lam = new
    return min(lam, LAMBDA_MAX), lam, int(keep.sum())
UNCLEAN_FLAGS = {"near_bright_star", "core_masked", "core_masked_all_epochs", "ambiguous_link", "near_edge"}
COMPLETENESS_FRACTION = 0.90
COMPLETENESS_MIN_TRACKS = 5
WEIGHTS = {"motion": 0.45, "variability": 0.30, "no_catalogue_counterpart": 0.25}
CAPS = (
    ("near_bright_star", 0.05, "within the Lanczos ringing zone of a much brighter star"),
    ("core_masked_all_epochs", 0.05, "core masked or saturated in every epoch"),
    ("ambiguous_link", 0.10, "the cross-epoch association is ambiguous"),
    ("two_epochs_only", 0.30, "only two epochs"),
)


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


def snr_bin(snr: float) -> int:
    for i, (lo, hi) in enumerate(SNR_BINS):
        if lo <= snr < hi:
            return i
    return len(SNR_BINS) - 1


def coverage_bin(n_epochs: int) -> str:
    return "2" if n_epochs <= 2 else "3" if n_epochs == 3 else "4+"


def is_peer(t: dict) -> bool:
    return (t.get("reliability") == "high" and t.get("n_epochs", 0) >= 2
            and not (UNCLEAN_FLAGS & set(t.get("flags", []))))


def motion_stat(t: dict) -> Optional[float]:
    s = (t.get("motion") or {}).get("pm_significance")
    return None if s is None else float(s) ** 2


def variability_stat(t: dict) -> Optional[float]:
    p = (t.get("variability") or {}).get("p_value")
    if p is None:
        return None
    return float(chi2_dist.isf(min(max(p, 1e-300), 1.0), 1))


def track_flux(t: dict) -> Optional[float]:
    f = [d["flux_mjy"] for d in t.get("detections", []) if d.get("phot_status") == "ok" and d.get("flux_mjy") is not None]
    return float(np.median(f)) if f else None


def _peer_calibration(t: dict, tracks: list[dict], stat_fn, dof: int) -> dict:
    x = stat_fn(t)
    if x is None:
        return {"available": False}
    sb, cb = snr_bin(t["detection_snr_median"]), coverage_bin(t["n_epochs"])
    peers = [p for p in tracks if p is not t and is_peer(p) and stat_fn(p) is not None]
    levels = (
        ("snr_and_coverage_bin", [p for p in peers if snr_bin(p["detection_snr_median"]) == sb and coverage_bin(p["n_epochs"]) == cb]),
        ("snr_bin", [p for p in peers if snr_bin(p["detection_snr_median"]) == sb]),
        ("all_peers", peers),
    )
    level, used = "analytic", []
    n_kept = 0
    for name, group in levels:
        if len(group) >= PEER_MIN:
            level, used = name, group
            break
    capped = False
    if used:
        px = np.array([stat_fn(p) for p in used])
        lam, lam_raw, n_kept = robust_inflation(px, dof)
        capped = lam_raw > LAMBDA_MAX
        k = int(np.sum(px >= x))
        rarity = (k + 1) / (len(used) + 1)
    else:
        lam, k, rarity = 1.0, None, None
    x_cal = x / lam
    return {"available": True, "statistic": x, "inflation": lam, "statistic_calibrated": x_cal,
            "p_analytic": float(chi2_dist.sf(x, dof)), "p_calibrated": float(chi2_dist.sf(x_cal, dof)),
            "peer_level": level, "n_peers": len(used), "n_peers_used_for_inflation": n_kept,
            "n_peers_at_least_as_extreme": k, "inflation_capped": capped,
            "empirical_rarity_bound": rarity, "snr_bin": [SNR_BINS[sb][0], SNR_BINS[sb][1]], "coverage_bin": cb}


NO_COVERAGE = -1.0   # sentinel: the catalogue matches none of >= COMPLETENESS_MIN_TRACKS clean tracks


def _completeness_limit(tracks: list[dict], exclude: dict, key: str) -> Optional[float]:
    rows = []
    for t in tracks:
        if t is exclude or t.get("reliability") == "low":
            continue
        st = (t.get(key) or {}).get("status")
        f = track_flux(t)
        if f is None or f <= 0 or st not in ("matched", "ambiguous", "unmatched"):
            continue
        rows.append((f, st != "unmatched"))
    if len(rows) < COMPLETENESS_MIN_TRACKS:
        return None
    rows.sort(key=lambda r: -r[0])
    if not any(m for _, m in rows):
        # e.g. CatWISE2020 has no rows at all within 4' of R Dra: a coverage
        # hole, so "unmatched" there carries no information
        return NO_COVERAGE
    best = None
    n_match = 0
    for i, (f, m) in enumerate(rows, start=1):
        n_match += m
        if i >= COMPLETENESS_MIN_TRACKS and n_match / i >= COMPLETENESS_FRACTION:
            best = f
    return best


def _completeness(t: dict, tracks: list[dict]) -> dict:
    out = {}
    flux = track_flux(t)
    for key, name in (("gaia", "gaia_dr3"), ("catwise", "catwise2020")):
        st = (t.get(key) or {}).get("status")
        if st in ("matched", "ambiguous"):
            out[name] = {"status": st, "regime": "matched", "weight": None}
            continue
        if st != "unmatched":
            out[name] = {"status": st, "regime": "unavailable", "weight": None}
            continue
        lim = _completeness_limit(tracks, t, key)
        if lim == NO_COVERAGE:
            out[name] = {"status": st, "regime": "catalogue_incomplete_here", "weight": 0.0,
                         "completeness_flux_mjy": None, "source_flux_mjy": flux}
            continue
        if lim is None or flux is None:
            regime, w = "completeness_unknown", 0.5
        elif flux >= lim:
            regime, w = "above_completeness", 1.0
        else:
            regime, w = "below_completeness", 0.0
        out[name] = {"status": st, "regime": regime, "weight": w, "completeness_flux_mjy": lim, "source_flux_mjy": flux}
    return out


def calibrate_track(t: dict, tracks: list[dict]) -> dict:
    m = _peer_calibration(t, tracks, motion_stat, 2)
    v = _peer_calibration(t, tracks, variability_stat, 1)
    comp = _completeness(t, tracks)
    raw = t.get("score_components", {})
    explanation: list[str] = []
    suppression: list[str] = []

    # motion
    M = 0.0
    if m["available"] and t["n_epochs"] >= 2:
        z = math.sqrt(m["statistic_calibrated"])
        M = _clip01((z - 3.0) / 7.0)
        if t["n_epochs"] == 2:
            M *= 0.5
        if m["inflation"] > 1.05 and raw.get("motion", 0) > M:
            explanation.append(f"Motion significance deflated by {math.sqrt(m['inflation']):.2f}x: peers of similar S/N and "
                               f"coverage ({m['peer_level'].replace('_', ' ')}, n = {m['n_peers']}) scatter more than the error model.")
    # variability
    V = 0.0
    if v["available"]:
        mlog = -math.log10(max(v["p_calibrated"], 1e-300))
        V = _clip01((mlog - 2.0) / 6.0)
        if (t.get("variability") or {}).get("wavelength_mismatch"):
            V *= 0.3
        if v["inflation"] > 1.05 and raw.get("variability", 0) > V:
            explanation.append(f"Variability statistic deflated by {v['inflation']:.2f}x relative to peers "
                               f"({v['peer_level'].replace('_', ' ')}, n = {v['n_peers']}).")
    # catalogue
    unmatched_w = [c["weight"] for c in comp.values()
                   if c["regime"] in ("above_completeness", "below_completeness", "completeness_unknown", "catalogue_incomplete_here")]
    C = float(raw.get("no_catalogue_counterpart", 0.0)) * (min(unmatched_w) if unmatched_w else 0.0)
    regimes = {c["regime"] for c in comp.values()}
    if "catalogue_incomplete_here" in regimes:
        regime = "catalogue_incomplete_here"
    elif "below_completeness" in regimes:
        regime = "below_completeness"
    elif "completeness_unknown" in regimes:
        regime = "completeness_unknown"
    elif "above_completeness" in regimes:
        regime = "above_completeness"
    elif regimes == {"matched"} or "matched" in regimes:
        regime = "matched"
    else:
        regime = "unavailable"
    if raw.get("no_catalogue_counterpart", 0) > 0 and C < raw["no_catalogue_counterpart"]:
        why = []
        for name, c in comp.items():
            if c["regime"] == "below_completeness":
                why.append(f"{name} ({c['source_flux_mjy']:.3g} mJy is below the local completeness limit {c['completeness_flux_mjy']:.3g} mJy)")
            elif c["regime"] == "completeness_unknown":
                why.append(f"{name} (completeness could not be estimated in this field)")
            elif c["regime"] == "catalogue_incomplete_here":
                why.append(f"{name} (matches none of the clean sources in this field: the catalogue does not cover it)")
        tail = (" A source fainter than the catalogue's completeness limit is expected to be missing, not anomalous."
                if any(c["regime"] == "below_completeness" for c in comp.values()) else
                " Without a completeness estimate a missing counterpart is not treated as fully anomalous.")
        suppression.append("Catalogue-unmatched boost reduced: " + "; ".join(why) + "." + tail)

    score = WEIGHTS["motion"] * M + WEIGHTS["variability"] * V + WEIGHTS["no_catalogue_counterpart"] * C
    cap = 1.0
    flags = set(t.get("flags", []))
    if t["n_epochs"] < 2:
        cap = 0.0
        suppression.append("Seen in fewer than two epochs: nothing to rank.")
    if t.get("reliability") == "low":
        cap = min(cap, 0.05)
        suppression.append("Capped at 0.05: reliability is low.")
    for flag, c, why in CAPS:
        if flag in flags and c < cap:
            cap = c
            suppression.append(f"Capped at {c:.2f}: {why}.")
        elif flag in flags:
            suppression.append(f"Also: {why}.")
    calibrated = min(score, cap)
    return {
        "M_norm": M, "V_norm": V, "C_norm": C,
        "calibrated_score": calibrated, "uncapped_calibrated_score": score, "quality_cap": cap,
        "completeness_regime": regime, "completeness": comp,
        "motion_calibration": m, "variability_calibration": v,
        "explanation": explanation, "suppression_reasons": suppression,
        "note": "Calibrated ranking device, not a probability. Raw Phase Two fields are unchanged.",
    }


def calibrate_series(tracks: list[dict]) -> None:
    """Adds `calibration`, `calibrated_score` and `calibrated_rank` to every
    track in place. Does not modify any existing field."""
    for t in tracks:
        cal = calibrate_track(t, tracks)
        t["calibration"] = cal
        t["calibrated_score"] = cal["calibrated_score"]
    order = sorted(tracks, key=lambda t: (-t["calibrated_score"], -t["detection_snr_median"]))
    for i, t in enumerate(order):
        t["calibrated_rank"] = i + 1
