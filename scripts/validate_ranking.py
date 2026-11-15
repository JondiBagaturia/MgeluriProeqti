#!/usr/bin/env python3
"""Validate the calibrated ranking (app/ranking.py) against the raw Phase Two
score by injecting SYNTHETIC point sources into REAL cached SPHEREx cutouts
(backend/tests/data/real) and running the unchanged discovery pipeline on them,
with the cached REAL Gaia DR3 / CatWISE2020 rows for catalogue matching.

Experiments per field (fixed random seed, reproducible):
  stationary  constant, non-moving sources at detection S/N ~ 6, 10, 20, 50.
              Any motion- or variability-driven score is a false positive. A
              catalogue-driven score is reported separately by completeness
              regime: an injected source is truly absent from the catalogues, so a
              bright one IS a missing-counterpart anomaly, a faint one is not.
  moving      2 and 5"/yr in random directions at S/N ~ 10, 20, 50.
  variable    one epoch dimmed by 40% at S/N ~ 10, 20, 50.
              Two anomalous sources per run, among six stationary ones at the
              same S/N (which also add to the stationary sample).
  near_bright stationary S/N ~ 10 sources 20" from bright (S/N > 300) stars,
              where Lanczos-3 ringing lives.
  sparse_stationary  one S/N ~ 50 and one S/N ~ 10 stationary source per run:
              the catalogue-completeness estimate needs a field that is not
              crowded with injected uncatalogued sources (the main runs inject
              8 per field, which biases the matched fraction down).

Detection S/N is the pipeline's per-frame peak S/N (detection_snr_median).
Injected flux per detector is set from a pilot injection (S/N assumed
proportional to flux). Positions avoid real SPHEREx detections (30"),
catalogue positions (6", i.e. any chance of a catalogue match), the edges
(on a 20" grid within +-100"), and each other (40").

Writes docs/data/ranking_validation.json and prints a summary.
Usage: python scripts/validate_ranking.py [--fields barnard eps_ind_b ...] [--reps 2]
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
warnings.filterwarnings("ignore")

from astropy.table import Table  # noqa: E402

from app.injection import InjectedSource, make_transform  # noqa: E402
from app.jsonsafe import clean  # noqa: E402
from app.offline import REAL_DATA_DIR, load_manifest, run_cached  # noqa: E402

FIELDS = ("barnard", "eps_ind_b", "wise0855", "rw_aur", "r_dra")
SNR_TARGETS = (6, 10, 20, 50)
THRESHOLDS = (0.1, 0.2, 0.3, 0.4)
MATCH_ARCSEC = 5.0
PILOT_FLUX = {"D3": 3.0, "D5": 1.0, "D6": 2.0}
SEED = 20261115
# CatWISE is far deeper than SPHEREx (~1 source per 300 arcsec^2 here), so only
# positions that could be MATCHED to a catalogue source are avoided (the match
# gate is a few arcsec); real SPHEREx detections are avoided by 30".
CAT_EXCLUSION_ARCSEC = 6.0


def offsets_to_radec(e, n, ra0, dec0):
    return ra0 + e / 3600 / math.cos(math.radians(dec0)), dec0 + n / 3600


def sep(ra1, dec1, ra2, dec2):
    return math.hypot((ra1 - ra2) * math.cos(math.radians(dec2)), dec1 - dec2) * 3600


def all_tracks(run):
    for s in run.analysis["series"]:
        for t in s["tracks"]:
            yield s["series"], t


def field_setup(field):
    m = load_manifest(field)
    base = run_cached(field, use_catalogs="cached")
    ra0, dec0 = m["cutout_centre_deg"]
    times = sorted(e["time_mjd"] for e in m["exposures"])
    ref_mjd = float(np.mean(times))
    # epoch edges: midpoints of the large gaps
    edges = [0.5 * (a + b) for a, b in zip(times[:-1], times[1:]) if b - a > 45]
    avoid = [(d["ra_deg"], d["dec_deg"]) for _, t in all_tracks(base) for d in t["detections"]]
    cat_pos = []
    for table, (rc, dc) in (("gaia_dr3_source", ("ra", "dec")), ("catwise_2020", ("ra_pm", "dec_pm"))):
        p = REAL_DATA_DIR / field / "catalogs" / f"{table}.csv"
        if p.exists():
            t = Table.read(p, format="ascii.csv")
            cat_pos += [(float(a), float(b)) for a, b in zip(t[rc], t[dc])]
    cands = []
    for e in np.arange(-100, 101, 20.0):
        for n in np.arange(-100, 101, 20.0):
            ra, dec = offsets_to_radec(e, n, ra0, dec0)
            if any(sep(ra, dec, a, b) < 30 for a, b in avoid):
                continue
            if any(sep(ra, dec, a, b) < CAT_EXCLUSION_ARCSEC for a, b in cat_pos):
                continue
            cands.append((ra, dec))
    bright = [(t["ra_deg"], t["dec_deg"]) for _, t in all_tracks(base) if t["detection_snr_median"] > 300 and t["n_epochs"] >= 2]
    return {"manifest": m, "base": base, "ra0": ra0, "dec0": dec0, "ref_mjd": ref_mjd, "edges": edges,
            "candidates": cands, "bright": bright, "detectors": m["detectors"]}


def recovered(run, src, series):
    best = None
    for s_key, t in all_tracks(run):
        if s_key != series:
            continue
        ra, dec = src.position_at(t["epoch_mjd"])
        d = sep(t["ra_deg"], t["dec_deg"], ra, dec)
        if d <= MATCH_ARCSEC and (best is None or d < best[0]):
            best = (d, t)
    return best[1] if best else None


def run_injection(field, sources):
    return run_cached(field, use_catalogs="cached", cutout_transform=make_transform(sources))


def summarise_track(t):
    c = t["calibration"]
    return {"n_epochs": t["n_epochs"], "snr": t["detection_snr_median"], "label": t["label"], "reliability": t["reliability"],
            "flags": t["flags"], "raw_score": t["anomaly_score"], "calibrated_score": t["calibrated_score"],
            "raw_rank": t["rank"], "calibrated_rank": t["calibrated_rank"],
            "raw_components": t["score_components"], "calibrated_components": {"M": c["M_norm"], "V": c["V_norm"], "C": c["C_norm"]},
            "completeness_regime": c["completeness_regime"], "pm_significance": t["motion"].get("pm_significance"),
            "motion_inflation": c["motion_calibration"].get("inflation"), "peer_level": c["motion_calibration"].get("peer_level"),
            "variability_p": t["variability"].get("p_value"), "suppression": c["suppression_reasons"]}


class SourceFactory:
    def __init__(self, setup, fluxscale, rng):
        self.s, self.k, self.rng = setup, fluxscale, rng

    def flux_for(self, snr):
        # one flux per detector: a source with a "colour", sized for the target S/N in each detector
        return {d: snr / self.k[d] for d in self.k}

    def make(self, pos, snr, kind, speed=0.0, factors=()):
        ang = self.rng.uniform(0, 2 * math.pi)
        fl = self.flux_for(snr)
        src = InjectedSource(ra_deg=pos[0], dec_deg=pos[1], flux_mjy=1.0, ref_mjd=self.s["ref_mjd"],
                             pm_east_mas_yr=1000 * speed * math.cos(ang), pm_north_mas_yr=1000 * speed * math.sin(ang),
                             epoch_edges_mjd=tuple(self.s["edges"]), epoch_factors=tuple(factors), kind=kind,
                             label=f"{kind}:snr{snr}:v{speed}")
        src.flux_by_detector = fl
        return src


def spaced(cands, rng, k, min_sep=40.0):
    order = rng.permutation(len(cands))
    out = []
    for i in order:
        p = cands[i]
        if all(sep(p[0], p[1], q[0], q[1]) >= min_sep for q in out):
            out.append(p)
        if len(out) == k:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", nargs="*", default=list(FIELDS))
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--sparse", type=int, default=6,
                    help="runs per field of the sparse experiment (one S/N 50 + one S/N 10 stationary source per run)")
    ap.add_argument("--out", default=str(REPO / "docs" / "data" / "ranking_validation.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    results = []
    t_start = time.time()
    for field in args.fields:
        t0 = time.time()
        setup = field_setup(field)
        dets = setup["detectors"]
        if len(setup["candidates"]) < 4:
            print(f"{field}: only {len(setup['candidates'])} clean positions; skipped")
            continue
        # pilot: S/N per mJy per detector
        pos = spaced(setup["candidates"], rng, 8)
        pilot = []
        for p in pos:
            src = InjectedSource(ra_deg=p[0], dec_deg=p[1], flux_mjy=1.0, ref_mjd=setup["ref_mjd"], label="pilot")
            src.flux_by_detector = {d: PILOT_FLUX.get(d, 2.0) for d in dets}
            pilot.append(src)
        prun = run_injection(field, pilot)
        k = {}
        for d in dets:
            ratios = [t["detection_snr_median"] / PILOT_FLUX.get(d, 2.0) for s in pilot if (t := recovered(prun, s, d))]
            k[d] = float(np.median(ratios)) if ratios else None
        if any(v is None for v in k.values()):
            print(f"{field}: pilot not recovered in {[d for d, v in k.items() if v is None]}; skipped")
            continue
        fac = SourceFactory(setup, k, rng)
        plan = []
        for rep in range(args.reps):
            for snr in SNR_TARGETS:
                plan.append([fac.make(p, snr, "stationary") for p in spaced(setup["candidates"], rng, 8)])
        # anomalous injections: 2 per run among 6 stationary ones at the same S/N,
        # so that (as in real fields) the peers are dominated by null sources
        n_ep = len(setup["edges"]) + 1
        dip = [1.0] * n_ep
        dip[n_ep // 2] = 0.6
        for rep in range(args.reps):
            for snr in (10, 20, 50):
                for kind, kw in (("moving", {"speed": 2.0}), ("moving", {"speed": 5.0}), ("variable", {"factors": dip})):
                    ps = spaced(setup["candidates"], rng, 8)
                    plan.append([fac.make(p, snr, kind, **kw) if i < 2 else fac.make(p, snr, "stationary")
                                 for i, p in enumerate(ps)])
        # sparse: realistic for catalogue completeness, where the main runs (8
        # uncatalogued injections per field) bias the matched fraction down
        for rep in range(args.sparse):
            ps = spaced(setup["candidates"], rng, 2)
            if len(ps) == 2:
                plan.append([fac.make(ps[0], 50, "sparse_stationary"), fac.make(ps[1], 10, "sparse_stationary")])
        if setup["bright"]:
            nb = []
            for (bra, bdec) in setup["bright"][:6]:
                ang = rng.uniform(0, 2 * math.pi)
                p = offsets_to_radec(20 * math.cos(ang), 20 * math.sin(ang), bra, bdec)
                if all(sep(p[0], p[1], q.ra_deg, q.dec_deg) >= 40 for q in nb):
                    nb.append(fac.make(p, 10, "near_bright"))
            plan.append(nb)
        n_inj = 0
        for sources in plan:
            run = run_injection(field, sources)
            for s in sources:
                for d in dets:
                    t = recovered(run, s, d)
                    n_inj += 1
                    results.append({"field": field, "series": d, "kind": s.kind, "target_snr": int(s.label.split(":")[1][3:]),
                                    "speed_arcsec_yr": float(s.label.split(":v")[1]), "flux_mjy": s.flux_by_detector[d],
                                    "recovered": t is not None, "track": summarise_track(t) if t else None})
        print(f"{field}: S/N per mJy {({d: round(v, 2) for d, v in k.items()})}; {len(setup['candidates'])} clean positions; "
              f"{len(plan)} injection runs, {n_inj} source-detector cases; {time.time() - t0:.0f}s")

    summary = summarise(results)
    out = {"note": ("SYNTHETIC point sources injected into REAL cached SPHEREx cutouts; real cached Gaia/CatWISE rows; "
                    "production pipeline. Detection S/N = per-frame peak S/N."),
           "seed": SEED, "thresholds": THRESHOLDS, "match_arcsec": MATCH_ARCSEC, "runtime_s": time.time() - t_start,
           "summary": summary, "cases": results}
    Path(args.out).write_text(json.dumps(clean(out), indent=1))
    print_summary(summary)
    print("wrote", args.out)


def summarise(results):
    def rate(xs):
        return {"n": len(xs), "fraction": (sum(xs) / len(xs)) if xs else None}

    out = {}
    for kind in ("stationary", "moving", "variable", "near_bright", "sparse_stationary"):
        rows = [r for r in results if r["kind"] == kind]
        by = {}
        for snr in sorted({r["target_snr"] for r in rows}):
            sub = [r for r in rows if r["target_snr"] == snr]
            rec = [r for r in sub if r["recovered"] and r["track"]["n_epochs"] >= 2]
            entry = {"n_injected": len(sub), "recovered_2plus_epochs": rate([r["recovered"] and r["track"]["n_epochs"] >= 2 for r in sub]),
                     "median_measured_snr": float(np.median([r["track"]["snr"] for r in rec])) if rec else None}
            for thr in THRESHOLDS:
                key = f"thr_{thr}"
                if kind in ("stationary", "near_bright", "sparse_stationary"):
                    # false positives: anything flagged; split by what drove it
                    entry[key] = {
                        "raw_flagged": rate([r["track"]["raw_score"] >= thr for r in rec]),
                        "calibrated_flagged": rate([r["track"]["calibrated_score"] >= thr for r in rec]),
                        "raw_flagged_by_motion_or_variability": rate([
                            0.45 * r["track"]["raw_components"]["motion"] + 0.30 * r["track"]["raw_components"]["variability"] >= thr for r in rec]),
                        "calibrated_flagged_by_motion_or_variability": rate([
                            0.45 * r["track"]["calibrated_components"]["M"] + 0.30 * r["track"]["calibrated_components"]["V"] >= thr for r in rec]),
                    }
                else:
                    entry[key] = {"raw_flagged": rate([r["track"]["raw_score"] >= thr for r in rec]),
                                  "calibrated_flagged": rate([r["track"]["calibrated_score"] >= thr for r in rec])}
            # component level: does calibration remove the signal itself?
            comp_key = {"moving": ("motion", "M"), "variable": ("variability", "V")}.get(kind)
            if comp_key:
                entry["component"] = {}
                for lvl in (0.1, 0.3, 0.5):
                    entry["component"][f"ge_{lvl}"] = {
                        "raw": rate([r["track"]["raw_components"][comp_key[0]] >= lvl for r in rec]),
                        "calibrated": rate([r["track"]["calibrated_components"][comp_key[1]] >= lvl for r in rec])}
            regimes = {}
            for r in rec:
                g = r["track"]["completeness_regime"]
                regimes.setdefault(g, {"n": 0, "raw_C_positive": 0, "calibrated_C_positive": 0})
                regimes[g]["n"] += 1
                regimes[g]["raw_C_positive"] += r["track"]["raw_components"]["no_catalogue_counterpart"] > 0
                regimes[g]["calibrated_C_positive"] += r["track"]["calibrated_components"]["C"] > 0
            entry["completeness_regimes"] = regimes
            entry["flags"] = {f: sum(f in r["track"]["flags"] for r in rec) for f in ("near_bright_star", "two_epochs_only", "ambiguous_link")}
            by[str(snr)] = entry
        out[kind] = by
    return out


def print_summary(summary):
    for kind, by in summary.items():
        print(f"\n== {kind}")
        for snr, e in by.items():
            line = f"  S/N~{snr:>2}: injected {e['n_injected']}, recovered(>=2 ep) {e['recovered_2plus_epochs']['fraction']:.2f}, measured S/N {e['median_measured_snr']}"
            print(line)
            for thr in THRESHOLDS:
                x = e[f"thr_{thr}"]
                s = f"     thr {thr}: raw {x['raw_flagged']['fraction']}  calibrated {x['calibrated_flagged']['fraction']}"
                if "raw_flagged_by_motion_or_variability" in x:
                    s += (f"  | M/V-driven raw {x['raw_flagged_by_motion_or_variability']['fraction']} "
                          f"cal {x['calibrated_flagged_by_motion_or_variability']['fraction']}")
                print(s)
            if "component" in e:
                print("     component " + "  ".join(f"{k}: raw {v['raw']['fraction']} cal {v['calibrated']['fraction']}" for k, v in e["component"].items()))
            print(f"     completeness regimes {e['completeness_regimes']}; flags {e['flags']}")


if __name__ == "__main__":
    main()
