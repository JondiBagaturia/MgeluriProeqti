#!/usr/bin/env python3
"""Spectro-temporal validation of the five NASA targets on cached REAL cutouts.

Runs the production pipeline offline (app/offline.py, catalogues disabled so
it needs no network) with the spectro-temporal step, and writes
docs/data/spectrotemporal_report.json plus a printed summary.
"""
import json
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
warnings.filterwarnings("ignore")

from app.jsonsafe import clean  # noqa: E402
from app.offline import run_cached  # noqa: E402

TARGETS = ("barnard", "eps_ind_b", "wise0855", "rw_aur", "r_dra")


def fmt(v, d=3):
    return "—" if v is None else (f"{v:.{d}g}" if isinstance(v, float) else str(v))


def main():
    report = {"note": "Cached REAL SPHEREx cutouts; production code; catalogues disabled (offline).", "targets": {}}
    for tgt in TARGETS:
        run = run_cached(tgt, spectrotemporal=True)
        st = run.analysis["spectrotemporal"]
        target = next(s for s in st["sources"] if s["kind"] == "target")
        entry = {
            "query": run.manifest["query"], "n_exposures": st["n_exposures"],
            "retrieval_failed_at_caching": st.get("retrieval_failed_at_caching"),
            "reference_stars": st["reference"]["n_stars"],
            "state_counts": target["state_counts"], "coverage": target["coverage"],
            "fits": target["fits"],
            "metrics": [{k: v for k, v in m.items() if k not in ("pairs",)} for m in target["metrics"]],
            "measurements": target["measurements"],
        }
        report["targets"][tgt] = entry
        print(f"\n== {tgt}: {st['n_exposures']} exposures, {entry['retrieval_failed_at_caching']} failed at caching, "
              f"{entry['reference_stars']} reference stars; target states {target['state_counts']}")
        for c in target["coverage"]:
            rng = "no usable data" if c["lambda_min_um"] is None else f"{c['lambda_min_um']:.3f}-{c['lambda_max_um']:.3f} um"
            print(f"   coverage {c['detector']} epoch {c['epoch']}: {c['n_usable']}/{c['n_exposures']} usable, {rng}, {c['states']}")
        for d, f in target["fits"].items():
            if f:
                amps = ", ".join(f"{a:.4g}±{e:.2g}" for a, e in zip(f["amplitudes_mjy"], f["amplitude_errs_mjy"]))
                print(f"   fit {d}: lambda0 {f['lambda0_um']:.3f} um; a_e = [{amps}] mJy; {len(f['knots_um'])} shape knots; "
                      f"chi2/dof {f['chi2']:.1f}/{f['dof']}; adequate {f['model_adequate']}; floor {100 * f['sys_floor']:.1f}%")
            else:
                print(f"   fit {d}: none (not enough usable measurements)")
        for m in target["metrics"]:
            stat = m.get("delta_chi2", m.get("chi2"))
            print(f"   [{m['data']}] {m['metric']} {m.get('detector') or m.get('detectors')}: {m['status']}; "
                  f"stat {fmt(stat)} dof {fmt(m.get('dof'))} p {fmt(m.get('p_value'))}; "
                  f"ref frac {fmt(m.get('reference_fraction_at_least_as_large'))} -> {m['assessment']}")
    out = REPO / "docs" / "data" / "spectrotemporal_report.json"
    out.write_text(json.dumps(clean(report), indent=1))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
