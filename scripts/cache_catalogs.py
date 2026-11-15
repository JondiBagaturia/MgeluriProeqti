#!/usr/bin/env python3
"""Cache REAL Gaia DR3 and CatWISE2020 cone-search rows for the five cached
validation fields (backend/tests/data/real/<target>/catalogs/<table>.csv), so
offline runs and tests can do catalogue matching without the network.

Only the columns the pipeline reads are kept (app/catalogs.py NEEDED_COLUMNS).
The cone is centred on the cutout centre with radius 0.07 deg, which covers
the radius the analysis asks for (0.0625 deg for a 5' field). Rows are not
modified otherwise. Retries on IRSA 502/503 (seen often).
"""
import io
import json
import sys
import time
from pathlib import Path

import requests
from astropy.table import Table

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
from app.catalogs import NEEDED_COLUMNS, SCS_URL  # noqa: E402

ROOT = REPO / "backend" / "tests" / "data" / "real"
RADIUS = 0.07


def fetch(table, ra, dec):
    for attempt in range(8):
        try:
            r = requests.get(SCS_URL, params={"table": table, "RA": ra, "DEC": dec, "SR": RADIUS, "format": "csv"},
                             timeout=(10, 90))
            if r.status_code == 200:
                return r.text
            print(f"   {table}: HTTP {r.status_code}, retry {attempt + 1}")
        except requests.RequestException as exc:
            print(f"   {table}: {exc}, retry {attempt + 1}")
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{table} failed after retries")


def main():
    for d in sorted(p for p in ROOT.iterdir() if (p / "manifest.json").exists()):
        m = json.loads((d / "manifest.json").read_text())
        ra, dec = m["cutout_centre_deg"]
        out = d / "catalogs"
        out.mkdir(exist_ok=True)
        for table, cols in NEEDED_COLUMNS.items():
            dest = out / f"{table}.csv"
            if dest.exists():
                print(f"{d.name} {table}: already cached")
                continue
            t = Table.read(fetch(table, ra, dec), format="ascii.csv")
            t = t[[c for c in cols if c in t.colnames]]
            buf = io.StringIO()
            t.write(buf, format="ascii.csv")
            dest.write_text(buf.getvalue())
            meta = {"table": table, "centre_deg": [ra, dec], "radius_deg": RADIUS, "n_rows": len(t),
                    "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "note": "REAL catalogue rows from IRSA /SCS; only the columns the pipeline uses are kept."}
            (out / f"{table}.json").write_text(json.dumps(meta, indent=1))
            print(f"{d.name} {table}: {len(t)} rows")


if __name__ == "__main__":
    main()
