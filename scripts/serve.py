#!/usr/bin/env python3
"""Run the app (backend + frontend) with a chosen data source.

    python scripts/serve.py                 # auto: IRSA, falling back to downloaded data
    python scripts/serve.py --mode cached   # offline: downloaded data only, no network
    python scripts/serve.py --mode live     # IRSA only

The downloaded data are the REAL cutouts and catalogue rows under
backend/tests/data/real (or --data DIR). See app/localdata.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("auto", "cached", "live"), default="auto")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--data", help="directory of downloaded fields (default backend/tests/data/real)")
    args = ap.parse_args()

    os.environ["SPHEREX_DATA_MODE"] = args.mode
    if args.data:
        os.environ["SPHEREX_LOCAL_DATA"] = str(Path(args.data).resolve())
    sys.path.insert(0, str(REPO / "backend"))

    from app import localdata  # noqa: E402

    fields = localdata.fields()
    print(f"Data mode: {args.mode}. Downloaded fields in {localdata.data_root()}:")
    for f in fields:
        print(f"  - {f.manifest.get('query', f.name)}: {'+'.join(f.detectors)}, {f.fov_arcmin:g}', "
              f"{len(f.manifest['exposures'])} exposures, catalogues {'yes' if f.has_catalogues() else 'no'}")
    if not fields:
        print("  (none)")
    print(f"Open http://{args.host}:{args.port}/")

    import uvicorn  # noqa: E402

    uvicorn.run("app.main:app", host=args.host, port=args.port, app_dir=str(REPO / "backend"))


if __name__ == "__main__":
    main()
