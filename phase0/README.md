# Phase 0 measurement scripts

Run from the repo root with the `backend` venv active
(`backend/requirements.txt`, Python 3.11). Each script targets Barnard's Star
(RA=269.452, Dec=4.693) unless noted. Findings are summarized in
[`../PLAN.md`](../PLAN.md#phase-0--measurements-done); these scripts are kept
so the numbers are reproducible, not as library code the app imports.

| Script | What it measures |
|---|---|
| `01_discover_collections.py` | Runtime SIA collection discovery (confirms no hard-coded release) |
| `02_tap_query.py` | TAP query timing (cold/warm), band/release distribution |
| `03_sia_query.py` | SIA query timing (cold/warm) — compare against TAP |
| `04_cutout_bytes_and_flags.py` | Per-extension byte breakdown of a cutout (the EPSF finding) + live FLAGS bit names |
| `05_s3_range_probe.py` | Anonymous S3 HEAD/Range latency + CORS header on the bucket |
| `06_s3_lazy_section_experiment.py` | The row-alignment pitfall: naive `s3fs` + astropy `.section` lazy reads, timed |
| `07_simbad_mirror_pyvo.py` | SIMBAD astrometry (pmra/pmdec/plx) via the Harvard TAP mirror for all 5 acceptance targets |
