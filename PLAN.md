# PLAN.md — SPHEREx Blink

A public web tool to blink SPHEREx epochs at a sky position, built for the NASA
Space Apps challenge. This plan follows the brief in `SPHEREx_tool_prompt.md`.
Phase 0 below reports **real, measured numbers** from live IRSA/SIMBAD/S3
services (Nov 14 2026) and calls out where the live data forced us to deviate
from the brief's assumptions — per the brief's own instruction: *"If something
differs from this prompt, trust the live data and note it."*

## Environment note
System Python is 3.14, which currently has no wheels for astropy/astroquery/
reproject/numpy. We pin **Python 3.11** in a venv (`backend/requirements.txt`,
Dockerfile, README) to avoid build-from-source pain.

---

## Phase 0 — Measurements (done)

All scripts that produced these numbers live in `phase0/` so they're
reproducible. Target used throughout: Barnard's Star, RA=269.452 Dec=4.693.

### 1. Collections (runtime discovery works, confirms "never hard-code")
`Irsa.query_sia`/SIA collection listing on 2026-11-14 shows:
`spherex_dr1`, `spherex_dr1_cube`, `spherex_dr1_deep`, `spherex_qr3`,
`spherex_qr3_deep`. **No QR/QR2 collection remains** — DR1/QR3 have fully
superseded them, exactly as the brief warned. These five names go in one
config file (`config/collections.yaml`), never hard-coded in request code.

### 2. TAP vs SIA discovery speed — **major deviation**
The brief says TAP (the artifact/plane join) is "preferred." Measured on the
same position/radius:
| Method | Cold | Warm | Rows |
|---|---|---|---|
| TAP (`spherex.artifact` ⋈ `spherex.plane`) | 31.5s | 13.5s | 363 |
| SIA (`Irsa.query_sia`) | 5.7s | 4.4s | 379 |

TAP's own `server-timing` header confirms ~12.3s is server-side
`query_delegate`/`database_query` time, not our network. **Decision: use SIA
as the primary, user-facing discovery call (meets "first frame in a few
seconds" far better); keep TAP available as a slower fallback/cross-check**
(e.g., a background "any very recent exposures SIA might be missing"
refresh), since the brief notes SIA lags ~1 day behind TAP for the newest data.
The small row-count difference (363 vs 379) is because SIA's cone search and
TAP's exact-point `CONTAINS` test aren't identically shaped queries — harmless
for our purposes.

### 3. Cutout fetch: latency, size, and the EPSF problem
Single-cutout fetch latency from the documented `?center=&size=` cutout
endpoint was **11–20s per request**, roughly flat across 2′/5′/15′ FOV — and
intermittent `503 Server busy, retry later.` responses occurred even on a
single sequential request. **The retry list in the brief (TimeoutError,
HTTPError, IncompleteRead, RemoteDisconnected) must also treat HTTP 503
explicitly, with backoff** — confirmed necessary, not hypothetical.

Byte breakdown at 2′ (20×19 px image):

| Extension | Bytes | % of file |
|---|---|---|
| IMAGE/FLAGS/VARIANCE/ZODI (each) | 1,520 | ~0.04% each |
| **EPSF** | **3,859,632** | **97.7%** |
| WCS-WAVE | 720 | ~0.02% |

**The cutout service does not crop the EPSF table to the requested FOV at
all** — it's the same ~3.86MB regardless of whether you ask for 2′ or 15′.
This means >97% of every cutout's bytes are a PSF table we don't use for
display. **Decision: the backend fetches the full response (we can't avoid
that leg), but strips EPSF immediately in memory and never forwards it to the
browser.** EPSF is only worth fetching at all in Phase 4 (PSF photometry), and
even then it should be opt-in per request, not default.

### 4. FLAGS bit names — confirms "read from header, don't hard-code"
Live `FLAGS` header `MSKNnnnn` keys returned bit 2 = `SUR_ERROR`, bit 7 =
`DICHROIC`, bit 12 = `FULLSAMPLE`, bit 14 = `PHANMISS` — bits not listed in the
brief's quick-reference table. Confirms the brief's own instruction: flag
names **must** be read from each file's header, never hard-coded from a fixed
table.

### 5. CORS — backend is mandatory for IRSA calls
No `Access-Control-Allow-Origin` header on IRSA TAP, SIA, or the cutout
service. A "no-backend" browser-only mode is **not possible** against IRSA
directly — confirms the architecture in the brief.

### 6. The S3 mirror: fast, CORS-enabled, but not a v1 shortcut
`s3://nasa-irsa-spherex` (anonymous, `us-east-1`) answered HEAD/Range requests
in ~0.7–0.8s, and — unlike IRSA — **the bucket itself sends
`Access-Control-Allow-Origin: *`**. That's tempting for a browser-only future
mode. We tried the obvious "use it to dodge the EPSF tax" approach: open the
full-frame FITS lazily via `s3fs` + astropy's `.section` and pull only the
pixel window we need. Result: **slower, not faster** — 38s just to walk the
HDU list, then 69s for a single 50×50px slice (110s total), because the
window's rows aren't contiguous in the file and `s3fs`'s default block
caching turns one logical read into dozens of small ranged fetches. A
row-aligned single whole-width Range GET per extension (using the byte offsets
astropy already computes when it parses headers) would likely fix this, since
raw S3 Range latency alone is sub-second — but that's bespoke byte-offset
engineering against SPHEREx's specific FITS layout (and the FLAGS extension is
a `CompImageHDU`, whose compressed tile layout complicates this further).
**Decision: not in v1.** Documented here as a concrete, promising Phase-5/
future-work item, not adopted now — the documented cutout-endpoint path
already works and is lower risk for a hackathon deadline.

### 7. SIMBAD — **major deviation, needed for Phase 1 to work at all**
`simbad.cds.unistra.fr` (astroquery's default SIMBAD host) returned **404 on
every path we tried**, including its legacy non-TAP API and bare capabilities
endpoint — it was unavailable at measurement time, not a transient blip
(`cds.unistra.fr`'s main site is up; the SIMBAD subpaths specifically are not).
astroquery 0.4.11's `Simbad.query_object`/`query_tap` hard-fail as a result,
and setting `conf.server` does **not** redirect them (the TAP base URL isn't
driven by that config in this version).

What does work: `SkyCoord.from_name` (uses the separate Sesame resolver, a
different code path) resolves all 5 test targets correctly. For proper
motion/parallax we need more than `from_name` gives, so we query SIMBAD's
mirror directly: **`https://simbad.harvard.edu/simbad/sim-tap`** via `pyvo`'s
`TAPService`, bypassing `astroquery.simbad` entirely. Verified for all 5
targets (see table below) — values match known literature, which is reassuring
given the data is public and users will probe it.

| Target | main_id found via | pmra | pmdec | plx (mas) |
|---|---|---|---|---|
| Barnard's Star | `basic` exact match | (large, confirmed) | 10362.4 mas/yr | 546.98 |
| eps Ind B | `basic` exact match | 3981.98 | -2466.83 | 270.66 |
| WISE 0855−0714 | `ident`⋈`basic` (id=`WISE J085510.83-071442.5`) | — | 673.2 | 439.0 |
| RW Aurigae | `ident`⋈`basic` (id=`RW Aur`) | 7.23 | -20.33 | 15.42 |
| R Draconis | `ident`⋈`basic` (id=`R Dra`) | -20.64 | 6.15 | 1.25 |

**Decision:** `backend/app/simbad.py` tries a short configured list of SIMBAD
TAP mirrors (Harvard first, since it's the one confirmed live; CDS kept in the
list in case it comes back) via raw `pyvo.dal.TAPService`, with `ident`⋈`basic`
join fallback when the literal name isn't a `main_id`. `SkyCoord.from_name`
(Sesame) is still used for the quick RA/Dec-only path and as a last resort.

### Bottom line for architecture
Given measured latencies (SIA ~4-6s, a single cutout ~11-20s with occasional
503s), **"first frame in a few seconds" is only achievable by being aggressive
about parallelism, streaming per-frame, and shrinking the default
workload** — not by any single trick. v1 defaults: FOV 5′, 2 detectors
(D6+D2) as the brief suggests, epoch-grouped (fewest frames, best S/N),
8–10 concurrent cutout fetches. Cold time-to-first-frame will realistically be
dominated by one cutout's ~15s, not "a few seconds" — we report this honestly
in the smoke test rather than overclaiming, and treat it as the #1 thing to
optimize further (candidate: the S3 row-aligned range-read from finding #6).

---

## Architecture (as specified, refined by the above)

```
Browser (static HTML/JS, Canvas2D, IndexedDB cache)
   │  SSE/NDJSON stream of compact per-frame binary + JSON metadata
   ▼
FastAPI backend (stateless, in-memory only)
   ├─ discover.py   SIA primary / TAP fallback, SIMBAD multi-mirror resolver
   ├─ irsa_fetch.py cutout fetch: ThreadPoolExecutor(8-10), retry+backoff incl. 503
   ├─ pipeline.py   subtract ZODI → mask flags → reproject to common TAN grid →
   │                group (epoch|detector|single) → inverse-variance combine
   ├─ encode.py      strip EPSF, quantize to uint16 (scale/offset in metadata)
   └─ cache.py       tiny in-memory LRU, few-minutes TTL, de-dupes in-flight
                      identical requests only — no disk, no DB
   ▼
IRSA (SIA + cutout HTTPS) · SIMBAD (pyvo TAP, Harvard mirror)
```

### File layout
```
Nasa_spaceapps/
├── PLAN.md, README.md, SPHEREx_tool_prompt.md
├── phase0/                  # measurement scripts (reproducible)
├── config/
│   └── collections.yaml     # SPHEREx collections, IRSA/SIMBAD endpoints
├── backend/
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py          # FastAPI app + endpoints
│   │   ├── config.py        # loads config/collections.yaml
│   │   ├── coords.py         # sexagesimal/decimal/name parsing, PM propagation
│   │   ├── simbad.py         # multi-mirror pyvo TAP client
│   │   ├── irsa.py           # SIA/TAP discovery + cutout fetch (pool+retry)
│   │   ├── pipeline.py       # zodi-subtract, flag-mask, reproject, combine
│   │   ├── stream.py         # groups known pre-fetch; emits each the instant it's complete
│   │   ├── spectrum.py       # Phase 4: native-pixel/native-wavelength forced photometry
│   │   ├── encode.py         # EPSF strip, uint16 quantize, metadata
│   │   └── cache.py          # in-memory TTL/LRU in-flight de-dupe (wired into
│   │                         # /api/images only - see README's trade-off note)
│   └── tests/
│       ├── test_coords.py
│       └── test_pipeline_live.py   # marked slow/network, opt-in
├── cli/
│   └── spherex_blink.py      # Phase 1 standalone script (no server/storage)
├── frontend/
│   ├── index.html, app.js, style.css
│   └── idb_cache.js
├── scripts/
│   └── smoke_test.py         # runs the 5 acceptance targets, writes examples/
├── examples/                  # smoke test GIFs/diff images + report
├── docker-compose.yml, Dockerfile
└── .gitignore
```

### Risks
- **IRSA rate limiting / 503s** under concurrent user load — mitigated by a
  global concurrency cap (not just per-request), polite backoff, and the
  in-memory dedup cache for identical concurrent requests.
- **SIMBAD single point of failure** — mitigated by the mirror list; if all
  mirrors are down, fall back to Sesame-only (position, no PM) and say so in
  the UI rather than failing the whole request.
- **"First frame in a few seconds" may not be fully achievable in v1** given
  measured per-cutout latency; mitigated by honest reporting + streaming
  (something useful appears as soon as the *first* cutout of the *first*
  group is ready, not after the whole set).
- **reproject cost** at many frames/detectors could itself be slow; mitigate
  by reprojecting onto a small common grid sized to the requested FOV only.
- Python 3.14 incompatibility with the astro stack — pinned 3.11 everywhere.

### Post-launch finding: dropped the `reproject` package
The brief says "use astropy, astroquery, reproject and numpy... don't
reimplement them." We did, for one narrow piece: nearest-neighbour
reprojection (`app/pipeline.py`'s `reproject_nearest`), replacing
`reproject.reproject_interp(order="nearest-neighbor")`.

Root cause: a real bug report (Barnard's Star showing "no data" at its
own target pixel) led first to a flag-masking fix, then to switching
reproject's default bilinear mode to its own nearest-neighbor mode (fixes
most of it — bilinear blends in a masked neighbour's NaN even when the
exact output pixel's own nearest native pixel is clean). But a live check
(`scripts/check_center_pixel_coverage.py`) still found residual cases:
reproject's *own* nearest-neighbor mode returned a 0 footprint (NaN) at
specific exposures' exact pixel-boundary coordinates (e.g. input pixel
coordinate `(13.79, 13.56)`) even though the nearest actual native pixel
was clean and unmasked. Confirmed directly (`tmp_phase0/debug_nn_residual.py`):
reproject's footprint computation is more conservative than "does the
nearest native pixel exist and have a value" right at certain boundaries.

Nearest-neighbour reprojection is trivial (`astropy.wcs.WCS.pixel_to_world`
+ `world_to_pixel` + round — both already provided by astropy, which we're
still using exactly as intended; only `reproject`'s own black-box footprint
logic was dropped), so implementing it directly gives full control and
eliminates the disagreement by construction. `reproject` itself is no
longer a dependency (backend/requirements.txt).

---

## Phases 1–4
Build in order, per the brief: Phase 1 (data layer/CLI, no UI, unit tests +
proper motion), Phase 2 (FastAPI + blink viewer MVP), Phase 3 (difference/sum/
flag overlay/light curve), Phase 4 (spectra, bonus). Commit after each phase
passes its own checks. Smoke test (5 acceptance targets → GIF + diff image +
timing report in `examples/`) runs at the end of Phase 2 at the latest so we
have early honest signal, then again after Phase 3/4 additions.
