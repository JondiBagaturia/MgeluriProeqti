# SPHEREx Blink

A public web tool to blink NASA SPHEREx epochs at any sky position, so moving
or variable objects pop out visually — the same idea volunteers used on
[WiseView](http://byw.tools/wiseview) to find brown dwarfs in WISE data.

Built for a NASA Space Apps hackathon. See [PLAN.md](PLAN.md) for the
architecture and the real, measured numbers behind every design decision,
and [SPHEREx_tool_prompt.md](SPHEREx_tool_prompt.md) for the original brief.

## What is SPHEREx?

SPHEREx (Spectro-Photometer for the History of the Universe, Epoch of
Reionization and Ices Explorer) is a NASA all-sky survey satellite. Unlike a
camera with a few fixed color filters, each of its 6 detectors has a
*linear variable filter* — the wavelength it sees changes smoothly across
the detector. As the satellite scans the sky, every point ends up imaged at
many different wavelengths (0.75–5.0 µm) across six detectors (D1–D6), and
the whole sky gets covered again roughly every 6 months. So a single cutout
from SPHEREx is really "one detector, one wavelength (at this position),
one moment in time" — and a few years of the mission gives you a handful of
independent snapshots of the same patch of sky, which is exactly what a
blink comparator needs.

## Using the tool

1. Run it (see **Run it** below) and open the URL it prints.
2. Type a position — sexagesimal (`08h55m10.831s -07d14m42.53s`), decimal
   degrees (`269.452 4.693`), or an object name (`Barnard's star`,
   `WISE J085510.83-071442.5`) — and press **Go**.
3. Pick a field of view (1′–30′; above 15′ is noticeably slower) and which
   of the 6 detectors to include (fewer = faster).
4. Watch frames stream in. Play/pause, step, or drag the frame slider to
   blink through survey epochs. Use the contrast sliders (asinh is usually
   the most revealing for faint sources), zoom, and the crosshair/pixel
   readout to inspect.
5. Try **Difference (B−A)** or **Sum all epochs** under "Discovery" — a
   mover shows up as a dark/bright dipole in the difference view.
6. Click anywhere in the viewer for a flux-vs-wavelength spectrum at that
   exact pixel (bonus feature — slow, since it re-queries IRSA).
7. The URL updates as you search — copy it, or use the generated `<iframe>`
   snippet, to share or embed the exact view.

### Run it
```bash
# Python 3.11 — astropy/astroquery/reproject/numpy have no wheels yet for
# newer Pythons as of Nov 2026 (see PLAN.md).
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000/
```
Or with Docker:
```bash
docker compose up --build
# open http://localhost:8000/
```
The backend is stateless (no database, no disk cache — see PLAN.md), so it
scales horizontally behind any load balancer with zero shared state.

### Command-line / data layer only
```bash
python cli/spherex_blink.py "Barnard's star" --fov 3 --detectors D6 -o ./out
```
Writes `frames.npz` (the combined epoch images) and `metadata.json` — useful
for testing the pipeline without a browser.

## What's actually in each frame

- **Grouping**: per survey epoch (default, best signal/noise — all exposures
  in a ~6-month pass combined), per detector per epoch, or every single
  exposure.
- **Processing**: each exposure has its zodiacal-light model subtracted,
  bad pixels masked using that file's *own* FLAGS header (bit names are read
  from the header, never hard-coded — SPHEREx's bit assignments turned out
  to not exactly match the quick-reference table in the original brief; see
  PLAN.md), then everything is reprojected onto one common grid centred on
  your target and combined with an inverse-variance weighted mean.
- **Compactness**: the server sends a 16-bit quantised array + scale/offset,
  never a pre-rendered image — contrast/stretch changes are instant,
  computed from the same cached array, no new network request.

## Honest results — the 5 acceptance targets

Ran via `scripts/smoke_test.py` (single detector, D6, 3′ FOV, epoch
grouping — see that script for why: speed and being polite to IRSA, which
we'd already leaned on heavily during development). Outputs and full timing
are in [`examples/`](examples/) (`smoke_test_report.json` has the raw
numbers). IRSA throttled us noticeably harder as the run went on — by
target 5 we were seeing more `503`s than at the start of the day — which
is itself a useful data point (see the engineering-rules note in PLAN.md).

| Target | Found visible? | Notes |
|---|---|---|
| [Barnard's Star](examples/Barnards_Star_blink.gif) | **Yes** — clean dipole | [Difference image](examples/Barnards_Star_difference.png) shows a textbook dark/bright pair between MJD 60908 and 61272 (~1 yr), matching its ~10.3″/yr proper motion. |
| [Epsilon Indi B](examples/Epsilon_Indi_B_blink.gif) | **Yes** — clean dipole | [Difference image](examples/Epsilon_Indi_B_difference.png) shows the same signature for B's own large proper motion (~4.7″/yr). Note: Eps Ind A is ~24′ away — far outside a 3′ FOV — so this shows B's motion alone, not "near the primary" as literally pictured in the brief. |
| [WISE 0855−0714](examples/WISE_0855-0714_blink.gif) | **No** — honest non-detection | [Difference image](examples/WISE_0855-0714_difference.png) is pure noise, no dipole. Single detector (D6), single epoch per frame, 3′ FOV — this target is right at the edge of SPHEREx's sensitivity even in the best band; the brief explicitly allows "if it isn't detectable, the tool must say so rather than fail." Worth trying with D5+D6 combined and/or the "sum all epochs" mode before concluding it's truly undetectable. |
| [RW Aurigae](examples/RW_Aurigae_blink.gif) | **Partial** — point source detected, shape/brightness changes | [Difference image](examples/RW_Aurigae_difference.png) shows an asymmetric, overlapping (not cleanly separated) light/dark pattern — consistent with brightness variability rather than motion, which is what's expected for this target. Not quantitatively confirmed as a light curve in this run (see note below). |
| [R Draconis](examples/R_Draconis_blink.gif) | **Yes, with a real caveat** | The blink GIF shows a dark hole surrounded by a bright ring. Checked directly against a live cutout: the core pixel's `IMAGE` value is `NaN` with `FLAGS` bits `OVERFLOW` + `BLOOM` set — SPHEREx's own Level-2 pipeline has already blanked out this pixel because the star **saturates and blooms the detector**. R Draconis is a Mira variable bright enough to overflow SPHEREx's D6 — the difference image is accordingly a flat grey disc (NaN − NaN, both epochs saturated the same way), which is itself evidence of how bright this source gets, just not a measurement of *how much* brighter. |

A real gap this surfaced: `scripts/smoke_test.py` doesn't record the
target-pixel flux per frame (the light curve feature does this live in the
browser, reading it straight from the already-decoded array — that path is
not separately re-verified against live data in this run). Quantitatively
confirming RW Aurigae's/R Draconis's variability numerically, not just
visually, is a reasonable next step for the smoke test script itself.

**Timing honesty**: PLAN.md's Phase 0 measurements already show that a
single IRSA cutout fetch takes 11–20s with occasional `503 Server busy`
responses, and a full TAP/SIA discovery call takes several seconds more —
so "first frame in a few seconds," while the design target, is not
consistently achievable against the documented IRSA endpoints as they exist
today. A live end-to-end run (curl, Barnard's Star, 3′ FOV, D6 only, 59
images) measured **first frame at 37s, full set at 122.6s**. The backend
streams progressively specifically to soften this: something useful appears
well before the whole set finishes, and a fast epoch doesn't wait on a slow
one (verified: groups completed out of chronological order in that same
run). The browser's IndexedDB cache makes a *repeat* view of the same
position instant; that path is exercised by the frontend's code (and its
own tests) but intentionally wasn't re-verified against live IRSA again
here to avoid hitting their service a third/fourth time for the same query.

## Architecture, briefly

Stateless FastAPI backend (discover via IRSA SIA → fetch cutouts in
parallel with retry/backoff → zodi-subtract → flag-mask → reproject →
group → inverse-variance combine → stream as compact NDJSON) feeding a
plain-JS frontend that does all display work (stretching, blinking, zoom,
difference/sum, light curve, spectrum) from cached typed arrays, plus an
IndexedDB cache keyed by release+ra+dec+fov+group+detector so a new SPHEREx
data release invalidates stale cache automatically. Full rationale and
every measured number behind these choices: [PLAN.md](PLAN.md).

One deliberate trade-off: the backend's in-memory request-coalescing cache
(`app/cache.py`) is only wired into `/api/images` (the fast, metadata-only
lookup). `/api/frames` and `/api/spectrum` deliberately do **not** share
it, even though that means two users asking for the same position seconds
apart both hit IRSA — caching those would mean fully materializing the
result before replying, which would sacrifice genuine incremental
streaming (the brief's top priority) just to save IRSA a duplicate request.
Given the choice, speed-to-first-frame won.

## What works

- Coordinate parsing (sexagesimal, decimal, object name) and SIMBAD proper
  motion propagation — verified live for all 5 acceptance targets.
- Runtime SPHEREx collection discovery (no hard-coded release).
- The full pipeline end-to-end against live IRSA data: discovery, parallel
  fetch with retry/backoff, zodi subtraction, header-driven flag masking,
  reprojection, epoch grouping, inverse-variance combination.
- Progressive streaming to the browser with out-of-order group completion.
- Blink playback, A/B flip, difference image, sum-all-epochs, light curve
  (target-pixel flux vs time), contamination summary, pixel readout,
  URL state + embed snippet, IndexedDB caching with LRU eviction.
- Bonus: click-to-spectrum (forced photometry at one pixel, all 6
  detectors, correct per-exposure wavelength), linked to IRSA's own
  Spectrophotometry Tool for comparison.
- A per-stage timing report (one log line per `/api/frames` request:
  resolve/discover/fetch/reproject/combine/encode, plus fetch's own byte
  count and effective parallelism) that settled where the time actually
  goes — fetch is ~97% of it, everything else is noise — instead of
  guessing. Led directly to the Quality slider below.
- A **Quality slider** (Low/Mid/High) that caps exposures fetched *per
  epoch*, evenly spaced across the epoch's time span (first/last exposure
  always kept, so the baseline is preserved) — capped before fetching,
  not after, since that's where the time actually goes. Measured on
  Barnard's Star (D6, 3′ FOV): High fetched 56 images / 211.6 MB / 155.7s;
  Low fetched 12 images / 45.4 MB / 38.7s — same 3 epochs either way, just
  fewer (noisier) exposures stacked into each.
- The Asinh display mode is a ported renderer from a teammate's parallel
  prototype ([DmaniseliMgeli/planet-x](https://github.com/DmaniseliMgeli/planet-x),
  `phase1-check/phase1/blink.py`): astropy's `AsinhStretch(a=0.05)` with
  levels fixed at the median/99.8th percentile across every loaded frame
  (not recomputed per frame, so contrast stays consistent through the
  blink), plus its red/salmon gapped crosshair. Their script independently
  found and fixed the same bright-source-core masking bug documented
  below, which is a good independent sanity check on both of our pipelines.

## What doesn't (yet)

- **Speed**: "first frame in a few seconds" isn't met against the
  documented IRSA cutout endpoint as it behaves today — see the honesty
  note above. PLAN.md documents a promising-but-unproven fix (row-aligned
  HTTP Range reads straight from the public S3 mirror, skipping the ~98%
  of every cutout's bytes that's an EPSF table we never use) that a naive
  implementation of actually made *slower*, not faster — it needs real
  byte-offset engineering against SPHEREx's FITS layout, not more time
  pressure.
- **Flag overlay as a translucent per-pixel mask**: Phase 3 asked for a
  toggleable overlay per flag type. What's implemented is the textual
  contamination summary (flag name + pixel count per frame), not a visual
  overlay — that needs the backend to also reproject and ship a combined
  flags layer per frame, which wasn't done.
- **PSF/aperture photometry**: the spectrum feature reads a single native
  pixel per exposure, not a PSF-fit or aperture sum. Good enough to see
  whether a position has detectable flux and roughly what shape the
  spectrum has; not science-grade photometry.
- **RGB colour composite**: not implemented.
- **Light curve**: reads the same single target pixel as the spectrum
  feature, not aperture photometry — noted in the UI itself. The smoke
  test doesn't record this number either (see the acceptance-target table
  above), so RW Aurigae's/R Draconis's variability is confirmed visually,
  not with an actual flux-vs-time plot from that script.
- **Single-pixel photometry fails outright on saturated sources**: R
  Draconis (see the table above) is bright enough to overflow/bloom
  SPHEREx's D6 detector, and SPHEREx's own Level-2 pipeline already blanks
  that pixel to `NaN`. Forced single-pixel photometry has nothing to read
  at the core for a source this bright — only PSF/aperture photometry on
  the (still valid) wings would recover a usable brightness, which is the
  same missing capability as the PSF/aperture point above.

## Suggested next steps

1. Prototype the S3 Range-read optimization properly (row-aligned whole-
   width reads per needed extension, skip EPSF entirely, handle the
   `CompImageHDU` compression on FLAGS) — PLAN.md has the groundwork.
2. Ship a real per-pixel flag overlay: reproject FLAGS with
   nearest-neighbour (not bilinear — it's a bitmask), OR-combine within a
   group, send as a second compact payload per frame.
3. Add PSF-based forced photometry for the spectrum and light-curve
   features, using the EPSF extension this tool currently always skips.
4. Load-test the global concurrency cap under multiple simultaneous users
   — right now it's a config value, not something observed under load.
5. Have `scripts/smoke_test.py` record target-pixel flux per frame (cheap —
   the data's already in memory) so RW Aurigae/R Draconis variability has
   an actual number attached, not just "look at the GIF."
6. Try WISE 0855−0714 again with D5+D6 combined and/or "sum all epochs" —
   this run's non-detection used one detector, one epoch per frame.

## Credits

> This publication makes use of data products from the Spectro-Photometer
> for the History of the Universe, Epoch of Reionization and Ices Explorer
> (SPHEREx), which is a joint project of the Jet Propulsion Laboratory and
> the California Institute of Technology, and is funded by the National
> Aeronautics and Space Administration.
>
> — required acknowledgement, [IRSA's SPHEREx mission page](https://irsa.ipac.caltech.edu/Missions/spherex.html)

This research has made use of the SIMBAD database, operated at CDS,
Strasbourg, France, for object name resolution and proper motion data.

Data access: [NASA/IPAC Infrared Science Archive](https://irsa.ipac.caltech.edu/).
Tutorials this tool's IRSA access patterns were built from:
[caltech-ipac.github.io/irsa-tutorials/spherex](https://caltech-ipac.github.io/irsa-tutorials/spherex/).
