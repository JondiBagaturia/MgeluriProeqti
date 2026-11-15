# SPHEREx Discovery Lab — Codebase & Scientific Audit

Date: 2026-11-15. Scope: everything under `MgeluriProeqti/` at commit `627c463`.

> Historical record of the original state and its fixes. For the current state see
> [PHASE2.md](PHASE2.md) and [../PROJECT_STATUS.md](../PROJECT_STATUS.md).

**How this was verified.** Every backend and frontend file was read. The offline test suite
was run (31 passed, 6 live tests deselected). These checks were made against live services
on 2026-11-15: an IRSA SIA query with 377 rows for Barnard's Star, inspection of a real D6
cutout's headers, a check for overlap between DR1 and QR3 and for gaps between epochs at
three targets, and SIMBAD resolution and proper-motion propagation for all five targets. The
full `/api/frames` stream was **not** re-run end to end in this audit. Statements about it
come from reading the code and from the team's own `examples/smoke_test_report.json`.
Claims marked *(verified)* were tested here. Claims marked *(code reading)* come from reading
the code only.

---

## 1. What the application is today

"SPHEREx Blink" is a WiseView-style **blink comparator**, not yet a discovery platform.
A user enters a position or name. The backend finds SPHEREx Level-2 spectral images with
IRSA SIA and downloads cutouts in parallel. For each cutout it subtracts the ZODI extension,
masks four flag bits and resamples onto a common TAN grid using nearest-neighbour
resampling. It then groups the cutouts by survey "epoch", combines each group with an
inverse-variance weighted mean, quantises the result to uint16 and streams it as NDJSON. A
plain-JS frontend blinks the frames and can show B−A differences, a mean of all epochs, a
single-pixel "light curve" and a single-pixel spectrum when you click the image.

## 2. Architecture map

```
frontend/ (vanilla JS, Canvas2D, IndexedDB cache)
  index.html, app.js (UI, stretch, blink, diff, light curve, spectrum plot), idb_cache.js
        │  GET /api/resolve, /api/images, /api/frames (NDJSON stream), /api/spectrum, /api/releases
        ▼
backend/app/ (FastAPI, stateless, run_in_threadpool)
  main.py      endpoints, CORS *, static mount
  coords.py    sexagesimal/decimal/name parsing, PM propagation (astropy apply_space_motion)
  simbad.py    raw pyvo TAP against SIMBAD mirrors (Harvard, CDS)
  irsa.py      collection discovery, SIA query, cutout fetch (ThreadPool, retry, global semaphore)
  pipeline.py  flag parsing, zodi subtraction, masking, nearest-neighbour reprojection,
               epoch clustering, inverse-variance coadd, small-hole filling
  stream.py    progressive per-group emission + timing report
  encode.py    uint16 quantisation + metadata
  spectrum.py  single-native-pixel forced "photometry" per exposure
  cache.py     in-memory TTL de-dupe (only /api/images)
config/collections.yaml   releases, mirrors, detectors, FOV, fetch, quality presets, grid scale
cli/, scripts/ (smoke test, coverage checks), phase0/ (measurement scripts), examples/
```

No database, no source catalogue, no persistent storage. Docker image uses Python 3.11; the
local `.venv` is Python 3.12 (README says 3.11 — harmless mismatch).

## 3. Feature matrix (intended product vs. today)

> This table is the state **at the time of the audit**. Detection, tracking, catalogue
> matching, ranking, classification, export and the coordinate-epoch lookup were added
> afterwards: see [DISCOVERY.md](DISCOVERY.md).

| Capability | Status | Notes |
|---|---|---|
| Coordinate input (decimal, `h m s`) | **Works** *(verified)* | `hh:mm:ss` colon form was not parsed as coordinates and fell through to Sesame name resolution (fixed, see §8) |
| Name resolution + PM | **Works** *(verified, 5/5)* | SIMBAD Harvard mirror |
| Coordinate **epoch** handling | **Missing** | Typed coordinates are assumed fixed. NASA's WISE 0855 coordinates are the ~2010 position; the object is now ~2.1′ away |
| SIA discovery, DR1+QR3 | **Works** *(verified)* | DR1 and QR3 are time-disjoint at the three targets checked. No duplicate exposures |
| Cutout fetch, retry, EPSF strip | **Works** *(code reading + team logs)* | ~4 MB per cutout, 5–20 s each |
| FITS units / WCS | **Mostly correct** *(verified headers)* | IMAGE is MJy/sr (surface brightness), VARIANCE is (MJy/sr)², TAN-SIP |
| Multi-epoch blink | **Works** | |
| Image alignment | **Partial** | WCS-only, nearest-neighbour onto a 2× oversampled grid. No registration check |
| Difference imaging | **Partial / misleading** | No PSF matching, no flux scaling, ±½-native-pixel resampling jitter, mixed wavelengths |
| Addition | **Mislabelled** | "Sum" is a per-pixel mean of epochs |
| Detector co-addition | **Scientifically wrong by default** | see C1 |
| Light curve | **Broken for its main use cases** | see C3, C4 |
| Spectrum F(λ) | **Partial + crashes** | see C2, H3 |
| F(λ, t) | **Missing** | The spectrum plot colours points by time, nothing more |
| Contamination flags | **Partial** | Text summary of the whole cutout, not the target. No overlay |
| Source detection | **Missing** | No photutils, no catalogue |
| Temporal tracking / PM fit | **Missing** | |
| Catalogue cross-match (Gaia/WISE) | **Missing** | |
| Candidate ranking / explanation | **Missing** | |
| Citizen-science classification | **Missing** | |
| Export | **Partial** | CLI writes `frames.npz`. No export from the web UI |
| Aladin Lite / survey overlays | **Missing** | |

## 4. Critical issues

**C1 — The default "epoch" grouping co-adds different detectors, so frames mix wavelengths.**
`pipeline.group_indices(mode="epoch")` and `stream.py`. The UI defaults to D2+D6 and
inverse-variance-averages 1.1–1.6 µm and 4.4–5.0 µm images into one frame. A star's surface
brightness differs by large factors between these bands, and the inverse-variance weights
differ per detector. So a frame's brightness depends on the mix of detectors and LVF
positions that happened to be observed in that epoch. Differences between epochs and the
light curve can then show **spurious variability**, and the detectors get different weights
from one epoch to the next. This directly contradicts "do not blindly add data from different
detector arrays."
*Fix:* default to grouping by detector and epoch, and blink one detector at a time. Keep
cross-detector stacking only as an explicitly labelled display aid.

**C2 — `/api/spectrum` returns HTTP 500 whenever any exposure has a NaN pixel** *(verified)*.
`spectrum.py` puts `float(NaN)` into the response. Starlette's `JSONResponse` uses
`allow_nan=False` and raises `ValueError`. Saturated targets (R Dra, whose core is NaN per
the team's own finding) and edge pixels trigger it. *Fix:* convert NaN to null and set a flag.

**C3 — The pixel readout and click-to-spectrum use the J2000 position, not the position the
grid is centred on** *(verified numerically)*.
`app.js:pixelAtEvent` measures offsets from `state.resolvedCoord`, the SIMBAD J2000 position.
The grid, however, is centred on the position propagated to the mean observation epoch.
For Barnard's Star these differ by **272″ (4.5′)**, for WISE 0855 by 213″ and for ε Ind B
by 122″. Clicking the star therefore asks for a spectrum several arcminutes away. On top of
that, the pixel scale is derived as `fov*60/width` instead of the real grid scale, which is
~1.5 % off, and the centre is off by half a pixel. *Fix:* send the grid WCS with each frame
and invert the gnomonic projection properly on the client.

**C4 — The light curve reads one fixed pixel, and for saturated sources that pixel is
fabricated.**
`app.js:centrePixelFlux` and `pipeline.fill_small_gaps`.
(a) The grid is fixed at the mean-epoch position, so a high-PM target moves off the centre
pixel. Barnard's Star moves ~10″/yr, about 1.7 native pixels per year. The "light curve" then
records the star leaving the pixel, not a change in brightness.
(b) `fill_small_gaps` replaces NaN holes of up to 6 grid pixels with the mean of their
neighbours. One saturated native pixel covers ~4 grid pixels, so a saturated core is filled
with the value of its fainter **wings**. That invented value is what the light curve plots.
(c) The quantity is surface brightness in a single pixel, which depends on sub-pixel phase
because the PSF is undersampled (header `PSF_FWHM` ≈ 9″ against 6.15″ pixels).
*Fix:* predict the target pixel per frame from its proper motion. Measure with background-
subtracted aperture photometry with errors, and refuse to measure (and say why) when any
aperture pixel is masked or interpolated.

**C5 — The viewer shows a mirror image of the sky, and the pixel readout has Dec with the
wrong sign** *(verified)*.
`make_target_wcs` sets `cdelt2 > 0`, so array row 0 is the **southern** edge (checked: row 0
is at Dec₀ − 2.5′ for a 5′ field). `app.js:renderGrid` draws row 0 at the top. The result is
south up and east left, which is a parity-flipped sky that cannot be compared directly with
Aladin, WiseView or finder charts. `pixelAtEvent` assumes north is up, so every RA/Dec it
reports for an off-centre pixel has its Dec offset mirrored. Clicking a source 1′ north of
the target requests a spectrum 1′ **south** of it.
*Fix:* draw array row `h−1−y` at the top (north up, east left), and do all readouts through
the frame's WCS.

## 5. High-severity issues

**H1 — Typed coordinates have no epoch, and the NASA WISE 0855 coordinates are out of date.**
`coords.parse_position`. With PM −8.1″/yr, WISE 0855 is ~128″ from
`08h55m10.831s −07d14m42.53s` in 2026. A 3′ field (radius 1.5′) around the published
coordinates barely contains it, and its position at each epoch is unknown to the app.
Name resolution works (*verified*). *Fix (this pass):* note it in the UI and README and
recommend the name. *Later:* an epoch-aware Gaia/CatWISE lookup in a cone around typed
coordinates.

**H2 — Nearest-neighbour resampling makes difference images unreliable for science.**
`pipeline.reproject_nearest`. Each exposure lands on the common grid with a random error of up
to ±0.5 native pixels (±3″). That makes every bright stationary star leave dipole-like
residuals in B−A, which look like the signature of a moving source. The PSF also differs between exposures
(each exposure puts the source at a different detector position and wavelength), and nothing
matches it. The reason nearest-neighbour was adopted is real (bilinear interpolation spreads
NaN), but it is better solved by interpolating with normalised weights. *Fix (Phase 3):*
flux-conserving or bilinear resampling with weight renormalisation, a significance map
`(B−A)/sqrt(varA+varB)`, and an explicit "residuals of stationary stars" check.

**H3 — Spectrum quality.** `spectrum.py`
- It is a single pixel of surface brightness, labelled "flux", with no uncertainty even though
  VARIANCE is downloaded.
- It mixes all epochs into one F(λ). This is wrong for RW Aur and R Dra unless the points are
  separated by time. The plot colours points by time but draws no conclusion.
- A clicked position has no PM. For movers the spectrum then samples empty sky at most epochs.
- It downloads **every** exposure without a cap: about 190 exposures × 4 MB ≈ 760 MB per click
  at Barnard's Star.

*Fix:* aperture photometry with errors, PM propagation when the click is on the target, a
cap on the number of exposures, and output that carries epoch so that F(λ, t) can be built
honestly.

**H4 — Epoch clustering spans ~90 days near the ecliptic poles** *(verified for R Dra:
continuous coverage blocks of 67–90 d)*. R Dra's period is ~245 d, so an "epoch" coadd smears
a large fraction of its cycle. *Fix:* for variability, use single exposures or short bins
(user-selectable bin width), not survey passes.

## 6. Medium / Low issues

| ID | Where | Problem | Severity | Fix |
|---|---|---|---|---|
| M1 | `stream.py` → `json.dumps` | NaN in metadata (e.g. wavelength) gives invalid JSON. The frontend `JSON.parse` silently drops the frame | Medium | Sanitise NaN to null |
| M2 | `app.js` status text | Says "proper motion known, propagated per epoch". The frames are **not** propagated per epoch; the grid is centred on one mean-epoch position | Medium | Correct the wording, show the predicted track |
| M3 | `pipeline.flag_contamination_summary` | Counts flags across the whole native cutout, not at the target. It also doesn't say whether the target itself is flagged | Medium | Add flags at the target aperture |
| M4 | `main.py` | No caching of name resolution (SIMBAD is hit on every call). No per-client rate limit; 30′ × 6 detectors can trigger hundreds of 4 MB IRSA downloads | Medium | Cache name resolution with `lru_cache`, cap the exposure count |
| M5 | `app.js:recomputeGlobalFiniteValues` | Sorts every pixel of every frame on each arriving frame and on every render | Low | Subsample, compute once |
| M6 | Frontend "Sum" | It is a mean, not a sum | Low | Rename |
| M7 | Variance after coadd | Not sent to the client, so no significance can be computed in the browser | Medium | Send variance (or a σ map) |
| M8 | Tests | No tests for stream/encode/spectrum/main, the NaN path or the frontend. Live tests are only opt-in | Medium | Add offline tests with labelled synthetic FITS fixtures |
| M9 | Units | `pixelReadout` shows MJy/sr, correct. The spectrum and light curve call it "flux", which is wrong | Low | Label units correctly |
| L1 | `coords._SEXAGESIMAL_RE` | `hh:mm:ss ±dd:mm:ss` and space-separated forms go to Sesame by accident | Low | Extend the parser |
| L2 | `.idea/`, `__pycache__`, `.pytest_cache` in the tree | Clutter | Low | `.gitignore` |
| L3 | CORS `*` + GET only | Acceptable for public read-only data | Info | — |
| L4 | SIMBAD ADQL | Escapes literals correctly. No injection found | Info | — |

## 7. What is genuinely good (and should be kept)

- Flag bit names are read from each file's `MSKN*` header (*verified*: the bits differ from
  the brief's table).
- Zodi subtraction, use of VARIANCE in the coadd, and an odd-sized grid with an exact centre pixel.
- Runtime collection discovery. The DR1/QR3 split is handled without duplicates (*verified*).
- Honest documentation of failures (WISE 0855 non-detection, R Dra saturation).
- Progressive streaming, retry with backoff, a global fetch semaphore and the timing report.

## 8. Stabilisation done in this pass (Phase 2)

See the git history for details. Summary:
1. C1: detector-separated grouping by default. The frontend blinks one detector at a time,
   and the cross-detector stack is relabelled as display-only.
2. C2/M1: NaN-safe JSON everywhere.
3. C3: each frame carries its grid WCS. The client inverts TAN exactly.
4. C4: a new `photometry.py` does background-subtracted aperture photometry with
   uncertainties at the target position predicted for each frame. It returns null with a reason
   when the aperture is masked or interpolated. The light curve uses it.
5. H3 (part): the spectrum uses the same aperture photometry, with uncertainties, NaN safety,
   per-exposure PM propagation when you click the target, and a cap on exposures.
6. L1: colon-separated sexagesimal parsing. M2/M6/M9: wording and units.
7. Tests for all of the above, using clearly labelled synthetic fixtures.

### Live verification after the fixes (2026-11-15, Barnard's Star, D6, 3′, Low quality)

- `/api/frames` streamed 3 D6 frames: 2025-08-15…27, 2026-03-19…05-05 and 2026-08-16…09-22.
  The crosshair sits on the star in every frame and moves **north** between epochs, as
  expected for pmDec = +10.4″/yr. This confirms both the north-up display and the per-frame
  PM tracking.
- Aperture fluxes (r = 9″): **2406 ± 3.8, 2189 ± 3.4, 2981 ± 5.5 mJy** at λ ≈ 4.72, 4.72 and
  4.54 µm. The two frames at the same wavelength differ by **10 %**, about 60× their
  statistical errors. Barnard's Star is not known to vary like that, so the difference
  measures the **systematic floor** of this photometry (pixel phase, undersampled PSF,
  nearest-neighbour resampling). The third frame also sits at a different wavelength, so its
  difference may be spectral rather than temporal. **Conclusion: as things stand, this light
  curve cannot claim variability below roughly 10–20 % in SPHEREx.** The UI now says so
  explicitly. Order-of-magnitude check: WISE W2 ≈ 4 mag corresponds to ~4 Jy in total, so
  ~2.2–3 Jy inside a 9″ aperture is plausible (no aperture correction is applied).
- `/api/spectrum`, clicking the star: the request used proper-motion tracking. It returned
  58 exposures: 19 measured, 35 refused (masked or saturated aperture), 4 off-image. Before
  the fix any NaN pixel produced HTTP 500.
- The client-side TAN inverse agrees with astropy to about 1e-12° (including at Dec 89.9°).

Not re-run in this pass: the other four targets, and the live pytest markers.

## 9. Roadmap adapted to this codebase

**Phase 3a (foundation for discovery):** `detect.py` (photutils `DAOStarFinder`/segmentation
on each detector-epoch coadd, with significance from VARIANCE); a cross-epoch association
module (KD-tree with PM-aware gating); a weighted least-squares PM fit with errors; and
validation on Barnard's Star (expected 10.4″/yr) and ε Ind B.
**Phase 3b:** resampling that preserves weights, and difference significance maps (H2).
**Phase 3c:** Gaia DR3 cross-match through IRSA/ESA TAP, with PM propagated to each epoch and
an uncertainty-aware match radius. CatWISE2020 for cold brown dwarfs that Gaia misses
(WISE 0855).
**Phase 3d:** an explainable candidate score with separate fields for detection S/N, PM
significance, variability χ² and catalogue-match status. These are never merged into a
"discovery" label.
**Phase 3e:** F(λ, t) built from single exposures binned by epoch, overlaid with IRSA's
Spectrophotometry Tool output as a cross-check.
**Phase 4:** a validation suite over the five targets, recorded as regression fixtures (cached
real cutouts).
**Later:** a citizen-science classification store (needs persistence, which breaks the
"stateless" design and is a deliberate architecture decision for the team), and Aladin Lite.
