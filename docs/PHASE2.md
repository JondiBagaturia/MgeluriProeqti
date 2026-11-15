# Phase Two report — SPHEREx Discovery Lab

Date: 2026-11-15. Earlier records: [AUDIT.md](AUDIT.md) (the original problems and their
fixes) and [DISCOVERY.md](DISCOVERY.md) (the first discovery pipeline). This report
supersedes DISCOVERY.md where they differ.

Verification levels used throughout:

- **implemented**: code exists and is wired into the app;
- **offline-tested**: covered by an automated test on synthetic data;
- **real-data-tested**: covered by an automated test on the cached real SPHEREx cutouts;
- **live-tested**: exercised against the live IRSA / SIMBAD / catalogue services;
- **not verified**: none of the above.

## A. Objectives

Turn the blink viewer into a reproducible discovery workflow, validated on the five NASA
targets:

- detect and track sources across epochs, and measure their motion and brightness with
  honest uncertainties;
- compare them with Gaia DR3 and CatWISE2020 and rank them with stated reasons;
- let volunteers classify the ranked sources;
- replace nearest-neighbour resampling with a method chosen by measurement;
- add a search for faint movers (shift-and-stack) and a parallax-aware fit;
- validate everything on real cached data.

## B. Implemented functionality

| Component | Where | Status |
|---|---|---|
| Source detection (photutils segmentation + deblending, 5σ) | `app/discovery.py` | implemented, offline-tested, real-data-tested |
| Cross-epoch association (nearest-first, motion-aware gate) | `app/discovery.py` | implemented, offline-tested, real-data-tested |
| Linear proper-motion fit (weighted least squares, χ²-rescaled errors) | `app/discovery.py` | implemented, offline-tested, real-data-tested |
| Gaia registration of each frame (median offset of isolated slow Gaia stars, applied before linking) | `app/discovery.py` | implemented, offline-tested, live-tested (final validation run, all five fields) |
| Variability (χ² of constant flux + systematic floor, wavelength-mismatch flag) | `app/discovery.py` | implemented, offline-tested, real-data-tested (RW Aur) |
| Gaia DR3 / CatWISE2020 cross-match, propagated to the observation epoch | `app/catalogs.py`, `app/discovery.py` | implemented, offline-tested, live-tested (final run: Gaia for all five fields, CatWISE for three; IRSA returned 502 for the other two) |
| Typed-coordinate epoch lookup (CatWISE track matching) | `app/catalogs.py`, `app/coords.py` | implemented, offline-tested, live-tested |
| Explainable ranking (anomaly score, reliability, reasons and caveats) | `app/discovery.py` | implemented, offline-tested, real-data-tested |
| Candidate store and volunteer classification (SQLite) | `app/store.py`, `/api/candidates*` | implemented, offline-tested; checked in the browser (vote saved, survives reload) |
| Lanczos-3 resampling with variance propagation | `app/resample.py`, `app/pipeline.py` | implemented, offline-tested, benchmarked (synthetic), real-data-tested |
| Shift-and-stack search with time-shuffled controls | `app/shiftstack.py`, `app/stream.py` (`shiftstack=true`) | implemented, offline-tested, real-data-tested (WISE 0855), live-tested once in the browser |
| Parallax-aware 5-parameter fit (reported, never adopted unless constrained) | `app/astrometry.py`, `app/discovery.py` | implemented, offline-tested, real-data-tested (always unconstrained on current data) |
| Frontend: tracked-sources table, overlays, details, classification, JSON/CSV export | `frontend/` | implemented, checked in the browser |
| Frontend: shift-and-stack panel and S/N-map view, parallax block | `frontend/` | implemented, checked in the browser (cached-data server and one live run) |
| Offline runs on cached real cutouts | `app/offline.py`, `backend/tests/data/real/` | implemented, used by the regression tests |

Not implemented in Phase Two: spectro-temporal analysis F(λ, t) beyond the existing
per-exposure spectrum endpoint, brightness-normalised anomaly scores, and the false-positive
rate study for the ranking (see section H).

## C. Scientific methodology

### Detection significance vs anomaly score
These are separate fields in every record and are never combined:

- **Detection S/N** is the segment's peak divided by the frame's robust background scatter.
- **Anomaly score** (0–1) = 0.45 × motion + 0.30 × variability + 0.25 × missing catalogue
  counterpart. Each indicator is clipped to 0–1. It is a ranking device, **not a probability**.
- **Reliability** (high / medium / low) counts problems with the measurement itself: number
  of epochs, masked or saturated cores, image edges, ambiguous links, faintness, and
  proximity to a much brighter star.
- **Class probability** is not estimated; the field is always null.

Labels: `catalogued` (uniquely matched), `candidate` (unmatched or unusual, measurement
trustworthy), `uncertain` (reliability low), `unverified` (a catalogue could not be
queried). No label means "new object".

### Positional uncertainty
Per detection: the centroid's statistical error, FWHM/(2.355 × S/N), combined with a floor of
**0.4″** (3″ if the core is masked). The floor was measured on real data: 34 field stars with
S/N ≥ 30 in the five fields, observed at 3–4 epochs, repeat to 0.17–0.23″ (median) and
0.31–0.39″ (84th percentile). It was 1.0″ before Lanczos resampling.

### Flux uncertainty and the variability limit
Aperture photometry (r = 9″, sky from an annulus 18–30″) with propagated variance. Resampled
pixels are correlated, so the aperture variance is multiplied by **5.0** for Lanczos-3. That
factor was measured on synthetic data and checked on real data: for field stars with
10 ≤ S/N < 30 the measured scatter equals the propagated error (D3 4.4 % vs 4.5 %,
D5 9.3 % vs 9.3 %, D6 4.9 % vs 5.1 %).

Bright stars scatter more than their statistical error, so the variability test adds a
**6 %** systematic floor (84th percentile of bright field stars, 5.4–7.1 %; it was 10 %).
This floor includes the wavelength difference between frames of one detector. When those
frames differ by more than 0.05 µm, the result is flagged (`wavelength_mismatch`) and its
weight in the score is cut to 30 %. **Variability below roughly 6–10 % cannot be claimed.**

### Association
Linking runs frame by frame, nearest first, inside a gate of 4σ of the combined positional
errors plus room for motion: up to 12″/yr while the motion is unknown, a quarter of that once
a velocity is measured. A track with more than one detection in its gate is flagged
`ambiguous_link`. Frames are registered to Gaia **before** linking; a shifted frame would
otherwise break the links.

### Catalogue matching
Gaia DR3 (epoch 2016.0) and CatWISE2020 (MJD 57170) positions are propagated to the track's
mean epoch with their proper motions, and their errors grow with the propagation time. A
source matches if it lies within 3.5σ of the combined uncertainty. More than one source in
the gate gives `ambiguous`; more than one inside one PSF width (9″) gives a blend caveat.
A failed query gives `catalog_unavailable`, which is never treated as "unmatched".

### Lanczos-3 resampling and its ringing
`scripts/benchmark_resampling.py` (synthetic; 150 trials; seeded and reproducible) puts
dithered, rotated exposures of a 9″-FWHM star on 6.15″ pixels onto the 3.075″ grid. Results
for 5 exposures per epoch:

| Method | centroid error, moment / fit | aperture scatter | difference-image residual (median, of peak) | aperture variance factor |
|---|---|---|---|---|
| nearest (old) | 0.21″ / 0.23″ | 2.5 % | 30 % | 3.8 |
| bilinear | 0.05″ / 0.04″ | 2.0 % | 9 % | 7.3 |
| **lanczos3 (adopted)** | 0.06″ / 0.05″ | **1.4 %** | **4 %** | 5.0 |
| reproject adaptive | 0.06″ / 0.05″ | 1.8 % | 13 % | 6.1 |
| reproject exact | 0.07″ / 0.06″ | 1.7 % | 17 % | 5.1 |

Lanczos-3 was chosen for the lowest photometric scatter and the smallest false residuals in
difference images. Flux is preserved to about 1 % in the tests (`test_resample.py`); it is
**not** exactly flux-conserving.

**Ringing.** On an undersampled PSF, Lanczos-3 leaves a trough of −3.5 % of the peak at
12–18″ and a positive lobe of +1.3–2.8 % at 18–24″. On real data this produced a 3-epoch
"source" at 2 % of a field star's peak, 19″ from it (RW Aur field, D3), which does not appear
with bilinear resampling. Such detections are kept but flagged `near_bright_star` (within
12–32″ of a star at least 25× brighter), get low reliability, and are never ranked as
candidates. This is covered by a regression test on real data.

### Shift-and-stack
For each trial motion, every exposure's matched-filter map is shifted so that a source with
that motion stays put, and the maps are summed with inverse-variance weights. Each velocity
plane is normalised by its own robust noise, which absorbs resampling-correlated noise.

- **Search grid:** a coarse grid (step = one third of the PSF over the baseline, |v| ≤ 12″/yr;
  49 trial motions over one year), then a five-times finer local grid around every peak.
- **"Moving":** a peak counts as moving only if (1) S/N² beats the best zero-motion S/N²
  nearby by at least 11.8, (2) S/N is at least 1.05 × the zero-motion S/N, and (3) the motion
  is at least half a PSF width over the baseline (about 4.4″/yr for one year).
- **Controls:** the same search repeated with the observation dates shuffled among the
  exposures (20 in the app, 30 in the validation script). The empirical false-alarm
  probability is the fraction of controls with a moving peak at least as strong as the real
  one, (1 + k)/(n + 1). With no control as strong, it can only be stated as **≤ 1/(n + 1)**:
  0.048 for 20 controls and 0.032 for 30, **never zero**.
- **Status words:** `passes_controls`, `tentative`, `no_moving_candidate`. Even
  `passes_controls` is described as tentative: shift-and-stack is a search, not a confirmation.
- **Known stars:** a forced stack along a known trajectory is compared with the same position
  at zero and reversed motion, with the same motion at 200 other positions, and with rotated
  directions.

Fixed during close-out:

- Refinement had been skipped for peaks whose coarse velocity was zero. This hid WISE 0855
  in D5, where the coarse grid narrowly preferred zero motion.
- The returned list was cut to the 10 highest-S/N peaks, so bright static stars pushed out
  a faint mover.

Both are covered by tests now.

### Parallax
`fit_motion_parallax` fits position, proper motion and parallax (5 parameters) using
parallax factors from the Earth's barycentric position (astropy). SPHEREx's own orbit adds
about 5×10⁻⁵ au, which is negligible. The parallax is believed only if all of the following
hold:

- at least 4 epochs;
- at least 20 % of the parallax signal is distinguishable from linear motion;
- parallax ≥ 3σ and positive.

Otherwise the linear fit stays the reported result, and the parallax is shown as
**UNCONSTRAINED**, with the reason.

## D. Validation results (five targets, cached real cutouts)

Data: 181 real SPHEREx cutouts (5′ field, D3 or D5 plus D6, at most 6 exposures per
detector-epoch), cached on 2026-11-15 in `backend/tests/data/real/`. With one
`manifest.json` per target, the fixture directory holds **186 files**: 181 `.fits.gz`
cutouts + 5 manifests. Per target: Barnard 36, ε Ind B 47 (one of 48 requested failed to
download), WISE 0855 33, RW Aur 36, R Dra 29 (7 of 36 failed).

Results below are from the **final** run with the final code
(`docs/data/validation_report.json`, `docs/data/validation_log_2026-11-15_final.txt`), with
live catalogues. Gaia DR3 answered for all five fields. CatWISE2020 answered for ε Ind B,
WISE 0855 and RW Aur; for Barnard's Star and R Dra, IRSA's cone search returned HTTP 502,
which is reported as `catalog_unavailable`. An earlier run that day
(`validation_log_2026-11-15_live_catalogues.txt`) gave identical motions and photometry.
"Offline" numbers are from the regression tests, which run without catalogues and therefore
without Gaia registration.

| Target | Pipeline | Detection / photometry | Proper motion (mas/yr) vs reference | Catalogue | Variability | Shift-and-stack | Main limitation |
|---|---|---|---|---|---|---|---|
| **Barnard's Star** | completed | D6 detected (3340, 3141, 3487 mJy at 4.70/4.69/4.63 µm). D3 saturated in all 3 epochs: **refused**, no flux; its D3 track is `uncertain`, reliability low | Gaia-registered: **(−1061, +10358) ± 959**, 10.9σ. Offline (no registration): (−1205, +10428) ± 1202. Gaia (−802, +10362): east within 0.3σ, north agrees | Gaia matched; CatWISE unavailable (502) | none (p = 0.68) | n/a | Linear χ² = 9.3 on 2 dof, explained by the unmodelled 0.547″ parallax: removing Gaia's parallax gives χ² = 1.8 and error 558 (regression test). Parallax fit 751 ± 286 mas: **unconstrained** (3 epochs) |
| **ε Indi B** | completed | D3 16.3–17.5 mJy; D6 19.5 → 17.3 mJy (4 epochs) | D6 **(+4009, −2703) ± 425** (11.4σ), D3 (+3635, −2168) ± 455. Offline: D6 (+3654, −2743) ± 379, D3 (+3511, −2344) ± 391. Gaia (+3982, −2467): every component within 1.2σ | Gaia matched; CatWISE unmatched | none (D6 p = 0.32, D3 p = 0.83); the D6 frames span 0.14 µm | n/a | 4 epochs; parallax fit D6 448 ± 229 mas (true 271): **unconstrained** (< 3σ) |
| **WISE 0855** | completed | Blind detections in all 3 D5 epochs (0.42, 0.37, 0.47 mJy, S/N ≈ 10–13) and 2 D6 epochs | D5 track: **(−6275, +895) ± 711** (8.9σ); offline (−6130, +894) ± 711. CatWISE/SIMBAD (−8124, +673): east is **2.6σ low** | Gaia unmatched (as expected for a Y dwarf); CatWISE **ambiguous** (CatWISE lists this object in 3 rows) | none (p = 0.33) | see below | Faint (S/N ≈ 10 per epoch), 3 epochs, CatWISE duplicates |
| **RW Aurigae** | completed | D6 **2489 → 1808 → 1282 mJy** at 4.751/4.746/4.742 µm (same wavelength) | 0.9σ from zero (not moving), as expected (7, −20) | CatWISE matched; Gaia ambiguous (close binary) | **D6 p = 8.6×10⁻¹³** with the 6 % floor (4.5×10⁻⁵ under the old 10 % floor). A real 49 % decline at fixed wavelength. D3 p = 0.011 but flagged `wavelength_mismatch` (2.06 vs 1.93 µm), so not claimed | n/a | 3 epochs; no period |
| **R Draconis** | completed | **Saturated in all 6 frames (D3 and D6): photometry refused, no flux reported** | Not meaningful: ± 4400 mas/yr from masked cores | — | **none claimed** (no flux) | n/a | Saturated; reliable photometry impossible with these data. Its own track is labelled `uncertain`, reliability low (`core_masked_all_epochs`) |

### WISE 0855 shift-and-stack (real data, 30 controls)

| Data | Exposures | Blind moving candidate | S/N (zero motion) | Controls | Status |
|---|---|---|---|---|---|
| D5 | 18 | (−6.5, 0.0)″/yr at the predicted position | 12.1 (11.2) | 1 of 30 as strong, p = 0.065 | tentative |
| D6 | 15 | (−8.9, 0.0)″/yr at the predicted position | 11.6 (10.1) | none of 30, p ≤ 0.032 | passes controls |
| D5 + D6 | 33 | (−7.0, 0.0)″/yr at the predicted position | 13.4 (12.2) | none of 30, p ≤ 0.032 | passes controls |

Forced stack along the catalogue motion (D5 + D6):

- S/N 13.3, against 12.2 with zero motion and 9.4 with reversed motion.
- At the catalogue positions 11 and 5 years earlier, where the object no longer is,
  S/N 0.3–0.5.
- The same motion reaches that S/N at 4.5 % of 200 other positions, all of them real stars.

The live browser run (D6, Low quality, 13 exposures) gave (−8.28, 0.00)″/yr and passed all
20 controls.

**Comparison with CatWISE (−8.12, +0.67)″/yr, and why they don't agree perfectly.** The
diagnostic `scripts/diagnose_wise0855_velocity.py` maps S/N on a 0.25″/yr grid:

- The peak is at **(−7.75, −0.25)″/yr**.
- **East:** the 1σ range is −9.5 to −7.5″/yr, consistent with the catalogue.
- **North:** the statistic constrains it tightly near −0.25, while the catalogue says +0.67.
  The catalogue motion is disfavoured by Δ(S/N²) = 5.1, about 2σ for two parameters.
- The blind velocity also moves by about 1″/yr between equivalent refinement schemes:
  (−8.2, −0.6) this morning, (−7.0, 0.0) after the fix, and (−7.75, −0.25) on the fine map.
  That spread is within the east width.

So the error the search quotes (±0.6″/yr, the refinement step) **understates the true
uncertainty**. Systematics not in the statistic are:

- candidate positions are whole 3.1″ grid pixels;
- exposures are not registered against Gaia inside shift-and-stack;
- one Gaussian PSF is used for every exposure.

**Conclusion:** shift-and-stack recovers WISE 0855 at the right place, with motion correct
to roughly 1–1.5″/yr, which is far from zero and distinguishable from the shuffled controls.
This is a recovery of a known object, not an independent measurement of its motion.

**How the interface presents this.** The shift-and-stack panel shows each candidate's
velocity as approximate (≈, one decimal). In a separate "Velocity uncertainty" box it shows:

- the **formal** uncertainty, labelled "statistical only … NOT the full measurement error";
- the **estimated total** of about 1–1.5″/yr per component, stated as an estimate from this
  one validated object, not a per-source calculation;
- the systematics the formal error leaves out;
- a note that the tracked-source fit is the better motion measurement when the source is
  detected in each epoch.

The S/N-map caption repeats the ~1–1.5″/yr caveat. The JSON export keeps the raw fields
`pm_east_err_arcsec_yr` / `pm_north_err_arcsec_yr`. These are the **formal** widths only
and must be read with the same caveat. No per-source total error is computed; that needs
the Phase Three work (per-exposure registration, PSF model).

### Field-star repeatability (real data)

| Detector, S/N | n | flux scatter (median / p84) | propagated error (median) | position rms (median / p84) |
|---|---|---|---|---|
| D3, ≥ 30 | 22 | 5.3 % / 7.1 % | 0.9 % | 0.23″ / 0.39″ |
| D3, 10–30 | 15 | 4.4 % / 8.7 % | 4.5 % | 0.39″ / 0.50″ |
| D5, ≥ 30 | 3 | 4.0 % / 5.7 % | 1.5 % | 0.16″ / 0.19″ |
| D5, 10–30 | 7 | 9.3 % / 11.6 % | 9.3 % | 0.46″ / 0.72″ |
| D6, ≥ 30 | 9 | 2.8 % / 5.4 % | 1.6 % | 0.17″ / 0.31″ |
| D6, 10–30 | 7 | 4.9 % / 9.7 % | 5.1 % | 0.35″ / 0.45″ |

## E. Test results (final run, 2026-11-15)

| Suite | Result |
|---|---|
| Offline suite (`pytest`), project venv, Python 3.12.7 | **107 passed, 0 failed** (6 live tests deselected) |
| Same suite in a brand-new venv built only from `requirements-dev.txt` (Python 3.12.7, photutils 3.1.1, astropy 8.0.2, numpy 2.5.4, scipy 1.18.1; `reproject` absent), run on the final code | **107 passed, 0 failed** (6 live deselected) |
| Of which real-data regression tests (`test_real_targets.py`, cached cutouts, no network) | **10 passed, 0 failed** |
| Live tests (`pytest -m live`: SIMBAD name resolution ×5, IRSA pipeline) | **6 passed, 0 failed** (IRSA collection discovery returned 502; the configured fallback list was used) |

Tests that need external services: only those marked `live` (deselected by default). The
validation script `scripts/validate_targets.py` also queries Gaia and CatWISE (disable with
`--no-catalogs`).

Test limitations:

- All real-data tests use 3–4 epochs of 5 targets.
- The variability tests cover one real variable (RW Aur).
- Catalogue matching is tested automatically only on synthetic catalogues; against the live
  services it was exercised by the validation runs, not by a test.
- Shift-and-stack false-alarm rates are known only to ≤ 1/(n + 1).

## F. Known limitations

- **Few epochs:** 3–4 per target over about one year. Motions are good to roughly 0.4–1″/yr,
  and parallax cannot be constrained for any target yet.
- **Undersampled PSF:** 9″ FWHM on 6.15″ pixels. Lanczos ringing at 2–3 % of a star's peak
  is flagged but not removed.
- **Deblending:** a neighbour with a few percent of a bright star's flux that falls inside
  its 5σ isophote is merged and lost (contrast 0.05). This was seen at 20–28″ in synthetic
  tests at very high S/N.
- **Photometric systematics:** 3–7 % between frames for bright stars, plus wavelength
  differences between frames of one detector. Variability below about 6–10 % cannot be
  claimed.
- **Faint sources:** blind detection needs about 5σ per frame-coadd. Shift-and-stack helps
  only moderately with a 9″ PSF and a one-year baseline (WISE 0855: S/N 12.2 → 13.4).
- **Crowded fields:** links can be wrong; they are flagged, not corrected.
- **Saturation:** Barnard's Star in D3 and R Draconis in D3/D6 cannot be measured; no flux
  is reported, and no wing-based photometry is attempted.
- **Catalogues:** CatWISE duplicates fast movers (WISE 0855: 3 rows → `ambiguous`), and IRSA
  catalogue access was intermittent on 2026-11-15 (HTTP 502 for hours; partly during the
  final run).
- **Shift-and-stack:** limited control samples (20–30); velocity errors are understated (see
  above); no per-exposure registration; a single Gaussian PSF.
- **No brightness-normalised anomaly scores** and no false-positive rate study for the ranking.

## G. Reproduction

```bash
cd MgeluriProeqti
python -m venv .venv                  # tested with Python 3.12.7; the Docker image uses 3.11 (not re-verified)
.venv/Scripts/pip install -r backend/requirements-dev.txt     # Windows; on Linux/macOS: .venv/bin/pip
```

Run the app (backend and frontend are one process):

```bash
cd backend && ../.venv/Scripts/python -m uvicorn app.main:app --port 8000
```

Then open http://localhost:8000/ (or use `docker compose up --build`).

Run the offline tests (includes the cached real-data regression tests; no network needed):

```bash
cd backend && ../.venv/Scripts/python -m pytest
```

Run the optional live tests (SIMBAD and IRSA):

```bash
cd backend && ../.venv/Scripts/python -m pytest -m live
```

Reproduce the five-target validation (writes `docs/data/validation_report.json`; add
`--no-catalogs` to skip Gaia/CatWISE):

```bash
.venv/Scripts/python scripts/validate_targets.py
```

Reproduce the WISE 0855 velocity diagnostic:

```bash
.venv/Scripts/python scripts/diagnose_wise0855_velocity.py
```

Reproduce the resampling benchmark (needs `reproject`):

```bash
.venv/Scripts/pip install -r backend/requirements-benchmark.txt
```

```bash
.venv/Scripts/python scripts/benchmark_resampling.py --trials 150 --nexp 1 5 --json docs/data/resampling_benchmark.json
```

Refresh the cached cutouts (network; the expected values in the tests may then need review):

```bash
.venv/Scripts/python scripts/cache_validation_cutouts.py
```

## H. Completion status

**Complete:**

- every component in section B;
- the five-target validation on real data;
- the regression tests;
- frontend access to shift-and-stack and parallax;
- the documentation.

**Incomplete within Phase Two's original scope** (deferred, see below):

- spectro-temporal analysis F(λ, t);
- brightness- and coverage-normalised anomaly scores;
- a false-positive rate study for the ranking;
- comparison with IRSA's Spectrophotometry Tool.

**Not verified:**

- CatWISE matching for Barnard's Star and R Dra in the final run (IRSA returned 502 for those
  two queries).
- The Docker image with Python 3.11.
- The appearance of the new panels was checked by screenshots on the cached-data dev
  server; the live run was checked through page content only.

**Deferred to Phase Three:**

- spectro-temporal F(λ, t) pipeline, band-dependent light curves, spectral anomaly scores;
- anomaly scores normalised against sources of similar brightness, coverage and quality,
  with a synthetic-injection false-positive study;
- per-exposure Gaia registration and a PSF-aware fit inside shift-and-stack (to fix the
  understated velocity errors);
- parallax once SPHEREx provides at least 4–6 epochs per field;
- more control samples, or an analytic false-alarm calibration, for shift-and-stack.
