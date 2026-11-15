# Calibrated anomaly ranking (Phase Three, Priority 2)

Status: implemented and validated by synthetic injection. **Not committed yet; the numbers
below are waiting for review.**

Code: `backend/app/ranking.py` (calibration), `backend/app/injection.py` (synthetic sources),
`scripts/validate_ranking.py` (injection study), `scripts/cache_catalogs.py` (real catalogue
rows for offline runs), `backend/tests/test_ranking_calibration.py` (19 tests).
Data: `docs/data/ranking_validation.json` (every injected case), `docs/data/ranking_validation_log.txt`.

## 1. What changed, and what did not

The Phase Two outputs are unchanged: `anomaly_score`, `score_components` (raw M, V, C),
`reliability`, `rank`, labels and the SQLite schema. All 137 existing tests pass unmodified.
Each track also gets:

| field | meaning |
|---|---|
| `calibrated_score` | `min(0.45 M_norm + 0.30 V_norm + 0.25 C_norm, quality cap)`. Same weights and the same significance-to-[0,1] mapping as the raw score; only the inputs are calibrated. Not a probability. |
| `calibrated_rank` | rank within the series by `calibrated_score` |
| `calibration.M_norm`, `V_norm`, `C_norm` | calibrated components |
| `calibration.completeness_regime` | the least favourable per-catalogue regime (see §2.2) |
| `calibration.completeness` | per catalogue: status, regime, weight, local completeness flux, source flux |
| `calibration.motion_calibration`, `variability_calibration` | statistic, peer level, number of peers, inflation λ, analytic and calibrated p, empirical rarity bound |
| `calibration.quality_cap`, `uncapped_calibrated_score` | the cap applied, and the score before it |
| `calibration.explanation`, `suppression_reasons` | plain-language reasons, shown in the UI |

## 2. Method

### 2.1 Motion and variability: peer calibration

- **Statistics.** Under the Phase Two error model the motion statistic x_M = (PM significance)² is χ² with 2 dof, and the variability statistic x_V = χ²-isf(p_var, 1) is χ² with 1 dof.
- **Peers.** Clean tracks in the same series: reliability high, at least 2 epochs, and no near_bright_star, core_masked, ambiguous_link or near_edge flag. A track is never its own peer.
- **Binning.** Peers are grouped by detection S/N (0–10, 10–30, 30–100, 100+) and by epoch coverage (2, 3, 4+ epochs).
- **Fallback.** With fewer than 5 peers in a bin, the calibration falls back to the S/N bin over all coverages, then to all peers of the series, then to the analytic error model (λ = 1). The level used is reported.
- **Inflation estimate.**
  - Start from λ₀ = max(1, Q25(x) / χ²₀.₂₅).
  - Iterate:
    - keep the peers with x/λ ≤ the null's 99.7% point;
    - set λ = median(kept) / median of the truncated null.
  - Clip λ to [1, 4]; hitting 4 is reported as capped.
  - The clipping is there so that genuine movers or variables among the peers are not counted as noise. In the unit tests it recovers the true λ within about 0.1 with 0%, 30% or 60% of peers anomalous (n = 20).
- **Calibrated values.** x_cal = x/λ and p_cal = χ²-sf(x_cal), mapped to [0,1] exactly as in Phase Two.
- **Empirical rarity.** If k of n peers are at least as extreme, the reported bound is (k+1)/(n+1). It is never reported as zero.

### 2.2 Catalogue term: completeness gating

- **Why gate it.** Missing from Gaia DR3 or CatWISE2020 means something only if the source is bright enough that the catalogue should contain it.
- **Local completeness flux F_c.** Estimated per series and per catalogue from the non-low-reliability tracks that have a measured flux and a catalogue answer. F_c is the lowest flux at which at least 90% of the tracks at or above it are matched, needing at least 5 tracks.
- **Leave-one-out.** F_c is computed without the source being scored, so an unmatched source cannot raise the limit above itself.
- **Weights.** Each catalogue a track is unmatched in gets a weight:

| regime | weight | when |
|---|---|---|
| `above_completeness` | 1.0 | flux ≥ F_c: a missing counterpart is meaningful |
| `below_completeness` | 0.0 | flux < F_c: expected to be missing ("below catalogue completeness") |
| `completeness_unknown` | 0.5 | F_c could not be estimated (fewer than 5 usable tracks, or the bright end is never 90% matched) |
| `catalogue_incomplete_here` | 0.0 | the catalogue matches none of at least 5 usable tracks (a coverage hole) |
| `unavailable` | n/a | the query failed or was not run; raw C is already 0 |

- **C_norm** = raw C × the smallest weight among the catalogues the track is unmatched in.

### 2.3 Quality caps (calibrated score only)

| condition | cap |
|---|---|
| reliability low | 0.05 |
| near_bright_star (Lanczos ringing zone) | 0.05 |
| core masked in every epoch | 0.05 |
| ambiguous_link | 0.10 |
| two_epochs_only | 0.30 |
| seen in fewer than 2 epochs | 0 |

Each applied cap is listed in `suppression_reasons`, so a capped source can never be a top
candidate without saying why.

### 2.4 Frontend

- **Table.** Raw and Calibrated columns. ▼ marks a calibrated score below the raw one; the tooltip gives the reasons.
- **Ordering.** An "Order by" selector, defaulting to calibrated.
- **Detail panel.** A "Calibrated ranking" block showing:
  - components raw → calibrated;
  - motion and variability peer calibration (λ, p before and after, rarity bound);
  - per-catalogue completeness regime with the local limit;
  - suppression reasons.
- **CSV export.** Gains calibrated_score, calibrated_rank, M_norm, V_norm, C_norm, completeness_regime, quality_cap and suppression_reasons.
- **Checked.** In the browser on cached real data (ε Ind B and RW Aur, cached real catalogue rows): no NaN, undefined or Infinity anywhere on the page.

## 3. Synthetic injection study

**Everything in this section is synthetic:** artificial point sources added to real cutouts.
It measures how the pipeline treats sources whose truth is known; it is not a measurement of
real astrophysical sources.

### 3.1 Setup

- **Where sources go.** PSF-matched point sources are added to the image extension of the real cached native cutouts (`backend/tests/data/real/`), before registration, resampling, detection and photometry, so they pass through the unchanged pipeline.
- **PSF.** Pixel-integrated circular Gaussian with each exposure's header PSF_FWHM.
- **Positions.** Moved by the injected proper motion to each exposure's MJD-AVG.
- **Catalogues.** Matching uses the cached real Gaia DR3 and CatWISE2020 rows for each field (radius 0.07°). There is no network access.
- **Placement.** A 20″ grid within ±100″ of the centre, at least 30″ from any real SPHEREx detection, 6″ from any catalogue position, and 40″ from each other.
- **Flux.** Set per detector from a pilot injection (S/N assumed proportional to flux). Pilot S/N per mJy:

  | field | detector: S/N per mJy |
  |---|---|
  | barnard | D6 7.36, D3 3.28 |
  | eps_ind_b | D6 12.81, D3 13.0 |
  | wise0855 | D6 12.93, D5 21.78 |
  | rw_aur | D6 8.88, D3 18.11 |

- **R Dra skipped.** The pilot source was not recovered in D6 or D3.
- **Run parameters.** Seed 20261115, `--reps 2 --sparse 6`, 1358 injected source-detector cases, runtime 1435 s.
- **Experiments.**

  | experiment | what is injected |
  |---|---|
  | stationary | constant, non-moving sources at target S/N 6, 10, 20, 50 (8 per run) |
  | moving | 2 or 5″/yr, random direction (2 per run, among 6 stationary) |
  | variable | one epoch dimmed by 40% (2 per run, among 6 stationary) |
  | near_bright | S/N 10, 20″ from stars with S/N > 300 |
  | sparse_stationary | one S/N 50 and one S/N 10 source per run, 6 runs per field, so the field is not crowded with uncatalogued injections |

- **Measured.** Recovery is a track with at least 2 epochs within 5″. A source counts as flagged if its score is at or above the threshold (0.1–0.4).

### 3.2 Results: stationary constant sources (false positives)

All injected sources are truly absent from the catalogues, so the raw catalogue term C = 1
(score 0.25) is technically correct for every one. The question is whether that boost belongs
to sources too faint to be catalogued. Columns: fraction flagged, raw → calibrated; in
brackets, the fraction flagged by motion or variability alone (raw = calibrated in every row).

| S/N (measured median) | recovered | n | thr 0.1 | thr 0.2 | thr 0.3 | thr 0.4 |
|---|---|---|---|---|---|---|
| ~6 (7.6) | 16% of 96 | 15 | 1.00 → 0.27 [0.13] | 1.00 → 0.13 [0.13] | 0.13 → 0.13 [0.13] | 0.13 → 0.00 [0] |
| ~10 (10.2) | 93% of 288 | 269 | 0.941 → 0.416 [0.022] | 0.941 → 0.033 [0.022] | 0.037 → 0.022 [0.022] | 0.022 → 0.022 [0] |
| ~20 (19.9) | 91% of 288 | 261 | 0.916 → 0.406 [0.015] | 0.916 → 0.019 [0] | 0.019 → 0.000 [0] | 0 → 0 |
| ~50 (49.7) | 88% of 288 | 253 | 0.933 → 0.802 [0] | 0.933 → 0.004 [0] | 0.004 → 0.000 [0] | 0 → 0 |
| sparse ~10 (10.4) | 85% of 48 | 41 | 0.854 → 0.683 [0.049] | 0.854 → 0.049 [0.049] | 0.049 → 0.049 [0.049] | 0.049 → 0.049 [0] |
| sparse ~50 (50.5) | 94% of 48 | 45 | 0.844 → 0.733 [0] | 0.844 → 0.067 [0] | 0 → 0 | 0 → 0 |

Completeness regimes of the recovered stationary sources (count, of which C > 0 raw → calibrated):

| S/N | below_completeness | completeness_unknown | above_completeness |
|---|---|---|---|
| ~10 | 148 (148 → 0) | 120 (120 → 120) | 1 (1 → 1) |
| ~20 | 144 (144 → 0) | 115 (115 → 115) | 2 (2 → 2) |
| ~50 | 34 (34 → 0) | 219 (219 → 219) | 0 |
| sparse ~10 | 7 (7 → 0) | 34 (34 → 34) | 0 |
| sparse ~50 | 5 (5 → 0) | 33 (33 → 33) | 7 (7 → 7) |

What this shows:

- **The raw score's large flag rate at 0.1–0.2 is entirely the catalogue term.** Calibration
  removes it for sources below completeness and halves it where completeness is unknown.
  - At threshold 0.2 the flagged fraction falls from 92–94% to 0.4–3.3% (main runs) and from 84–85% to 4.9–6.7% (sparse runs).
  - The calibrated residual at 0.2 is the sources with C weight 1 plus the motion/variability cases.
- **Motion- and variability-driven false positives are unchanged by calibration in this study.**
  - The measured peer inflation λ was 1.00 for the median track; the 95th percentile was 1.38, the maximum 2.59, and 12.8% of 1120 tracks had λ > 1.05.
  - In other words, the Phase Two error model was already close to right in these fields at these S/N levels.
- **Every one of those residual stationary false positives is in RW Aur D6** (V = 0.34–1.0 for constant sources).
  - That field has only 4 clean injection positions, so the same few positions recur.
  - Its calibration fell back to the all-peers level with λ = 1.0.
  - A field-specific systematic there, likely from the bright, variable RW Aur itself, is not captured by the peers. This is a limitation, not a solved case.
- **"Completeness unknown" is common even in the sparse runs** (33 of 45 S/N 50 sources); see §4. Where completeness was estimable, the bright S/N 50 sparse sources were correctly classed above completeness (7: barnard D6 1, wise0855 D5 6).

### 3.3 Results: movers and variables (recovery)

Fraction of recovered anomalous sources whose score reaches each threshold, raw → calibrated,
and the fraction whose own component (M for movers, V for variables) is at least 0.3,
raw → calibrated.

| kind, S/N | recovered | n | thr 0.1 | thr 0.2 | thr 0.3 | thr 0.4 | component ≥ 0.3 |
|---|---|---|---|---|---|---|---|
| moving ~10 | 80% of 64 | 51 | 0.96 → 0.78 | 0.94 → 0.35 | 0.47 → 0.27 | 0.37 → 0.20 | 0.373 → 0.373 |
| moving ~20 | 80% of 64 | 51 | 0.96 → 0.78 | 0.96 → 0.53 | 0.67 → 0.37 | 0.47 → 0.29 | 0.510 → 0.510 |
| moving ~50 | 75% of 64 | 48 | 0.98 → 0.88 | 0.96 → 0.65 | 0.73 → 0.52 | 0.58 → 0.42 | 0.604 → 0.604 |
| variable ~10 | 75% of 32 | 24 | 0.96 → 0.46 | 0.96 → 0.04 | 0.04 → 0 | 0.04 → 0 | 0 → 0 |
| variable ~20 | 84% of 32 | 27 | 1.00 → 0.48 | 0.96 → 0.30 | 0.85 → 0 | 0 → 0 | 0.481 → 0.481 |
| variable ~50 | 100% of 32 | 32 | 1.00 → 0.75 | 0.94 → 0.56 | 0.94 → 0.13 | 0.13 → 0.06 | 0.969 → 0.844 |

- **Movers.**
  - Motion sensitivity, the M component itself, is identical raw and calibrated.
  - The drop in score-threshold recovery is the removal or halving of the catalogue boost; it is not a loss of motion sensitivity.
  - By speed (recovered cases with M ≥ 0.3):

    | speed | S/N 10 | S/N 20 | S/N 50 |
    |---|---|---|---|
    | 5″/yr | 17/26 | 22/25 | 24/24 |
    | 2″/yr | 2/25 | 4/26 | 5/24 |

  - With 3–4 epochs spanning about a year, 2″/yr is barely above the motion errors.
- **Variables.**
  - A one-epoch 40% dip is undetectable at S/N 10.
  - At S/N 50 the variability component reaches ≥ 0.3 in 97% (raw) and 84% (calibrated) of cases. The difference is peer inflation > 1 in some series.
  - Because the variability weight is 0.30, a pure variable without a catalogue boost scores at most 0.30. At thresholds ≥ 0.3 the calibrated score therefore loses variables unless they are also unmatched above completeness. This is a property of the Phase Two weights, which were kept on purpose.

### 3.4 Results: near bright stars

- **Recovery.** 3 of 14 injections recovered (21%); the rest fall inside the bright star's isophote and are lost, as already documented for Phase Two.
- **Flagged near_bright_star:** 1 of the 3.
- **Flag rates.** Raw: 1.00 at thresholds 0.1 and 0.2, 0.33 at 0.3. Calibrated: 0.67 at 0.1, 0 at 0.2 and above.
- **No motion- or variability-driven flags.**
- In the first run (same code before the estimator change) one ε Ind B D6 case was not flagged near_bright_star, because its neighbour was less than 25× brighter in D6. Its calibrated score was 0.20.

## 4. Real-data regression (no injection)

Everything here is real data: the cached real cutouts and real catalogue rows, with no
synthetic sources.

- **Regression tests.**
  - The 137 Phase Two and Phase Three tests, including the real-cutout regression tests on the five targets, pass unchanged.
  - With the 19 new tests the suite stands at 156 passed, 6 live-network tests deselected.
- **Local completeness limits** on the five real fields (cached real catalogue rows, all tracks, no leave-one-out exclusion):

| field / detector | Gaia DR3 F_c (mJy) | CatWISE2020 F_c (mJy) | why unknown |
|---|---|---|---|
| barnard D6 | 3.33 | 22.0 | |
| barnard D3 | 0.90 | unknown | Barnard's Star (3342 mJy) itself is unmatched in CatWISE (very high proper motion) |
| eps_ind_b D6 | 2.31 | unknown | the second-brightest track is unmatched in CatWISE |
| eps_ind_b D3 | 0.51 | unknown | brightest track unmatched in CatWISE |
| wise0855 D6 | 0.48 | 0.36 | |
| wise0855 D5 | unknown | 0.55 | only 6 tracks; one unmatched |
| rw_aur D6 | unknown | unknown | crowded star-forming field: about 1 in 5 of the 43 tracks unmatched at all flux levels, including the third-brightest |
| rw_aur D3 | unknown | unknown | 8 tracks, unmatched near the top |
| r_dra D6, D3 | unknown | unknown | 1 and 0 usable tracks; CatWISE2020 has no rows within 0.07° |

- **Why "unknown" is common.**
  - Real bright sources are often unmatched for reasons that have nothing to do with catalogue depth: high proper motion (catalogue epochs differ), blends, saturation and crowding.
  - With the 90%-from-the-top definition, one such source at the bright end prevents an estimate.
  - The calibration then applies half weight and says so, rather than guessing.
- **What the UI shows on real data** (RW Aur D6, cached real catalogues): the highest raw-ranked track (0.47, reliability low) is capped to 0.05; most candidates go from 0.25 to 0.13 with "completeness could not be estimated in this field".

## 5. Limitations

1. **Injected sources add no photon noise of their own** (the variance plane is untouched). This is negligible at S/N ≤ 20 and slightly underestimates errors at S/N 50.
2. **The PSF is a Gaussian with the header FWHM, not the real SPHEREx ePSF.** Recovery near bright stars and of S/N ~6 sources depends on this.
3. **Completeness is often unknown on real fields** (§4); the 0.5 weight is a convention, not a measurement.
   - The main experiment's 8 uncatalogued injections per field also bias the matched fraction down.
   - The sparse experiment reduces that bias but cannot remove the real-field causes.
4. **R Dra could not be included:** the pilot was not recovered.
5. **CatWISE2020 has no rows near R Dra.** That is handled by the `catalogue_incomplete_here` regime, but not exercised by injection.
6. **Peer calibration caught no inflation in this study** (median λ = 1.00), so its benefit on these data is not demonstrated. The unit tests show it works when inflation is present.
   - The RW Aur D6 systematic shows that a field-specific problem around one bright variable star is not caught by series-wide peers.
7. **Sample sizes are small:** 15 recovered S/N 6 sources, 24–51 per mover or variable bin, 3 near-bright recoveries.
   - Rates of 0 are upper limits of roughly 1/(n+1), not zero false-alarm rates: for example ≤ 0.004 for 253 S/N 50 stationary sources at threshold 0.3.
8. **The variability weight caps a pure variable at 0.30**, so threshold ≥ 0.3 selections exclude variables unless they are also unmatched above completeness.

## 6. Reproduce

```bash
cd backend && OPENBLAS_NUM_THREADS=2 ../.venv/Scripts/python.exe -m pytest -p no:cacheprovider -q
```

```bash
./.venv/Scripts/python.exe scripts/validate_ranking.py --reps 2 --sparse 6
```

`scripts/cache_catalogs.py` refreshes the cached real catalogue rows. It needs IRSA, and the
validation numbers above refer to the rows now in `backend/tests/data/real/*/catalogs/`.
