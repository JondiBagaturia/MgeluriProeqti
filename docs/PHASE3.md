# Phase Three report — spectro-temporal analysis F(λ, t)

Date: 2026-11-15. Builds on [PHASE2.md](PHASE2.md). Code: `backend/app/spectrotemporal.py`
(measurements, model, metrics), `backend/app/stream.py` (`run_spectrotemporal`), and the
"Spectro-temporal analysis" panel in `frontend/`. Validation:
`scripts/validate_spectrotemporal.py` (results in `docs/data/spectrotemporal_report.json` and
`docs/data/spectrotemporal_log.txt`). Floor calibration: `scripts/calibrate_spectrotemporal.py`.

## 1. What the data can and cannot give (audit)

- **One wavelength per exposure.** SPHEREx's linear-variable filters put a different
  wavelength at every detector position. A source therefore gets **one wavelength per
  exposure**, read from the WCS-WAVE lookup at the source's position. IRSA describes that
  lookup as accurate to about 1 nm and "intended for visualization"; for science IRSA
  points to the CWAVE/CBAND calibration products, which are not used here (deferred).
  A "spectrum" is therefore assembled from many exposures taken at different times.
- **Simultaneity.** Only the dichroic detector pairs D1/D4, D2/D5 and D3/D6 observe at
  the same instant: they share an `obs_id` and a timestamp. Nothing else is simultaneous.
- **Timing.** Each exposure's mid-time comes from the FITS header (`MJD-AVG`); the SIA
  `t_min` is used only as a fallback. Exposure time is **not** in the cached files.
- **Coverage.** Each target was cached in **2 of the 6 detectors** (D3 + D6, or D5 + D6
  for WISE 0855). Each epoch gives about 6 exposures per detector, at scattered
  wavelengths (e.g. 4.50–4.96 µm in D6), against a resolution element of about
  0.037–0.05 µm. A per-channel light curve would almost never land on the same wavelength
  twice, so light curves need an explicit model.
- **The "102 channels".** These belong to the All-Sky Cube product (17 per detector), not
  to the spectral images used here. No channel IDs are invented: each measurement keeps its
  own wavelength and bandwidth.
- **Flag definitions.** The SPHEREx flag definitions (Explanatory Supplement, Table 16)
  could not be read. OVERFLOW, BLOOM, SUR_ERROR and NONLINEAR appear in the apertures of
  every bright target. They are treated as a separate quality state (below), never as
  plain "valid".
- **Database.** No migration is needed. Spectro-temporal results travel in the analysis
  record; the candidates table and its stored summaries are unchanged, and a test checks
  this.

## 2. Measurement model

Each measurement is forced aperture photometry of one source in one exposure:

- **Pixels:** the exposure's own native pixels, so there is no resampling and no
  resampling-correlated noise.
- **Aperture:** an exact-overlap aperture of r = 9″ (photutils) with a sigma-clipped sky
  annulus at 18–30″.
- **Position:** propagated to the exposure's mid-time with the catalogue proper motion
  (target) or the measured motion (tracked sources with motion ≥ 3σ).
- **Units:** F<sub>ν</sub> in mJy (aperture sum of MJy/sr × pixel solid angle, × 10⁹).
- **Aperture correction:** **none applied**. An approximate Gaussian correction from the
  header `PSF_FWHM` is stored separately.
- **Statistical error:** from the VARIANCE plane plus the sky error.

Every measurement has exactly one state:

| State | Meaning | Flux used? |
|---|---|---|
| `valid` | finite aperture, no saturation-related flag, S/N ≥ 3 | yes |
| `valid_flagged` | finite, but OVERFLOW/BLOOM/SUR_ERROR/NONLINEAR in the aperture | yes, marked; every result is also computed without these |
| `below_threshold` | finite, unflagged, S/N < 3 | no; 3σ upper limit = max(F, 0) + 3σ |
| `saturated` | blanked (NaN) aperture pixels **and** saturation flags | no flux |
| `masked` | NONFUNC/MISSING_DATA/HOT/COLD or NaN without saturation flags | no flux |
| `off_image` / `no_background` | aperture off the cutout / too little sky | no flux |
| retrieval failed | requested exposure not downloaded (counted per series) | no row |
| missing | a wavelength/epoch combination SPHEREx did not observe | shown as a coverage gap |

**Systematic floor (2.5% per exposure).** Calibrated on real field stars
(`calibrate_spectrotemporal.py`): the robust residual scatter about the shared-shape model,
minus the statistical error, for 206 reference-star exposures with S/N ≥ 50. Per S/N band
it is 2.3%, 2.9% and 1.8%; the adopted value is 2.43%, rounded to 2.5%.

A first calibration (median reduced χ² = 1 per detector) gave 0.7% for D6. Faint exposures
dominate that method, and it made a quiet star look variable. It was rejected.

## 3. The spectro-temporal model and the metrics

For each detector:

> F<sub>i</sub> = a<sub>e(i)</sub> · S(λ<sub>i</sub>) · (1 + t<sub>e(i)</sub> (λ<sub>i</sub> − λ₀)),
> with σ<sub>i</sub>² = σ<sub>stat,i</sub>² + (0.025 · model<sub>i</sub>)²

- **S(λ)** is one spectral shape shared by all epochs: piecewise-linear through knots at
  quantiles of the observed wavelengths. There is at most one knot per three bandwidths
  and per three data points, and S(λ₀) = 1.
- **a<sub>e</sub>** is the flux at λ₀ in epoch e. This is the band light curve.
- **t<sub>e</sub>** is an optional per-epoch tilt, with t = 0 in the first epoch.

**Why not a straight line?** The first version used a straight line across each detector.
On real data it made a quiet M dwarf (Barnard's Star) and a T dwarf (ε Ind B) look
variable. Their spectra have molecular bands, and each epoch samples different
wavelengths, so the shape error leaked into the epoch amplitudes. That is the
"wavelength mismatch read as variability" error.

| Metric | Definition | Requires | Can establish | Cannot establish |
|---|---|---|---|---|
| **M1 brightness change at λ₀** | Δχ² = χ²(a<sub>e</sub> = a) − χ²(a<sub>e</sub> free); dof = E − 1; plus the fractional range (max − min)/mean of a<sub>e</sub> with propagated error | ≥ 2 epochs with ≥ 2 usable points each; enough points for the shape | that the flux at λ₀ differs between epochs beyond the errors | the cause; anything about other wavelengths |
| **M2 spectral slope change** | Δχ² = χ²(t<sub>e</sub> = 0) − χ²(t<sub>e</sub> free); dof = E − 1 | ≥ 2 epochs with ≥ 3 points spanning ≥ 0.1 µm | a change of slope within the detector, relative to the shared shape | changes narrower than the knot spacing |
| **M3 colour change** | C<sub>e</sub> = ln(a<sub>e</sub>(A) / a<sub>e</sub>(B)); χ² of a constant (dof = n − 1); errors propagated (σ<sub>a</sub>/a in quadrature) | both detectors fitted, ≥ 2 common epochs | wavelength-dependent (chromatic) change between detectors | which detector changed (see M1 per detector) |
| **M4 same-wavelength pairs** (model-free) | pairs of exposures from different epochs with \|Δλ\| ≤ ¼ bandwidth; z = (F₂ − F₁)/√(σ₁² + σ₂²); χ² = Σz² (dof = number of pairs; approximate when pairs share an exposure) | at least one matched pair | change at a wavelength without any spectral model | anything where no pairs exist; steep spectral features still matter |

**Interpretation rules.** These are applied in code, in this order:

1. **Model adequacy.** If the shape fit's reduced χ² exceeds the limit, M1–M3 are reported
   as **inconclusive**. The limit is the reference stars' 84th percentile, clipped to 2–3,
   and 2 when there are no reference stars.
   - Synthetic check: with a limit of 2, 0 of 900 evaluations of a **constant** source with
     a deep resolved absorption band claimed a change; a limit of 3 allowed 9.
   - The 95th percentile was tried first and is not robust: one odd reference star in the
     RW Aur field pushed it to 119.
2. **Reference stars.** These are stationary, S/N ≥ 30, high-reliability field stars in the
   same field, measured identically. A metric is compared by p-value. If k of n reference
   stars are at least as significant, k = 0 is reported as "rarer than about 1 in (n + 1)",
   not as zero; k > 0 gives "not unusual for this field".
3. **Flagged data.** If more than 50% of the inputs are `valid_flagged`, a significant
   result becomes **conditional**. Its size is compared with an empirical control: the
   quiet Barnard's Star, all of whose usable D6 data are flagged, shows a 7.0% range. This
   is one star, so it indicates scale and is not a calibration.
4. **Structured spectra.** If the detector's shape model is inadequate, an M4 result is
   **conditional**, because wavelength offsets within ¼ bandwidth can still matter on steep
   bands.
5. **Significance.** Only then: p < 10⁻³ gives "significant change", described as a
   measurement statement and not as a physical cause.

No combined spectro-temporal score was built. The four metrics stay separate and are not
merged with the Phase Two anomaly score. No p-value is described as the probability that a
source is variable.

## 4. Validation on the five targets

Cached real cutouts, offline, final code (`docs/data/spectrotemporal_log.txt`).

| Target | Data | Spectrum (SED) | Result | What can be concluded |
|---|---|---|---|---|
| **Barnard's Star** | D3: 16 saturated, 2 off-image. D6: 14 `valid_flagged`, 2 masked, 2 off-image | D6 only, 4.41–5.00 µm | D6 flux at 4.59 µm: 3365 ± 120, 3274 ± 100, 3511 ± 120 mJy (7% range). Shape model inadequate (reduced χ² 2.85 vs limit), so M1/M2 are inconclusive. M4: no significant change (p = 0.03) | Moving source handled (position propagated per exposure). Saturation is refused in D3. **No variability claim** for a quiet star. This target is the empirical control for flagged data |
| **ε Indi B** | 39 valid, 2 flagged, 3 masked, 3 off-image; 4 epochs | D3 1.64–2.37 µm, D6 4.42–4.98 µm (T-dwarf molecular structure) | Both shape models inadequate (D3 reduced χ² 15), so M1–M3 are inconclusive. M4: D3 no change (p = 0.97); D6 conditional (structured spectrum, p = 8×10⁻⁴) | Its spectral structure cannot be separated from time variability with this sampling. **No variability claim; inconclusive.** The epoch-4 D6 drop (16.8 vs ~19.4 mJy) is noted but not supported |
| **WISE 0855** | 22 valid, 6 **below threshold** (upper limits), 4 masked, 1 flagged | D5 3.82–4.41 µm, D6 4.42–4.97 µm; fluxes 0.2–0.7 mJy | Fits adequate. Flux at 4.25 µm: 0.52, 0.59, 0.62 (± 0.08) mJy; at 4.69 µm: 0.69, 0.62 (± 0.1) mJy. M1–M4 all **no significant change** (p 0.29–0.93) | A valid **negative** result: constant within about 15–20% at S/N ~ 6 per epoch. Low flux and upper limits handled honestly |
| **RW Aurigae** | 34 `valid_flagged`, 2 masked (all usable data flagged) | D3 1.64–2.35 µm, D6 4.52–4.96 µm | D6 at 4.80 µm: 2333 → 1633 → 1148 mJy; D3 at 1.93 µm: 468, 535, 481 mJy. Both shape models inadequate (D3 reduced χ² 11.5), so M1–M3 are inconclusive. Model-free M4: **D6 median ratio later/earlier 0.73; D3 1.02**; both **conditional** (flagged data) | The dimming is **wavelength-dependent**: large at 4.8 µm, not at 2 µm. Conditional on OVERFLOW-flagged photometry, though the D6 change (~38% typical between matched exposures) is far above the 7% flagged-data control. The fits fail because RW Aur also varies **within** epochs: two D6 exposures in epoch 2, 0.011 µm and 9.8 days apart, differ by 15% (1717 ± 4 mJy at 4.663 µm on MJD 61091.7 vs 1971 ± 4 mJy at 4.652 µm on MJD 61101.6). The per-epoch model cannot represent that |
| **R Draconis** | 27 saturated, 2 off-image; 7 more exposures failed to download at caching | none | No fits; all metrics inconclusive | **No spectral or variability conclusion is possible** with these data, and none is made |

Phase Two coadd fluxes and these per-exposure fluxes differ by a few percent (e.g. RW Aur
D6 epoch 1: 2489 vs 2333 mJy). They are different measurements: a grid coadd at the mean
wavelength of an epoch versus native-pixel exact apertures at λ₀ from a model.

## 5. External reference (IRSA)

IRSA documents no SPHEREx "Spectrophotometry Tool" that could be found, and the Explanatory
Supplement could not be read (a 29 MB PDF over the fetch limit). No external spectral
comparison was implemented. A comparison would only be valid against a product with the
same calibration, apertures and epochs; none was identified. This is deferred.

## 6. Tests

| Suite | Result |
|---|---|
| Full offline suite, project venv | **137 passed, 0 failed** (6 live tests deselected) |
| New synthetic tests (`test_spectrotemporal.py`) | 20 passed |
| New real-data tests (`test_real_spectrotemporal.py`) | 9 passed |
| New store / schema-compatibility test (`test_store.py`) | 1 passed |
| Phase Two tests | all 107 still pass, unchanged |

The synthetic tests cover:

- flux and unit conversion (exact-aperture expectation, error calibration over 150 noise
  draws);
- every measurement state, and the NaN-outside-aperture regression;
- epochs from time gaps;
- known-answer recovery for M1 (30% dip), M2 (injected tilt), M3 (chromatic vs grey) and
  M4 (pair tolerance);
- uniform p-values for a constant source;
- **no false claims for 300 constant structured sources**;
- insufficient-data handling, the interpretation rules, and the reference distribution.

The real-data tests cover:

- wavelength metadata matches the FITS files;
- dichroic pairs share a timestamp;
- R Dra refuses everything;
- Barnard is never claimed variable;
- WISE 0855 upper limits and its negative result;
- ε Ind B is inconclusive;
- RW Aur's chromatic but conditional dimming.

## 7. Limitations

- **Coverage:** only 2 of 6 detectors are cached per target, and 3–4 epochs. The SED shows
  coverage gaps; it is not a full 0.75–5 µm spectrum.
- **Within-epoch variability** (RW Aur) violates the per-epoch model; it is detected only
  indirectly (inadequate fit).
- **Flagged photometry** (OVERFLOW etc.) has unverified reliability. Conclusions from it
  are conditional, controlled by one quiet star.
- **Structured spectra** (brown dwarfs) leave the model-based metrics inconclusive with
  this sampling. That is a real degeneracy, not a bug.
- **Absolute SED shape** carries aperture-loss differences between detectors (PSF FWHM
  7.4″ in D3 vs 9″ in D6, about a 5% effect). Temporal ratios within a detector are not
  affected.
- **Wavelengths** come from the WCS-WAVE lookup (about 1 nm, "for visualization"), not
  from CWAVE/CBAND.
- **Reference sets** are small (0–11 stars per field). The rarity bound is at best about
  1 in 12.
- **Cost:** the analysis adds about 5–10 s per field offline and about 0.7 MB to the
  analysis record. In the browser it is opt-in.

## 8. Deferred

- CWAVE/CBAND wavelength calibration, and per-exposure PSF (ePSF) photometry with aperture
  corrections.
- More detectors per target (cache D1, D2, D4).
- A within-epoch variability metric.
- A comparison with any official IRSA SPHEREx spectral product, if one with comparable
  calibration exists.
- Larger reference samples (wider fields).
- A combined spectro-temporal score (deliberately not built). The calibrated ranking of
  Priority 2 ([RANKING.md](RANKING.md)) recalibrates the existing motion, variability and
  catalogue terms only; it adds no spectro-temporal term.

## 9. Reproduction

Run from the project root, `MgeluriProeqti`.

Offline tests (no network):

```bash
cd backend && ../.venv/Scripts/python -m pytest
```

Five-target validation (writes `docs/data/spectrotemporal_report.json`; log as in
`docs/data/spectrotemporal_log.txt`):

```bash
.venv/Scripts/python scripts/validate_spectrotemporal.py
```

Systematic-floor calibration:

```bash
.venv/Scripts/python scripts/calibrate_spectrotemporal.py
```

In the app, tick "Spectro-temporal analysis F(λ, t)", or call
`/api/frames?...&spectrotemporal=true`.
