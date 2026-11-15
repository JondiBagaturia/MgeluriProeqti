# Project status — SPHEREx Discovery Lab

Last updated: 2026-11-15. **Current phase: Phase Two complete (pending commit); Phase Three
not started.**

Details: [docs/PHASE2.md](docs/PHASE2.md) (methods, validation, limitations),
[docs/AUDIT.md](docs/AUDIT.md) (the original audit and its fixes), and
[docs/DISCOVERY.md](docs/DISCOVERY.md) (the first discovery pipeline, partly superseded).

## What the application does

- Takes a sky position or object name. Typed coordinates are given an epoch through a CatWISE
  fast-mover lookup.
- Fetches SPHEREx cutouts from IRSA and blinks them per detector, north up, with the target
  followed by its proper motion.
- Measures aperture photometry with errors, refusing saturated or masked apertures.
- Detects and tracks sources across epochs, registers each frame to Gaia, fits proper
  motions, and tests variability against measured systematic floors.
- Cross-matches with Gaia DR3 and CatWISE2020 at the observation epoch.
- Ranks sources with an explainable anomaly score kept separate from detection S/N and
  reliability. No class probability is estimated.
- Optionally runs a shift-and-stack search for faint movers, with time-shuffled controls.
- Shows a parallax-aware fit, which is labelled unconstrained unless the data constrain it.
- Stores ranked sources and volunteer classifications in SQLite, and exports JSON/CSV.

## Latest validation (cached real cutouts, final code, live catalogues)

| Target | Result |
|---|---|
| Barnard's Star | PM (−1061, +10358) ± 959 mas/yr vs Gaia (−802, +10362). Saturated in D3 (refused). Parallax unconstrained (3 epochs) |
| ε Indi B | PM (+4009, −2703) ± 425 mas/yr vs Gaia (+3982, −2467). Not variable |
| WISE 0855 | Blind detection in 3 D5 epochs (0.37–0.47 mJy). Shift-and-stack recovers it at the predicted position, passing 30 controls (p ≤ 0.032). Velocity (−7.75, −0.25)″/yr vs catalogue (−8.12, +0.67): east agrees, north differs by ~2σ |
| RW Aurigae | 2489 → 1808 → 1282 mJy at 4.75 µm, p = 8.6×10⁻¹³ for constant flux. Not moving |
| R Draconis | Saturated in every frame: no photometry, no variability claim |

Tests: 107 offline tests pass (10 of them on real data), both in the project venv and in a
brand-new venv built only from `backend/requirements-dev.txt`, on the final code. 6 live
tests pass.

## Known limitations (summary)

- 3–4 epochs per field, so motions are good to about 0.4–1″/yr and parallaxes are
  unconstrained.
- Variability below roughly 6–10 % cannot be claimed.
- The PSF is undersampled: Lanczos ringing is flagged, and faint neighbours inside a bright
  star's isophote are lost.
- Saturated targets cannot be measured.
- Shift-and-stack velocities are uncertain by about 1–1.5″/yr (estimated from WISE 0855); the formal
  refinement width (~0.6″/yr) is shown in the UI only as "statistical only, not the full error",
  and its controls only bound false-alarm rates to ≤ 1/(n + 1).
- IRSA catalogue access is intermittent.

## Phase Three (not started)

1. Spectro-temporal F(λ, t) per source and epoch; band-dependent light curves; comparison
   with IRSA's Spectrophotometry Tool.
2. Anomaly scores normalised against sources of similar brightness, coverage and quality,
   plus a synthetic-injection false-positive study.
3. Per-exposure Gaia registration and a PSF model inside shift-and-stack; honest velocity
   errors.
4. Parallax once SPHEREx provides at least 4–6 epochs per field.
5. Re-verify the Docker image (Python 3.11).
