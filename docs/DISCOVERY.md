# Discovery pipeline — what it does, how far to trust it, what was verified

Date: 2026-11-15. Follows [AUDIT.md](AUDIT.md). The feature matrix in the audit describes the
state **before** this work; this file describes what was added after it.

> **Superseded in part by [PHASE2.md](PHASE2.md).** Kept as the record of the first discovery
> pipeline. Since this was written: resampling is Lanczos-3 (was nearest-neighbour); the
> positional floor is 0.4″ (was 1″) and the flux floor 6 % (was 10 %), both measured on real
> field stars; the variability test has been shown on a real variable (RW Aurigae); and
> shift-and-stack and a parallax-aware fit were added. Numbers below are as measured at the time.

## What was added

| Capability | Where | Status |
|---|---|---|
| Epoch for typed coordinates (CatWISE2020 track lookup) | `backend/app/catalogs.py`, `coords.py` | Working, verified live on WISE 0855 |
| Source detection per frame (photutils segmentation + deblending) | `backend/app/discovery.py` | Working, verified live |
| Cross-epoch tracking and weighted proper-motion fit | `discovery.py` | Working, verified live on Barnard's Star |
| Registration check against Gaia stars | `discovery.py` | Working, verified live |
| Variability test with a systematic error floor | `discovery.py` | Working on synthetic data; not yet shown on a real variable (since shown on RW Aurigae: see PHASE2.md) |
| Gaia DR3 + CatWISE2020 cross-match, propagated to the observation epoch | `catalogs.py`, `discovery.py` | Working, verified live |
| Explainable ranking | `discovery.py` | Working |
| Candidate store and volunteer classification (SQLite) | `backend/app/store.py`, `/api/candidates*` | Working, verified live |
| Tracked-sources panel, overlay, classification buttons, JSON/CSV export | `frontend/` | Working; layout and overlay not checked by eye (see below) |

The analysis runs at the end of the `/api/frames` stream, on the frames that were just
streamed. It does not download anything from IRSA a second time. It is sent as one final
NDJSON line, `{"type": "analysis", ...}`.

## How it works

1. **Detection.** Each frame of a detector series is searched for connected groups of pixels
   above 5σ of the frame's own background scatter. Each source gets a centroid and an aperture
   flux. Its positional error is the centroid's statistical error combined with a 1″ floor
   (3″ if the core is masked or saturated).
2. **Tracking.** Detections are linked from frame to frame, nearest first, inside a gate that
   allows up to 12″/yr of motion. If more than one detection could continue a track, the
   track is flagged.
3. **Registration check.** For every frame, isolated Gaia stars with motion below 50 mas/yr
   give the median offset between measured and catalogue positions. That offset is removed
   before fitting. A frame with fewer than three such stars is reported as unchecked.
4. **Proper motion.** A weighted straight-line fit of position against time. With two epochs
   the fit has no spare measurements to check itself, and says so.
5. **Variability.** A χ² test of constant flux, with a 10 % systematic error added to every
   point. A wavelength difference between frames above 0.05 µm is flagged, because the change
   may then come from the spectrum's shape.
6. **Catalogues.** Gaia DR3 and CatWISE2020 positions are moved to the observation epoch
   using their proper motions. A source matches if it lies within 3.5σ of the combined
   uncertainty. Several sources inside that gate are reported as ambiguous.
7. **Ranking.** Anomaly score = 0.45 × motion + 0.30 × variability + 0.25 × missing catalogue
   counterpart, each between 0 and 1. Every track also carries the reasons for its score and
   the reasons for caution, in plain sentences.

Four quantities are kept apart in every record:

- **Detection S/N**: how strongly the source stands out.
- **Anomaly score**: how unusual its measured behaviour is. It is not a probability.
- **Reliability** (high / medium / low): how far the measurement itself can be trusted.
- **Class probability**: not estimated. The field is always null.

Status words: `catalogued` (matched uniquely to Gaia or CatWISE), `candidate` (not matched,
measurement trustworthy), `uncertain` (measurement not trustworthy), `unverified` (a
catalogue could not be queried). None of them means "new object".

## Typed coordinates and CatWISE

Typed coordinates carry no date. For a fast-moving object, a published position is where it
was when measured. The lookup asks CatWISE2020 for sources within 1.5′ and accepts one only if:

- its motion is at least 0.5″/yr and at least 5σ significant, and
- its straight-line track passes within 3″ (+3σ) of the typed position at some date between
  2010 and 2021.

Rows belonging to the same object are combined by weighted mean. If two different objects
qualify, nothing is applied and the status line says the result is ambiguous. The status
line always says what was done. The lookup can be switched off per request with
`pm_lookup=false`.

Things seen in CatWISE while building this, all handled:

- WISE 0855 appears as **three** rows (motions −7.92, −7.97, −8.02″/yr in RA).
- A slow source sits exactly at WISE 0855's 2010 position, which made the first version of the
  lookup report an ambiguity.
- Near Barnard's Star there is an 18th-magnitude row with a 10″/yr motion, which looks like a
  fragment of the bright star and not a real object.
- CatWISE's motion for ε Ind B is wrong (it does not resemble the Gaia value).

Catalogue access uses IRSA's cone-search service (`/SCS`). The same cone through IRSA TAP did
not return within 60–150 s when tested; `/SCS` took 1.5–10 s.

## Live results (2026-11-15)

**Barnard's Star, D6, 5.5′ field, Low quality, 3 epochs over 1.01 yr**

| Quantity | Measured from SPHEREx | Gaia DR3 |
|---|---|---|
| Total proper motion | 10.19 ± 1.39″/yr (7.3σ from zero) | 10.39″/yr |
| East component | −1506 ± 1387 mas/yr | −802 mas/yr |
| North component | 10076 ± 1387 mas/yr | 10362 mas/yr |
| Position vs Gaia at the observation epoch | 0.45″ (0.5σ) | — |

- Ranked first of 17 tracked sources and labelled `catalogued`, with the reason "measured
  motion agrees with Gaia DR3 (0.5σ difference): a known moving object".
- The 13 other sources seen in all three epochs have motions between 0.1σ and 1.1σ from zero.
- Registration was checked against Gaia stars in every frame.
- An earlier run, during which the Gaia query failed with HTTP 502, gave 9.62 ± 1.39″/yr with
  no registration correction, and reported Gaia as "n/a" (not as "unmatched").
- The motion error is set almost entirely by the 1″ positional floor. Three epochs over one
  year cannot do better with this floor.

**WISE 0855−0714, entered as NASA's coordinates `08h55m10.831s -07d14m42.53s`, D6, 3′, Low**

- The lookup matched CatWISE J085507.96−071439.3 (three rows combined), whose track passes
  2.2″ from the typed position at epoch 2010.4. Motion used: (−7989, 877) mas/yr.
- The position predicted for 2026 is 2.5″ from the one predicted from SIMBAD's astrometry.
  Before this change the typed coordinates were about 128″ from the object.
- Forced aperture photometry at the predicted position, per epoch:
  0.19 ± 0.10 mJy (1.9σ, 4.94 µm), 0.47 ± 0.09 mJy (5.6σ, 4.69 µm), 0.47 ± 0.06 mJy (7.7σ, 4.72 µm).
- Its WISE W2 magnitude (about 13.9) corresponds to roughly 0.47 mJy, so the measured flux is
  the size expected. **This is a tentative detection, not a confirmed one.** Blind detection
  found no source above 5σ in this field. The second and third positions are only ~3.5″ apart,
  inside one 9″ aperture, so a faint background source there would produce the same numbers.
  A check at the 2010 and 2015 positions (where the object no longer is) would settle it.

**Classification.** A "Moving source" vote on the Barnard's Star track was stored and read
back with the agreement summary. That one test vote is still in the local development
database (`data/discovery.sqlite`, not tracked by git).

## Not verified

- The tracked-sources panel and the image overlay were checked through the page's content and
  the absence of console errors. Screenshots timed out, so their appearance was **not**
  checked by eye.
- JSON/CSV export: the buttons enable and the code runs without error; the downloaded files
  were not opened.
- ε Ind B, RW Aurigae and R Draconis were not run through the new pipeline.
- The variability test has only been shown to work on synthetic data.
- Tests: 67 offline tests pass. All images and catalogue rows in them are synthetic and
  labelled as such. The six live tests in `test_pipeline_live.py` were not run.

## Known limits

- **Sensitivity.** Blind detection needs ~5σ per pixel group in a coadd of a few exposures.
  Faint movers such as WISE 0855 are below that; they are reachable only by forced photometry
  at a predicted position, or by stacking along a trial motion (not implemented).
- **Variability below ~10–20 %** cannot be claimed (see the audit's measurement on Barnard's
  Star), and frames of one detector differ in wavelength.
- **Crowded fields.** The 12″/yr gate can link the wrong detections. Such tracks are flagged
  but not corrected.
- **Two epochs.** Many positions have only two or three SPHEREx passes so far.
- **Parallax** is ignored in the motion fit (at most 0.55″ for the nearest stars).
- **Classification** has no accounts: one vote per browser token, which stops accidental
  double votes and nothing more.
- **Resampling** is still nearest-neighbour (audit item H2), which is where the 1″ floor and
  the unreliable difference images come from.

## Suggested next steps

1. Run ε Ind B, RW Aurigae and R Draconis; record all five targets as regression fixtures
   from cached real cutouts.
2. Off-position control photometry for forced detections (settles the WISE 0855 question).
3. Replace nearest-neighbour resampling, then lower the positional floor.
4. Shift-and-stack along trial motions to reach faint movers.
5. Per-exposure F(λ, t) binned by epoch, compared with IRSA's Spectrophotometry Tool.
