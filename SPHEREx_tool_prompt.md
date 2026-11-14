# Build: "SPHEREx Blink" — a public web tool to see how the sky changes in SPHEREx data

You are building a hackathon project (NASA Space Apps challenge). Work in phases, commit after each one, and don't start a phase until the previous one passes its checks. Begin by writing a short PLAN.md (architecture, file layout, risks), then build.

## Goal
A public web tool where anyone types sky coordinates (RA/Dec) or an object name. The tool fetches every NASA SPHEREx image covering that spot, aligns them, and plays them as a blinking animation (a "flipbook") so moving or variable objects are easy to see. Inspiration: WiseView (http://byw.tools/wiseview), which volunteers used to find brown dwarfs in WISE data. Copy its good ideas: fast blinking between epochs, contrast sliders, zoom, and a shareable URL.

**Top priority: speed.** The first frame should appear within a few seconds, and everything after that (contrast, blinking, adding or subtracting epochs) should feel instant.

**Out of scope for v1:** overlays from other telescopes (Gaia, WISE, SIMBAD markers) and embedded third-party sky viewers (Aladin etc.). Don't build them.

## Domain facts you must respect (verified from IRSA docs, Nov 2026)
- Data lives at NASA/IPAC IRSA: https://irsa.ipac.caltech.edu/Missions/spherex.html
  Tutorials (read them first and copy their patterns): https://caltech-ipac.github.io/irsa-tutorials/spherex/
  Especially: `spherex-intro` and `spherex-cutouts`.
- Releases: QR (Jul 2025), QR2 (Oct 2025), QR3 (Sep 2026), DR1 (Nov 2026). QR3/DR1 supersede QR2. New releases keep coming, so **never hard-code a release**. Discover collections at runtime (`Irsa.list_collections(servicetype='SIA', filter='spherex')`) or keep them in one config file.
- **No API keys or accounts are needed.** All SPHEREx data is public. IRSA TAP/SIA/cutouts are plain HTTPS. There is also an anonymous S3 mirror (`s3://nasa-irsa-spherex/`, us-east-1, `--no-sign-request` / `anonymous=True`), but it serves only whole files, so use the IRSA cutout service for the live tool. SIMBAD (name resolution) is also keyless.
- **Finding images, two options:**
  - SIA: `Irsa.query_sia(pos=(coord, radius), collection=['spherex_dr1','spherex_qr3'])` (`_deep` variants exist too). It lags new data by about a day.
  - TAP (preferred, used by the cutouts tutorial):
    ```sql
    SELECT 'https://irsa.ipac.caltech.edu/' || a.uri || '?center={ra},{dec}d&size={size_deg}' AS uri,
           p.time_bounds_lower, p.energy_bandpassname
    FROM spherex.artifact a JOIN spherex.plane p ON a.planeid = p.planeid
    WHERE 1 = CONTAINS(POINT('ICRS', {ra}, {dec}), p.poly)
    ORDER BY p.time_bounds_lower
    ```
    `energy_bandpassname` looks like `SPHEREx-D1` … `SPHEREx-D6` (detector). `time_bounds_lower` is MJD.
- **Cutouts:** append `?center=RA,DECd&size=DEG` to the image URL. Never download full frames (2040×2040, about 6 arcsec/pixel, roughly 3.5°). Fetch in parallel (a ThreadPoolExecutor with about 8–10 workers worked in the tutorial). Retry on TimeoutError, HTTPError, IncompleteRead and RemoteDisconnected.
- **Each file is multi-extension FITS:** `IMAGE` (MJy/sr, zodiacal light NOT subtracted), `FLAGS` (int32 bitmask), `VARIANCE`, `ZODI` (zodi model: subtract it), `EPSF`, `WCS-WAVE`.
  - Sky WCS is TAN-SIP in the IMAGE header.
  - Wavelength per pixel: `WCS(hdul['IMAGE'].header, fobj=hdul, key='W')`, set `.sip = None`, then `pixel_to_world(x, y)` returns (wavelength, bandwidth) in µm.
  - Flag bits include 0 TRANSIENT, 1 OVERFLOW, 6 NONFUNC, 9 MISSING_DATA, 10 HOT, 11 COLD, 15 NONLINEAR, 17 PERSIST, 19 OUTLIER, 20 CROSSTALK, 21 SOURCE (benign, means "a source is here"), 22–24 GHOST*, 25 STREAK, 26 BLOOM, 27 SNOWBALL, 28 HALO. Read names from the FLAGS header (`MSKNnnnn`/`MSKDnnnn`) rather than hard-coding.
- **Key physics: SPHEREx has no 102 discrete filters.** It has 6 detectors, each with a linear variable filter: wavelength changes across the detector, and the sky is scanned so each point gets imaged at many wavelengths. So one cutout = one detector, one wavelength (for that position), one time.
  - Bands: D1 0.75–1.09, D2 1.10–1.62, D3 1.63–2.41, D4 2.42–3.82, D5 3.83–4.41, D6 4.42–5.00 µm.
  - The whole sky is covered once per ~6 months, so a position has a few **survey epochs** since May 2025, each containing many exposures spread over days.
- Pixel scale (~6″) is coarse. Barnard's Star (~10″/yr) and WISE 0855 (~8″/yr) shift about 1–2 pixels between epochs a year apart. The display must make sub-pixel to 2-pixel shifts obvious: nearest-neighbour upscaling, zoom, crosshair, difference images.

## Architecture
Libraries: use astropy, astroquery, reproject and numpy for FITS, WCS and alignment. Don't reimplement them.

### Backend: a stateless processing pipe (stores nothing)
- Python + FastAPI. It fetches cutouts from IRSA, processes them **in memory**, streams the result to the browser, and discards it. **No disk cache, no database, no stored images.** The only exception is a small in-memory LRU (a few minutes TTL) that de-duplicates identical in-flight requests.
- Pipeline: query → cutouts → subtract ZODI → mask bad flags (configurable) → reproject all onto ONE common TAN grid centred on the target → group → combine (inverse-variance weighted mean, honouring masks) → output.
- **Output compact data, not PNGs:** for each frame, send a 16-bit quantised array (store scale/offset in metadata) or float16, plus JSON metadata (time, epoch, detector, wavelength at target, number of exposures combined, flag summary, release version). The browser does the stretching, so contrast changes never need a new request.
- **Stream frames as they're ready** (NDJSON or Server-Sent Events, or per-frame endpoints the frontend calls in parallel). The first epoch should be on screen before the rest finish.
- Endpoints, for example:
  - `GET /api/images?ra&dec&fov` → metadata list only (fast)
  - `GET /api/frames?ra&dec&fov&group=epoch|detector|single&detectors=…` → streamed frames
  - `GET /api/spectrum?ra&dec` → wavelength vs flux at the position
- Send `Cache-Control: public, max-age=…, immutable` on responses (the release version is in the URL), so the normal browser HTTP cache helps too.

### Frontend: does display work and caching
- Plain static HTML/JS (or a small Vite app), no heavy framework. Draw frames with Canvas 2D or WebGL. Stretch, blink, sum and difference are all computed client-side from the cached arrays.
- **Browser cache in IndexedDB:**
  - Key: `release + ra + dec + fov + group + detector`. A new data release changes the key, so stale data is ignored automatically.
  - Size cap ~500 MB with LRU eviction. Call `navigator.storage.persist()`. Show cache size and a "clear cache" button in settings.
  - Treat the cache as a speed-up only. Browsers (Safari especially) may evict it, and the app must work normally when it's empty.
- Sensible defaults for speed: FOV 5′ (allowed 1′–30′, warn above 15′ that it'll be slower); start with 1–2 detectors (e.g. D6 + D2) instead of all six, and let users add more.

### Ops
- One config file lists collections/releases and the IRSA endpoints, so a new data release is a one-line change (or automatic via runtime discovery).
- Provide `docker-compose` or a single `uvicorn` command, plus a README for deploying (e.g. Render/Fly/any VM). The backend is stateless, so it can scale horizontally.

## Features
### Phase 0: measure before building (report the numbers in PLAN.md)
- Run one real TAP query and download a few real cutouts. Record:
  - actual column names and headers
  - **bytes per cutout** at 2′, 5′ and 15′, and how much of that is the EPSF extension
  - latency per cutout
  - how many images a typical position has per detector
- **Check CORS:** do IRSA TAP and the cutout service send `Access-Control-Allow-Origin` for browser requests? If yes, note it as a possible future "no backend" mode, but still build the backend design above for v1. Downloading 80+ raw cutouts per detector into each browser is too heavy.
- If something differs from this prompt, trust the live data and note it.

### Phase 1: data layer (no UI)
- CLI/script: given RA/Dec/FOV, list matching images and write aligned, grouped frames plus a JSON metadata file (for testing only; the server itself stores nothing).
- Unit tests for coordinate parsing (sexagesimal like `08h55m10.831s -7d14m42.53s`, decimal, and object names resolved via Sesame/SIMBAD with `SkyCoord.from_name`).
- **Proper motion:** when an object name is given, get its proper motion from SIMBAD (`pmra`, `pmdec`, plus parallax if available) and propagate the position to the SPHEREx epochs (`SkyCoord.apply_space_motion`). Centre the cutouts on it, and optionally draw the predicted track. Catalogue coordinates are often J2000 epoch, and Barnard's Star has moved about 4.5′ since then.

### Phase 2: the blink viewer (MVP, must be solid)
- Input box: coordinates or name, FOV selector, detector/band selector.
- Grouping modes:
  - (a) per survey epoch (coadd all exposures of a ~6-month pass; best S/N; the default)
  - (b) per detector per epoch
  - (c) every single exposure in time order
- Animation: play/pause, speed slider, step forward/back, keyboard shortcuts, and loop or "A/B flip" between two chosen frames.
- Display: contrast/stretch sliders (asinh, linear, zscale), invert, zoom with crisp pixels, crosshair on the target, and a pixel readout (RA/Dec, flux, wavelength, flags).
- Frame label overlay: date, epoch, detector, wavelength range, number of exposures combined.
- **URL state:** `?ra=…&dec=…&fov=…&band=…&mode=…` restores the exact view, so it can be embedded or shared. Also provide an `<iframe>` embed snippet.
- Progress indicator showing frames arriving.

### Phase 3: discovery helpers
- Difference image (epoch B − epoch A) and "sum all epochs" mode, computed in the browser. Moving sources show as a dipole (dark/bright pair).
- Flag overlay: toggle a translucent mask per flag type. Show a per-frame contamination summary (ghosts, streaks, persistence) so users don't "discover" artifacts.
- Light curve: flux at the target vs time per detector (catches variable stars).

### Phase 4: spectra (bonus)
- Click a pixel → on-the-fly spectrum: aperture or forced photometry at that position in every cutout, plotted against each image's wavelength (from the W WCS) and coloured by epoch. Use the EPSF extension for PSF photometry if time permits. Link to IRSA's official Spectrophotometry Tool for comparison.
- Optional: "colour composite" using three wavelength ranges as RGB.

## Test targets (acceptance criteria)
Resolve names with SIMBAD and propagate proper motion. The tool must visibly show:
1. **Barnard's Star**: clear motion between epochs (a dipole in difference mode).
2. **Epsilon Indi B** (brown dwarf pair near the star Epsilon Indi A): visible near the bright primary, with motion shared by both.
3. **WISE 0855−0714**: RA 08h55m10.831s, Dec −07°14′42.53″ (as given; check its epoch and propagate). A very faint, very red, fast mover: it should be visible mainly in the long-wavelength detectors (D5/D6). If it isn't detectable, the tool must say so rather than fail.
4. **RW Aurigae** and **R Draconis**: brightness changes between epochs in the light curve and blink.

Write an automated smoke test that runs the pipeline on all five and saves a GIF and a difference image for each into `examples/`. Look at the outputs yourself and report honestly what is and isn't visible. Also report time-to-first-frame and total load time for each target, cold (empty browser cache) and warm.

## Engineering rules
- Be polite to IRSA: cap concurrency (about 8–10 parallel cutouts per request, plus a global cap across users), set timeouts.
- Keep the code simple and documented. This is for the public and for judges, so the README should explain what SPHEREx is, how to use the tool, and include screenshots or GIFs.
- Credit: include the SPHEREx/IRSA acknowledgement from the IRSA mission page, and credit SIMBAD/CDS for name resolution.
- At the end, list what works, what doesn't, and suggested next steps.
