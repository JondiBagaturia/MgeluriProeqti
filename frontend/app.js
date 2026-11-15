/**
 * SPHEREx Blink — frontend application logic.
 *
 * Talks to the FastAPI backend's /api/resolve, /api/images and (streaming)
 * /api/frames endpoints. All stretching/blinking/zoom happens here, client
 * side, on cached typed arrays — the backend is only asked once per frame
 * set, never again just because the user moved a contrast slider.
 */
(function () {
  "use strict";

  const API_BASE = window.SPHEREX_API_BASE || "";
  const INVALID_SENTINEL = 65535;

  // ---------------------------------------------------------------------
  // DOM
  // ---------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const form = $("search-form");
  const positionInput = $("position-input");
  const resolveStatus = $("resolve-status");
  const fovSlider = $("fov-slider");
  const fovValue = $("fov-value");
  const fovWarning = $("fov-warning");
  const modeSelect = $("mode-select");
  const detectorFieldset = $("detector-fieldset");
  const qualitySlider = $("quality-slider");
  const qualityValue = $("quality-value");
  const shiftstackCheckbox = $("shiftstack-checkbox");
  const stCheckbox = $("st-checkbox");
  const QUALITY_LEVELS = ["low", "mid", "high"];
  const QUALITY_LABELS = ["Low (fastest)", "Mid (balanced)", "High (full quality)"];
  const stretchSelect = $("stretch-select");
  const asinhNote = $("asinh-note");
  const blackSlider = $("black-slider");
  const whiteSlider = $("white-slider");
  const blackValue = $("black-value");
  const whiteValue = $("white-value");
  const invertCheckbox = $("invert-checkbox");
  const zoomSlider = $("zoom-slider");
  const zoomValue = $("zoom-value");
  const cacheStatus = $("cache-status");
  const clearCacheBtn = $("clear-cache-btn");
  const shareUrlInput = $("share-url");
  const embedSnippet = $("embed-snippet");
  const progressWrap = $("progress-wrap");
  const progressBar = $("progress-bar");
  const progressText = $("progress-text");
  const progressAction = $("progress-action");
  const canvas = $("viewer-canvas");
  const canvasCtx = canvas.getContext("2d");
  const frameLabel = $("frame-label");
  const pixelReadout = $("pixel-readout");
  const emptyState = $("empty-state");
  const stepBackBtn = $("step-back-btn");
  const playBtn = $("play-btn");
  const stepFwdBtn = $("step-fwd-btn");
  const speedSlider = $("speed-slider");
  const loopCheckbox = $("loop-checkbox");
  const abCheckbox = $("ab-checkbox");
  const frameSlider = $("frame-slider");
  const viewModeSelect = $("view-mode-select");
  const abPick = $("ab-pick");
  const diffASlider = $("diff-a-slider");
  const diffBSlider = $("diff-b-slider");
  const contaminationSummary = $("contamination-summary");
  const lightcurveCanvas = $("lightcurve-canvas");
  const lightcurveCtx = lightcurveCanvas.getContext("2d");
  const spectrumCanvas = $("spectrum-canvas");
  const spectrumCtx = spectrumCanvas.getContext("2d");
  const spectrumStatus = $("spectrum-status");
  const detectorViewSelect = $("detector-view-select");
  // Bump when the frame wire format changes, so stale IndexedDB entries
  // (e.g. frames without grid WCS / photometry) are never revived.
  const FRAME_SCHEMA = "v7"; // v7: frames carry saturation_mask_b64
  // Click within this distance of the predicted target position -> spectrum
  // follows the target's proper motion per exposure (2 native pixels).
  const TRACK_CLICK_RADIUS_ARCSEC = 12.3;

  const stage = $("stage");
  const crosshair = $("crosshair");
  const compass = $("compass");
  const overlay = $("overlay-canvas");
  const overlayCtx = overlay.getContext("2d");
  const sourcesBtn = $("sources-btn");
  const spectroBtn = $("spectro-btn");
  const fillCheckbox = $("fill-checkbox");
  const fullscreenBtn = $("fullscreen-btn");

  // ---------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------
  const state = {
    allFrames: [], // every frame received, all detectors
    frames: [], // the subset currently shown (one detector series), time-ordered: {groupIndex, timeMjdMean, detectors, nExposures, centralWavelengthUm, bandwidthUm, contamination, width, height, scale, offset, data: Uint16Array}
    currentIndex: 0,
    playing: false,
    playTimer: null,
    resolvedCoord: null, // {ra, dec} at the catalogue reference epoch (J2000) - NOT the grid centre
    resolved: null, // full /api/resolve response
    analysis: null, // end-of-stream {"type":"analysis"} record
    selectedRank: null, // rank of the tracked source selected in the table (current series)
    query: "",
    fovArcmin: 5,
    releaseTag: "unknown",
    abFrames: [0, 0],
    abPhase: 0,
    lastRenderedGrid: null, // {width, height, values: Float32Array (NaN = no data)}
    globalFiniteValues: null, // flat array across ALL loaded frames, for Asinh's fixed levels
    searchGeneration: 0, // bumped by every new search; stale streams/retries check it and stop
    retryTimer: null, // countdown to the next automatic attempt while the archive is down
    loading: "idle", // "busy" from Go until the search ends, "stopped" after Stop, else "idle"
    abort: null, // AbortController of the search in flight (its resolve + frame requests)
    pillTimer: null, // pending "hide the progress bar" timeout
  };

  // ---------------------------------------------------------------------
  // Utilities
  // ---------------------------------------------------------------------
  function mjdToDate(mjd) {
    const MJD_EPOCH_MS = Date.UTC(1858, 10, 17);
    return new Date(MJD_EPOCH_MS + mjd * 86400000);
  }

  function fmtDate(mjd) {
    if (mjd == null) return "?";
    return mjdToDate(mjd).toISOString().slice(0, 10);
  }

  function base64ToUint16Array(b64) {
    const binary = atob(b64);
    const len = binary.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
    return new Uint16Array(bytes.buffer);
  }

  function selectedDetectors() {
    return Array.from(detectorFieldset.querySelectorAll("input[type=checkbox]:checked")).map((el) => el.value);
  }

  function setDetectors(list) {
    const set = new Set(list);
    detectorFieldset.querySelectorAll("input[type=checkbox]").forEach((el) => {
      el.checked = set.has(el.value);
    });
  }

  // ---------------------------------------------------------------------
  // URL state
  // ---------------------------------------------------------------------
  function readUrlState() {
    const params = new URLSearchParams(location.search);
    const quality = params.get("quality") || "mid";
    return {
      position: params.get("position") || "",
      fov: parseFloat(params.get("fov")) || 5,
      detectors: (params.get("detectors") || "D2,D6").split(",").filter(Boolean),
      mode: params.get("mode") || "detector_epoch",
      quality: QUALITY_LEVELS.includes(quality) ? quality : "mid",
      shiftstack: params.get("shiftstack") === "1",
      spectrotemporal: params.get("st") === "1",
    };
  }

  function writeUrlState({ position, fov, detectors, mode, quality, shiftstack, spectrotemporal }) {
    const params = new URLSearchParams();
    params.set("position", position);
    params.set("fov", fov);
    params.set("detectors", detectors.join(","));
    params.set("mode", mode);
    params.set("quality", quality);
    if (shiftstack) params.set("shiftstack", "1");
    if (spectrotemporal) params.set("st", "1");
    const url = `${location.pathname}?${params.toString()}`;
    history.replaceState(null, "", url);
    const fullUrl = location.origin + url;
    shareUrlInput.value = fullUrl;
    embedSnippet.value = `<iframe src="${fullUrl}" width="600" height="700" loading="lazy"></iframe>`;
  }

  // ---------------------------------------------------------------------
  // Form wiring
  // ---------------------------------------------------------------------
  fovSlider.addEventListener("input", () => {
    fovValue.textContent = `${fovSlider.value}′`;
    fovWarning.hidden = parseFloat(fovSlider.value) <= 15;
  });
  qualitySlider.addEventListener("input", () => {
    qualityValue.textContent = QUALITY_LABELS[parseInt(qualitySlider.value, 10)];
  });
  blackSlider.addEventListener("input", () => {
    blackValue.textContent = `${blackSlider.value}%`;
    renderView();
  });
  whiteSlider.addEventListener("input", () => {
    whiteValue.textContent = `${whiteSlider.value}%`;
    renderView();
  });
  function updateStretchControlsEnabled() {
    const isAsinh = stretchSelect.value === "asinh";
    blackSlider.disabled = isAsinh;
    whiteSlider.disabled = isAsinh;
    asinhNote.hidden = !isAsinh;
  }
  stretchSelect.addEventListener("change", () => {
    updateStretchControlsEnabled();
    renderView();
  });
  updateStretchControlsEnabled();
  invertCheckbox.addEventListener("change", renderView);
  // The canvas holds exactly one canvas pixel per grid pixel; its on-screen
  // size is pure CSS (crisp nearest-neighbour scaling via
  // image-rendering: pixelated). Zoom 1× fits the whole field inside the
  // window - or covers the window when "Fill the screen" is ticked - and
  // higher zoom magnifies about the centre of the field.
  function layoutCanvas() {
    const rect = stage.getBoundingClientRect();
    const fit = fillCheckbox.checked ? Math.max(rect.width, rect.height) : Math.min(rect.width, rect.height);
    const size = Math.round(fit * parseFloat(zoomSlider.value));
    canvas.style.width = `${size}px`;
    canvas.style.height = `${Math.round((size * canvas.height) / canvas.width)}px`;
    drawOverlay();
  }

  // ---------------------------------------------------------------------
  // Overlay: everything drawn ON TOP of the image - the crosshair at the
  // target's predicted position in this frame, markers for tracked
  // sources, shift-and-stack candidates. The image canvas is one pixel per
  // grid pixel and only scaled by CSS, so strokes drawn into it would be
  // as blocky as the image; these go on a second canvas that covers the
  // stage at screen resolution instead.
  // ---------------------------------------------------------------------
  const LABEL_COLORS = { candidate: "#f0b64e", catalogued: "#8b9bea", uncertain: "#8f8160", unverified: "#c0a2ea" };
  const SELECTED_COLOR = "#8fe6a0";

  /** Returns a function mapping a 0-based grid ARRAY position (row 0 is the
   * south edge - the display is north-up, see renderGrid) to CSS pixels in
   * the stage, plus the on-screen size of one grid pixel. */
  function gridToStageMapper(grid) {
    const c = canvas.getBoundingClientRect();
    const s = stage.getBoundingClientRect();
    const sx = c.width / grid.width;
    const sy = c.height / grid.height;
    const toStage = (x, y) => [c.left - s.left + (x + 0.5) * sx, c.top - s.top + (grid.height - 1 - y + 0.5) * sy];
    return { toStage, pixel: sx };
  }

  /** False-colour composite: grid pixels with no usable value in some channel
   * get a diagonal hatch ("/"), saturation-flagged ones a cross ("✕"), drawn
   * at screen resolution over the grey the composite already put there. */
  function drawRgbMaskOverlay(grid, toStage, pixel) {
    if (pixel < 4) return;   // too small to hatch; the grey checker in the image still shows them
    const m = grid.rgb.masked, W = grid.width, h = pixel / 2;
    overlayCtx.strokeStyle = "rgba(0, 0, 0, 0.75)";
    overlayCtx.lineWidth = 1;
    overlayCtx.beginPath();
    for (let i = 0; i < m.length; i++) {
      if (!m[i]) continue;
      const ay = Math.floor(i / W), ax = i - ay * W;
      const [cx, cy] = toStage(ax, ay);
      overlayCtx.moveTo(cx - h, cy + h); overlayCtx.lineTo(cx + h, cy - h);
      if (m[i] === 2) { overlayCtx.moveTo(cx - h, cy - h); overlayCtx.lineTo(cx + h, cy + h); }
    }
    overlayCtx.stroke();
  }

  function drawOverlay() {
    const s = stage.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(s.width * dpr));
    const h = Math.max(1, Math.round(s.height * dpr));
    if (overlay.width !== w) overlay.width = w;
    if (overlay.height !== h) overlay.height = h;
    overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    overlayCtx.clearRect(0, 0, s.width, s.height);

    const grid = state.lastRenderedGrid;
    if (!grid || state.frames.length === 0) {
      crosshair.hidden = true;
      compass.hidden = true;
      return;
    }
    const { toStage, pixel } = gridToStageMapper(grid);

    // Crosshair at the target's PREDICTED position in this frame (follows
    // proper motion); falls back to the grid centre. A CSS element
    // (#crosshair in style.css): gapped, red, constant size - the same look
    // as the planet-x renderer's blink.py (fill=(255,80,80)).
    const tp = grid.frame && grid.frame.targetPixel ? grid.frame.targetPixel : { x: (grid.width - 1) / 2, y: (grid.height - 1) / 2 };
    const [tx, ty] = toStage(tp.x, tp.y);
    crosshair.style.left = `${tx}px`;
    crosshair.style.top = `${ty}px`;
    crosshair.hidden = false;
    compass.hidden = false; // the display is always north up, east left

    const viewMode = viewModeSelect.value;
    overlayCtx.font = '15px "Almendra", Georgia, serif';
    overlayCtx.lineJoin = "round";
    overlayCtx.textBaseline = "middle";

    if (viewMode === "shiftstack") {
      // Moving candidates of the shift-and-stack search.
      const e = currentShiftStack();
      overlayCtx.strokeStyle = SELECTED_COLOR;
      overlayCtx.lineWidth = 1.5;
      for (const c of (e && e.blind && e.blind.candidates) || []) {
        if (!c.moving) continue;
        const [cx, cy] = toStage(c.x, c.y);
        overlayCtx.beginPath();
        overlayCtx.arc(cx, cy, Math.max(9, 3 * pixel), 0, 2 * Math.PI);
        overlayCtx.stroke();
      }
      return;
    }

    // Tracked sources (not on the mean image: a mover has no single place there).
    const series = currentSeriesAnalysis();
    if (!series || viewMode === "sum") return;
    if (viewMode === "rgb" && grid.rgb) drawRgbMaskOverlay(grid, toStage, pixel);
    // False colour: the markers of the selected detector series in the epoch shown.
    const frameIdx = viewMode === "blink" ? state.currentIndex : viewMode === "rgb" ? rgb.overlayFrameIdx : null;
    if (viewMode === "rgb" && frameIdx == null) return;   // that series has no frame in this epoch
    const sel = selectedTrack();

    if (showTracksCheckbox.checked) {
      for (const t of series.tracks) {
        if (sel && t.rank === sel.rank) continue;
        const d = frameIdx == null ? t.detections[0] : t.detections.find((q) => q.frame_index === frameIdx);
        if (!d) continue;
        const [cx, cy] = toStage(d.x, d.y);
        const radius = Math.max(7, 2.2 * pixel);
        overlayCtx.strokeStyle = LABEL_COLORS[t.label] || "#fff";
        overlayCtx.lineWidth = 1.25;
        overlayCtx.beginPath();
        overlayCtx.arc(cx, cy, radius, 0, 2 * Math.PI);
        overlayCtx.stroke();
        // Rank number beside the ring, outlined in black so it reads on stars too.
        overlayCtx.fillStyle = overlayCtx.strokeStyle;
        overlayCtx.lineWidth = 3;
        overlayCtx.strokeStyle = "rgba(0, 0, 0, 0.85)";
        overlayCtx.strokeText(String(t.rank), cx + radius + 3, cy - radius * 0.5);
        overlayCtx.fillText(String(t.rank), cx + radius + 3, cy - radius * 0.5);
      }
    }
    if (sel) {
      // Measured path across all epochs, then the detection in this frame.
      overlayCtx.strokeStyle = SELECTED_COLOR;
      overlayCtx.lineWidth = 1.5;
      overlayCtx.beginPath();
      sel.detections.forEach((d, i) => {
        const [cx, cy] = toStage(d.x, d.y);
        if (i === 0) overlayCtx.moveTo(cx, cy);
        else overlayCtx.lineTo(cx, cy);
      });
      overlayCtx.stroke();
      for (const d of sel.detections) {
        const [cx, cy] = toStage(d.x, d.y);
        const here = frameIdx != null && d.frame_index === frameIdx;
        overlayCtx.lineWidth = here ? 2.5 : 1;
        overlayCtx.beginPath();
        overlayCtx.arc(cx, cy, here ? Math.max(10, 3 * pixel) : 3, 0, 2 * Math.PI);
        overlayCtx.stroke();
      }
    }
  }
  zoomSlider.addEventListener("input", () => {
    zoomValue.textContent = `${zoomSlider.value}×`;
    layoutCanvas();
  });
  fillCheckbox.addEventListener("change", layoutCanvas);
  window.addEventListener("resize", layoutCanvas);
  stage.addEventListener(
    "wheel",
    (ev) => {
      if (state.frames.length === 0) return;
      ev.preventDefault();
      const step = ev.deltaY < 0 ? 0.25 : -0.25;
      const next = Math.min(parseFloat(zoomSlider.max), Math.max(parseFloat(zoomSlider.min), parseFloat(zoomSlider.value) + step));
      zoomSlider.value = next;
      zoomSlider.dispatchEvent(new Event("input"));
    },
    { passive: false }
  );
  fullscreenBtn.addEventListener("click", () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else if (document.documentElement.requestFullscreen) document.documentElement.requestFullscreen().catch(() => {});
  });
  layoutCanvas();
  blackValue.textContent = `${blackSlider.value}%`;
  whiteValue.textContent = `${whiteSlider.value}%`;

  clearCacheBtn.addEventListener("click", async () => {
    await window.SpherexCache.clearAll();
    updateCacheStatus();
  });

  async function updateCacheStatus() {
    const bytes = await window.SpherexCache.totalSizeBytes();
    cacheStatus.textContent = `${(bytes / (1024 * 1024)).toFixed(1)} MB cached locally`;
  }

  // ---------------------------------------------------------------------
  // Search / streaming
  // ---------------------------------------------------------------------
  form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    runSearch();
  });

  /** Banner whenever the data shown are the locally downloaded copy
   * (app/localdata.py) rather than a live IRSA query. null hides it. */
  function showDataSource(note, field) {
    if (!note) {
      dataSourceBanner.hidden = !state.offlineModeNote;
      dataSourceBanner.textContent = "";
      if (state.offlineModeNote) dataSourceBanner.appendChild(el("span", state.offlineModeNote));
      return;
    }
    dataSourceBanner.textContent = "";
    dataSourceBanner.appendChild(el("strong", "Local data. "));
    dataSourceBanner.appendChild(el("span", note));
    dataSourceBanner.hidden = false;
  }

  async function checkDataMode() {
    try {
      const r = await fetch(`${API_BASE}/api/local-data`);
      if (!r.ok) return;
      const d = await r.json();
      if (d.mode === "cached") {
        const names = (d.fields || []).map((f) => `${f.query || f.name} (${f.detectors.join("+")})`).join(", ");
        state.offlineModeNote = `Offline mode (SPHEREX_DATA_MODE=cached): no network; only the downloaded fields can be searched: ${names || "none found"}.`;
        showDataSource(null);
      }
    } catch (e) {
      /* older backend without /api/local-data: nothing to show */
    }
  }
  checkDataMode();

  async function fetchReleaseTag() {
    try {
      const resp = await fetch(`${API_BASE}/api/releases`);
      if (!resp.ok) return "unknown";
      const data = await resp.json();
      return (data.collections || []).slice().sort().join(",") || "unknown";
    } catch (e) {
      return "unknown";
    }
  }

  async function runSearch() {
    const position = positionInput.value.trim();
    if (!position) return;
    const fov = parseFloat(fovSlider.value);
    const detectors = selectedDetectors();
    const mode = modeSelect.value;
    const quality = QUALITY_LEVELS[parseInt(qualitySlider.value, 10)];
    const shiftstack = shiftstackCheckbox.checked;
    const spectrotemporal = stCheckbox.checked;

    if (detectors.length === 0) {
      resolveStatus.textContent = "Pick at least one detector.";
      resolveStatus.className = "status-text error";
      return;
    }

    cancelRetry();
    if (state.abort) state.abort.abort(); // a search still in flight is replaced by this one
    const controller = new AbortController();
    state.abort = controller;
    const generation = ++state.searchGeneration; // invalidates any earlier stream or pending retry
    setLoading("busy");
    progressWrap.hidden = false;
    progressBar.style.width = "0%";
    progressText.textContent = "Resolving the position…";
    stopPlayback();
    state.frames = [];
    state.allFrames = [];
    state.analysis = null;
    showDataSource(null);
    state.selectedRank = null;
    state.currentIndex = 0;
    detectorViewSelect.innerHTML = "";
    renderAnalysis();
    frameSlider.max = 0;
    frameSlider.value = 0;
    emptyState.hidden = true;
    frameLabel.textContent = "";
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
    state.lastRenderedGrid = null;
    drawOverlay();

    resolveStatus.textContent = "Resolving…";
    resolveStatus.className = "status-text";

    let resolved;
    try {
      const resp = await fetch(`${API_BASE}/api/resolve?position=${encodeURIComponent(position)}`, { signal: controller.signal });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || resp.statusText);
      }
      resolved = await resp.json();
    } catch (e) {
      if (generation !== state.searchGeneration) return; // stopped, or replaced by a newer search
      resolveStatus.textContent = `Could not resolve: ${e.message}`;
      resolveStatus.className = "status-text error";
      progressWrap.hidden = true;
      setLoading("idle");
      return;
    }
    if (generation !== state.searchGeneration) return;

    state.resolvedCoord = { ra: resolved.ra_deg, dec: resolved.dec_deg };
    state.resolved = resolved;
    state.query = position;
    state.fovArcmin = fov;

    const epochYear = resolved.ref_epoch_mjd != null ? (2000 + (resolved.ref_epoch_mjd - 51544.5) / 365.25).toFixed(1) : "2000.0";
    let statusMsg = `${resolved.name || position} → RA ${resolved.ra_deg.toFixed(5)}, Dec ${resolved.dec_deg.toFixed(5)} (position at epoch ${epochYear})`;
    if (resolved.has_proper_motion) {
      statusMsg +=
        ` · proper motion (${resolved.pmra_mas_yr.toFixed(0)}, ${resolved.pmdec_mas_yr.toFixed(0)}) mas/yr: ` +
        `the field is centred on the mean-epoch position and the crosshair follows the predicted position in each frame.`;
    }
    if (resolved.note) statusMsg += ` · ${resolved.note}`;
    resolveStatus.textContent = statusMsg;
    resolveStatus.className = "status-text";

    writeUrlState({ position, fov, detectors, mode, quality, shiftstack, spectrotemporal });

    if (state.releaseTag === "unknown") {
      state.releaseTag = await fetchReleaseTag();
      if (generation !== state.searchGeneration) return;
    }

    const cacheKey = window.SpherexCache.makeKey({
      release: state.releaseTag,
      ra: resolved.ra_deg,
      dec: resolved.dec_deg,
      fov,
      group: `${FRAME_SCHEMA}:${mode}:${quality}:ss${shiftstack ? 1 : 0}:st${spectrotemporal ? 1 : 0}`, // quality changes the actual pixel content, so it must be part of the key
      detector: detectors.slice().sort().join(""),
    });

    progressText.textContent = "Checking local cache…";

    const cached = await window.SpherexCache.get(cacheKey);
    if (generation !== state.searchGeneration) return;
    if (cached && cached.length) {
      setLoading("idle");
      const cachedAnalysis = cached.find((c) => c && c.__analysis);
      state.analysis = cachedAnalysis ? cachedAnalysis.__analysis : null;
      state.allFrames = cached.filter((c) => !(c && c.__analysis)).map(reviveFrame);
      const localFrame = state.allFrames.find((f) => f.dataSource === "local_cache");
      if (localFrame) showDataSource(localFrame.dataSourceNote);
      applyDetectorFilter(false);
      progressText.textContent = `Loaded ${state.allFrames.length} frame(s) from local cache`;
      progressBar.style.width = "100%";
      hideProgressLater(1500);
      return; // cache hit: no network call for pixels at all
    }

    progressText.textContent = "Fetching from IRSA…";
    streamFrames({ position, fov, detectors, mode, quality, cacheKey, shiftstack, spectrotemporal }, generation, 1);
  }

  // ---------------------------------------------------------------------
  // Stop / Start again
  //
  // A search can take minutes when IRSA is slow. While one is in flight
  // (resolving, downloading, retrying, or waiting for the source analysis)
  // the progress bar carries a Stop button: it abandons the search - the
  // browser cancels its requests, and the server then drops the downloads
  // it has not started (backend/app/irsa.py, fetch_cutouts). Frames that
  // already arrived stay on screen. The button then reads "Start again",
  // which runs the search in the form afresh (so does Go).
  // ---------------------------------------------------------------------
  function setLoading(phase) {
    state.loading = phase;
    progressAction.hidden = phase === "idle";
    progressAction.textContent = phase === "stopped" ? "Start again" : "Stop";
    progressWrap.classList.toggle("has-action", phase !== "idle");
    if (phase !== "idle") {
      clearTimeout(state.pillTimer); // a "hide" left over from the previous search
      state.pillTimer = null;
    }
  }

  function hideProgressLater(ms) {
    clearTimeout(state.pillTimer);
    state.pillTimer = setTimeout(() => {
      state.pillTimer = null;
      if (state.loading === "idle") progressWrap.hidden = true;
    }, ms);
  }

  function stopLoading() {
    cancelRetry();
    state.searchGeneration++; // whatever is still running now belongs to a dead search
    if (state.abort) state.abort.abort();
    state.abort = null;
    emptyState.hidden = true;
    progressWrap.hidden = false;
    progressText.textContent = state.allFrames.length ? `Stopped: ${state.allFrames.length} frame(s) loaded` : "Stopped";
    setLoading("stopped");
    renderAnalysis();
  }

  progressAction.addEventListener("click", () => {
    if (state.loading === "busy") stopLoading();
    else runSearch();
  });

  // ---------------------------------------------------------------------
  // Automatic retry while the archive is down
  //
  // IRSA's search service drops out for a while at a time (the server then
  // reports e.g. "SIA query failed: 502 Proxy Error"). A search that ended
  // with an archive error and no frames is repeated quietly: the progress
  // pill stays up, noting which attempt this is, and nothing else appears
  // on screen. Only when every retry has failed does a
  // message come up, with a button to start over. A new search cancels it.
  // ---------------------------------------------------------------------
  const RETRY_DELAYS_S = [3, 5, 8, 12, 15]; // pause before each automatic retry
  const MAX_AUTO_RETRIES = RETRY_DELAYS_S.length; // 5 retries after the first attempt

  function cancelRetry() {
    if (state.retryTimer) {
      clearTimeout(state.retryTimer);
      state.retryTimer = null;
    }
  }

  function showFetchProgress(attempt) {
    emptyState.hidden = true;
    progressWrap.hidden = false;
    progressBar.style.width = "0%";
    // From the second attempt on the bar says so: a slow archive should not
    // look like a frozen page. Still no pop-up, and Stop is right there.
    progressText.textContent =
      attempt > 1 ? `Fetching from IRSA… the archive is slow to answer (attempt ${attempt} of ${MAX_AUTO_RETRIES + 1})` : "Fetching from IRSA…";
  }

  function showArchiveFailure(request, generation, errorText) {
    progressWrap.hidden = true;
    setLoading("idle");
    emptyState.textContent = "";
    const message = document.createElement("span");
    message.textContent =
      `The IRSA archive is not answering right now.\n\nDetails: ${errorText.slice(0, 160)}`;
    const againBtn = document.createElement("button");
    againBtn.type = "button";
    againBtn.className = "text-btn";
    againBtn.textContent = "Try again";
    againBtn.addEventListener("click", () => {
      if (generation !== state.searchGeneration) return;
      setLoading("busy");
      showFetchProgress(1);
      streamFrames(request, generation, 1);
    });
    emptyState.append(message, againBtn);
    emptyState.hidden = false;
  }

  function scheduleRetry(request, generation, attempt, errorText) {
    cancelRetry();
    const retriesDone = attempt - 1;
    if (retriesDone >= MAX_AUTO_RETRIES) {
      showArchiveFailure(request, generation, errorText);
      return;
    }
    showFetchProgress(attempt + 1); // same pill as the first attempt, with a note that it is a retry
    state.retryTimer = setTimeout(() => {
      state.retryTimer = null;
      if (generation !== state.searchGeneration) return;
      streamFrames(request, generation, attempt + 1);
    }, RETRY_DELAYS_S[retriesDone] * 1000);
  }

  function reviveFrame(stored) {
    return { ...stored, data: new Uint16Array(stored.data), gapFilled: new Set(stored.gapFilledIndices || []) };
  }

  async function streamFrames(request, generation, attempt) {
    const { position, fov, detectors, mode, quality, cacheKey, shiftstack, spectrotemporal } = request;
    const url = `${API_BASE}/api/frames?position=${encodeURIComponent(position)}&fov=${fov}&detectors=${detectors.join(",")}&group=${mode}&quality=${quality}${shiftstack ? "&shiftstack=true" : ""}${spectrotemporal ? "&spectrotemporal=true" : ""}&progress=1`;
    let resp;
    try {
      // no-store: the server marks this response cacheable for 5 minutes
      // before it knows how the stream will end, so without this a failed
      // attempt (IRSA down) is replayed from the browser's HTTP cache on
      // every retry. Successful sets are cached in IndexedDB instead.
      resp = await fetch(url, { cache: "no-store", signal: state.abort ? state.abort.signal : undefined });
    } catch (e) {
      if (generation !== state.searchGeneration) return; // stopped, or replaced by a newer search
      progressText.textContent = `Network error: ${e.message}`;
      setLoading("idle");
      return;
    }
    if (generation !== state.searchGeneration) return;
    if (!resp.ok || !resp.body) {
      progressText.textContent = `Server error (${resp.status})`;
      setLoading("idle");
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let nGroups = null;
    let received = 0;
    let firstFrameShown = false;
    let download = null; // latest {"progress": ...} line: images downloaded / failed / total

    // A frame is only complete once every exposure of its epoch is in, which
    // can take minutes - so the bar follows the individual image downloads
    // (when the server reports them) and the text gives both counts.
    const showProgress = () => {
      const framesText = nGroups ? `${received} / ${nGroups} frame(s) ready` : `${received} frame(s) ready`;
      if (download && download.total > 0) {
        const finished = download.downloaded + download.failed;
        const percent = Math.floor((download.downloaded / download.total) * 100);
        progressBar.style.width = `${Math.min(100, (finished / download.total) * 100)}%`;
        let text = `${download.downloaded} / ${download.total} images downloaded (${percent}%)`;
        if (download.failed) text += `, ${download.failed} failed`;
        if (received) text += `, ${framesText}`;
        // The source analysis (detection, tracking, catalogue queries) runs
        // on the server after the last download and can take a while.
        if (finished >= download.total && received && !state.analysis) text = `${framesText}, now detecting and tracking sources…`;
        progressText.textContent = text;
      } else if (received) {
        if (nGroups) progressBar.style.width = `${Math.min(100, (received / nGroups) * 100)}%`;
        progressText.textContent = framesText;
      }
    };
    let streamError = null; // last {"error": ...} line the server sent, if any

    while (true) {
      let value, done;
      try {
        ({ value, done } = await reader.read());
      } catch (e) {
        // Stop pressed (the request was aborted) or the connection dropped.
        if (generation !== state.searchGeneration) return;
        streamError = `connection to the server was lost (${e.message})`;
        break;
      }
      if (generation !== state.searchGeneration) {
        reader.cancel().catch(() => {}); // stopped, or a newer search has started; drop this one
        return;
      }
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newlineIdx;
      while ((newlineIdx = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newlineIdx);
        buffer = buffer.slice(newlineIdx + 1);
        if (!line.trim()) continue;
        let obj;
        try {
          obj = JSON.parse(line);
        } catch (e) {
          continue;
        }
        if (obj.progress) {
          download = obj.progress;
          showProgress();
          continue;
        }
        if (obj.type === "data_source") {
          showDataSource(obj.note, obj.field);
          continue;
        }
        if (obj.type === "analysis") {
          state.analysis = obj;
          renderAnalysis();
          renderView();
          continue;
        }
        if (obj.error) {
          // Not shown here: with no frames it is retried quietly, with
          // some frames it is reported once the stream ends (both below).
          streamError = obj.error;
          continue;
        }
        nGroups = obj.n_groups || nGroups;
        const frame = {
          groupIndex: obj.group_index,
          timeMjdMean: obj.time_mjd_mean,
          timeMjdMin: obj.time_mjd_min,
          timeMjdMax: obj.time_mjd_max,
          detectors: obj.detectors,
          nExposures: obj.n_exposures,
          centralWavelengthUm: obj.central_wavelength_um,
          bandwidthUm: obj.bandwidth_um,
          contamination: obj.contamination,
          releaseVersions: obj.release_versions,
          partial: obj.partial,
          nGapFilledPixels: obj.n_gap_filled_pixels,
          gapFilledIndices: obj.gap_filled_indices || [],
          gapFilled: new Set(obj.gap_filled_indices || []),
          saturationMaskB64: obj.saturation_mask_b64 || null,
          dataSource: obj.data_source || "irsa_live",
          dataSourceNote: obj.data_source_note || "",   // kept packed: survives the IndexedDB cache as-is
          nSaturationFlaggedPixels: obj.n_saturation_flagged_pixels != null ? obj.n_saturation_flagged_pixels : null,
          photometry: obj.photometry || null,
          targetPixel: obj.target_pixel || null,
          gridWcs: obj.grid_wcs || null,
          width: obj.width,
          height: obj.height,
          scale: obj.scale,
          offset: obj.offset,
          data: base64ToUint16Array(obj.payload_b64),
        };
        state.allFrames.push(frame);
        received++;
        showProgress();
        // Frames complete out of chronological order; applyDetectorFilter
        // re-sorts by time and keeps the frame on screen by identity.
        applyDetectorFilter(firstFrameShown);
        firstFrameShown = true;
      }
    }

    if (state.allFrames.length === 0) {
      // Two different situations end with nothing to show; say which one.
      // The server reported a failure (IRSA unreachable or busy): that is
      // not "no images here", and trying again shortly usually works - so
      // that case retries by itself, quietly (see scheduleRetry).
      if (streamError) {
        scheduleRetry(request, generation, attempt, streamError);
        return;
      }
      progressWrap.hidden = true;
      setLoading("idle");
      emptyState.hidden = false;
      emptyState.textContent = "No SPHEREx images found at this position/FOV/detector selection.";
      return;
    }
    setLoading("idle");
    if (streamError) {
      // Keep the error visible: an unreachable archive is not an empty sky.
      progressText.textContent = `Stopped with an error after ${state.allFrames.length} frame(s): ${streamError.slice(0, 160)}`;
      renderAnalysis(); // no analysis record is coming
      return; // never cache an incomplete result
    }
    progressText.textContent = `Done: ${state.allFrames.length} frame(s)`;
    hideProgressLater(2000);
    // Never store a local-copy result (app/localdata.py) under the live key:
    // once IRSA is back, the same search must query it again.
    if (state.allFrames.some((f) => f.dataSource === "local_cache")) return;

    // Cache the whole set now that the stream is complete.
    const serializable = state.allFrames.map(({ gapFilled, ...f }) => ({ ...f, data: f.data.buffer.slice(0) }));
    const approxBytes = serializable.reduce((s, f) => s + f.data.byteLength, 0);
    if (state.analysis) serializable.push({ __analysis: state.analysis });
    window.SpherexCache.put(cacheKey, serializable, approxBytes);
    updateCacheStatus();
  }

  // ---------------------------------------------------------------------
  // Detector series. Frames from different detectors sample very different
  // wavelengths (0.75-5 um), so they are never blinked or differenced
  // against each other: the viewer shows one detector series at a time
  // (docs/AUDIT.md C1).
  // ---------------------------------------------------------------------
  function seriesKey(f) {
    return (f.detectors || []).join("+") || "?";
  }

  function rebuildDetectorOptions() {
    const keys = [...new Set(state.allFrames.map(seriesKey))].sort();
    const existing = Array.from(detectorViewSelect.options).map((o) => o.value);
    if (keys.join("|") !== existing.join("|")) {
      const previous = detectorViewSelect.value;
      detectorViewSelect.innerHTML = "";
      for (const k of keys) {
        const opt = document.createElement("option");
        opt.value = k;
        detectorViewSelect.appendChild(opt);
      }
      if (keys.includes(previous)) detectorViewSelect.value = previous;
    }
    Array.from(detectorViewSelect.options).forEach((o) => {
      o.textContent = `${o.value} (${state.allFrames.filter((f) => seriesKey(f) === o.value).length})`;
    });
  }

  function applyDetectorFilter(keepCurrent) {
    const shown = keepCurrent ? state.frames[state.currentIndex] : null;
    const previousKey = detectorViewSelect.value;
    rebuildDetectorOptions();
    const key = detectorViewSelect.value;
    state.frames = state.allFrames
      .filter((f) => seriesKey(f) === key)
      .sort((a, b) => a.timeMjdMean - b.timeMjdMean);
    const idx = shown ? state.frames.indexOf(shown) : -1;
    state.currentIndex = idx >= 0 ? idx : 0;
    if (key !== previousKey || !keepCurrent) {
      diffASlider.value = 0;
      diffBSlider.value = Math.max(0, state.frames.length - 1);
      state.selectedRank = null;
    }
    updateFrameSlider();
    renderAnalysis();
    renderView();
  }

  detectorViewSelect.addEventListener("change", () => {
    stopPlayback();
    applyDetectorFilter(false);
  });

  function updateFrameSlider() {
    const maxIdx = Math.max(0, state.frames.length - 1);
    frameSlider.max = maxIdx;
    frameSlider.value = state.currentIndex;
    diffASlider.max = maxIdx;
    diffBSlider.max = maxIdx;
    if (diffBSlider.value == 0 && maxIdx > 0) diffBSlider.value = maxIdx; // default B to the latest epoch
    if (state.frames.length) emptyState.hidden = true;
    recomputeGlobalFiniteValues();
    drawLightCurve();
  }

  // ---------------------------------------------------------------------
  // Stretch / rendering
  // ---------------------------------------------------------------------
  // Asinh is a faithful port of a teammate's renderer
  // (github.com/DmaniseliMgeli/planet-x, phase1-check/phase1/blink.py):
  // astropy's AsinhStretch(a=0.05) - y = asinh(x/a)/asinh(1/a), same form
  // as our existing formula with A = 1/a = 20 - with vmin/vmax fixed at
  // the median and 99.8th percentile, computed once across every loaded
  // frame (not per-frame), so contrast stays consistent through the
  // blink instead of jumping epoch to epoch. That global computation
  // lives in recomputeGlobalFiniteValues() below; the black/white
  // sliders are inactive for this mode (their script has no adjustable
  // levels), active as before for Linear/Zscale.
  const ASINH_A = 1 / 0.05; // = 20

  function computeLevels(finiteValues, mode, blackPct, whitePct) {
    if (finiteValues.length === 0) return { black: 0, white: 1 };
    const sorted = Array.from(finiteValues).sort((a, b) => a - b);
    const pct = (p) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor((p / 100) * (sorted.length - 1))))];
    if (mode === "zscale") {
      const median = pct(50);
      const absDev = sorted.map((v) => Math.abs(v - median));
      absDev.sort((a, b) => a - b);
      const mad = absDev[Math.floor(absDev.length / 2)] || 1e-6;
      const sigma = mad * 1.4826;
      return { black: median - 1 * sigma, white: median + 5 * sigma };
    }
    if (mode === "asinh") {
      return { black: pct(50), white: pct(99.8) };
    }
    return { black: pct(blackPct), white: pct(whitePct) };
  }

  function stretchValue(phys, black, white, mode, invert) {
    let t;
    const range = white - black || 1e-9;
    if (mode === "asinh") {
      const lin = (phys - black) / range;
      t = Math.asinh(lin * ASINH_A) / Math.asinh(ASINH_A);
    } else {
      t = (phys - black) / range;
    }
    t = Math.min(1, Math.max(0, t));
    if (invert) t = 1 - t;
    return Math.round(t * 255);
  }

  /** Levels for Asinh mode come from every currently-loaded frame
   * combined, not just the one on screen - recomputed whenever the frame
   * set changes (see updateFrameSlider). */
  function recomputeGlobalFiniteValues() {
    if (state.frames.length === 0) {
      state.globalFiniteValues = null;
      return;
    }
    // Percentiles only need a representative sample; sorting every pixel of
    // every frame on each update was O(N log N) per arriving frame.
    const totalPixels = state.frames.reduce((acc, f) => acc + f.data.length, 0);
    const stride = Math.max(1, Math.floor(totalPixels / 200000));
    const all = [];
    for (const f of state.frames) {
      const v = physicalValuesForFrame(f);
      for (let i = 0; i < v.length; i += stride) if (Number.isFinite(v[i])) all.push(v[i]);
    }
    state.globalFiniteValues = all;
  }

  /** Raw uint16+scale/offset -> physical values, NaN where there's no data. */
  function physicalValuesForFrame(frame) {
    const out = new Float32Array(frame.data.length);
    for (let i = 0; i < frame.data.length; i++) {
      out[i] = frame.data[i] === INVALID_SENTINEL ? NaN : frame.data[i] * frame.scale + frame.offset;
    }
    return out;
  }

  /** Entry point for all three view modes: dispatches, then draws a
   * physical-value grid common to blink/difference/sum so the stretch,
   * zoom, and crosshair code only has to exist once. */
  function renderView() {
    if (state.frames.length === 0) return;
    const viewMode = viewModeSelect.value;

    if (viewMode === "difference") {
      const a = state.frames[Math.min(diffASlider.value, state.frames.length - 1)];
      const b = state.frames[Math.min(diffBSlider.value, state.frames.length - 1)];
      const va = physicalValuesForFrame(a);
      const vb = physicalValuesForFrame(b);
      const diff = new Float32Array(va.length);
      for (let i = 0; i < diff.length; i++) diff[i] = vb[i] - va[i]; // NaN propagates if either side is NaN
      renderGrid(a.width, a.height, diff, { useGlobalAsinhLevels: false, frame: b, wcsFrame: a });
      frameLabel.textContent =
        `Difference (${seriesKey(a)}): ${fmtDate(b.timeMjdMean)}  −  ${fmtDate(a.timeMjdMean)}\n` +
        `A mover shows as a dark/bright pair. Stationary bright stars also leave residuals (no PSF matching).`;
    } else if (viewMode === "shiftstack") {
      renderShiftStackView();
    } else if (viewMode === "rgb") {
      renderRgbView();
    } else if (viewMode === "sum") {
      const first = state.frames[0];
      const sum = new Float32Array(first.width * first.height);
      const count = new Float32Array(sum.length);
      for (const f of state.frames) {
        const v = physicalValuesForFrame(f);
        for (let i = 0; i < v.length; i++) {
          if (Number.isFinite(v[i])) {
            sum[i] += v[i];
            count[i] += 1;
          }
        }
      }
      const mean = new Float32Array(sum.length);
      for (let i = 0; i < sum.length; i++) mean[i] = count[i] > 0 ? sum[i] / count[i] : NaN;
      renderGrid(first.width, first.height, mean, { useGlobalAsinhLevels: false, frame: null, wcsFrame: first });
      frameLabel.textContent = `Mean of all ${state.frames.length} ${seriesKey(first)} frame(s): a persistent faint source stands out, a mover smears out.`;
    } else {
      const frame = state.frames[state.currentIndex];
      renderGrid(frame.width, frame.height, physicalValuesForFrame(frame), { useGlobalAsinhLevels: true, frame, wcsFrame: frame });
      updateFrameLabel(frame);
    }
    updateContaminationSummary();
  }
  function renderGrid(width, height, values, { useGlobalAsinhLevels = true, frame = null, wcsFrame = null } = {}) {
    state.lastRenderedGrid = { width, height, values, frame, wcsFrame };
    const mode = stretchSelect.value;

    // Asinh's levels come from every loaded frame combined (see
    // recomputeGlobalFiniteValues) ONLY for the plain blink view - those
    // are raw per-epoch flux values (roughly 0..300 MJy/sr). Difference
    // and Sum pass useGlobalAsinhLevels:false because their arrays are a
    // different quantity entirely (differences centred near 0, including
    // negatives; sums on a different scale) - using the blink levels on
    // those made the display look badly broken (everything clipped to
    // one extreme, since the white point was calibrated for raw flux, not
    // for small +/- differences). Falls back to this grid's own values if
    // the global cache is somehow empty.
    let physValues;
    if (mode === "asinh" && useGlobalAsinhLevels && state.globalFiniteValues && state.globalFiniteValues.length) {
      physValues = state.globalFiniteValues;
    } else {
      physValues = [];
      for (let i = 0; i < values.length; i++) if (Number.isFinite(values[i])) physValues.push(values[i]);
    }

    const blackPct = parseFloat(blackSlider.value);
    const whitePct = parseFloat(whiteSlider.value);
    const { black, white } = computeLevels(physValues, mode, blackPct, whitePct);
    const invert = invertCheckbox.checked;

    // Array row 0 is the SOUTH edge of the field (grid cdelt2 > 0) and
    // column 0 the EAST edge, so drawing array row (h-1-r) on display row r
    // gives the standard north-up / east-left sky (docs/AUDIT.md C5).
    const rgba = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < values.length; i++) {
      const ay = Math.floor(i / width);
      const ax = i - ay * width;
      const o = ((height - 1 - ay) * width + ax) * 4;
      if (!Number.isFinite(values[i])) {
        rgba[o + 3] = 0;
        continue;
      }
      const g = stretchValue(values[i], black, white, mode, invert);
      rgba[o] = g;
      rgba[o + 1] = g;
      rgba[o + 2] = g;
      rgba[o + 3] = 255;
    }

    // One canvas pixel per grid pixel; on-screen size and zoom are CSS
    // (see layoutCanvas). Setting width/height also clears the canvas.
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    canvasCtx.putImageData(new ImageData(rgba, width, height), 0, 0);
    layoutCanvas();

    // The crosshair, the tracked-source markers and the compass are drawn
    // by drawOverlay (called from layoutCanvas above), at screen resolution,
    // so they keep a constant on-screen size at any zoom.
  }

  function updateContaminationSummary() {
    const viewMode = viewModeSelect.value;
    if (viewMode !== "blink" || state.frames.length === 0) {
      contaminationSummary.textContent = "";
      return;
    }
    const c = state.frames[state.currentIndex].contamination || {};
    const parts = Object.entries(c)
      .filter(([name]) => name !== "SOURCE") // benign - "a source is here", not an artifact
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([name, count]) => `${name} (${count}px)`);
    const ph = state.frames[state.currentIndex].photometry;
    const phStr = ph
      ? ph.status === "ok"
        ? ` Target aperture: ${fmtFlux(ph.flux_mjy)} ± ${fmtFlux(ph.flux_err_mjy)} mJy (S/N ${ph.snr.toFixed(1)}; statistical error only; frames of a constant star differ by ~3-5%).`
        : ` Target aperture: not measured (${ph.status.replace(/_/g, " ")}).`
      : "";
    contaminationSummary.textContent =
      (parts.length ? `Flagged pixels in this frame's cutouts (whole field): ${parts.join(", ")}.` : "No notable flags in this frame.") + phStr;
  }

  function updateFrameLabel(frame) {
    const waveStr =
      frame.centralWavelengthUm != null
        ? `${(frame.centralWavelengthUm - (frame.bandwidthUm || 0) / 2).toFixed(2)}–${(
            frame.centralWavelengthUm + (frame.bandwidthUm || 0) / 2
          ).toFixed(2)} µm`
        : "?";
    const dateRange =
      frame.timeMjdMin === frame.timeMjdMax
        ? fmtDate(frame.timeMjdMin)
        : `${fmtDate(frame.timeMjdMin)}…${fmtDate(frame.timeMjdMax)}`;
    const partial = frame.partial ? "  [partial: some exposures failed to fetch]" : "";
    const gapFilled = frame.nGapFilledPixels
      ? `  [${frame.nGapFilledPixels}px interpolated - SPHEREx flags the peak of bright sources in every exposure]`
      : "";
    frameLabel.textContent =
      `Frame ${state.currentIndex + 1}/${state.frames.length}   ${dateRange}\n` +
      `Detectors: ${frame.detectors.join(",")}   λ ${waveStr}\n` +
      `${frame.nExposures} exposure(s) combined${partial}${gapFilled}` +
      (frame.dataSource === "local_cache" ? "\nData: local copy downloaded earlier (not live IRSA)" : "");
  }

  // ---------------------------------------------------------------------
  // Light curve: background-subtracted aperture flux (mJy, +-1 sigma) at the
  // target's predicted position in each frame of the CURRENT detector
  // series, computed server-side (app/photometry.py) before any display
  // hole-filling. Frames where the measurement was refused (masked or
  // saturated aperture, off-image) are drawn hollow on the baseline, never
  // given an invented value.
  // ---------------------------------------------------------------------
  function fmtFlux(v) {
    if (v == null || !Number.isFinite(v)) return "?";
    const a = Math.abs(v);
    return a >= 100 ? v.toFixed(0) : a >= 1 ? v.toFixed(2) : v.toPrecision(2);
  }

  function drawLightCurve() {
    const w = lightcurveCanvas.width;
    const h = lightcurveCanvas.height;
    lightcurveCtx.clearRect(0, 0, w, h);
    lightcurveCtx.font = '12px "Almendra", Georgia, serif';
    if (state.frames.length === 0) return;

    const points = state.frames.map((f, i) => {
      const ph = f.photometry;
      const ok = ph && ph.status === "ok" && Number.isFinite(ph.flux_mjy);
      return {
        t: f.timeMjdMean,
        flux: ok ? ph.flux_mjy : NaN,
        err: ok ? ph.flux_err_mjy || 0 : 0,
        status: ph ? ph.status : "unavailable",
        i,
      };
    });
    const good = points.filter((p) => Number.isFinite(p.flux));
    if (good.length === 0) {
      lightcurveCtx.fillStyle = "#8f8160";
      const reasons = [...new Set(points.map((p) => p.status))].join(", ").replace(/_/g, " ");
      lightcurveCtx.fillText(`No valid aperture measurement of the target in this detector series (${reasons}).`, 10, h / 2);
      return;
    }

    const padL = 56, padR = 12, padT = 10, padB = 18;
    const tMin = Math.min(...points.map((p) => p.t));
    const tMax = Math.max(...points.map((p) => p.t));
    const lo = Math.min(...good.map((p) => p.flux - p.err));
    const hi = Math.max(...good.map((p) => p.flux + p.err));
    const span = hi - lo || Math.abs(hi) || 1;
    const fMin = lo - 0.08 * span;
    const fMax = hi + 0.08 * span;
    const xOf = (t) => padL + ((t - tMin) / (tMax - tMin || 1)) * (w - padL - padR);
    const yOf = (f) => h - padB - ((f - fMin) / (fMax - fMin || 1)) * (h - padT - padB);

    lightcurveCtx.fillStyle = "#8f8160";
    lightcurveCtx.fillText(`${fmtFlux(fMax)} mJy`, 2, padT + 8);
    lightcurveCtx.fillText(`${fmtFlux(fMin)} mJy`, 2, h - padB);
    lightcurveCtx.fillText(fmtDate(tMin), padL, h - 4);
    const lastLabel = fmtDate(tMax);
    lightcurveCtx.fillText(lastLabel, w - padR - lightcurveCtx.measureText(lastLabel).width, h - 4);

    for (const p of points) {
      const x = xOf(p.t);
      const isCurrent = viewModeSelect.value === "blink" && p.i === state.currentIndex;
      if (!Number.isFinite(p.flux)) {
        lightcurveCtx.strokeStyle = "#6b6048";
        lightcurveCtx.lineWidth = 1;
        lightcurveCtx.beginPath();
        lightcurveCtx.arc(x, h - padB, isCurrent ? 5 : 3.5, 0, 2 * Math.PI);
        lightcurveCtx.stroke();
        continue;
      }
      const color = isCurrent ? "#f3dc9a" : "#8b9bea";
      lightcurveCtx.strokeStyle = color;
      lightcurveCtx.lineWidth = 1;
      lightcurveCtx.beginPath();
      lightcurveCtx.moveTo(x, yOf(p.flux - p.err));
      lightcurveCtx.lineTo(x, yOf(p.flux + p.err));
      lightcurveCtx.stroke();
      lightcurveCtx.fillStyle = color;
      lightcurveCtx.beginPath();
      lightcurveCtx.arc(x, yOf(p.flux), isCurrent ? 4 : 3, 0, 2 * Math.PI);
      lightcurveCtx.fill();
    }
  }

  // ---------------------------------------------------------------------
  // Pixel readout (approximate tangent-plane RA/Dec from the known centre)
  // ---------------------------------------------------------------------
  /** Map a mouse event on #viewer-canvas to {px, py, ra, dec, phys} in the
   * currently-rendered grid, or null if the cursor is outside it / there's
   * nothing rendered yet. Shared by the hover readout and the spectrum
   * click handler below. */
  /** FITS TAN (gnomonic) pixel -> world for the grid WCS the backend sends
   * with every frame. x, y are 0-based ARRAY indices. These grids carry no
   * SIP terms. Checked in the browser against astropy's
   * WCS.pixel_to_world on make_target_wcs grids (2026-11-15). */
  function gridPixelToWorld(wcs, x, y) {
    const D = Math.PI / 180;
    const xi = (x + 1 - wcs.crpix[0]) * wcs.cdelt[0] * D;
    const eta = (y + 1 - wcs.crpix[1]) * wcs.cdelt[1] * D;
    const d0 = wcs.crval[1] * D;
    const denom = Math.cos(d0) - eta * Math.sin(d0);
    let ra = wcs.crval[0] + Math.atan2(xi, denom) / D;
    const dec = Math.atan2(Math.sin(d0) + eta * Math.cos(d0), Math.hypot(xi, denom)) / D;
    ra = ((ra % 360) + 360) % 360;
    return { ra, dec };
  }

  function angularSepArcsec(ra1, dec1, ra2, dec2) {
    const D = Math.PI / 180;
    const hav =
      Math.sin(((dec2 - dec1) * D) / 2) ** 2 +
      Math.cos(dec1 * D) * Math.cos(dec2 * D) * Math.sin(((ra2 - ra1) * D) / 2) ** 2;
    return (2 * Math.asin(Math.sqrt(hav)) * 180 * 3600) / Math.PI;
  }

  /** Map a mouse event on #viewer-canvas to the rendered grid. The display
   * is north-up (array rows flipped), see renderGrid. */
  function pixelAtEvent(ev) {
    const grid = state.lastRenderedGrid;
    if (!grid) return null;
    const rect = canvas.getBoundingClientRect();
    const canvasX = ((ev.clientX - rect.left) / rect.width) * canvas.width;
    const canvasY = ((ev.clientY - rect.top) / rect.height) * canvas.height;
    const px = Math.floor(canvasX); // canvas pixels are grid pixels (see renderGrid)
    const py = Math.floor(canvasY);
    if (px < 0 || py < 0 || px >= grid.width || py >= grid.height) return null;
    const ax = px;
    const ay = grid.height - 1 - py;
    const idx = ay * grid.width + ax;
    const wcsFrame = grid.wcsFrame;
    const world = wcsFrame && wcsFrame.gridWcs ? gridPixelToWorld(wcsFrame.gridWcs, ax, ay) : null;
    const interpolated = !!(grid.frame && grid.frame.gapFilled && grid.frame.gapFilled.has(idx));
    return { ax, ay, ra: world ? world.ra : NaN, dec: world ? world.dec : NaN, phys: grid.values[idx], interpolated };
  }

  canvas.addEventListener("mousemove", (ev) => {
    const hit = pixelAtEvent(ev);
    if (!hit) {
      pixelReadout.textContent = "";
      return;
    }
    const pos = Number.isFinite(hit.ra) ? `RA ${hit.ra.toFixed(5)}  Dec ${hit.dec.toFixed(5)}` : "(no WCS - reload)";
    const rg = state.lastRenderedGrid && state.lastRenderedGrid.rgb;
    if (rg) {
      const idx = hit.ay * state.lastRenderedGrid.width + hit.ax;
      const parts = rg.channels.map((c) => c.values ? `${c.c} ${c.label} ${Number.isFinite(c.values[idx]) ? c.values[idx].toFixed(3) : "—"}` : `${c.c} ${c.label}`);
      pixelReadout.textContent = rg.masked[idx]
        ? `${pos}   ${rg.masked[idx] === 2 ? "masked: saturation / non-linearity flag in at least one channel" : "masked: no usable value in at least one channel"}`
        : `${pos}   background-subtracted MJy/sr: ${parts.join(" · ")}`;
      return;
    }
    const val = Number.isFinite(hit.phys)
      ? `${hit.phys.toFixed(4)} MJy/sr${hit.interpolated ? " (interpolated for display, not data)" : ""}`
      : "no data";
    pixelReadout.textContent = `${pos}   surface brightness ${val}`;
  });
  canvas.addEventListener("mouseleave", () => (pixelReadout.textContent = ""));

  // ---------------------------------------------------------------------
  // Spectrum (Phase 4, bonus): click a pixel -> flux vs wavelength
  // ---------------------------------------------------------------------
  canvas.addEventListener("click", (ev) => {
    const hit = pixelAtEvent(ev);
    if (!hit || !Number.isFinite(hit.ra)) return;
    // If the click is on the target's predicted position and the target has
    // a known proper motion, ask the backend to resolve the TARGET (so it
    // propagates the position to every exposure's own epoch). Otherwise
    // measure the fixed clicked position.
    let track = false;
    const f = state.lastRenderedGrid.frame;
    if (state.resolved && state.resolved.has_proper_motion && f && f.targetPixel && f.gridWcs) {
      const t = gridPixelToWorld(f.gridWcs, f.targetPixel.x, f.targetPixel.y);
      track = angularSepArcsec(hit.ra, hit.dec, t.ra, t.dec) <= TRACK_CLICK_RADIUS_ARCSEC;
    }
    fetchAndDrawSpectrum(hit.ra, hit.dec, track);
  });

  async function fetchAndDrawSpectrum(ra, dec, track) {
    const what = track
      ? `${state.resolved.name || state.query} (following its proper motion)`
      : `RA ${ra.toFixed(5)}, Dec ${dec.toFixed(5)}`;
    spectrumStatus.textContent = `Fetching spectrum for ${what} from IRSA (all 6 detectors, capped exposures, so this is slow)…`;
    spectrumCtx.clearRect(0, 0, spectrumCanvas.width, spectrumCanvas.height);
    state.lastSpectrumPoints = null;
    try {
      const position = track ? state.query : `${ra.toFixed(6)} ${dec.toFixed(6)}`;
      // A clicked sky position is a fixed position: never let the backend
      // swap it for a catalogued mover (pm_lookup only when following the target).
      const url = `${API_BASE}/api/spectrum?position=${encodeURIComponent(position)}&fov=${Math.min(state.fovArcmin, 3)}&pm_lookup=${track ? "true" : "false"}`;
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`server error (${resp.status})`);
      const body = await resp.json();
      const points = body.points || [];
      state.lastSpectrumPoints = points;
      drawSpectrum(points);
      const ok = points.filter((p) => p.status === "ok" && p.flux_mjy != null);
      const byStatus = {};
      points.forEach((p) => (byStatus[p.status] = (byStatus[p.status] || 0) + 1));
      const refused = Object.entries(byStatus)
        .filter(([k]) => k !== "ok")
        .map(([k, n]) => `${n} ${k.replace(/_/g, " ")}`)
        .join(", ");
      spectrumStatus.textContent = ok.length
        ? `${ok.length} of ${points.length} exposure(s) measured for ${what}${refused ? `; refused: ${refused}` : ""}. Colour = date (blue early → red late).`
        : `No usable aperture measurement for ${what}${refused ? ` (${refused})` : ""}.`;
    } catch (e) {
      spectrumStatus.textContent = `Spectrum failed: ${e.message}`;
    }
  }

  // The light curve and the spectrum sit in the right window, whose width
  // can be changed: redraw them at the new width.
  if ("ResizeObserver" in window) {
    const ro = new ResizeObserver(() => {
      let changed = false;
      for (const c of [lightcurveCanvas, spectrumCanvas]) {
        const w = Math.floor(c.parentElement.clientWidth);
        if (w > 100 && c.width !== w) {
          c.width = w;
          changed = true;
        }
      }
      if (!changed) return;
      drawLightCurve();
      if (state.lastSpectrumPoints) drawSpectrum(state.lastSpectrumPoints);
    });
    ro.observe(lightcurveCanvas.parentElement);
  }

  function drawSpectrum(points) {
    const w = spectrumCanvas.width;
    const h = spectrumCanvas.height;
    spectrumCtx.clearRect(0, 0, w, h);
    const good = points.filter((p) => p.status === "ok" && p.flux_mjy != null && p.wavelength_um != null);
    if (good.length === 0) return;

    const wMin = Math.min(...good.map((p) => p.wavelength_um));
    const wMax = Math.max(...good.map((p) => p.wavelength_um));
    const lo = Math.min(...good.map((p) => p.flux_mjy - (p.flux_err_mjy || 0)));
    const hi = Math.max(...good.map((p) => p.flux_mjy + (p.flux_err_mjy || 0)));
    const tMin = Math.min(...good.map((p) => p.time_mjd));
    const tMax = Math.max(...good.map((p) => p.time_mjd));
    const padL = 34;
    const pad = 12;
    const xOf = (wl) => padL + ((wl - wMin) / (wMax - wMin || 1)) * (w - padL - pad);
    const yOf = (f) => h - pad - ((f - lo) / (hi - lo || 1)) * (h - 2 * pad);

    spectrumCtx.font = '11px "Almendra", Georgia, serif';
    spectrumCtx.fillStyle = "#8f8160";
    spectrumCtx.fillText(fmtFlux(hi), 1, pad);
    spectrumCtx.fillText(fmtFlux(lo), 1, h - pad);
    spectrumCtx.fillText("mJy", 1, h / 2);
    spectrumCtx.fillText(`${wMin.toFixed(2)} µm`, padL, h - 1);
    const lbl = `${wMax.toFixed(2)} µm`;
    spectrumCtx.fillText(lbl, w - pad - spectrumCtx.measureText(lbl).width, h - 1);

    for (const p of good) {
      const hue = 220 - 220 * ((p.time_mjd - tMin) / (tMax - tMin || 1));
      const color = `hsl(${hue}, 80%, 60%)`;
      const x = xOf(p.wavelength_um);
      spectrumCtx.strokeStyle = color;
      spectrumCtx.lineWidth = 1;
      spectrumCtx.beginPath();
      spectrumCtx.moveTo(x, yOf(p.flux_mjy - (p.flux_err_mjy || 0)));
      spectrumCtx.lineTo(x, yOf(p.flux_mjy + (p.flux_err_mjy || 0)));
      spectrumCtx.stroke();
      spectrumCtx.fillStyle = color;
      spectrumCtx.beginPath();
      spectrumCtx.arc(x, yOf(p.flux_mjy), 2.2, 0, 2 * Math.PI);
      spectrumCtx.fill();
    }
  }

  // ---------------------------------------------------------------------
  // Tracked sources: detection / tracking / catalogue match / ranking from
  // the backend's end-of-stream analysis record (app/discovery.py), plus
  // volunteer classification (app/store.py).
  //
  // Wording rule for everything shown here: a "candidate" is something for
  // a person to look at. Nothing in this panel says an object is new.
  // ---------------------------------------------------------------------
  const analysisStatus = $("analysis-status");
  const analysisTableBody = $("analysis-table-body");
  const analysisDetail = $("analysis-detail");
  const showTracksCheckbox = $("show-tracks-checkbox");
  const dataSourceBanner = $("data-source-banner");
  const rankOrderSelect = $("rank-order-select");
  const exportJsonBtn = $("export-json-btn");
  const exportCsvBtn = $("export-csv-btn");

  const VOTE_LABELS = [
    ["moving", "Moving source"],
    ["variable", "Variable source"],
    ["stationary", "Stationary source"],
    ["artifact", "Potential artifact"],
    ["uncertain", "Uncertain"],
  ];

  function voterToken() {
    // Anonymous random token so one browser counts as one volunteer.
    let token = null;
    try {
      token = localStorage.getItem("spherex-voter");
    } catch (e) {
      /* storage unavailable: fall through to a per-page token */
    }
    if (!token) {
      const bytes = new Uint8Array(12);
      crypto.getRandomValues(bytes);
      token = "v-" + Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
      try {
        localStorage.setItem("spherex-voter", token);
      } catch (e) {
        /* keep the in-memory token */
      }
    }
    voterToken = () => token;
    return token;
  }

  function currentSeriesAnalysis() {
    if (!state.analysis || !state.analysis.series) return null;
    return state.analysis.series.find((s) => s.series === detectorViewSelect.value) || null;
  }

  function selectedTrack() {
    const series = currentSeriesAnalysis();
    if (!series || state.selectedRank == null) return null;
    return series.tracks.find((t) => t.rank === state.selectedRank) || null;
  }

  function el(tag, text, className) {
    const node = document.createElement(tag);
    if (text != null) node.textContent = text;
    if (className) node.className = className;
    return node;
  }

  function fmtPm(track) {
    const m = track.motion;
    if (!m || m.pm_total_mas_yr == null) return "—";
    return `${(m.pm_total_mas_yr / 1000).toFixed(2)} ± ${(m.pmra_err_mas_yr / 1000).toFixed(2)}″/yr (${m.pm_significance.toFixed(1)}σ)`;
  }

  function fmtCatalog(track) {
    const parts = [];
    for (const [key, name] of [["gaia", "Gaia"], ["catwise", "CatWISE"]]) {
      const m = track[key];
      if (!m) continue;
      if (m.status === "matched") parts.push(`${name} ✓`);
      else if (m.status === "ambiguous") parts.push(`${name} ? (${m.n_in_gate})`);
      else if (m.status === "unmatched") parts.push(`${name} –`);
      else parts.push(`${name} n/a`);
    }
    return parts.join("  ");
  }

  function renderAnalysis() {
    analysisTableBody.innerHTML = "";
    analysisDetail.innerHTML = "";
    const a = state.analysis;
    const shownSeries = currentSeriesAnalysis();
    sourcesBtn.textContent = shownSeries ? `Sources (${shownSeries.tracks.length})` : "Sources";
    exportJsonBtn.disabled = exportCsvBtn.disabled = !a;
    if (!a) {
      analysisStatus.textContent = !state.allFrames.length
        ? ""
        : state.loading === "busy"
          ? "Analysis runs after the last frame arrives…"
          : "No source analysis for this search: loading was stopped or ended with an error. Start it again to get one.";
      return;
    }
    if (a.error) {
      analysisStatus.textContent = `Analysis failed: ${a.error}`;
      return;
    }
    const series = currentSeriesAnalysis();
    if (!series) {
      analysisStatus.textContent = "No analysis for this detector series.";
      return;
    }
    const reg = series.registration || [];
    const checked = reg.filter((r) => r.applied).length;
    const notes = [];
    notes.push(`${series.tracks.length} source(s) tracked across ${series.n_frames} frame(s) of ${series.series}.`);
    notes.push(
      checked === reg.length && reg.length
        ? "Image registration checked against Gaia stars in every frame."
        : `Image registration could be checked against Gaia stars in ${checked} of ${reg.length} frame(s) (needs 3 isolated stars).`
    );
    if (series.n_frames < 2) notes.push("Only one frame: no motion or variability can be measured.");
    if (series.warning) notes.push(series.warning);
    if (a.catalog_errors && a.catalog_errors.length) notes.push(`Catalogue problem: ${a.catalog_errors.join("; ")}`);
    const localCats = Object.entries(a.catalog_source || {}).filter(([, v]) => v === "local_cache").map(([k]) => k.replace("gaia_dr3", "Gaia DR3").replace("catwise2020", "CatWISE2020"));
    if (localCats.length) notes.push(`Catalogue matching used ${localCats.join(" and ")} rows downloaded earlier (local copy), not a live query.`);
    if (a.store_error) notes.push(`Classification storage unavailable: ${a.store_error}`);
    notes.push("Two rankings are shown: the raw Phase Two anomaly score (0–1: weighted motion, variability, missing catalogue counterpart) and a calibrated score with the same weights whose inputs are corrected for brightness, coverage, catalogue completeness and quality. Neither is a probability, and no object type is estimated.");
    analysisStatus.textContent = notes.join(" ");

    const ordered = series.tracks.slice().sort((a, b) =>
      rankOrderSelect.value === "calibrated" && a.calibrated_rank != null ? a.calibrated_rank - b.calibrated_rank : a.rank - b.rank);
    for (const t of ordered) {
      const tr = document.createElement("tr");
      if (t.rank === state.selectedRank) tr.className = "selected";
      const labelCell = el("td", t.label);
      labelCell.style.color = LABEL_COLORS[t.label] || "";
      tr.appendChild(el("td", rankOrderSelect.value === "calibrated" && t.calibrated_rank != null ? String(t.calibrated_rank) : String(t.rank)));
      tr.appendChild(labelCell);
      tr.appendChild(el("td", t.anomaly_score.toFixed(2)));
      const cal = t.calibration;
      const reduced = cal && (cal.calibrated_score + 1e-9 < t.anomaly_score);
      const calCell = el("td", cal ? `${cal.calibrated_score.toFixed(2)}${reduced ? " ▼" : ""}` : "—");
      if (cal && cal.suppression_reasons.length) calCell.title = cal.suppression_reasons.join(" ");
      tr.appendChild(calCell);
      tr.appendChild(el("td", t.reliability));
      tr.appendChild(el("td", t.detection_snr_median.toFixed(0)));
      tr.appendChild(el("td", `${t.n_epochs}/${t.n_frames_in_series}`));
      tr.appendChild(el("td", fmtPm(t)));
      const v = t.variability;
      tr.appendChild(el("td", v && v.p_value != null ? `p=${v.p_value.toExponential(0)}${v.wavelength_mismatch ? " λ≠" : ""}` : "—"));
      tr.appendChild(el("td", fmtCatalog(t)));
      tr.appendChild(el("td", `${t.ra_deg.toFixed(5)} ${t.dec_deg >= 0 ? "+" : ""}${t.dec_deg.toFixed(5)}`));
      tr.addEventListener("click", () => {
        state.selectedRank = state.selectedRank === t.rank ? null : t.rank;
        renderAnalysis();
        renderView();
      });
      analysisTableBody.appendChild(tr);
    }
    renderAnalysisDetail();
    renderShiftStackPanel();
    renderSpectroTemporal();
  }

  function renderAnalysisDetail() {
    analysisDetail.innerHTML = "";
    const t = selectedTrack();
    if (!t) {
      analysisDetail.appendChild(el("p", "Select a row to see why it was ranked there, its catalogue matches, and to classify it.", "status-text"));
      return;
    }
    analysisDetail.appendChild(el("h3", `#${t.rank} · ${t.label} · anomaly ${t.anomaly_score.toFixed(2)} · reliability ${t.reliability} · detection S/N ${t.detection_snr_median.toFixed(0)}`));

    const why = el("div");
    why.appendChild(el("strong", "Why it is ranked here"));
    const ul = el("ul");
    (t.reasons.length ? t.reasons : ["Nothing unusual measured."]).forEach((r) => ul.appendChild(el("li", r)));
    why.appendChild(ul);
    analysisDetail.appendChild(why);

    // --- astrometry: linear fit (reported) and parallax-aware fit (shown, not adopted unless constrained) ---
    // --- calibrated ranking (Phase 3, Priority 2), next to the raw Phase 2 numbers ---
    const cal = t.calibration;
    if (cal) {
      const cb = el("div");
      cb.appendChild(el("strong", `Calibrated ranking: ${cal.calibrated_score.toFixed(2)} (raw Phase 2 score ${t.anomaly_score.toFixed(2)}; calibrated rank #${t.calibrated_rank}, raw rank #${t.rank})`));
      const cl = el("ul");
      const rc = t.score_components;
      cl.appendChild(el("li", `Components, raw → calibrated: motion ${rc.motion.toFixed(2)} → ${cal.M_norm.toFixed(2)}, variability ${rc.variability.toFixed(2)} → ${cal.V_norm.toFixed(2)}, ` +
        `no catalogue counterpart ${rc.no_catalogue_counterpart.toFixed(2)} → ${cal.C_norm.toFixed(2)} (weights 0.45 / 0.30 / 0.25).`));
      const mc = cal.motion_calibration;
      if (mc && mc.available) {
        const lvl = mc.peer_level === "analytic" ? "too few peers: the analytic error model is used (no correction)"
          : `${mc.n_peers} clean peers (${mc.peer_level.replace(/_/g, " ")}; S/N ${mc.snr_bin[0]}–${Number.isFinite(mc.snr_bin[1]) ? mc.snr_bin[1] : "∞"}, ${mc.coverage_bin} epochs)`;
        cl.appendChild(el("li", `Motion significance calibrated against ${lvl}: inflation ×${mc.inflation.toFixed(2)}` +
          `${mc.inflation_capped ? " (capped)" : ""}; p = ${mc.p_analytic.toExponential(1)} → ${mc.p_calibrated.toExponential(1)}` +
          `${mc.empirical_rarity_bound != null ? `; ${mc.n_peers_at_least_as_extreme} of ${mc.n_peers} peers at least as extreme (≤ ${mc.empirical_rarity_bound.toFixed(2)})` : ""}.`));
      }
      const vc = cal.variability_calibration;
      if (vc && vc.available) {
        cl.appendChild(el("li", `Variability calibrated (${vc.peer_level.replace(/_/g, " ")}, ${vc.n_peers} peers): inflation ×${vc.inflation.toFixed(2)}; p = ${vc.p_analytic.toExponential(1)} → ${vc.p_calibrated.toExponential(1)}.`));
      }
      const REG = { matched: "matched", above_completeness: "unmatched, and bright enough that it should be in the catalogue",
        below_completeness: "unmatched, but fainter than the local completeness limit: expected to be missing",
        completeness_unknown: "unmatched; completeness could not be estimated here (half weight)",
        catalogue_incomplete_here: "unmatched, but the catalogue matches none of the clean sources in this field (no coverage)",
        unavailable: "catalogue query failed or not run" };
      for (const [name, c] of Object.entries(cal.completeness)) {
        const lim = c.completeness_flux_mjy != null ? ` (local limit ${fmtFlux(c.completeness_flux_mjy)} mJy, source ${fmtFlux(c.source_flux_mjy)} mJy)` : "";
        cl.appendChild(el("li", `${name}: ${REG[c.regime] || c.regime}${lim}.`));
      }
      for (const r of cal.explanation.concat(cal.suppression_reasons)) cl.appendChild(el("li", r));
      cb.appendChild(cl);
      analysisDetail.appendChild(cb);
    }

    const ast = el("div");
    ast.appendChild(el("strong", "Astrometry"));
    const al = el("ul");
    const m = t.motion;
    if (m && m.pmra_mas_yr != null) {
      al.appendChild(el("li",
        `Proper motion (linear fit, the reported result): east ${m.pmra_mas_yr.toFixed(0)}, north ${m.pmdec_mas_yr.toFixed(0)} ± ${m.pmra_err_mas_yr.toFixed(0)} mas/yr ` +
        `(${m.pm_significance.toFixed(1)}σ from zero; χ² ${fmtNum(m.chi2, 1)} for ${m.dof} degrees of freedom` +
        `${m.errors_rescaled ? "; errors scaled up because the fit is worse than the position errors predict" : ""}).`));
    } else {
      al.appendChild(el("li", "No proper motion: seen in fewer than two epochs."));
    }
    const p = t.parallax_fit;
    if (!p) {
      al.appendChild(el("li", "Parallax-aware fit not run (needs at least 3 epochs)."));
    } else if (p.constrained) {
      al.appendChild(el("li",
        `Parallax (5-parameter fit, constrained): ${p.parallax_mas.toFixed(0)} ± ${p.parallax_err_mas.toFixed(0)} mas; ` +
        `motion (${p.pmra_mas_yr.toFixed(0)}, ${p.pmdec_mas_yr.toFixed(0)}) ± ${p.pmra_err_mas_yr.toFixed(0)} mas/yr.`));
    } else {
      const item = el("li");
      item.appendChild(el("span", "UNCONSTRAINED", "badge badge-dim"));
      item.appendChild(document.createTextNode(
        ` Parallax-aware fit: ${p.parallax_mas.toFixed(0)} ± ${p.parallax_err_mas.toFixed(0)} mas is not a measurement (${p.reason}). ` +
        `The linear proper-motion fit above remains the reported result. ` +
        `(${p.n_epochs} epochs, ${p.dof} degree(s) of freedom; ${Math.round(100 * p.factor_independence)}% of the parallax signal is distinguishable from linear motion.)`));
      al.appendChild(item);
    }
    ast.appendChild(al);
    analysisDetail.appendChild(ast);

    if (t.caveats.length) {
      const cav = el("div");
      cav.appendChild(el("strong", "Reasons for caution"));
      const cl = el("ul");
      t.caveats.forEach((c) => cl.appendChild(el("li", c)));
      cav.appendChild(cl);
      analysisDetail.appendChild(cav);
    }

    const cat = el("div");
    cat.appendChild(el("strong", "Catalogue comparison (positions propagated to the observation epoch)"));
    const catList = el("ul");
    for (const [key, name] of [["gaia", "Gaia DR3"], ["catwise", "CatWISE2020"]]) {
      const m = t[key];
      if (!m) continue;
      let text;
      if (m.status === "catalog_unavailable") text = `${name}: query failed, not compared.`;
      else if (m.status === "unmatched") text = `${name}: no source inside the positional uncertainty (unmatched).`;
      else {
        const s = m.source;
        const pm = s.pmra_mas_yr != null ? `, PM (${s.pmra_mas_yr.toFixed(0)}, ${s.pmdec_mas_yr.toFixed(0)}) mas/yr` : ", no PM";
        const plx = s.parallax_mas != null ? `, parallax ${s.parallax_mas.toFixed(1)} mas` : "";
        const mags = Object.entries(s.mags || {}).map(([k, v]) => `${k}=${v.toFixed(2)}`).join(" ");
        text = `${name}: ${s.source_id}, ${m.separation_arcsec.toFixed(2)}″ away (${m.normalized_separation.toFixed(1)}σ)${pm}${plx}${mags ? ", " + mags : ""}${m.status === "ambiguous" ? ` (${m.n_in_gate} sources fit, not unique)` : ""}.`;
      }
      catList.appendChild(el("li", text));
    }
    cat.appendChild(catList);
    analysisDetail.appendChild(cat);

    const dets = t.detections
      .map((d) => `${fmtDate(d.time_mjd)}: ${d.phot_status === "ok" ? `${fmtFlux(d.flux_mjy)} ± ${fmtFlux(d.flux_err_mjy)} mJy` : d.phot_status.replace(/_/g, " ")} @ ${d.wavelength_um != null ? d.wavelength_um.toFixed(2) + " µm" : "?"}`)
      .join("   ·   ");
    analysisDetail.appendChild(el("p", `Per-frame aperture flux: ${dets}`, "status-text"));

    // --- volunteer classification ---
    const voteBox = el("div", null, "vote-box");
    voteBox.appendChild(el("strong", "Your classification"));
    if (!t.candidate_id) {
      voteBox.appendChild(el("p", "This source was not stored (only the top-ranked sources per series are), so it cannot be classified.", "status-text"));
    } else {
      const buttons = el("div", null, "vote-buttons");
      const agreement = el("p", "Loading classifications…", "status-text");
      const showAgreement = (info, mine) => {
        const c = info.classifications || info;
        const counts = VOTE_LABELS.filter(([k]) => c.counts[k]).map(([k, name]) => `${name}: ${c.counts[k]}`).join(", ");
        agreement.textContent = c.n_voters
          ? `${c.n_voters} volunteer(s). ${counts}. Agreement on the most common label: ${(100 * c.agreement_fraction).toFixed(0)}%. This shows how people voted, not what the object is.`
          : "No classifications yet.";
        Array.from(buttons.children).forEach((b) => b.classList.toggle("chosen", b.dataset.label === mine));
      };
      for (const [key, name] of VOTE_LABELS) {
        const b = el("button", name, "text-btn");
        b.type = "button";
        b.dataset.label = key;
        b.addEventListener("click", async () => {
          try {
            const resp = await fetch(`${API_BASE}/api/candidates/${t.candidate_id}/classifications`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ label: key, voter: voterToken() }),
            });
            if (!resp.ok) throw new Error(`server error (${resp.status})`);
            const body = await resp.json();
            showAgreement(body, body.your_label);
          } catch (e) {
            agreement.textContent = `Could not save your classification: ${e.message}`;
          }
        });
        buttons.appendChild(b);
      }
      voteBox.appendChild(buttons);
      voteBox.appendChild(agreement);
      fetch(`${API_BASE}/api/candidates/${t.candidate_id}?voter=${encodeURIComponent(voterToken())}`)
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`server error (${r.status})`))))
        .then((body) => showAgreement(body, body.your_label))
        .catch((e) => (agreement.textContent = `Could not load classifications: ${e.message}`));
    }
    analysisDetail.appendChild(voteBox);
  }

  function downloadText(filename, text, mime) {
    const url = URL.createObjectURL(new Blob([text], { type: mime }));
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function exportName(ext) {
    const base = (state.query || "field").replace(/[^A-Za-z0-9+_.-]+/g, "_").slice(0, 60);
    return `spherex_sources_${base}.${ext}`;
  }

  exportJsonBtn.addEventListener("click", () => {
    if (!state.analysis) return;
    const payload = { query: state.query, resolved: state.resolved, fov_arcmin: state.fovArcmin, releases: state.releaseTag, analysis: state.analysis };
    downloadText(exportName("json"), JSON.stringify(payload, null, 2), "application/json");
  });

  exportCsvBtn.addEventListener("click", () => {
    if (!state.analysis) return;
    const cols = [
      "series", "rank", "label", "anomaly_score", "reliability", "detection_snr_median", "n_epochs", "ra_deg", "dec_deg", "epoch_mjd",
      "pos_err_arcsec", "pmra_mas_yr", "pmdec_mas_yr", "pm_err_mas_yr", "pm_significance",
      "parallax_fit_mas", "parallax_fit_err_mas", "parallax_constrained", "mean_flux_mjy", "variability_p_value",
      "wavelength_mismatch", "gaia_status", "gaia_source_id", "gaia_separation_arcsec", "catwise_status", "catwise_source_id",
      "catwise_separation_arcsec", "flags", "reasons", "caveats", "candidate_id",
      "calibrated_score", "calibrated_rank", "M_norm", "V_norm", "C_norm", "completeness_regime", "quality_cap", "suppression_reasons",
    ];
    const quote = (v) => (v == null ? "" : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
    const lines = [cols.join(",")];
    for (const s of state.analysis.series || []) {
      for (const t of s.tracks) {
        const row = {
          series: t.series, rank: t.rank, label: t.label, anomaly_score: t.anomaly_score, reliability: t.reliability,
          detection_snr_median: t.detection_snr_median, n_epochs: t.n_epochs, ra_deg: t.ra_deg, dec_deg: t.dec_deg, epoch_mjd: t.epoch_mjd,
          pos_err_arcsec: t.pos_err_arcsec, pmra_mas_yr: t.motion.pmra_mas_yr, pmdec_mas_yr: t.motion.pmdec_mas_yr,
          pm_err_mas_yr: t.motion.pmra_err_mas_yr, pm_significance: t.motion.pm_significance,
          parallax_fit_mas: t.parallax_fit ? t.parallax_fit.parallax_mas : "",
          parallax_fit_err_mas: t.parallax_fit ? t.parallax_fit.parallax_err_mas : "",
          parallax_constrained: t.parallax_fit ? t.parallax_fit.constrained : "",
          mean_flux_mjy: t.variability.mean_flux_mjy, variability_p_value: t.variability.p_value,
          wavelength_mismatch: t.variability.wavelength_mismatch,
          gaia_status: t.gaia.status, gaia_source_id: t.gaia.source ? t.gaia.source.source_id : "",
          gaia_separation_arcsec: t.gaia.separation_arcsec,
          catwise_status: t.catwise.status, catwise_source_id: t.catwise.source ? t.catwise.source.source_id : "",
          catwise_separation_arcsec: t.catwise.separation_arcsec,
          flags: t.flags.join("|"), reasons: t.reasons.join(" | "), caveats: t.caveats.join(" | "), candidate_id: t.candidate_id || "",
          calibrated_score: t.calibration ? t.calibration.calibrated_score : "", calibrated_rank: t.calibrated_rank != null ? t.calibrated_rank : "",
          M_norm: t.calibration ? t.calibration.M_norm : "", V_norm: t.calibration ? t.calibration.V_norm : "",
          C_norm: t.calibration ? t.calibration.C_norm : "", completeness_regime: t.calibration ? t.calibration.completeness_regime : "",
          quality_cap: t.calibration ? t.calibration.quality_cap : "",
          suppression_reasons: t.calibration ? t.calibration.suppression_reasons.join(" | ") : "",
        };
        lines.push(cols.map((c) => quote(row[c])).join(","));
      }
    }
    downloadText(exportName("csv"), lines.join("\n"), "text/csv");
  });

  showTracksCheckbox.addEventListener("change", renderView);
  rankOrderSelect.addEventListener("change", renderAnalysis);

  // ---------------------------------------------------------------------
  // Shift-and-stack (app/stream.py: run_shift_stack). Optional; requested
  // with the "Shift-and-stack search" checkbox. A search technique: the
  // panel says so and never calls a result a confirmation.
  // ---------------------------------------------------------------------
  const shiftstackPanel = $("shiftstack-panel");

  function fmtNum(v, digits = 1) {
    return v == null || !Number.isFinite(v) ? "—" : v.toFixed(digits);
  }

  function currentShiftStack() {
    const ss = state.analysis && state.analysis.shift_stack;
    if (!ss || !ss.series) return null;
    const det = detectorViewSelect.value;
    return ss.series.find((s) => s.series === det) || null;
  }

  const SS_BADGE = {
    passes_controls: ["Passes its controls (tentative)", "badge-ok"],
    tentative: ["Not distinguishable from chance", "badge-warn"],
    no_moving_candidate: ["No moving candidate", "badge-dim"],
    not_run: ["Not run", "badge-dim"],
  };

  function kvTable(rows) {
    const tbl = el("table", null, "kv-table");
    for (const [k, v] of rows) {
      const tr = el("tr");
      tr.appendChild(el("th", k));
      tr.appendChild(el("td", v));
      tbl.appendChild(tr);
    }
    return tbl;
  }

  function renderShiftStackPanel() {
    shiftstackPanel.innerHTML = "";
    const a = state.analysis;
    if (!a) return;
    if (!a.shift_stack) {
      shiftstackPanel.appendChild(
        el("p", "Shift-and-stack was not requested for this search. Tick “Shift-and-stack search” and search again to run it.", "status-text")
      );
      return;
    }
    const e = currentShiftStack();
    shiftstackPanel.appendChild(el("h3", `Shift-and-stack search (${detectorViewSelect.value || "?"})`));
    shiftstackPanel.appendChild(el("p", a.shift_stack.statement, "status-text"));
    if (!e) {
      shiftstackPanel.appendChild(el("p", "No shift-and-stack result for this detector.", "status-text"));
      return;
    }
    const [badgeText, badgeClass] = SS_BADGE[e.assessment] || [e.assessment, "badge-dim"];
    const head = el("p");
    head.appendChild(el("span", badgeText, `badge ${badgeClass}`));
    head.appendChild(document.createTextNode(" " + (e.assessment_text || "")));
    shiftstackPanel.appendChild(head);
    if (!e.blind) return;

    const b = e.blind;
    const wcsFrame = state.frames[0];
    const toSky = (x, y) => (wcsFrame && wcsFrame.gridWcs ? gridPixelToWorld(wcsFrame.gridWcs, x, y) : null);
    const fmtSky = (s) => (s ? `${s.ra.toFixed(5)}, ${s.dec >= 0 ? "+" : ""}${s.dec.toFixed(5)}` : "—");

    const rows = [
      ["Exposures stacked", `${e.n_exposures} over ${fmtNum(b.baseline_yr, 2)} yr (reference epoch ${fmtDate(b.epoch_ref_mjd)})`],
      ["Velocity search", `${b.n_velocities} trial motions up to ${fmtNum(b.v_max_arcsec_yr, 0)}″/yr, step ${fmtNum(b.velocity_step_arcsec_yr, 1)}″/yr, refined ×5 around each peak`],
      ["Slowest motion distinguishable from none", `${fmtNum(b.min_distinguishable_speed_arcsec_yr, 1)}″/yr (half a PSF width over the baseline)`],
    ];
    const f = e.forced;
    let predSky = null;
    if (f) {
      predSky = toSky(f.predicted_xy[0], f.predicted_xy[1]);
      rows.push(["Predicted (catalogue) position", `${fmtSky(predSky)} at the reference epoch`]);
      rows.push(["Predicted motion", `(${fmtNum(f.predicted_pm_arcsec_yr[0], 2)}, ${fmtNum(f.predicted_pm_arcsec_yr[1], 2)})″/yr east, north`]);
      rows.push([
        "Stack along predicted track",
        `S/N ${fmtNum(f.snr_on_trajectory)}; same place with zero motion ${fmtNum(f.snr_same_position_no_motion)}; reversed motion ${fmtNum(f.snr_same_position_reversed_motion)}`,
      ]);
      rows.push([
        "Same motion elsewhere in the field",
        `${(100 * f.off_position.fraction_at_least_as_high).toFixed(1)}% of ${f.off_position.n} random positions reach that S/N (95th percentile ${fmtNum(f.off_position.p95)}; includes real stars)`,
      ]);
    } else {
      rows.push(["Predicted track", "None: the target has no known proper motion, so only the blind search ran."]);
    }
    const moving = (b.candidates || []).filter((c) => c.moving);
    if (!moving.length) rows.push(["Moving candidates", "None"]);
    moving.forEach((c, i) => {
      const sky = toSky(c.x, c.y);
      const off = predSky && sky ? angularSepArcsec(sky.ra, sky.dec, predSky.ra, predSky.dec) : null;
      const q = b.position_quantum_arcsec || 3.075; // candidate positions are whole grid pixels
      const offText = off == null ? "" : off < q ? ` (within one ${q.toFixed(1)}″ grid pixel of the prediction)` : ` (${fmtNum(off, 1)}″ from prediction)`;
      rows.push([
        `Moving candidate ${i + 1}`,
        `${fmtSky(sky)}${offText}; motion ≈ (${fmtNum(c.pm_east_arcsec_yr, 1)}, ${fmtNum(c.pm_north_arcsec_yr, 1)})″/yr east, north ` +
          `(see “Velocity uncertainty” below); S/N ${fmtNum(c.snr)} vs ${fmtNum(c.snr_static)} with zero motion (S/N² gain ${fmtNum(c.delta_chi2)})`,
      ]);
    });
    const ctrl = b.controls || {};
    const ctrlVals = (ctrl.best_moving_snr || []).filter((v) => v > 0).sort((x, y) => y - x);
    const floor = ctrl.n ? 1 / (ctrl.n + 1) : null;
    let pText = "not defined (no moving peak in the real search)";
    if (ctrl.false_alarm_probability != null) {
      pText = Math.abs(ctrl.false_alarm_probability - floor) < 1e-9
        ? `≤ ${floor.toFixed(3)} (no control as strong; the smallest value ${ctrl.n} controls can show)`
        : ctrl.false_alarm_probability.toFixed(3);
    }
    rows.push([
      "Time-shuffled controls",
      `${ctrl.n || 0} searches with the observation dates shuffled. ${ctrlVals.length} produced a moving peak` +
        (ctrlVals.length ? ` (strongest S/N ${ctrlVals.slice(0, 3).map((v) => v.toFixed(1)).join(", ")})` : "") +
        `; best real moving peak S/N ${fmtNum(b.best_moving_snr_coarse)} on the same coarse grid. False-alarm probability ${pText}.`,
    ]);
    shiftstackPanel.appendChild(kvTable(rows));
    // Velocity uncertainty: formal (statistical, from this search) and the
    // estimated total (from the WISE 0855 validation, docs/PHASE2.md) are
    // shown separately; the formal number alone understates the error.
    const unc = el("div", null, "uncertainty-box");
    unc.appendChild(el("strong", "Velocity uncertainty"));
    const ul2 = el("ul");
    const formal = moving.map((c, i) =>
      `candidate ${i + 1}: ±${fmtNum(c.pm_east_err_arcsec_yr, 2)} east, ±${fmtNum(c.pm_north_err_arcsec_yr, 2)} north`).join("; ");
    ul2.appendChild(el("li",
      `Formal (statistical only): ${formal || "—"} ″/yr. This is the width of the S/N peak on the refinement grid ` +
      `and cannot be smaller than one grid step. It is NOT the full measurement error.`));
    ul2.appendChild(el("li",
      "Estimated total: about 1–1.5″/yr per component. Basis: the WISE 0855 validation on real data, where the " +
      "velocity changed by ~1″/yr between equivalent refinement schemes and the north component differed from the " +
      "catalogue by ~0.9″/yr (~2σ). This is an estimate from one validated object, not a per-source calculation."));
    ul2.appendChild(el("li",
      "Not included in the formal error: candidate positions are whole 3.1″ grid pixels; exposures are not " +
      "individually registered against Gaia in this search; one Gaussian PSF is assumed for all exposures."));
    ul2.appendChild(el("li",
      "Use the velocity to say whether and roughly how a source moves, not as a precise proper motion. The tracked-source " +
      "fit (table below) is the better motion measurement when the source is detected in each epoch."));
    unc.appendChild(ul2);
    shiftstackPanel.appendChild(unc);
    if (e.image) {
      const btn = el("button", "Show the stacked S/N map in the viewer", "text-btn");
      btn.type = "button";
      btn.addEventListener("click", () => {
        viewModeSelect.value = "shiftstack";
        viewModeSelect.dispatchEvent(new Event("change"));
      });
      shiftstackPanel.appendChild(btn);
    }
  }

  /** Viewer branch for view mode "shiftstack". */
  function renderShiftStackView() {
    const e = currentShiftStack();
    if (!e || !e.image) {
      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
      state.lastRenderedGrid = null;
      drawOverlay();
      frameLabel.textContent = state.analysis && state.analysis.shift_stack
        ? "No shift-and-stack image for this detector (too few exposures)."
        : "Shift-and-stack was not requested for this search (tick “Shift-and-stack search” and search again).";
      return;
    }
    const img = e.image;
    const raw = base64ToUint16Array(img.payload_b64);
    const values = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) values[i] = raw[i] === INVALID_SENTINEL ? NaN : raw[i] * img.scale + img.offset;
    const pseudo = e.forced ? { targetPixel: { x: e.forced.predicted_xy[0], y: e.forced.predicted_xy[1] } } : null;
    renderGrid(img.width, img.height, values, { useGlobalAsinhLevels: false, frame: pseudo, wcsFrame: state.frames[0] });
    // (drawOverlay circles the moving candidates)
    frameLabel.textContent =
      `Shift-and-stack S/N map (${e.series}), stacked for motion ≈ (${img.pm_east_arcsec_yr.toFixed(1)}, ${img.pm_north_arcsec_yr.toFixed(1)})″/yr east, north (uncertain by ~1–1.5″/yr) ` +
      `[${img.velocity_source}], positions at ${fmtDate(img.epoch_mjd)}.\n` +
      `Values are S/N in units of this stack's measured noise. Sources with this motion add up to a point; static stars smear out. ` +
      `Red cross: predicted target position. Green circles: moving candidates. North up, east left.`;
  }

  // ---------------------------------------------------------------------
  // Spectro-temporal analysis F(lambda, t) (app/spectrotemporal.py).
  // Shows measured values only: no line is drawn through sparse points,
  // gaps in wavelength coverage stay visible, and points from different
  // epochs are never presented as one simultaneous spectrum.
  // ---------------------------------------------------------------------
  const stPanel = $("st-panel");
  const stSourceSelect = $("st-source-select");
  const stSedCanvas = $("st-sed-canvas");
  const stLcCanvas = $("st-lc-canvas");
  const stEpochBoxes = $("st-epoch-boxes");
  const stHover = $("st-hover");
  const stCoverage = $("st-coverage");
  const stMetrics = $("st-metrics");
  const stTableBody = $("st-table-body");
  const stStatus = $("st-status");
  const stBody = $("st-body");
  // Okabe-Ito (colour-blind safe), ordered for contrast on the dark background
  const EPOCH_COLORS = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#F0E442", "#D55E00", "#0072B2", "#FFFFFF"];
  const ST_FONT = '13px "Almendra", Georgia, serif';
  const STATE_TEXT = {
    valid: "valid",
    valid_flagged: "valid, saturation-related flag in aperture",
    below_threshold: "below 3σ (upper limit)",
    saturated: "saturated (no flux)",
    masked: "masked (no flux)",
    off_image: "off image (no flux)",
    no_background: "no sky estimate (no flux)",
  };
  const ASSESS = {
    significant_change: ["Significant change", "badge-ok"],
    conditional: ["Conditional", "badge-warn"],
    not_unusual: ["Not unusual for this field", "badge-dim"],
    no_significant_change: ["No significant change", "badge-dim"],
    inconclusive: ["Inconclusive", "badge-dim"],
  };
  const METRIC_TEXT = {
    grey_variability: ["Brightness change at λ₀", "Δχ² between a fit with one flux per epoch and a constant flux, both with one spectral shape shared by all epochs (dof = epochs − 1)."],
    slope_change: ["Spectral slope change", "Δχ² for adding a per-epoch tilt (1 + t_e(λ − λ₀)) to the shared shape (dof = epochs − 1)."],
    matched_wavelength_pairs: ["Same-wavelength pairs (model-free)", "Exposures from different epochs within ¼ bandwidth of each other; χ² of their differences."],
    colour_change: ["Colour change between detectors", "χ² of a constant ln(F_A(λ₀)/F_B(λ₀)) across epochs."],
  };
  const st = { selectedEpochs: null, hoverPoints: [] };

  function stData() {
    return state.analysis && state.analysis.spectrotemporal ? state.analysis.spectrotemporal : null;
  }
  function stSource() {
    const d = stData();
    if (!d || !d.sources) return null;
    return d.sources.find((s) => s.id === stSourceSelect.value) || d.sources[0] || null;
  }

  function renderSpectroTemporal() {
    const d = stData();
    stBody.hidden = true;
    spectroBtn.textContent = d && !d.error && d.sources ? `Spectra (${d.sources.length})` : "Spectra";
    if (!state.analysis) {
      stStatus.textContent = "";
      return;
    }
    if (!d) {
      stStatus.textContent = "Not requested for this search. Tick “Spectro-temporal analysis” and search again.";
      return;
    }
    if (d.error) {
      stStatus.textContent = `Spectro-temporal analysis failed: ${d.error}`;
      return;
    }
    const prev = stSourceSelect.value;
    stSourceSelect.innerHTML = "";
    for (const s of d.sources) {
      const o = document.createElement("option");
      o.value = s.id;
      o.textContent = `${s.kind === "target" ? "★ " : ""}${s.label}${s.is_reference_star ? " (reference star)" : ""}`;
      stSourceSelect.appendChild(o);
    }
    if (d.sources.some((s) => s.id === prev)) stSourceSelect.value = prev;
    const failed = Object.values(d.retrieval_failed_by_detector || {}).reduce((a, b) => a + b, 0) + (d.retrieval_failed_at_caching || 0);
    stStatus.textContent =
      `${d.n_exposures} exposure(s) measured on native pixels; ${failed} requested exposure(s) could not be retrieved. ` +
      `${d.reference.n_stars} reference star(s) in this field. ${d.statement}`;
    stBody.hidden = false;
    fitStCanvases();
    renderStSource();
  }

  // The two plots are redrawn at whatever size their window gives them
  // (the window can be resized), instead of stretching a fixed bitmap.
  function fitCanvasToBox(canvas, minW, minH) {
    const box = canvas.parentElement;
    if (!box.clientWidth || !box.clientHeight) return false; // tab not shown
    const w = Math.max(minW, Math.floor(box.clientWidth));
    const h = Math.max(minH, Math.floor(box.clientHeight));
    if (canvas.width === w && canvas.height === h) return false;
    canvas.width = w;
    canvas.height = h;
    return true;
  }
  function fitStCanvases() {
    const a = fitCanvasToBox(stSedCanvas, 420, 200);
    const b = fitCanvasToBox(stLcCanvas, 420, 200);
    return [a, b];
  }
  if ("ResizeObserver" in window) {
    const ro = new ResizeObserver(() => {
      const src = stSource();
      const [a, b] = fitStCanvases();
      if (!src || stBody.hidden) return;
      if (a) drawSed(src);
      if (b) drawStLightCurves(src);
    });
    ro.observe(stSedCanvas.parentElement);
    ro.observe(stLcCanvas.parentElement);
  }

  function renderStSource() {
    const s = stSource();
    if (!s) return;
    const epochs = s.epochs || [];
    if (!st.selectedEpochs || st.selectedEpochs.source !== s.id) {
      st.selectedEpochs = { source: s.id, set: new Set(epochs.map((e) => e.index)) };
    }
    stEpochBoxes.innerHTML = "";
    epochs.forEach((e, i) => {
      const lab = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = st.selectedEpochs.set.has(e.index);
      cb.addEventListener("change", () => {
        cb.checked ? st.selectedEpochs.set.add(e.index) : st.selectedEpochs.set.delete(e.index);
        drawSed(s);
      });
      lab.appendChild(cb);
      const col = EPOCH_COLORS[i % EPOCH_COLORS.length];
      const chip = el("span", "", "st-epoch-chip");
      chip.style.background = col;
      lab.appendChild(chip);
      const nE = s.measurements.filter((m) => m.epoch_index === e.index).length;
      const sw = el("span", ` Epoch ${e.index + 1}: ${fmtDate(e.mjd_min)}…${fmtDate(e.mjd_max)} (${(e.mjd_max - e.mjd_min).toFixed(1)} d, ${nE} exposures)`);
      sw.style.color = col;
      lab.appendChild(sw);
      stEpochBoxes.appendChild(lab);
    });
    drawSed(s);
    drawStLightCurves(s);
    renderStCoverage(s);
    renderStMetrics(s);
    renderStTable(s);
  }

  // Nominal SPHEREx detector bands (µm). Only used to SHADE the axis so coverage
  // gaps and detectors missing from this search are obvious; every point is
  // placed at its own measured wavelength_um, never at a band centre.
  const SPHEREX_BANDS = [
    ["D1", 0.75, 1.11], ["D2", 1.11, 1.64], ["D3", 1.64, 2.42],
    ["D4", 2.42, 3.82], ["D5", 3.82, 4.42], ["D6", 4.42, 5.00],
  ];
  const SPHEREX_RANGE = [0.75, 5.0];
  const DET_MARKERS = { D1: "circle", D2: "square", D3: "diamond", D4: "triangle", D5: "hexagon", D6: "star" };
  const DET_MARKER_TEXT = { circle: "●", square: "■", diamond: "◆", triangle: "▲", hexagon: "⬢", star: "★" };
  const stRangeSelect = $("st-range-select");
  const stScaleSelect = $("st-scale-select");
  const stShowFlagged = $("st-show-flagged");
  const stShowLimits = $("st-show-limits");
  const stShowNoflux = $("st-show-noflux");
  const stShowModel = $("st-show-model");
  const stSedCaption = $("st-sed-caption");
  const stTooltip = $("st-tooltip");

  function markerPath(ctx, shape, x, y, r) {
    ctx.beginPath();
    if (shape === "square") { ctx.rect(x - r * 0.85, y - r * 0.85, 1.7 * r, 1.7 * r); return; }
    if (shape === "diamond") { ctx.moveTo(x, y - 1.25 * r); ctx.lineTo(x + 1.05 * r, y); ctx.lineTo(x, y + 1.25 * r); ctx.lineTo(x - 1.05 * r, y); ctx.closePath(); return; }
    if (shape === "triangle") { ctx.moveTo(x, y - 1.25 * r); ctx.lineTo(x + 1.15 * r, y + 0.8 * r); ctx.lineTo(x - 1.15 * r, y + 0.8 * r); ctx.closePath(); return; }
    if (shape === "hexagon" || shape === "star") {
      const n = shape === "star" ? 10 : 6;
      for (let k = 0; k < n; k++) {
        const a = -Math.PI / 2 + (2 * Math.PI * k) / n;
        const rr = shape === "star" ? (k % 2 ? 0.75 * r : 1.7 * r) : 1.1 * r;
        k ? ctx.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a)) : ctx.moveTo(x + rr * Math.cos(a), y + rr * Math.sin(a));
      }
      ctx.closePath();
      return;
    }
    ctx.arc(x, y, r, 0, 2 * Math.PI);
  }

  function niceLinearTicks(lo, hi, n) {
    const span = hi - lo;
    if (!(span > 0)) return [lo];
    const raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((s) => span / s <= n) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * span; v += step) out.push(Math.abs(v) < 1e-12 * step ? 0 : v);
    return out;
  }
  /** Tick label with only the decimals the tick spacing needs (30, not 30.00). */
  function tickText(v, ticks) {
    if (v === 0) return "0";
    const steps = ticks.slice(1).map((t, i) => Math.abs(t - ticks[i])).filter((d) => d > 0);
    const step = steps.length ? Math.min(...steps) : Math.abs(v);
    let dec = 0;
    while (dec < 4 && Math.abs(step * 10 ** dec - Math.round(step * 10 ** dec)) > 1e-6 * 10 ** dec) dec++;
    return v.toFixed(dec);
  }
  function logTicks(lo, hi) {
    const out = [];
    const decades = Math.log10(hi) - Math.log10(lo);
    const mults = decades > 3 ? [1] : decades > 1.3 ? [1, 3] : [1, 2, 5];
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
      for (const m of mults) { const v = m * Math.pow(10, e); if (v >= lo && v <= hi) out.push(v); }
    }
    return out;
  }

  /** Spectrum by epoch: ONE shared wavelength axis for all detectors. Every
   * point is one exposure on native pixels at its own wavelength; nothing is
   * averaged or merged across exposures, epochs or detectors. */
  function drawSed(s) {
    const ctx = stSedCanvas.getContext("2d");
    const W = stSedCanvas.width, H = stSedCanvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.font = ST_FONT;
    st.hoverPoints = [];
    stTooltip.hidden = true;
    const ms = s.measurements.filter((m) => m.wavelength_um != null && Number.isFinite(m.wavelength_um));
    const epochIdx = new Map(s.epochs.map((e, i) => [e.index, i]));
    const showFlagged = stShowFlagged.checked, showLimits = stShowLimits.checked, showNoflux = stShowNoflux.checked;
    const hasFlux = (m) => m.flux_mjy != null && Number.isFinite(m.flux_mjy) && m.flux_err_mjy != null && Number.isFinite(m.flux_err_mjy);
    const kindOf = (m) => (m.state === "valid" || m.state === "valid_flagged") && hasFlux(m) ? m.state
      : m.state === "below_threshold" && m.upper_limit_mjy != null && Number.isFinite(m.upper_limit_mjy) ? "limit" : "noflux";
    const visible = ms.filter((m) => {
      if (!st.selectedEpochs.set.has(m.epoch_index)) return false;
      const k = kindOf(m);
      return k === "valid" || (k === "valid_flagged" && showFlagged) || (k === "limit" && showLimits) || (k === "noflux" && showNoflux);
    });

    // --- wavelength axis
    const padL = 62, padR = 14, padT = 22, nEp = s.epochs.length;
    const padB = 46 + 4 * nEp;                 // x labels + per-epoch sampling strips
    const plotW = W - padL - padR, plotH = H - padT - padB;
    let l0, l1;
    if (stRangeSelect.value === "observed" && ms.length) {
      const ls = ms.map((m) => m.wavelength_um);
      const a = Math.min(...ls), b = Math.max(...ls);
      const pad = Math.max(0.05, 0.04 * (b - a));
      l0 = Math.max(0.6, a - pad); l1 = Math.min(5.2, b + pad);
    } else {
      [l0, l1] = SPHEREX_RANGE;
    }
    const xOf = (l) => padL + ((l - l0) / (l1 - l0)) * plotW;

    // --- flux axis (log is chosen automatically when the plotted dynamic range > 5x)
    const fluxVals = [];
    for (const m of visible) {
      const k = kindOf(m);
      if (k === "valid" || k === "valid_flagged") fluxVals.push(m.flux_mjy - m.flux_err_mjy, m.flux_mjy + m.flux_err_mjy, m.flux_mjy);
      else if (k === "limit") fluxVals.push(m.upper_limit_mjy);
    }
    const pos = fluxVals.filter((v) => v > 0);
    const centralPos = visible.filter((m) => (kindOf(m) === "valid" || kindOf(m) === "valid_flagged") && m.flux_mjy > 0).map((m) => m.flux_mjy)
      .concat(visible.filter((m) => kindOf(m) === "limit" && m.upper_limit_mjy > 0).map((m) => m.upper_limit_mjy));
    const dynRange = centralPos.length ? Math.max(...centralPos) / Math.min(...centralPos) : 1;
    const useLog = stScaleSelect.value === "log" ? true : stScaleSelect.value === "linear" ? false : dynRange > 5;
    let yMin, yMax;
    if (useLog) {
      const lo = pos.length ? Math.min(...pos) : 0.1, hi = pos.length ? Math.max(...pos) : 10;
      yMin = lo / 1.4; yMax = hi * 1.4;
    } else {
      const lo = fluxVals.length ? Math.min(...fluxVals) : 0, hi = fluxVals.length ? Math.max(...fluxVals) : 1;
      const span = hi - lo || Math.abs(hi) || 1;
      yMin = Math.min(0, lo - 0.08 * span); yMax = hi + 0.08 * span;
    }
    const yOf = (f) => useLog
      ? padT + (1 - (Math.log10(Math.min(Math.max(f, yMin), yMax)) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * plotH
      : padT + (1 - (Math.min(Math.max(f, yMin), yMax) - yMin) / (yMax - yMin)) * plotH;

    // --- detector band shading and labels
    const present = new Set(ms.map((m) => m.detector));
    const failed = (stData() && stData().retrieval_failed_by_detector) || {};
    SPHEREX_BANDS.forEach(([det, a, b], k) => {
      const xa = Math.max(xOf(a), padL), xb = Math.min(xOf(b), padL + plotW);
      if (xb <= xa) return;
      if (present.has(det)) {
        ctx.fillStyle = k % 2 ? "rgba(125,107,69,0.07)" : "rgba(125,107,69,0.035)";
      } else {
        ctx.fillStyle = "rgba(125,107,69,0.025)";
      }
      ctx.fillRect(xa, padT, xb - xa, plotH);
      if (!present.has(det)) {                          // hatched: not observed in this search
        ctx.save(); ctx.beginPath(); ctx.rect(xa, padT, xb - xa, plotH); ctx.clip();
        ctx.strokeStyle = "rgba(125,107,69,0.12)"; ctx.lineWidth = 1;
        for (let x = xa - plotH; x < xb; x += 10) { ctx.beginPath(); ctx.moveTo(x, padT + plotH); ctx.lineTo(x + plotH, padT); ctx.stroke(); }
        ctx.restore();
      }
      ctx.strokeStyle = "rgba(125,107,69,0.35)"; ctx.setLineDash([2, 3]);
      for (const xb2 of [xOf(a), xOf(b)]) if (xb2 > padL && xb2 < padL + plotW) { ctx.beginPath(); ctx.moveTo(xb2, padT); ctx.lineTo(xb2, padT + plotH); ctx.stroke(); }
      ctx.setLineDash([]);
      const n = ms.filter((m) => m.detector === det).length;
      const lbl = present.has(det) ? `${DET_MARKER_TEXT[DET_MARKERS[det]]} ${det} · ${n}` : `${det} · ${failed[det] ? "retrieval failed" : "not in this search"}`;
      ctx.fillStyle = present.has(det) ? "#e6d2a4" : "#6b6048";
      const tw = ctx.measureText(lbl).width;
      if (tw < xb - xa - 4) ctx.fillText(lbl, (xa + xb) / 2 - tw / 2, padT - 7);
      else if (xb - xa > 18) { const t2 = det; ctx.fillText(t2, (xa + xb) / 2 - ctx.measureText(t2).width / 2, padT - 7); }
    });
    ctx.strokeStyle = "rgba(125,107,69,0.5)";
    ctx.strokeRect(padL, padT, plotW, plotH);

    // --- axes, ticks
    ctx.fillStyle = "#8f8160";
    const yt = useLog ? logTicks(yMin, yMax) : niceLinearTicks(yMin, yMax, 5);
    for (const v of yt) {
      const y = yOf(v);
      ctx.strokeStyle = "rgba(125,107,69,0.12)"; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();
      const t = tickText(v, yt); ctx.fillText(t, padL - 6 - ctx.measureText(t).width, y + 4);
    }
    const xt = niceLinearTicks(l0, l1, Math.max(4, Math.floor(plotW / 80)));
    for (const v of xt) {
      const x = xOf(v);
      ctx.strokeStyle = "rgba(125,107,69,0.5)"; ctx.beginPath(); ctx.moveTo(x, padT + plotH); ctx.lineTo(x, padT + plotH + 4); ctx.stroke();
      const t = tickText(v, xt); ctx.fillText(t, x - ctx.measureText(t).width / 2, padT + plotH + 15);
    }
    const xl = "λ (µm), each point at its own measured wavelength";
    ctx.fillText(xl, padL + plotW / 2 - ctx.measureText(xl).width / 2, H - 4);
    ctx.save(); ctx.translate(14, padT + plotH / 2 + 40); ctx.rotate(-Math.PI / 2);
    ctx.fillText(`F_ν (mJy${useLog ? ", log" : ", linear"})`, 0, 0); ctx.restore();

    // --- per-epoch sampling strips (what was actually observed, by bandwidth)
    s.epochs.forEach((e, i) => {
      if (!st.selectedEpochs.set.has(e.index)) return;
      ctx.fillStyle = EPOCH_COLORS[i % EPOCH_COLORS.length];
      ctx.globalAlpha = 0.6;
      for (const m of ms) {
        if (m.epoch_index !== e.index) continue;
        const bw = m.bandwidth_um != null && Number.isFinite(m.bandwidth_um) ? m.bandwidth_um : 0.02;
        const xa = Math.max(padL, xOf(m.wavelength_um - bw / 2)), xb = Math.min(padL + plotW, xOf(m.wavelength_um + bw / 2));
        if (xb > xa) ctx.fillRect(xa, padT + plotH + 21 + 4 * i, Math.max(1, xb - xa), 3);
      }
      ctx.globalAlpha = 1;
    });

    // --- optional model guide: the per-detector shared-shape fit a_e * S(lambda)
    let modelDrawn = 0;
    if (stShowModel.checked) {
      for (const det of s.detectors) {
        const fit = s.fits && s.fits[det];
        if (!fit || !fit.knots_um || fit.knots_um.length < 2) continue;
        fit.epochs.forEach((ei, j) => {
          if (!st.selectedEpochs.set.has(ei)) return;
          const a = fit.amplitudes_mjy[j];
          if (!Number.isFinite(a)) return;
          const col = EPOCH_COLORS[(epochIdx.get(ei) || 0) % EPOCH_COLORS.length];
          ctx.strokeStyle = col; ctx.globalAlpha = fit.model_adequate ? 0.55 : 0.3; ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]);
          ctx.beginPath();
          let started = false;
          fit.knots_um.forEach((k, q) => {
            const f = a * fit.shape[q];
            if (useLog && !(f > 0)) { started = false; return; }
            const x = xOf(k), y = yOf(f);
            started ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
            started = true;
          });
          ctx.stroke();
          ctx.setLineDash([]); ctx.globalAlpha = 1; ctx.lineWidth = 1;
          modelDrawn++;
        });
      }
    }

    // --- the measurements themselves (drawn last, so they stay on top)
    let hiddenNonPos = 0;
    const order = visible.slice().sort((a, b) => (kindOf(a) === "valid") - (kindOf(b) === "valid"));
    for (const m of order) {
      const i = epochIdx.has(m.epoch_index) ? epochIdx.get(m.epoch_index) : 0;
      const col = EPOCH_COLORS[i % EPOCH_COLORS.length];
      const shape = DET_MARKERS[m.detector] || "circle";
      const x = xOf(m.wavelength_um);
      if (x < padL - 1 || x > padL + plotW + 1) continue;
      const k = kindOf(m);
      let y;
      if (k === "valid" || k === "valid_flagged") {
        if (useLog && !(m.flux_mjy > 0)) { hiddenNonPos++; continue; }
        y = yOf(m.flux_mjy);
        ctx.strokeStyle = col; ctx.lineWidth = 1;
        if (k === "valid_flagged") ctx.setLineDash([2, 2]);
        ctx.beginPath(); ctx.moveTo(x, yOf(m.flux_mjy - m.flux_err_mjy)); ctx.lineTo(x, yOf(m.flux_mjy + m.flux_err_mjy)); ctx.stroke();
        ctx.setLineDash([]);
        markerPath(ctx, shape, x, y, 3.3);
        if (k === "valid") { ctx.fillStyle = col; ctx.fill(); ctx.strokeStyle = "rgba(0,0,0,0.55)"; ctx.stroke(); }
        else { ctx.fillStyle = "#000"; ctx.fill(); ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.stroke(); ctx.lineWidth = 1; }
      } else if (k === "limit") {
        if (useLog && !(m.upper_limit_mjy > 0)) { hiddenNonPos++; continue; }
        y = yOf(m.upper_limit_mjy);
        ctx.strokeStyle = col; ctx.lineWidth = 1.2; ctx.beginPath();
        ctx.moveTo(x - 4, y); ctx.lineTo(x + 4, y); ctx.moveTo(x, y); ctx.lineTo(x, y + 12);
        ctx.moveTo(x - 3, y + 8); ctx.lineTo(x, y + 12); ctx.lineTo(x + 3, y + 8); ctx.stroke();
        ctx.lineWidth = 1;
      } else {
        y = padT + plotH - 6;               // no flux: marked on the baseline, never given a value
        ctx.strokeStyle = m.state === "saturated" ? "#e0603c" : "#6b6048";
        ctx.beginPath(); ctx.moveTo(x - 3, y - 3); ctx.lineTo(x + 3, y + 3); ctx.moveTo(x + 3, y - 3); ctx.lineTo(x - 3, y + 3); ctx.stroke();
      }
      st.hoverPoints.push({ x, y, m });
    }
    if (!visible.length) {
      ctx.fillStyle = "#8f8160";
      ctx.fillText("Nothing to show with the current epoch and quality selection.", padL + 10, padT + plotH / 2);
    }

    // --- caption: what a point is, and what is missing
    const counts = { valid: 0, valid_flagged: 0, limit: 0, noflux: 0 };
    for (const m of visible) counts[kindOf(m)]++;
    const allCounts = { valid: 0, valid_flagged: 0, limit: 0, noflux: 0 };
    for (const m of ms) if (st.selectedEpochs.set.has(m.epoch_index)) allCounts[kindOf(m)]++;
    const missing = SPHEREX_BANDS.map((b) => b[0]).filter((d) => !present.has(d));
    const spans = s.epochs.map((e, i) => `E${i + 1} ${(e.mjd_max - e.mjd_min).toFixed(1)} d`).join(", ");
    const parts = [
      `Each point is ONE exposure measured on native pixels at its own wavelength; nothing is coadded, averaged or merged across exposures, epochs or detectors.`,
      `An epoch is a cluster of exposures taken over a span of time (${spans}), not one instant; colour shows the epoch, not a single time.`,
      `Shown: ${counts.valid} valid, ${counts.valid_flagged} flagged${showFlagged ? "" : ` (hidden: ${allCounts.valid_flagged})`}, ${counts.limit} upper limits${showLimits ? "" : ` (hidden: ${allCounts.limit})`}, ${counts.noflux} without flux${showNoflux ? "" : ` (hidden: ${allCounts.noflux})`}.`,
      `Detectors with data: ${[...present].sort().join(", ") || "none"}.` +
        (missing.length ? ` No exposures in this search for ${missing.join(", ")}: hatched bands are coverage gaps, not zero flux.` : ""),
      `Flux scale: ${useLog ? "log" : "linear"}${stScaleSelect.value === "auto" ? ` (auto; plotted dynamic range ${dynRange.toFixed(1)}×)` : ""}.` +
        (hiddenNonPos ? ` ${hiddenNonPos} point(s) with F ≤ 0 cannot be drawn on a log scale (switch to linear).` : ""),
    ];
    if (stShowModel.checked) parts.push(modelDrawn
      ? `Dashed lines: MODEL GUIDE only, each detector's shared-shape fit a_e·S(λ) per epoch (no per-epoch tilt), drawn only over the fitted wavelength range; faint where the model does not describe the source.`
      : `Model guide: no detector had enough data for the shared-shape fit.`);
    stSedCaption.textContent = parts.join(" ");
  }

  function stTooltipHtml(m) {
    const ep = stSource().epochs.find((e) => e.index === m.epoch_index);
    const rows = [
      ["Detector", m.detector],
      ["Epoch", `${m.epoch_index != null ? m.epoch_index + 1 : "?"}${ep ? ` (${fmtDate(ep.mjd_min)}…${fmtDate(ep.mjd_max)})` : ""}`],
      ["Time", Number.isFinite(m.time_mjd) ? `MJD ${m.time_mjd.toFixed(4)} (${fmtDate(m.time_mjd)})` : "?"],
      ["λ", m.wavelength_um != null ? `${m.wavelength_um.toFixed(4)} µm` : "?"],
      ["Δλ", m.bandwidth_um != null ? `${m.bandwidth_um.toFixed(4)} µm` : "?"],
    ];
    if (m.state === "below_threshold" && m.upper_limit_mjy != null) {
      rows.push(["Flux", `< ${fmtFlux(m.upper_limit_mjy)} mJy (3σ upper limit)`]);
      if (m.flux_mjy != null) rows.push(["Measured", `${fmtFlux(m.flux_mjy)} ± ${fmtFlux(m.flux_err_mjy)} mJy`]);
    } else if (m.flux_mjy != null) {
      rows.push(["Flux", `${fmtFlux(m.flux_mjy)} ± ${fmtFlux(m.flux_err_mjy)} mJy`]);
    } else {
      rows.push(["Flux", "none"]);
    }
    rows.push(["S/N", m.snr != null && Number.isFinite(m.snr) ? m.snr.toFixed(1) : "—"]);
    rows.push(["Quality", STATE_TEXT[m.state] || m.state]);
    rows.push(["Flags", (m.flags_in_aperture || []).join(", ") || "none"]);
    rows.push(["obs_id", m.obs_id]);
    const esc = (v) => String(v).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
    return rows.map(([k, v]) => `<div><span class="st-tt-k">${esc(k)}</span> ${esc(v)}</div>`).join("");
  }

  stSedCanvas.addEventListener("mousemove", (ev) => {
    const r = stSedCanvas.getBoundingClientRect();
    const sx = stSedCanvas.width / r.width, sy = stSedCanvas.height / r.height;
    const x = (ev.clientX - r.left) * sx, y = (ev.clientY - r.top) * sy;
    let best = null, bd = 9 * sx;
    for (const p of st.hoverPoints) {
      const dd = Math.hypot(p.x - x, p.y - y);
      if (dd < bd) { bd = dd; best = p; }
    }
    if (!best) {
      stTooltip.hidden = true;
      stHover.textContent = "Hover a point to see its measurement.";
      return;
    }
    stHover.textContent = describeMeasurement(best.m);
    stTooltip.innerHTML = stTooltipHtml(best.m);
    stTooltip.hidden = false;
    const wrap = stSedCanvas.parentElement.getBoundingClientRect();
    const px = best.x / sx + (r.left - wrap.left), py = best.y / sy + (r.top - wrap.top);
    const tw = stTooltip.offsetWidth, th = stTooltip.offsetHeight;
    stTooltip.style.left = `${px + 12 + tw > wrap.width ? Math.max(0, px - 12 - tw) : px + 12}px`;
    stTooltip.style.top = `${Math.max(0, Math.min(py - th / 2, wrap.height - th))}px`;
  });
  stSedCanvas.addEventListener("mouseleave", () => { stTooltip.hidden = true; });
  for (const c of [stRangeSelect, stScaleSelect, stShowFlagged, stShowLimits, stShowNoflux, stShowModel]) {
    c.addEventListener("change", () => { const s = stSource(); if (s) drawSed(s); });
  }

  function describeMeasurement(m) {
    const f = m.flux_mjy == null ? "no flux"
      : m.state === "below_threshold" ? `< ${fmtFlux(m.upper_limit_mjy)} mJy (3σ upper limit; measured ${fmtFlux(m.flux_mjy)} ± ${fmtFlux(m.flux_err_mjy)})`
      : `${fmtFlux(m.flux_mjy)} ± ${fmtFlux(m.flux_err_mjy)} mJy (S/N ${m.snr != null ? m.snr.toFixed(0) : "?"})`;
    return `${m.detector} λ = ${m.wavelength_um != null ? m.wavelength_um.toFixed(3) : "?"} µm (bandwidth ${m.bandwidth_um != null ? m.bandwidth_um.toFixed(3) : "?"}), ` +
      `${fmtDate(m.time_mjd)} (MJD ${m.time_mjd.toFixed(3)}), epoch ${m.epoch_index + 1}, ${m.obs_id}: ${f}; ${STATE_TEXT[m.state] || m.state}` +
      `${m.flags_in_aperture.length ? `; flags in aperture: ${m.flags_in_aperture.join(", ")}` : ""}.`;
  }

  /** Per-detector light curves: every usable exposure (small points, at its own
   * wavelength) and the shared-shape fit's flux at λ0 per epoch (large). */
  function drawStLightCurves(s) {
    const ctx = stLcCanvas.getContext("2d");
    const W = stLcCanvas.width, H = stLcCanvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.font = ST_FONT;
    const dets = s.detectors;
    const ph = (H - 10) / Math.max(dets.length, 1);
    const tAll = s.measurements.map((m) => m.time_mjd);
    const t0 = Math.min(...tAll), t1 = Math.max(...tAll);
    const padL = 58, padR = 10;
    const xOf = (t) => padL + ((t - t0) / (t1 - t0 || 1)) * (W - padL - padR);
    dets.forEach((det, k) => {
      const top = 5 + k * ph, bot = top + ph - 18;
      const fit = s.fits[det];
      const use = s.measurements.filter((m) => m.detector === det && m.flux_mjy != null && (m.state === "valid" || m.state === "valid_flagged"));
      ctx.strokeStyle = "rgba(125,107,69,0.7)"; ctx.strokeRect(padL, top, W - padL - padR, bot - top);
      ctx.fillStyle = "#8f8160";
      if (!use.length) { ctx.fillText(`${det}: no valid flux in any epoch`, padL + 6, (top + bot) / 2); return; }
      const vals = use.map((m) => m.flux_mjy).concat(fit ? fit.amplitudes_mjy.map((a, i) => a + fit.amplitude_errs_mjy[i]) : [])
        .concat(fit ? fit.amplitudes_mjy.map((a, i) => a - fit.amplitude_errs_mjy[i]) : []);
      const lo = Math.min(...vals), hi = Math.max(...vals), span = hi - lo || Math.abs(hi) || 1;
      const yOf = (f) => bot - ((f - (lo - 0.08 * span)) / (1.16 * span)) * (bot - top);
      ctx.fillText(`${fmtFlux(hi)}`, 4, top + 10); ctx.fillText(`${fmtFlux(lo)}`, 4, bot);
      const lmin = Math.min(...use.map((m) => m.wavelength_um)), lmax = Math.max(...use.map((m) => m.wavelength_um));
      for (const m of use) {
        const hue = 200 - 160 * ((m.wavelength_um - lmin) / (lmax - lmin || 1));
        ctx.fillStyle = `hsla(${hue},70%,60%,0.75)`;
        ctx.beginPath(); ctx.arc(xOf(m.time_mjd), yOf(m.flux_mjy), 2.2, 0, 2 * Math.PI); ctx.fill();
      }
      let label = `${det}: small dots = single exposures (colour = wavelength ${lmin.toFixed(2)}–${lmax.toFixed(2)} µm)`;
      if (fit) {
        fit.epochs.forEach((e, i) => {
          const ep = s.epochs.find((x) => x.index === e);
          const x = xOf(ep.mjd_mean), a = fit.amplitudes_mjy[i], ea = fit.amplitude_errs_mjy[i];
          ctx.strokeStyle = "#f3e3b8"; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.moveTo(x, yOf(a - ea)); ctx.lineTo(x, yOf(a + ea)); ctx.stroke();
          ctx.beginPath(); ctx.rect(x - 4, yOf(a) - 4, 8, 8);
          if (fit.model_adequate) { ctx.fillStyle = "#f3e3b8"; ctx.fill(); } else { ctx.stroke(); }
        });
        label += `; squares = flux at λ₀ = ${fit.lambda0_um.toFixed(2)} µm per epoch (shared-shape fit${fit.model_adequate ? "" : ", model does NOT describe this source: hollow, not reliable"})`;
      }
      ctx.fillStyle = "#8f8160"; ctx.fillText(label, padL + 4, bot + 13);
    });
  }

  function renderStCoverage(s) {
    stCoverage.innerHTML = "";
    const tbl = el("table", null, "analysis-table");
    const hr = el("tr");
    ["Detector", "Epoch", "Usable / exposures", "Wavelengths sampled (usable)", "Measurement states"].forEach((h) => hr.appendChild(el("th", h)));
    const thead = el("thead"); thead.appendChild(hr); tbl.appendChild(thead);
    const tb = el("tbody");
    for (const c of s.coverage) {
      const tr = el("tr");
      tr.appendChild(el("td", c.detector));
      tr.appendChild(el("td", String(c.epoch + 1)));
      tr.appendChild(el("td", `${c.n_usable} / ${c.n_exposures}`));
      tr.appendChild(el("td", c.lambda_min_um == null ? "none" : `${c.lambda_min_um.toFixed(3)}–${c.lambda_max_um.toFixed(3)} µm (${c.n_usable} points; gaps between them are not observed)`));
      tr.appendChild(el("td", Object.entries(c.states).map(([k, v]) => `${v} ${STATE_TEXT[k] || k}`).join(", ") || "—"));
      tb.appendChild(tr);
    }
    tbl.appendChild(tb);
    stCoverage.appendChild(tbl);
  }

  function renderStMetrics(s) {
    stMetrics.innerHTML = "";
    const main = s.metrics.filter((m) => m.data === "all_usable");
    const alt = s.metrics.filter((m) => m.data === "unflagged_only");
    for (const m of main) {
      const box = el("div", null, "st-metric");
      const [name, def] = METRIC_TEXT[m.metric] || [m.metric, ""];
      const head = el("p");
      const [bt, bc] = ASSESS[m.assessment] || [m.assessment, "badge-dim"];
      head.appendChild(el("span", bt, `badge ${bc}`));
      const where = m.detector || (m.detectors ? m.detectors.join("/") : "");
      head.appendChild(document.createTextNode(` ${name} — ${where}`));
      head.title = def;
      box.appendChild(head);
      const nums = [];
      if (m.status === "ok") {
        const stat = m.delta_chi2 != null ? `Δχ² = ${m.delta_chi2.toFixed(1)}` : `χ² = ${m.chi2.toFixed(1)}`;
        nums.push(`${stat}, dof = ${m.dof}, p = ${m.p_value != null ? m.p_value.toExponential(1) : "—"}`);
        if (m.amplitudes_mjy) nums.push(`flux at ${m.lambda0_um.toFixed(2)} µm per epoch: ${m.amplitudes_mjy.map((a, i) => `${fmtFlux(a)}±${fmtFlux(m.amplitude_errs_mjy[i])}`).join(", ")} mJy`);
        if (m.fractional_range != null && m.metric === "grey_variability") nums.push(`range ${(100 * m.fractional_range).toFixed(0)} ± ${(100 * m.fractional_range_err).toFixed(0)}%`);
        if (m.median_ratio_later_over_earlier != null) nums.push(`${m.n_pairs} pair(s); median ratio later/earlier ${m.median_ratio_later_over_earlier.toFixed(2)}`);
        if (m.ratio) nums.push(`flux ratio per epoch ${m.ratio.map((r) => r.toFixed(2)).join(", ")} (max change ${(100 * (Math.exp(m.max_change) - 1)).toFixed(0)}%); ${Math.round(100 * (m.simultaneous_fraction || 0))}% of exposures are simultaneous dichroic pairs`);
        if (m.tilts_per_um) nums.push(`tilt per epoch ${m.tilts_per_um.map((t, i) => `${t.toFixed(2)}±${m.tilt_errs_per_um[i].toFixed(2)}`).join(", ")} /µm (first epoch = 0)`);
        nums.push(`systematic floor ${(100 * (m.sys_floor || 0.025)).toFixed(1)}% per exposure`);
      }
      box.appendChild(el("p", nums.join(" · "), "status-text"));
      box.appendChild(el("p", m.assessment_text));
      const a = alt.find((x) => x.metric === m.metric && (x.detector || "") === (m.detector || ""));
      if (a) box.appendChild(el("p", `Without flagged measurements: ${a.status === "ok" ? `${(ASSESS[a.assessment] || [a.assessment])[0]} (p = ${a.p_value != null ? a.p_value.toExponential(1) : "—"})` : a.reason || "not testable"}.`, "status-text"));
      box.appendChild(el("p", `Needs: ${m.requires || "—"}.`, "status-text"));
      stMetrics.appendChild(box);
    }
  }

  function renderStTable(s) {
    stTableBody.innerHTML = "";
    for (const m of s.measurements) {
      const tr = el("tr");
      [fmtDate(m.time_mjd), m.time_mjd.toFixed(4), String(m.epoch_index + 1), m.obs_id, m.detector,
       m.wavelength_um != null ? m.wavelength_um.toFixed(4) : "—", m.bandwidth_um != null ? m.bandwidth_um.toFixed(4) : "—",
       m.flux_mjy != null ? fmtFlux(m.flux_mjy) : "—", m.flux_err_mjy != null ? fmtFlux(m.flux_err_mjy) : "—",
       m.snr != null ? m.snr.toFixed(1) : "—", m.upper_limit_mjy != null ? fmtFlux(m.upper_limit_mjy) : "—",
       STATE_TEXT[m.state] || m.state, m.flags_in_aperture.join(", ") || "—"].forEach((v) => tr.appendChild(el("td", v)));
      stTableBody.appendChild(tr);
    }
  }

  stSourceSelect.addEventListener("change", renderStSource);

  // ---------------------------------------------------------------------
  // False-colour IR composite (RGB). Built client-side from the per-detector
  // frames already resampled onto ONE common grid (resample.py), so pixel i
  // is the same sky position in every channel. Rules:
  //  * channels come from ONE observation epoch by default (frames whose mean
  //    times lie within RGB_EPOCH_GAP_DAYS; dichroic pairs D1/D4, D2/D5, D3/D6
  //    are simultaneous), because variability between epochs would otherwise
  //    show up as colour;
  //  * shortest wavelength -> blue, longest -> red (2 bands: G = 50/50 blend);
  //  * a pixel with no data or a display-interpolated value in ANY channel is
  //    drawn grey-hatched, never as zero flux in that channel;
  //  * default stretch is the colour-preserving asinh of Lupton et al. (2004).
  // ---------------------------------------------------------------------
  const rgbControls = $("rgb-controls");
  const rgbLegend = $("rgb-legend");
  const rgbEpochSelect = $("rgb-epoch-select");
  const rgbPrevBtn = $("rgb-prev");
  const rgbNextBtn = $("rgb-next");
  const rgbPlayBtn = $("rgb-play");
  const rgbStretchMode = $("rgb-stretch-mode");
  const rgbSoftSlider = $("rgb-soft-slider");
  const rgbBlackSlider = $("rgb-black-slider");
  const rgbWhiteSlider = $("rgb-white-slider");
  const rgbSoftValue = $("rgb-soft-value");
  const rgbBlackValue = $("rgb-black-value");
  const rgbWhiteValue = $("rgb-white-value");
  const rgbManual = { R: $("rgb-r-select"), G: $("rgb-g-select"), B: $("rgb-b-select") };
  const RGB_EPOCH_GAP_DAYS = 45;            // same gap as the backend's epoch clustering
  const RGB_SIMULTANEOUS_DAYS = 1;
  const SAT_FLAG_NAMES = ["OVERFLOW", "BLOOM", "NONLINEAR"];   // = pipeline.SATURATION_FLAG_NAMES (SUR_ERROR is mostly on empty sky)
  const rgb = { epochIndex: 0, signature: "", timer: null, overlayFrameIdx: null };

  const rgbSoftening = () => Math.pow(10, -0.5 + (3 * parseFloat(rgbSoftSlider.value)) / 100);   // 0.3 .. 300 sigma
  const rgbBlack = () => parseFloat(rgbBlackSlider.value);                                        // sigma
  const rgbWhite = () => parseFloat(rgbWhiteSlider.value);                                        // percentile of intensity
  const frameKey = (f) => `${f.detectors[0]}@${f.timeMjdMean.toFixed(5)}`;
  const frameText = (f, g) => `${f.detectors[0]} ${f.centralWavelengthUm.toFixed(2)} µm · epoch ${g != null ? g + 1 : "?"} · MJD ${f.timeMjdMean.toFixed(2)}`;

  /** Single-detector frames on the common grid, with a known wavelength. */
  function rgbFrames() {
    const all = (state.allFrames || []).filter((f) => f.detectors && f.detectors.length === 1 && f.centralWavelengthUm != null);
    if (!all.length) return [];
    const w = all[0].width, h = all[0].height;
    return all.filter((f) => f.width === w && f.height === h);
  }

  function rgbEpochGroups(frames) {
    const sorted = frames.slice().sort((a, b) => a.timeMjdMean - b.timeMjdMean);
    const groups = [];
    for (const f of sorted) {
      const g = groups[groups.length - 1];
      if (g && f.timeMjdMean - g.tMax <= RGB_EPOCH_GAP_DAYS) { g.frames.push(f); g.tMax = f.timeMjdMean; }
      else groups.push({ frames: [f], tMin: f.timeMjdMean, tMax: f.timeMjdMean });
    }
    groups.forEach((g, i) => (g.index = i));
    return groups;
  }

  /** One frame per detector, sorted by wavelength; mapping per the band count. */
  function rgbAutoMapping(group) {
    const seen = new Set();
    const uniq = group.frames.slice().sort((a, b) => a.centralWavelengthUm - b.centralWavelengthUm)
      .filter((f) => !seen.has(f.detectors[0]) && seen.add(f.detectors[0]));
    if (uniq.length < 2) return { error: `Epoch ${group.index + 1} has only ${uniq.length} wavelength band (${uniq.map((f) => f.detectors[0]).join(", ") || "none"}). A false-colour composite needs at least 2 distinct bands from the same epoch: search with 2 or more detectors (e.g. D3,D6).` };
    if (uniq.length === 2) return { R: [uniq[1]], G: "blend", B: [uniq[0]], bands: uniq.length };
    const n = uniq.length, a = Math.round(n / 3), b = Math.round((2 * n) / 3);
    return { B: uniq.slice(0, a), G: uniq.slice(a, b), R: uniq.slice(b), bands: n };
  }

  function sortedFinite(arr, mask) {
    const out = [];
    for (let i = 0; i < arr.length; i++) if (!mask[i] && Number.isFinite(arr[i])) out.push(arr[i]);
    return Float32Array.from(out).sort();
  }
  function pctl(sorted, p) {
    if (!sorted.length) return NaN;
    return sorted[Math.min(sorted.length - 1, Math.max(0, Math.round((p / 100) * (sorted.length - 1))))];
  }
  function robustStats(arr, mask) {
    const s = sortedFinite(arr, mask);
    const med = pctl(s, 50);
    const dev = Float32Array.from(s, (v) => Math.abs(v - med)).sort();
    return { median: med, sigma: 1.4826 * pctl(dev, 50), n: s.length };
  }

  /** 1 = this grid pixel was built partly from saturated / non-linear native
   * pixels (OVERFLOW, BLOOM, NONLINEAR) in at least one exposure. */
  function saturationMask(f) {
    if (!f.saturationMaskB64) return null;
    if (f._satMask) return f._satMask;
    const bytes = base64ToBytes(f.saturationMaskB64);
    const N = f.width * f.height, out = new Uint8Array(N);
    for (let i = 0; i < N; i++) out[i] = (bytes[i >> 3] >> (i & 7)) & 1;
    Object.defineProperty(f, "_satMask", { value: out, enumerable: false });
    return out;
  }
  function base64ToBytes(b64) {
    const bin = atob(b64), out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  /** Background-subtracted mean of the member frames; marks unusable pixels
   * in `masked`: 1 = no data or display-interpolated, 2 = saturation-flagged. */
  function rgbChannel(members, masked) {
    const N = members[0].width * members[0].height;
    const out = new Float32Array(N);
    const diag = [];
    for (const f of members) {
      const v = physicalValuesForFrame(f);
      const fmask = new Uint8Array(N);
      const sat = saturationMask(f);
      let nNan = 0, nInterp = 0, nSat = 0;
      for (let i = 0; i < N; i++) {
        if (!Number.isFinite(v[i])) { fmask[i] = 1; nNan++; }
        else if (f.gapFilled && f.gapFilled.has(i)) { fmask[i] = 1; nInterp++; }
        else if (sat && sat[i]) { fmask[i] = 2; nSat++; }
      }
      const bg = robustStats(v, fmask).median;
      for (let i = 0; i < N; i++) {
        if (fmask[i]) masked[i] = masked[i] === 1 ? 1 : fmask[i];
        else out[i] += (v[i] - (Number.isFinite(bg) ? bg : 0)) / members.length;
      }
      diag.push({ frame: f, bg, nNan, nInterp, nSat, hasMask: !!sat, sat: SAT_FLAG_NAMES.map((k) => [k, (f.contamination || {})[k] || 0]).filter(([, c]) => c > 0) });
    }
    return { values: out, diag };
  }

  /** Current channel assignment: auto for the selected epoch, with optional manual overrides. */
  function rgbAssignment(groups) {
    const group = groups[Math.min(rgb.epochIndex, groups.length - 1)];
    const auto = rgbAutoMapping(group);
    const byKey = new Map(rgbFrames().map((f) => [frameKey(f), f]));
    const manual = {};
    for (const c of ["R", "G", "B"]) {
      const v = rgbManual[c].value;
      if (v === "blend") manual[c] = "blend";
      else if (v !== "auto" && byKey.has(v)) manual[c] = [byKey.get(v)];
    }
    const anyManual = Object.keys(manual).length > 0;
    if (auto.error && !(manual.R && manual.B)) return { group, error: auto.error };
    const pick = (c) => manual[c] || (auto.error ? null : auto[c]);
    const map = { R: pick("R"), G: pick("G") || "blend", B: pick("B") };
    if (!map.R || !map.B || map.R === "blend" || map.B === "blend") return { group, error: "Red and blue must each be a frame (only green can be a 50/50 blend)." };
    return { group, map, anyManual, bands: auto.bands };
  }

  function refreshRgbSelectors(frames, groups) {
    const sig = frames.map(frameKey).join("|");
    if (sig === rgb.signature) return;
    rgb.signature = sig;
    rgb.epochIndex = Math.min(rgb.epochIndex, Math.max(groups.length - 1, 0));
    rgbEpochSelect.innerHTML = "";
    for (const g of groups) {
      const dets = Array.from(new Set(g.frames.map((f) => f.detectors[0]))).sort();
      const o = document.createElement("option");
      o.value = String(g.index);
      o.textContent = `Epoch ${g.index + 1}: ${fmtDate(g.tMin)}${g.tMax - g.tMin > 0.5 ? `…${fmtDate(g.tMax)}` : ""} (${dets.join("+")})`;
      rgbEpochSelect.appendChild(o);
    }
    rgbEpochSelect.value = String(rgb.epochIndex);
    const epochOf = (f) => groups.findIndex((g) => g.frames.includes(f));
    for (const c of ["R", "G", "B"]) {
      const sel = rgbManual[c], prev = sel.value;
      sel.innerHTML = "";
      const add = (v, t) => { const o = document.createElement("option"); o.value = v; o.textContent = t; sel.appendChild(o); };
      add("auto", "Auto (same epoch, by wavelength)");
      if (c === "G") add("blend", "50/50 blend of red and blue");
      frames.slice().sort((a, b) => a.timeMjdMean - b.timeMjdMean || a.centralWavelengthUm - b.centralWavelengthUm)
        .forEach((f) => add(frameKey(f), frameText(f, epochOf(f))));
      sel.value = Array.from(sel.options).some((o) => o.value === prev) ? prev : "auto";
    }
  }

  function renderRgbView() {
    const frames = rgbFrames();
    const groups = rgbEpochGroups(frames);
    refreshRgbSelectors(frames, groups);
    rgbSoftValue.textContent = `${rgbSoftening().toFixed(rgbSoftening() < 10 ? 1 : 0)}σ`;
    rgbBlackValue.textContent = `${rgbBlack().toFixed(1)}σ`;
    rgbWhiteValue.textContent = `${rgbWhite().toFixed(1)}%`;
    rgbLegend.innerHTML = "";
    rgbLegend.hidden = false;
    const fail = (msg) => {
      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
      state.lastRenderedGrid = null;
      drawOverlay();
      rgbLegend.appendChild(el("p", msg, "rgb-warning"));
      frameLabel.textContent = "False-colour IR composite unavailable.";
    };
    if ((state.allFrames || []).some((f) => f.detectors && f.detectors.length > 1)) {
      fail("These frames stack several detectors (wavelengths) together. A false-colour composite needs per-detector frames: choose “per detector, per epoch” grouping and search again.");
      return;
    }
    if (groups.length === 0) { fail("No frames with a known wavelength are loaded."); return; }
    const asg = rgbAssignment(groups);
    if (asg.error) { fail(asg.error); return; }

    const { map } = asg;
    const ref = map.R[0];
    const W = ref.width, H = ref.height, N = W * H;
    const masked = new Uint8Array(N);
    const chR = rgbChannel(map.R, masked), chB = rgbChannel(map.B, masked);
    const chG = map.G === "blend" ? null : rgbChannel(map.G, masked);
    let r = chR.values, b = chB.values;
    let g = chG ? chG.values : Float32Array.from(r, (v, i) => 0.5 * (v + b[i]));
    const perChannel = rgbStretchMode.value === "per_channel";
    const scales = {};
    if (perChannel) {
      const norm = (arr, name) => {
        const s = pctl(sortedFinite(arr, masked), 99.5);
        const k = s > 0 ? s : 1;
        scales[name] = k;
        return Float32Array.from(arr, (v) => v / k);
      };
      r = norm(r, "R"); b = norm(b, "B"); g = chG ? norm(g, "G") : Float32Array.from(r, (v, i) => 0.5 * (v + b[i]));
    }

    // Lupton et al. (2004): one intensity I = (r+g+b)/3 drives the stretch,
    // every channel is multiplied by the SAME factor asinh(I'/beta)/I, and a
    // pixel whose largest channel exceeds 1 is scaled down as a whole, so
    // R:G:B ratios survive in bright cores instead of saturating to white.
    const I = new Float32Array(N);
    for (let i = 0; i < N; i++) I[i] = (Math.max(r[i], 0) + Math.max(g[i], 0) + Math.max(b[i], 0)) / 3;
    const st0 = robustStats(Float32Array.from(r, (v, i) => (v + g[i] + b[i]) / 3), masked);
    const sigma = st0.sigma > 0 ? st0.sigma : 1e-6;
    const beta = rgbSoftening() * sigma, bp = rgbBlack() * sigma;
    const Ip = Float32Array.from(I, (v) => v - bp);
    const Iw = Math.max(pctl(sortedFinite(Ip, masked), rgbWhite()), beta * 1e-3);
    const normK = 1 / Math.asinh(Iw / beta);
    // A channel significantly BELOW the sky (< -3 sigma of that channel) is
    // not a colour: next to bright stars it is the negative lobe of the
    // Lanczos-3 resampling kernel. Clipping it to 0 would paint the pixel in
    // the pure colour of the other channel, so such pixels are drawn
    // neutral (grey at their intensity) and counted.
    const negLimit = [r, g, b].map((arr) => -3 * (robustStats(arr, masked).sigma || 0));
    const rgba = new Uint8ClampedArray(N * 4);
    let nMasked = 0, nNeutral = 0, nSatMasked = 0;
    for (let i = 0; i < N; i++) {
      const ay = Math.floor(i / W), ax = i - ay * W;
      const o = ((H - 1 - ay) * W + ax) * 4;
      if (masked[i]) {
        // neutral checker; saturation-flagged pixels lighter, with a cross-hatch drawn on top
        const c = masked[i] === 2 ? ((ax + ay) % 2 ? 150 : 175) : ((ax + ay) % 2 ? 92 : 118);
        rgba[o] = c; rgba[o + 1] = c; rgba[o + 2] = c; rgba[o + 3] = 255;
        if (masked[i] === 2) nSatMasked++; else nMasked++;
        continue;
      }
      let R = 0, G = 0, B = 0;
      if ((r[i] < negLimit[0] || g[i] < negLimit[1] || b[i] < negLimit[2]) && Math.max(r[i], g[i], b[i]) > 0) {
        const v = Ip[i] > 0 ? Math.min(1, Math.asinh(Ip[i] / beta) * normK) : 0;
        const q = Math.round(255 * v);
        rgba[o] = q; rgba[o + 1] = q; rgba[o + 2] = q; rgba[o + 3] = 255;
        nNeutral++;
        continue;
      }
      if (Ip[i] > 0 && I[i] > 0) {
        const k = (Math.asinh(Ip[i] / beta) * normK) / I[i];
        R = Math.max(r[i], 0) * k; G = Math.max(g[i], 0) * k; B = Math.max(b[i], 0) * k;
        const m = Math.max(R, G, B);
        if (m > 1) { R /= m; G /= m; B /= m; }
      }
      rgba[o] = Math.round(255 * R); rgba[o + 1] = Math.round(255 * G); rgba[o + 2] = Math.round(255 * B); rgba[o + 3] = 255;
    }

    // overlays: the detections of the selected detector series in THIS epoch
    const g0 = asg.group;
    rgb.overlayFrameIdx = null;
    state.frames.forEach((f, k) => {
      if (f.timeMjdMean >= g0.tMin - RGB_SIMULTANEOUS_DAYS && f.timeMjdMean <= g0.tMax + RGB_SIMULTANEOUS_DAYS) rgb.overlayFrameIdx = k;
    });

    const chans = [["R", "Red", map.R, chR], ["G", "Green", map.G, chG], ["B", "Blue", map.B, chB]];
    state.lastRenderedGrid = {
      width: W, height: H, values: r, frame: map.R[0], wcsFrame: map.R[0],
      rgb: { channels: chans.map(([c, , m, ch]) => ({ c, label: m === "blend" ? "50/50 blend" : m.map((f) => f.detectors[0]).join("+"), values: ch ? ch.values : null })), masked },
    };
    if (canvas.width !== W) canvas.width = W;
    if (canvas.height !== H) canvas.height = H;
    canvasCtx.putImageData(new ImageData(rgba, W, H), 0, 0);
    layoutCanvas();   // also redraws the overlay: crosshair, markers, hatching of unusable pixels

    // ---- legend: exact mapping, stretch, warnings
    const desc = (m) => m === "blend" ? "50/50 blend of the red and blue inputs"
      : m.map((f) => `${f.centralWavelengthUm.toFixed(2)} µm (${f.detectors[0]}, MJD ${f.timeMjdMean.toFixed(2)})`).join(" + ") + (m.length > 1 ? ", averaged" : "");
    const mapLine = el("p", null, "rgb-map");
    [["Red", "#ff6b6b", map.R], ["Green", "#7CFC9A", map.G], ["Blue", "#5fb0ff", map.B]].forEach(([n, col, m], k) => {
      const sp = el("span", `${n}: ${desc(m)}`);
      sp.style.color = col;
      mapLine.appendChild(sp);
      if (k < 2) mapLine.appendChild(document.createTextNode("  ·  "));
    });
    rgbLegend.appendChild(mapLine);
    rgbLegend.appendChild(el("p", "False-colour representation of near-infrared wavelengths (not human-visible colour). Bluer = relatively brighter at the shorter wavelength.", "rgb-note"));
    rgbLegend.appendChild(el("p", "The PSF is larger at longer wavelengths and the images are not PSF-matched, so stars show redder halos and bluer cores; read colours from the core of a source, and treat coloured rings around bright stars as optics and resampling, not as structure.", "status-text"));
    rgbLegend.appendChild(el("p", perChannel
      ? `Per-channel normalised asinh: each channel is first divided by its own 99.5th percentile (${Object.entries(scales).map(([k, v]) => `${k} ${v.toPrecision(3)} MJy/sr`).join(", ")}). This enhances contrast; colours do NOT show true flux ratios between wavelengths.`
      : `Colour-preserving asinh (Lupton et al. 2004) on background-subtracted surface brightness (MJy/sr): R:G:B ratios follow the measured ratios between wavelengths. Softening β = ${rgbSoftening().toFixed(1)}σ, black point ${rgbBlack().toFixed(1)}σ, white at the ${rgbWhite().toFixed(1)}th percentile of intensity (σ = ${sigma.toPrecision(2)} MJy/sr, the sky noise of the mean channel).`,
      "status-text"));
    const warnings = [];
    const used = chans.filter(([, , m]) => m !== "blend").flatMap(([, , m]) => m);
    const ts = used.map((f) => f.timeMjdMean);
    const spread = Math.max(...ts) - Math.min(...ts);
    const epochsUsed = new Set(used.map((f) => groups.findIndex((gg) => gg.frames.includes(f))));
    if (epochsUsed.size > 1) warnings.push(`CROSS-EPOCH: the channels come from different epochs (${spread.toFixed(0)} days apart). Any brightness change between those dates appears as a colour difference; this is not a single-time colour image.`);
    else if (spread > RGB_SIMULTANEOUS_DAYS) warnings.push(`The channels were observed ${spread.toFixed(1)} days apart within one epoch, so they are not simultaneous; a source varying on that timescale can change colour.`);
    if (asg.anyManual) warnings.push("Manual channel selection is active.");
    if (nNeutral) warnings.push(`${nNeutral} pixel(s) are drawn without colour (neutral grey): one channel is significantly below the sky there (< −3σ), typically the negative ringing lobe of the resampling kernel beside a bright star. They have no meaningful colour (not blue or red).`);
    if (nSatMasked) warnings.push(`${nSatMasked} pixel(s) are cross-hatched (light grey, ✕): at least one channel there is built from saturated or non-linear data (OVERFLOW, BLOOM or NONLINEAR in at least one exposure). Their colour would be wrong, so none is shown.`);
    if (nMasked) warnings.push(`${nMasked} pixel(s) are grey-hatched (dark grey, /): at least one channel has no data there (blank pixel) or only a value interpolated for display (typically a saturated or flagged core). They are not drawn as zero flux in that channel.`);
    for (const [, n, m, ch] of chans) {
      if (!ch) continue;
      for (const d of ch.diag) {
        const det = d.frame.detectors[0];
        if (d.nInterp) warnings.push(`${n} (${det}): ${d.nInterp} pixel(s) interpolated for display at a bright core: saturated or flagged data, masked in the composite.`);
        if (d.sat.length && !d.hasMask) warnings.push(`${n} (${det}): the input exposures carry ${d.sat.map(([k, c]) => `${k} ${c}`).join(", ")} flagged pixel(s), but this frame has no per-pixel saturation mask (cached before it existed: clear the cache and search again), so the brightest stars' colours may be unreliable even where not hatched.`);
        else if (d.nSat) warnings.push(`${n} (${det}): ${d.nSat} grid pixel(s) flagged for saturation or non-linearity (native flags: ${d.sat.map(([k, c]) => `${k} ${c}`).join(", ")}).`);
      }
    }
    if (warnings.length) {
      const box = el("div", null, "rgb-warning");
      box.appendChild(el("strong", "⚠ Check before interpreting colours"));
      const ul = el("ul");
      warnings.forEach((w) => ul.appendChild(el("li", w)));
      box.appendChild(ul);
      rgbLegend.appendChild(box);
    }
    const seriesNote = currentSeriesAnalysis() ? ` Circles: tracked sources of the ${detectorViewSelect.value} series in this epoch.` : "";
    frameLabel.textContent =
      `False-colour IR composite, epoch ${g0.index + 1}/${groups.length}: ${fmtDate(g0.tMin)}${g0.tMax - g0.tMin > 0.5 ? `…${fmtDate(g0.tMax)}` : ""}\n` +
      `${asg.bands || "?"} band(s) this epoch; ${perChannel ? "per-channel normalised" : "colour-preserving"} asinh.${seriesNote}`;
  }

  function rgbStep(d) {
    const groups = rgbEpochGroups(rgbFrames());
    if (!groups.length) return;
    rgb.epochIndex = (rgb.epochIndex + d + groups.length) % groups.length;
    rgbEpochSelect.value = String(rgb.epochIndex);
    renderView();
  }
  function stopRgbPlay() {
    if (rgb.timer) { clearInterval(rgb.timer); rgb.timer = null; }
    rgbPlayBtn.textContent = "▶ Blink epochs";
  }
  rgbPlayBtn.addEventListener("click", () => {
    if (rgb.timer) { stopRgbPlay(); return; }
    rgbPlayBtn.textContent = "⏸ Stop";
    const speed = parseInt(speedSlider.value, 10);
    rgb.timer = setInterval(() => rgbStep(1), Math.max(300, 1500 - speed * 110));
  });
  rgbPrevBtn.addEventListener("click", () => { stopRgbPlay(); rgbStep(-1); });
  rgbNextBtn.addEventListener("click", () => { stopRgbPlay(); rgbStep(1); });
  rgbEpochSelect.addEventListener("change", () => { stopRgbPlay(); rgb.epochIndex = parseInt(rgbEpochSelect.value, 10) || 0; renderView(); });
  [rgbStretchMode, rgbManual.R, rgbManual.G, rgbManual.B].forEach((c) => c.addEventListener("change", renderView));
  [rgbSoftSlider, rgbBlackSlider, rgbWhiteSlider].forEach((c) => c.addEventListener("input", renderView));

  // ---------------------------------------------------------------------
  // Playback
  // ---------------------------------------------------------------------
  function stopPlayback() {
    state.playing = false;
    playBtn.textContent = "▶";
    if (state.playTimer) {
      clearInterval(state.playTimer);
      state.playTimer = null;
    }
  }

  function startPlayback() {
    if (state.frames.length < 2) {
      // Previously a silent no-op - nothing told you why Play did nothing
      // if the search only turned up 0 or 1 usable frame.
      progressText.textContent =
        state.frames.length === 0
          ? "No frames loaded yet - nothing to play."
          : "Only 1 frame loaded - need at least 2 to blink. Try a larger FOV, more detectors, or a different grouping.";
      progressWrap.hidden = false;
      return;
    }
    state.playing = true;
    playBtn.textContent = "⏸";
    const tick = () => {
      if (abCheckbox.checked) {
        state.abPhase = 1 - state.abPhase;
        showFrame(state.abFrames[state.abPhase]);
      } else {
        let next = state.currentIndex + 1;
        if (next >= state.frames.length) {
          if (!loopCheckbox.checked) {
            stopPlayback();
            return;
          }
          next = 0;
        }
        showFrame(next);
      }
    };
    const speed = parseInt(speedSlider.value, 10); // 1 (slow) .. 10 (fast)
    const intervalMs = 900 - speed * 80;
    state.playTimer = setInterval(tick, Math.max(60, intervalMs));
  }

  function showFrame(index) {
    state.currentIndex = index;
    frameSlider.value = index;
    renderView();
    drawLightCurve();
  }

  playBtn.addEventListener("click", () => (state.playing ? stopPlayback() : startPlayback()));
  stepBackBtn.addEventListener("click", () => {
    stopPlayback();
    showFrame(Math.max(0, state.currentIndex - 1));
  });
  stepFwdBtn.addEventListener("click", () => {
    stopPlayback();
    showFrame(Math.min(state.frames.length - 1, state.currentIndex + 1));
  });
  frameSlider.addEventListener("input", () => {
    stopPlayback();
    showFrame(parseInt(frameSlider.value, 10));
  });
  speedSlider.addEventListener("input", () => {
    if (state.playing) {
      stopPlayback();
      startPlayback();
    }
  });
  abCheckbox.addEventListener("change", () => {
    if (abCheckbox.checked) state.abFrames = [state.currentIndex, Math.min(state.frames.length - 1, state.currentIndex + 1)];
  });

  // ---------------------------------------------------------------------
  // View mode (Phase 3: difference / sum, alongside the default blink)
  // ---------------------------------------------------------------------
  function setPlaybackControlsEnabled(enabled) {
    [stepBackBtn, playBtn, stepFwdBtn, speedSlider, loopCheckbox, abCheckbox, frameSlider].forEach((el) => {
      el.disabled = !enabled;
    });
  }

  viewModeSelect.addEventListener("change", () => {
    const isDiff = viewModeSelect.value === "difference";
    const isBlink = viewModeSelect.value === "blink";
    const isRgb = viewModeSelect.value === "rgb";
    abPick.hidden = !isDiff;
    rgbControls.hidden = !isRgb;
    rgbLegend.hidden = !isRgb;
    if (!isRgb) stopRgbPlay();
    setPlaybackControlsEnabled(isBlink);
    if (!isBlink) stopPlayback();
    renderView();
  });
  diffASlider.addEventListener("input", renderView);
  diffBSlider.addEventListener("input", renderView);

  document.addEventListener("keydown", (ev) => {
    if (document.activeElement === positionInput) return;
    if (viewModeSelect.value === "rgb" && (ev.key === "ArrowRight" || ev.key === "ArrowLeft")) {
      ev.preventDefault();
      stopRgbPlay();
      rgbStep(ev.key === "ArrowRight" ? 1 : -1);
      return;
    }
    if (ev.key === " ") {
      ev.preventDefault();
      state.playing ? stopPlayback() : startPlayback();
    } else if (ev.key === "ArrowRight") {
      stopPlayback();
      showFrame(Math.min(state.frames.length - 1, state.currentIndex + 1));
    } else if (ev.key === "ArrowLeft") {
      stopPlayback();
      showFrame(Math.max(0, state.currentIndex - 1));
    }
  });

  // ---------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------
  (async function init() {
    await window.SpherexCache.requestPersistence();
    updateCacheStatus();

    const urlState = readUrlState();
    fovSlider.value = urlState.fov;
    fovValue.textContent = `${urlState.fov}′`;
    fovWarning.hidden = urlState.fov <= 15;
    modeSelect.value = urlState.mode;
    setDetectors(urlState.detectors);
    const qualityIdx = QUALITY_LEVELS.indexOf(urlState.quality);
    qualitySlider.value = qualityIdx >= 0 ? qualityIdx : 1;
    qualityValue.textContent = QUALITY_LABELS[parseInt(qualitySlider.value, 10)];
    shiftstackCheckbox.checked = urlState.shiftstack;
    stCheckbox.checked = urlState.spectrotemporal;
    if (urlState.position) {
      positionInput.value = urlState.position;
      runSearch();
    }
  })();
})();
