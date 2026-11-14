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
  const QUALITY_LEVELS = ["low", "mid", "high"];
  const QUALITY_LABELS = ["Low (fastest)", "Mid (balanced)", "High (full quality)"];
  const stretchSelect = $("stretch-select");
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

  const offscreen = document.createElement("canvas");
  const offscreenCtx = offscreen.getContext("2d");

  // ---------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------
  const state = {
    frames: [], // {groupIndex, timeMjdMean, detectors, nExposures, centralWavelengthUm, bandwidthUm, contamination, width, height, scale, offset, data: Uint16Array}
    currentIndex: 0,
    playing: false,
    playTimer: null,
    resolvedCoord: null, // {ra, dec}
    fovArcmin: 5,
    releaseTag: "unknown",
    abFrames: [0, 0],
    abPhase: 0,
    lastRenderedGrid: null, // {width, height, values: Float32Array (NaN = no data)}
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
      mode: params.get("mode") || "epoch",
      quality: QUALITY_LEVELS.includes(quality) ? quality : "mid",
    };
  }

  function writeUrlState({ position, fov, detectors, mode, quality }) {
    const params = new URLSearchParams();
    params.set("position", position);
    params.set("fov", fov);
    params.set("detectors", detectors.join(","));
    params.set("mode", mode);
    params.set("quality", quality);
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
  stretchSelect.addEventListener("change", renderView);
  invertCheckbox.addEventListener("change", renderView);
  zoomSlider.addEventListener("input", () => {
    zoomValue.textContent = `${zoomSlider.value}×`;
    renderView();
  });
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

    if (detectors.length === 0) {
      resolveStatus.textContent = "Pick at least one detector.";
      resolveStatus.className = "status-text error";
      return;
    }

    stopPlayback();
    state.frames = [];
    state.currentIndex = 0;
    frameSlider.max = 0;
    frameSlider.value = 0;
    emptyState.hidden = true;
    frameLabel.textContent = "";
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

    resolveStatus.textContent = "Resolving…";
    resolveStatus.className = "status-text";

    let resolved;
    try {
      const resp = await fetch(`${API_BASE}/api/resolve?position=${encodeURIComponent(position)}`);
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || resp.statusText);
      }
      resolved = await resp.json();
    } catch (e) {
      resolveStatus.textContent = `Could not resolve: ${e.message}`;
      resolveStatus.className = "status-text error";
      return;
    }

    state.resolvedCoord = { ra: resolved.ra_deg, dec: resolved.dec_deg };
    state.fovArcmin = fov;

    let statusMsg = `${resolved.name || position} → RA ${resolved.ra_deg.toFixed(5)}, Dec ${resolved.dec_deg.toFixed(5)}`;
    if (resolved.has_proper_motion) statusMsg += ` (proper motion known, propagated per epoch)`;
    resolveStatus.textContent = statusMsg;
    resolveStatus.className = "status-text";

    writeUrlState({ position, fov, detectors, mode, quality });

    if (state.releaseTag === "unknown") {
      state.releaseTag = await fetchReleaseTag();
    }

    const cacheKey = window.SpherexCache.makeKey({
      release: state.releaseTag,
      ra: resolved.ra_deg,
      dec: resolved.dec_deg,
      fov,
      group: `${mode}:${quality}`, // quality changes the actual pixel content, so it must be part of the key
      detector: detectors.slice().sort().join(""),
    });

    progressWrap.hidden = false;
    progressBar.style.width = "0%";
    progressText.textContent = "Checking local cache…";

    const cached = await window.SpherexCache.get(cacheKey);
    if (cached && cached.length) {
      state.frames = cached.map(reviveFrame);
      sortAndShowFirst();
      progressText.textContent = `Loaded ${cached.length} frame(s) from local cache`;
      progressBar.style.width = "100%";
      setTimeout(() => (progressWrap.hidden = true), 1500);
      return; // cache hit: no network call for pixels at all
    }

    progressText.textContent = "Fetching from IRSA…";
    streamFrames(position, fov, detectors, mode, quality, cacheKey);
  }

  function reviveFrame(stored) {
    return { ...stored, data: new Uint16Array(stored.data) };
  }

  async function streamFrames(position, fov, detectors, mode, quality, cacheKey) {
    const url = `${API_BASE}/api/frames?position=${encodeURIComponent(position)}&fov=${fov}&detectors=${detectors.join(",")}&group=${mode}&quality=${quality}`;
    let resp;
    try {
      resp = await fetch(url);
    } catch (e) {
      progressText.textContent = `Network error: ${e.message}`;
      return;
    }
    if (!resp.ok || !resp.body) {
      progressText.textContent = `Server error (${resp.status})`;
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let nGroups = null;
    let received = 0;
    let firstFrameShown = false;

    while (true) {
      const { value, done } = await reader.read();
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
        if (obj.error) {
          progressText.textContent = `Error: ${obj.error}`;
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
          width: obj.width,
          height: obj.height,
          scale: obj.scale,
          offset: obj.offset,
          data: base64ToUint16Array(obj.payload_b64),
        };
        state.frames.push(frame);
        received++;
        if (nGroups) {
          progressBar.style.width = `${Math.min(100, (received / nGroups) * 100)}%`;
          progressText.textContent = `${received} / ${nGroups} frame(s) ready`;
        } else {
          progressText.textContent = `${received} frame(s) ready`;
        }
        if (!firstFrameShown) {
          firstFrameShown = true;
          sortAndShowFirst();
        } else {
          // Frames can complete out of chronological order (a later epoch
          // can finish before an earlier one). Re-sorting by groupIndex can
          // move the frame the user is currently looking at to a different
          // array slot, so track it by identity across the sort rather than
          // leaving `currentIndex` pointing at whatever slot it used to be.
          const shownGroupIndex = state.frames[state.currentIndex]
            ? state.frames[state.currentIndex].groupIndex
            : null;
          sortFrames();
          if (shownGroupIndex != null) {
            const newIdx = state.frames.findIndex((f) => f.groupIndex === shownGroupIndex);
            if (newIdx >= 0) state.currentIndex = newIdx;
          }
          updateFrameSlider();
          // Re-render: cheap for these small arrays, and keeps the "Frame
          // X/Y" count and light curve in sync as the total grows, even
          // when the user is sitting on an already-rendered frame.
          renderView();
        }
      }
    }

    progressText.textContent = `Done: ${state.frames.length} frame(s)`;
    setTimeout(() => (progressWrap.hidden = true), 2000);

    if (state.frames.length === 0) {
      emptyState.hidden = false;
      emptyState.textContent = "No SPHEREx images found at this position/FOV/detector selection.";
      return;
    }

    // Cache the whole set now that the stream is complete.
    const serializable = state.frames.map((f) => ({ ...f, data: f.data.buffer.slice(0) }));
    const approxBytes = serializable.reduce((s, f) => s + f.data.byteLength, 0);
    window.SpherexCache.put(cacheKey, serializable, approxBytes);
    updateCacheStatus();
  }

  function sortFrames() {
    state.frames.sort((a, b) => a.groupIndex - b.groupIndex);
  }

  function sortAndShowFirst() {
    sortFrames();
    state.currentIndex = 0;
    updateFrameSlider();
    renderView();
  }

  function updateFrameSlider() {
    const maxIdx = Math.max(0, state.frames.length - 1);
    frameSlider.max = maxIdx;
    frameSlider.value = state.currentIndex;
    diffASlider.max = maxIdx;
    diffBSlider.max = maxIdx;
    if (diffBSlider.value == 0 && maxIdx > 0) diffBSlider.value = maxIdx; // default B to the latest epoch
    if (state.frames.length) emptyState.hidden = true;
    drawLightCurve();
  }

  // ---------------------------------------------------------------------
  // Stretch / rendering
  // ---------------------------------------------------------------------
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
    return { black: pct(blackPct), white: pct(whitePct) };
  }

  function stretchValue(phys, black, white, mode, invert) {
    let t;
    const range = white - black || 1e-9;
    if (mode === "asinh") {
      const A = 10;
      const lin = (phys - black) / range;
      t = Math.asinh(lin * A) / Math.asinh(A);
    } else {
      t = (phys - black) / range;
    }
    t = Math.min(1, Math.max(0, t));
    if (invert) t = 1 - t;
    return Math.round(t * 255);
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
      renderGrid(a.width, a.height, diff);
      frameLabel.textContent =
        `Difference: epoch @ ${fmtDate(b.timeMjdMean)}  −  epoch @ ${fmtDate(a.timeMjdMean)}\n` +
        `A moving/variable source shows as a dark/bright dipole pair.`;
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
      renderGrid(first.width, first.height, mean);
      frameLabel.textContent = `Sum of all ${state.frames.length} epoch(s) — a persistent faint source stands out; a mover smears out.`;
    } else {
      const frame = state.frames[state.currentIndex];
      renderGrid(frame.width, frame.height, physicalValuesForFrame(frame));
      updateFrameLabel(frame);
    }
    updateContaminationSummary();
  }
  function renderGrid(width, height, values) {
    state.lastRenderedGrid = { width, height, values };
    const physValues = [];
    for (let i = 0; i < values.length; i++) if (Number.isFinite(values[i])) physValues.push(values[i]);

    const mode = stretchSelect.value;
    const blackPct = parseFloat(blackSlider.value);
    const whitePct = parseFloat(whiteSlider.value);
    const { black, white } = computeLevels(physValues, mode, blackPct, whitePct);
    const invert = invertCheckbox.checked;

    const rgba = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < values.length; i++) {
      const o = i * 4;
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

    offscreen.width = width;
    offscreen.height = height;
    offscreenCtx.putImageData(new ImageData(rgba, width, height), 0, 0);

    const zoom = parseInt(zoomSlider.value, 10);
    canvas.width = width * zoom;
    canvas.height = height * zoom;
    canvasCtx.imageSmoothingEnabled = false;
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
    canvasCtx.drawImage(offscreen, 0, 0, width, height, 0, 0, width * zoom, height * zoom);

    // crosshair on the target (always the grid centre, by construction)
    canvasCtx.strokeStyle = "rgba(95,176,255,0.85)";
    canvasCtx.lineWidth = 1;
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    canvasCtx.beginPath();
    canvasCtx.moveTo(cx - 10, cy);
    canvasCtx.lineTo(cx - 3, cy);
    canvasCtx.moveTo(cx + 3, cy);
    canvasCtx.lineTo(cx + 10, cy);
    canvasCtx.moveTo(cx, cy - 10);
    canvasCtx.lineTo(cx, cy - 3);
    canvasCtx.moveTo(cx, cy + 3);
    canvasCtx.lineTo(cx, cy + 10);
    canvasCtx.stroke();
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
    contaminationSummary.textContent = parts.length
      ? `Flagged in this frame: ${parts.join(", ")}`
      : "No notable flags in this frame.";
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
      `${frame.nExposures} exposure(s) combined${partial}${gapFilled}`;
  }

  // ---------------------------------------------------------------------
  // Light curve: flux at the target pixel vs time (Phase 3).
  //
  // This reads the single pixel at the grid centre, which by construction
  // (app/pipeline.py's make_target_wcs) is always the target position —
  // it is NOT aperture or PSF photometry (that's the Phase 4 bonus
  // feature), so a few mis-centred or noisy pixels can swing it. Good
  // enough to catch an obviously variable star like RW Aurigae or R
  // Draconis; not a substitute for real photometry.
  // ---------------------------------------------------------------------
  function centrePixelFlux(frame) {
    const cx = Math.floor(frame.width / 2);
    const cy = Math.floor(frame.height / 2);
    const raw = frame.data[cy * frame.width + cx];
    return raw === INVALID_SENTINEL ? NaN : raw * frame.scale + frame.offset;
  }

  function drawLightCurve() {
    const w = lightcurveCanvas.width;
    const h = lightcurveCanvas.height;
    lightcurveCtx.clearRect(0, 0, w, h);
    if (state.frames.length === 0) return;

    const points = state.frames.map((f, i) => ({ t: f.timeMjdMean, flux: centrePixelFlux(f), i }));
    const finitePoints = points.filter((p) => Number.isFinite(p.flux));
    if (finitePoints.length === 0) {
      lightcurveCtx.fillStyle = "#8b96ab";
      lightcurveCtx.fillText("No valid flux at the target pixel in any frame.", 10, h / 2);
      return;
    }

    const tMin = Math.min(...points.map((p) => p.t));
    const tMax = Math.max(...points.map((p) => p.t)) || tMin + 1;
    const fluxMin = Math.min(...finitePoints.map((p) => p.flux));
    const fluxMax = Math.max(...finitePoints.map((p) => p.flux));
    const pad = 16;
    const xOf = (t) => pad + ((t - tMin) / (tMax - tMin || 1)) * (w - 2 * pad);
    const yOf = (f) => h - pad - ((f - fluxMin) / (fluxMax - fluxMin || 1)) * (h - 2 * pad);

    lightcurveCtx.strokeStyle = "#5fb0ff";
    lightcurveCtx.lineWidth = 1.5;
    lightcurveCtx.beginPath();
    finitePoints.forEach((p, idx) => {
      const x = xOf(p.t);
      const y = yOf(p.flux);
      if (idx === 0) lightcurveCtx.moveTo(x, y);
      else lightcurveCtx.lineTo(x, y);
    });
    lightcurveCtx.stroke();

    for (const p of points) {
      const x = xOf(p.t);
      const y = Number.isFinite(p.flux) ? yOf(p.flux) : h - pad;
      const isCurrent = viewModeSelect.value === "blink" && p.i === state.currentIndex;
      lightcurveCtx.fillStyle = Number.isFinite(p.flux) ? (isCurrent ? "#ffb454" : "#5fb0ff") : "#555";
      lightcurveCtx.beginPath();
      lightcurveCtx.arc(x, y, isCurrent ? 4 : 2.5, 0, 2 * Math.PI);
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
  function pixelAtEvent(ev) {
    const grid = state.lastRenderedGrid;
    if (!grid || !state.resolvedCoord) return null;
    const rect = canvas.getBoundingClientRect();
    const canvasX = ((ev.clientX - rect.left) / rect.width) * canvas.width;
    const canvasY = ((ev.clientY - rect.top) / rect.height) * canvas.height;
    const zoom = parseInt(zoomSlider.value, 10);
    const px = Math.floor(canvasX / zoom);
    const py = Math.floor(canvasY / zoom);
    if (px < 0 || py < 0 || px >= grid.width || py >= grid.height) return null;

    const pixelScaleArcsec = (state.fovArcmin * 60) / grid.width;
    const dx = (px - grid.width / 2) * pixelScaleArcsec;
    const dy = (grid.height / 2 - py) * pixelScaleArcsec;
    const decRad = (state.resolvedCoord.dec * Math.PI) / 180;
    const ra = state.resolvedCoord.ra - dx / 3600 / Math.cos(decRad);
    const dec = state.resolvedCoord.dec + dy / 3600;
    const phys = grid.values[py * grid.width + px];
    return { px, py, ra, dec, phys };
  }

  canvas.addEventListener("mousemove", (ev) => {
    const hit = pixelAtEvent(ev);
    if (!hit) {
      pixelReadout.textContent = "";
      return;
    }
    pixelReadout.textContent = Number.isFinite(hit.phys)
      ? `RA ${hit.ra.toFixed(5)}  Dec ${hit.dec.toFixed(5)}   flux ${hit.phys.toFixed(4)} MJy/sr`
      : `RA ${hit.ra.toFixed(5)}  Dec ${hit.dec.toFixed(5)}   no data`;
  });
  canvas.addEventListener("mouseleave", () => (pixelReadout.textContent = ""));

  // ---------------------------------------------------------------------
  // Spectrum (Phase 4, bonus): click a pixel -> flux vs wavelength
  // ---------------------------------------------------------------------
  canvas.addEventListener("click", (ev) => {
    const hit = pixelAtEvent(ev);
    if (!hit) return;
    fetchAndDrawSpectrum(hit.ra, hit.dec);
  });

  async function fetchAndDrawSpectrum(ra, dec) {
    spectrumStatus.textContent = "Fetching spectrum from IRSA (re-queries all 6 detectors — this takes a while)…";
    spectrumCtx.clearRect(0, 0, spectrumCanvas.width, spectrumCanvas.height);
    try {
      const url = `${API_BASE}/api/spectrum?position=${ra.toFixed(6)},${dec.toFixed(6)}&fov=${state.fovArcmin}`;
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`server error (${resp.status})`);
      const points = await resp.json();
      drawSpectrum(points);
      const nGood = points.filter((p) => p.flux != null).length;
      spectrumStatus.textContent = nGood
        ? `${nGood} of ${points.length} exposure(s) usable at RA ${ra.toFixed(4)}, Dec ${dec.toFixed(4)}.`
        : `No unmasked flux found at RA ${ra.toFixed(4)}, Dec ${dec.toFixed(4)} — too faint, or always flagged.`;
    } catch (e) {
      spectrumStatus.textContent = `Spectrum failed: ${e.message}`;
    }
  }

  function drawSpectrum(points) {
    const w = spectrumCanvas.width;
    const h = spectrumCanvas.height;
    spectrumCtx.clearRect(0, 0, w, h);
    const good = points.filter((p) => p.flux != null && p.wavelength_um != null);
    if (good.length === 0) return;

    const wMin = Math.min(...good.map((p) => p.wavelength_um));
    const wMax = Math.max(...good.map((p) => p.wavelength_um));
    const fMin = Math.min(...good.map((p) => p.flux));
    const fMax = Math.max(...good.map((p) => p.flux));
    const tMin = Math.min(...good.map((p) => p.time_mjd));
    const tMax = Math.max(...good.map((p) => p.time_mjd)) || tMin + 1;
    const pad = 14;
    const xOf = (wl) => pad + ((wl - wMin) / (wMax - wMin || 1)) * (w - 2 * pad);
    const yOf = (f) => h - pad - ((f - fMin) / (fMax - fMin || 1)) * (h - 2 * pad);

    for (const p of good) {
      // Coloured by epoch (time), per the brief: blue (early) -> red (late).
      const hue = 220 - 220 * ((p.time_mjd - tMin) / (tMax - tMin || 1));
      spectrumCtx.fillStyle = `hsl(${hue}, 80%, 60%)`;
      spectrumCtx.beginPath();
      spectrumCtx.arc(xOf(p.wavelength_um), yOf(p.flux), 2.5, 0, 2 * Math.PI);
      spectrumCtx.fill();
    }
  }

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
    abPick.hidden = !isDiff;
    setPlaybackControlsEnabled(isBlink);
    if (!isBlink) stopPlayback();
    renderView();
  });
  diffASlider.addEventListener("input", renderView);
  diffBSlider.addEventListener("input", renderView);

  document.addEventListener("keydown", (ev) => {
    if (document.activeElement === positionInput) return;
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
    if (urlState.position) {
      positionInput.value = urlState.position;
      runSearch();
    }
  })();
})();
