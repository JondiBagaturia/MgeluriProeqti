/**
 * SPHEREx Blink - auto-hiding interface chrome.
 *
 * The image fills the screen; the four docks (top / left / right / bottom)
 * are shown only while the cursor is near their edge of the window, or over
 * the dock itself, or while a control inside is being used from the
 * keyboard. Two game-menu conventions sit on top of that: a window's pin
 * square keeps it open, and right-clicking the sky toggles "menu mode"
 * (every window open at once, Esc to leave). Two more windows, Tracked
 * sources and Spectro-temporal analysis, open from buttons in the bottom
 * bar; they float - drag the title bar to move one, drag any edge or
 * corner to resize it, double-click the title bar to put it back - and
 * their content is split into tabs. The two side windows can be made
 * wider or narrower from their inner edge. Sizes and positions are
 * remembered in this browser. This file only toggles classes on <body>
 * and sets window geometry - all visuals are in style.css, all viewer
 * logic stays in app.js.
 */
(function () {
  "use strict";

  const EDGE_PX = 88; // how close to a window edge the cursor must be
  const KEEP_MARGIN_PX = 16; // slack around an open dock before it hides
  const HIDE_DELAY_MS = 420;
  const EDGES = ["top", "left", "right", "bottom"];

  const body = document.body;
  const docks = {};
  document.querySelectorAll(".dock[data-edge]").forEach((el) => (docks[el.dataset.edge] = el));

  const emptyState = document.getElementById("empty-state");
  const resolveStatus = document.getElementById("resolve-status");
  const spectrumStatus = document.getElementById("spectrum-status");
  const crosshair = document.getElementById("crosshair");
  const edgeHint = document.getElementById("edge-hint");
  const chromeToggle = document.getElementById("chrome-toggle");
  const canvas = document.getElementById("viewer-canvas");
  const pixelReadout = document.getElementById("pixel-readout");

  const pointer = { x: 0, y: 0, inside: false };
  const pinned = { top: false, left: false, right: false, bottom: false }; // set by the app's state
  const userPinned = { top: false, left: false, right: false, bottom: false }; // set with a pin square
  const hideTimers = {};
  let spectrumPinVisited = false;
  let hintTimer = null;
  let hintDone = false;

  function dismissHint() {
    if (!edgeHint.hidden) edgeHint.hidden = true;
    clearTimeout(hintTimer);
  }

  const isShown = (edge) => body.classList.contains(`show-${edge}`);

  function hasKeyboardFocus(dock) {
    try {
      return dock.querySelector(":focus-visible") !== null;
    } catch (e) {
      return dock.contains(document.activeElement); // very old browsers
    }
  }

  function pointerOverDock(dock) {
    const r = dock.getBoundingClientRect();
    return (
      pointer.x >= r.left - KEEP_MARGIN_PX &&
      pointer.x <= r.right + KEEP_MARGIN_PX &&
      pointer.y >= r.top - KEEP_MARGIN_PX &&
      pointer.y <= r.bottom + KEEP_MARGIN_PX
    );
  }

  function wanted(edge) {
    const dock = docks[edge];
    if (!dock) return false;
    if (pinned[edge] || userPinned[edge] || hasKeyboardFocus(dock)) return true;
    if (!pointer.inside) return false;
    if (edge === "top" && pointer.y < EDGE_PX) return true;
    if (edge === "bottom" && pointer.y > window.innerHeight - EDGE_PX) return true;
    if (edge === "left" && pointer.x < EDGE_PX) return true;
    if (edge === "right" && pointer.x > window.innerWidth - EDGE_PX) return true;
    return isShown(edge) && pointerOverDock(dock);
  }

  function update() {
    for (const edge of EDGES) {
      if (wanted(edge)) {
        clearTimeout(hideTimers[edge]);
        hideTimers[edge] = null;
        if (!isShown(edge)) {
          body.classList.add(`show-${edge}`);
          dismissHint();
        }
      } else if (isShown(edge) && !hideTimers[edge]) {
        hideTimers[edge] = setTimeout(() => {
          hideTimers[edge] = null;
          if (!wanted(edge)) body.classList.remove(`show-${edge}`);
        }, HIDE_DELAY_MS);
      }
    }
  }

  document.addEventListener("mousemove", (ev) => {
    pointer.x = ev.clientX;
    pointer.y = ev.clientY;
    pointer.inside = true;

    // The spectrum panel opens itself when a spectrum is requested (see
    // below); it goes back to normal hiding once the cursor has been to it
    // and left again.
    if (pinned.right) {
      const over = pointerOverDock(docks.right);
      if (over) spectrumPinVisited = true;
      else if (spectrumPinVisited) pinned.right = false;
    }
    update();
  });
  document.documentElement.addEventListener("mouseleave", () => {
    pointer.inside = false;
    update();
  });
  document.addEventListener("focusin", update);
  document.addEventListener("focusout", () => setTimeout(update, 0));
  document.addEventListener("keydown", (ev) => {
    if (ev.key !== "Escape") return;
    pinned.right = false;
    body.classList.remove("show-all");
    closeAllFloating();
    if (document.activeElement && document.activeElement !== body) document.activeElement.blur();
    update();
  });

  // --- Pin squares: keep one window open while working with the image -----
  document.querySelectorAll(".pin-btn[data-pin]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const edge = btn.dataset.pin;
      userPinned[edge] = !userPinned[edge];
      btn.setAttribute("aria-pressed", String(userPinned[edge]));
      update();
    });
  });

  // --- Menu mode: right-click the sky to open every window, again to close -
  document.getElementById("stage").addEventListener("contextmenu", (ev) => {
    ev.preventDefault();
    body.classList.toggle("show-all");
    dismissHint();
  });

  // --- Stored window geometry ---------------------------------------------
  const store = {
    get(key) {
      try {
        return JSON.parse(localStorage.getItem(`spherex-ui-${key}`));
      } catch (e) {
        return null; // storage unavailable or damaged: fall back to the defaults
      }
    },
    set(key, value) {
      try {
        if (value == null) localStorage.removeItem(`spherex-ui-${key}`);
        else localStorage.setItem(`spherex-ui-${key}`, JSON.stringify(value));
      } catch (e) {
        /* not remembered, still usable */
      }
    },
  };

  /** Pointer drag helper: calls onMove(dx, dy) while the button is held. */
  function dragFrom(el, onStart, onMove, onEnd) {
    el.addEventListener("pointerdown", (down) => {
      if (down.button !== 0 || down.target.closest("button, input, select, label, a")) return;
      down.preventDefault();
      el.setPointerCapture(down.pointerId);
      const start = onStart();
      const move = (ev) => onMove(start, ev.clientX - down.clientX, ev.clientY - down.clientY);
      const up = () => {
        el.removeEventListener("pointermove", move);
        el.removeEventListener("pointerup", up);
        el.removeEventListener("pointercancel", up);
        if (onEnd) onEnd();
      };
      el.addEventListener("pointermove", move);
      el.addEventListener("pointerup", up);
      el.addEventListener("pointercancel", up);
    });
  }

  // --- Floating windows: Tracked sources, Spectro-temporal analysis --------
  // Opened on request rather than by an edge: a wide table or a plot needs
  // the middle of the screen. Each can be moved (title bar) and resized
  // (edges, corners), like a window in a game menu.
  const FLOAT_MIN_W = 380;
  const FLOAT_MIN_H = 220;
  const SMALL_SCREEN = window.matchMedia("(max-width: 720px)");
  const floats = [];
  let topZ = 13;

  function setupFloating(win) {
    const id = win.dataset.window;
    const showClass = `show-${id}`;
    const buttons = document.querySelectorAll(`[data-toggle-window="${id}"]`);
    const float = { win, id, setOpen, clamp };
    floats.push(float);

    const rectOf = () => {
      const r = win.getBoundingClientRect();
      return { left: r.left, top: r.top, width: r.width, height: r.height };
    };
    // Keeps a window inside the viewport and above its minimum size.
    function place(r) {
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      const width = Math.min(Math.max(r.width, FLOAT_MIN_W), vw - 16);
      const height = Math.min(Math.max(r.height, FLOAT_MIN_H), vh - 16);
      const left = Math.min(Math.max(r.left, 8), vw - width - 8);
      const top = Math.min(Math.max(r.top, 8), vh - height - 8);
      Object.assign(win.style, { left: `${left}px`, top: `${top}px`, width: `${width}px`, height: `${height}px` });
      win.classList.add("placed");
    }
    function clamp() {
      if (!win.classList.contains("placed")) return;
      if (SMALL_SCREEN.matches) return; // small screens use the fixed layout in style.css
      place(rectOf());
    }
    function reset() {
      win.classList.remove("placed");
      ["left", "top", "width", "height"].forEach((k) => (win.style[k] = ""));
      store.set(`window-${id}`, null);
    }
    const save = () => store.set(`window-${id}`, rectOf());

    function setOpen(open) {
      body.classList.toggle(showClass, open);
      buttons.forEach((btn) => btn.setAttribute("aria-pressed", String(open)));
      if (open) {
        dismissHint();
        win.style.zIndex = String(++topZ);
        clamp();
      }
    }

    const saved = store.get(`window-${id}`);
    if (saved && !SMALL_SCREEN.matches && [saved.left, saved.top, saved.width, saved.height].every(Number.isFinite)) place(saved);

    buttons.forEach((btn) => btn.addEventListener("click", () => setOpen(!body.classList.contains(showClass))));
    win.querySelectorAll("[data-close]").forEach((btn) => btn.addEventListener("click", () => setOpen(false)));
    win.addEventListener("pointerdown", () => (win.style.zIndex = String(++topZ)), true); // clicked window comes to the front

    // Move: drag the title bar. Double-click it to return to the default place and size.
    const bar = win.querySelector("[data-drag]");
    dragFrom(bar, rectOf, (start, dx, dy) => place({ ...start, left: start.left + dx, top: start.top + dy }), save);
    bar.addEventListener("dblclick", (ev) => {
      if (!ev.target.closest("button")) reset();
    });

    // Resize: a strip along each edge and a square on each corner.
    for (const dir of ["n", "s", "e", "w", "ne", "nw", "se", "sw"]) {
      const handle = document.createElement("div");
      handle.className = `resize-handle rh-${dir}`;
      win.appendChild(handle);
      dragFrom(
        handle,
        rectOf,
        (start, dx, dy) => {
          const r = { ...start };
          if (dir.includes("e")) r.width = Math.max(FLOAT_MIN_W, start.width + dx);
          if (dir.includes("s")) r.height = Math.max(FLOAT_MIN_H, start.height + dy);
          if (dir.includes("w")) {
            r.width = Math.max(FLOAT_MIN_W, start.width - dx);
            r.left = start.left + start.width - r.width;
          }
          if (dir.includes("n")) {
            r.height = Math.max(FLOAT_MIN_H, start.height - dy);
            r.top = start.top + start.height - r.height;
          }
          place(r);
        },
        save
      );
    }
  }
  document.querySelectorAll(".float-window").forEach(setupFloating);
  function closeAllFloating() {
    floats.forEach((f) => f.setOpen(false));
  }
  window.addEventListener("resize", () => floats.forEach((f) => f.clamp()));

  // --- Tabs inside a window -------------------------------------------------
  // Each tab button names the pane it shows (data-tab); panes are siblings
  // inside the same window.
  document.querySelectorAll(".tab-row").forEach((row) => {
    const tabs = Array.from(row.querySelectorAll(".tab-btn"));
    const select = (chosen) => {
      for (const tab of tabs) {
        const on = tab === chosen;
        tab.setAttribute("aria-selected", String(on));
        document.getElementById(tab.dataset.tab).hidden = !on;
      }
    };
    tabs.forEach((tab) => tab.addEventListener("click", () => select(tab)));
  });

  // --- Side windows: width can be changed from the inner edge --------------
  for (const [edge, dir] of [["left", "e"], ["right", "w"]]) {
    const dock = docks[edge];
    if (!dock) continue;
    const apply = (w) => {
      const width = Math.min(Math.max(w, 280), Math.max(280, window.innerWidth * 0.6));
      dock.style.width = `${width}px`;
      return width;
    };
    const saved = store.get(`dock-${edge}`);
    if (Number.isFinite(saved) && !SMALL_SCREEN.matches) apply(saved);
    const handle = document.createElement("div");
    handle.className = `resize-handle rh-${dir}`;
    dock.appendChild(handle);
    let last = null;
    dragFrom(
      handle,
      () => dock.getBoundingClientRect().width,
      (startWidth, dx) => (last = apply(startWidth + (dir === "e" ? dx : -dx))),
      () => store.set(`dock-${edge}`, last)
    );
    handle.addEventListener("dblclick", () => {
      dock.style.width = "";
      store.set(`dock-${edge}`, null);
    });
  }

  // --- States where hiding everything would leave the user stranded -------

  // Nothing loaded yet (or nothing found): keep the search bar on screen.
  // A failed name/coordinate lookup: keep its error message on screen.
  function syncTopPin() {
    pinned.top = !emptyState.hidden || resolveStatus.classList.contains("error");
    update();
  }
  new MutationObserver(syncTopPin).observe(emptyState, { attributes: true, attributeFilter: ["hidden"] });
  new MutationObserver(syncTopPin).observe(resolveStatus, { attributes: true, attributeFilter: ["class"] });
  syncTopPin();

  // A click on the image requests a spectrum, whose status and plot live in
  // the right dock - open it so the click visibly did something.
  new MutationObserver(() => {
    if (!spectrumStatus.textContent) return;
    pinned.right = true;
    spectrumPinVisited = false;
    update();
  }).observe(spectrumStatus, { childList: true, characterData: true, subtree: true });

  // --- One-time hint, the first time an image appears ---------------------
  new MutationObserver(() => {
    if (hintDone || crosshair.hidden) return;
    hintDone = true;
    if (!window.matchMedia("(hover: hover)").matches) return; // touch: the round button is the affordance
    edgeHint.hidden = false;
    hintTimer = setTimeout(dismissHint, 5000);
  }).observe(crosshair, { attributes: true, attributeFilter: ["hidden"] });

  // --- Touch screens: no cursor to bring near an edge ---------------------
  chromeToggle.addEventListener("click", () => body.classList.toggle("show-all"));

  // --- Publish the bar heights to CSS: keeps the progress pill clear of the
  // search bar, and on narrow screens fits the two panels between the bars.
  if ("ResizeObserver" in window) {
    const publish = () => {
      body.style.setProperty("--top-dock-h", `${docks.top.offsetHeight}px`);
      body.style.setProperty("--bottom-dock-h", `${docks.bottom.offsetHeight}px`);
    };
    const ro = new ResizeObserver(publish);
    ro.observe(docks.top);
    ro.observe(docks.bottom);
  }

  // --- Pixel readout follows the cursor -----------------------------------
  canvas.addEventListener("mousemove", (ev) => {
    const flip = ev.clientX > window.innerWidth - 340;
    const below = ev.clientY < window.innerHeight - 60;
    pixelReadout.style.transform =
      `translate(${ev.clientX + (flip ? -16 : 16)}px, ${ev.clientY + (below ? 18 : -18)}px) ` +
      `translate(${flip ? "-100%" : "0"}, ${below ? "0" : "-100%"})`;
  });
})();
