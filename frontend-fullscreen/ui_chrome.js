/**
 * SPHEREx Blink - auto-hiding interface chrome.
 *
 * The image fills the screen; the four docks (top / left / right / bottom)
 * are shown only while the cursor is near their edge of the window, or over
 * the dock itself, or while a control inside is being used from the
 * keyboard. Two game-menu conventions sit on top of that: a window's pin
 * square keeps it open, and right-clicking the sky toggles "menu mode"
 * (every window open at once, Esc to leave). This file only toggles
 * body.show-<edge> / body.show-all classes - all visuals are in style.css,
 * all viewer logic stays in app.js.
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
