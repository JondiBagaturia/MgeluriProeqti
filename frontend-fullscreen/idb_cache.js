/**
 * Browser-side frame cache in IndexedDB.
 *
 * Key = release + ra + dec + fov + group + detector (so a new SPHEREx data
 * release naturally invalidates everything under it — no manual bump
 * needed). Size-capped with LRU eviction. This is a speed-up only: every
 * call here must degrade gracefully (return null / no-op) if IndexedDB is
 * unavailable, blocked, or the quota is refused — Safari in particular may
 * evict this at any time, and the app must work fine starting from empty.
 */
(function () {
  const DB_NAME = "spherex_blink_cache";
  const DB_VERSION = 1;
  const STORE = "frames";
  const SIZE_CAP_BYTES = 500 * 1024 * 1024; // ~500MB, per the brief

  let dbPromise = null;

  function openDb() {
    if (dbPromise) return dbPromise;
    dbPromise = new Promise((resolve) => {
      if (!("indexedDB" in window)) {
        resolve(null);
        return;
      }
      let req;
      try {
        req = indexedDB.open(DB_NAME, DB_VERSION);
      } catch (e) {
        resolve(null);
        return;
      }
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE)) {
          const store = db.createObjectStore(STORE, { keyPath: "key" });
          store.createIndex("lastAccess", "lastAccess");
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    });
    return dbPromise;
  }

  /** Build the cache key described in the brief: release+ra+dec+fov+group+detector. */
  function makeKey({ release, ra, dec, fov, group, detector }) {
    const r = (x) => Number(x).toFixed(5);
    return `${release}|${r(ra)}|${r(dec)}|${r(fov)}|${group}|${detector}`;
  }

  async function get(key) {
    const db = await openDb();
    if (!db) return null;
    return new Promise((resolve) => {
      try {
        const tx = db.transaction(STORE, "readwrite");
        const store = tx.objectStore(STORE);
        const req = store.get(key);
        req.onsuccess = () => {
          const entry = req.result;
          if (entry) {
            entry.lastAccess = Date.now();
            store.put(entry);
          }
          resolve(entry ? entry.value : null);
        };
        req.onerror = () => resolve(null);
      } catch (e) {
        resolve(null);
      }
    });
  }

  async function put(key, value, approxBytes) {
    const db = await openDb();
    if (!db) return;
    try {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).put({
        key,
        value,
        bytes: approxBytes || 0,
        lastAccess: Date.now(),
      });
    } catch (e) {
      // quota exceeded or blocked - just skip caching this entry
    }
    evictIfNeeded().catch(() => {});
  }

  async function evictIfNeeded() {
    const db = await openDb();
    if (!db) return;
    const entries = await allEntries(db);
    let total = entries.reduce((s, e) => s + (e.bytes || 0), 0);
    if (total <= SIZE_CAP_BYTES) return;
    entries.sort((a, b) => a.lastAccess - b.lastAccess); // oldest first
    const tx = db.transaction(STORE, "readwrite");
    const store = tx.objectStore(STORE);
    for (const e of entries) {
      if (total <= SIZE_CAP_BYTES) break;
      store.delete(e.key);
      total -= e.bytes || 0;
    }
  }

  function allEntries(db) {
    return new Promise((resolve) => {
      try {
        const tx = db.transaction(STORE, "readonly");
        const store = tx.objectStore(STORE);
        const req = store.getAll();
        req.onsuccess = () => resolve(req.result || []);
        req.onerror = () => resolve([]);
      } catch (e) {
        resolve([]);
      }
    });
  }

  async function totalSizeBytes() {
    const db = await openDb();
    if (!db) return 0;
    const entries = await allEntries(db);
    return entries.reduce((s, e) => s + (e.bytes || 0), 0);
  }

  async function clearAll() {
    const db = await openDb();
    if (!db) return;
    return new Promise((resolve) => {
      try {
        const tx = db.transaction(STORE, "readwrite");
        tx.objectStore(STORE).clear();
        tx.oncomplete = () => resolve();
        tx.onerror = () => resolve();
      } catch (e) {
        resolve();
      }
    });
  }

  async function requestPersistence() {
    try {
      if (navigator.storage && navigator.storage.persist) {
        return await navigator.storage.persist();
      }
    } catch (e) {
      /* ignore */
    }
    return false;
  }

  window.SpherexCache = { makeKey, get, put, totalSizeBytes, clearAll, requestPersistence };
})();
