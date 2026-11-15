"""SQLite store for candidates and human classifications.

This is the one deliberate exception to the backend's "stateless, nothing on
disk" design: volunteer classifications have to outlive a request. Images
and frames are still never stored, only small JSON summaries of tracked
sources and the votes on them.

Location: env SPHEREX_DB_PATH, else <repo>/data/discovery.sqlite.

Classification model:
  * One vote per (candidate, voter). A voter is an anonymous random token
    the browser keeps; voting again replaces that voter's earlier vote.
    This is a convenience against accidental double votes, not protection
    against a determined ballot-stuffer (there are no accounts).
  * Agreement = share of voters choosing the most common label. It is
    reported with the number of voters, and never turned into a verdict:
    three people agreeing is not a confirmation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

LABELS = ("moving", "variable", "stationary", "artifact", "uncertain")
LABEL_TEXT = {
    "moving": "Moving source",
    "variable": "Variable source",
    "stationary": "Stationary source",
    "artifact": "Potential artifact",
    "uncertain": "Uncertain",
}
# Two analyses of the same source rarely give identical coordinates; a
# stored candidate within this distance (same detector series) is the same one.
SAME_CANDIDATE_ARCSEC = 6.0

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "discovery.sqlite"
_init_lock = threading.Lock()
_initialised: set[str] = set()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    series TEXT NOT NULL,
    ra_deg REAL NOT NULL,
    dec_deg REAL NOT NULL,
    field_query TEXT,
    label TEXT,
    anomaly_score REAL,
    reliability TEXT,
    summary_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_candidates_pos ON candidates (series, dec_deg, ra_deg);
CREATE TABLE IF NOT EXISTS classifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    voter TEXT NOT NULL,
    label TEXT NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE (candidate_id, voter)
);
"""


def db_path() -> Path:
    return Path(os.environ.get("SPHEREX_DB_PATH") or _DEFAULT_PATH)


@contextmanager
def _connect():
    path = db_path()
    key = str(path)
    with _init_lock:
        if key not in _initialised:
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(key, timeout=10)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(_SCHEMA)
                conn.commit()
            finally:
                conn.close()
            _initialised.add(key)
    conn = sqlite3.connect(key, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _sep_arcsec(ra1, dec1, ra2, dec2) -> float:
    dra = ((ra1 - ra2 + 180.0) % 360.0 - 180.0) * math.cos(math.radians(0.5 * (dec1 + dec2)))
    return math.hypot(dra, dec1 - dec2) * 3600.0


def upsert_candidate(track: dict, field_query: Optional[str] = None) -> str:
    """Store (or refresh) one tracked source; returns its stable id."""
    series, ra, dec = track["series"], float(track["ra_deg"]), float(track["dec_deg"])
    summary = {k: v for k, v in track.items() if k != "detections"}
    summary["detections"] = track.get("detections", [])
    now = time.time()
    box = SAME_CANDIDATE_ARCSEC / 3600.0
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, ra_deg, dec_deg FROM candidates WHERE series = ? AND dec_deg BETWEEN ? AND ?",
            (series, dec - box, dec + box),
        ).fetchall()
        existing = [r for r in rows if _sep_arcsec(ra, dec, r["ra_deg"], r["dec_deg"]) <= SAME_CANDIDATE_ARCSEC]
        if existing:
            cid = min(existing, key=lambda r: _sep_arcsec(ra, dec, r["ra_deg"], r["dec_deg"]))["id"]
            conn.execute(
                "UPDATE candidates SET label=?, anomaly_score=?, reliability=?, summary_json=?, updated_at=?, field_query=COALESCE(?, field_query) WHERE id=?",
                (track.get("label"), track.get("anomaly_score"), track.get("reliability"), json.dumps(summary), now, field_query, cid),
            )
            return cid
        cid = hashlib.sha1(f"{series}:{ra:.5f}:{dec:.5f}".encode()).hexdigest()[:16]
        conn.execute(
            "INSERT OR REPLACE INTO candidates (id, series, ra_deg, dec_deg, field_query, label, anomaly_score, reliability, summary_json, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (cid, series, ra, dec, field_query, track.get("label"), track.get("anomaly_score"), track.get("reliability"), json.dumps(summary), now, now),
        )
        return cid


def _agreement(conn, candidate_id: str) -> dict:
    rows = conn.execute(
        "SELECT label, COUNT(*) AS n FROM classifications WHERE candidate_id = ? GROUP BY label", (candidate_id,)
    ).fetchall()
    counts = {label: 0 for label in LABELS}
    for r in rows:
        counts[r["label"]] = r["n"]
    total = sum(counts.values())
    top = max(counts, key=counts.get) if total else None
    return {
        "n_voters": total,
        "counts": counts,
        "majority_label": top,
        "agreement_fraction": (counts[top] / total) if total else None,
        "note": "Share of volunteers choosing the most common label. Not a confirmation of what the object is.",
    }


def get_candidate(candidate_id: str, voter: Optional[str] = None) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
        if row is None:
            return None
        out = {
            "id": row["id"], "series": row["series"], "ra_deg": row["ra_deg"], "dec_deg": row["dec_deg"],
            "field_query": row["field_query"], "label": row["label"], "anomaly_score": row["anomaly_score"],
            "reliability": row["reliability"], "created_at": row["created_at"], "updated_at": row["updated_at"],
            "summary": json.loads(row["summary_json"]),
            "classifications": _agreement(conn, candidate_id),
        }
        if voter:
            mine = conn.execute(
                "SELECT label FROM classifications WHERE candidate_id = ? AND voter = ?", (candidate_id, voter)
            ).fetchone()
            out["your_label"] = mine["label"] if mine else None
        return out


def list_candidates(limit: int = 50, min_score: float = 0.0, label: Optional[str] = None) -> list[dict]:
    q = "SELECT id, series, ra_deg, dec_deg, field_query, label, anomaly_score, reliability, updated_at FROM candidates WHERE COALESCE(anomaly_score, 0) >= ?"
    args: list = [min_score]
    if label:
        q += " AND label = ?"
        args.append(label)
    q += " ORDER BY anomaly_score DESC, updated_at DESC LIMIT ?"
    args.append(int(limit))
    with _connect() as conn:
        rows = conn.execute(q, args).fetchall()
        return [{**dict(r), "classifications": _agreement(conn, r["id"])} for r in rows]


def classify(candidate_id: str, voter: str, label: str) -> dict:
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    voter = (voter or "").strip()
    if not (8 <= len(voter) <= 64) or not all(c.isalnum() or c in "-_" for c in voter):
        raise ValueError("voter must be an 8-64 character token of letters, digits, '-' or '_'")
    with _connect() as conn:
        if conn.execute("SELECT 1 FROM candidates WHERE id = ?", (candidate_id,)).fetchone() is None:
            raise KeyError(candidate_id)
        conn.execute(
            "INSERT INTO classifications (candidate_id, voter, label, created_at) VALUES (?,?,?,?) "
            "ON CONFLICT(candidate_id, voter) DO UPDATE SET label = excluded.label, created_at = excluded.created_at",
            (candidate_id, voter, label, time.time()),
        )
        return _agreement(conn, candidate_id)
