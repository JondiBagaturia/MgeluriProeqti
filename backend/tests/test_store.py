"""SQLite candidate/classification store. Uses a temporary database file;
the track dictionaries are synthetic."""
import pytest

from app import store


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("SPHEREX_DB_PATH", str(tmp_path / "test.sqlite"))
    return tmp_path


def _track(ra=100.0, dec=20.0, series="D6", score=0.4):
    return {"series": series, "ra_deg": ra, "dec_deg": dec, "label": "candidate", "anomaly_score": score,
            "reliability": "high", "reasons": ["synthetic"], "detections": [{"frame_index": 0}]}


def test_upsert_is_stable_for_the_same_source(db):
    a = store.upsert_candidate(_track(), "field A")
    b = store.upsert_candidate(_track(ra=100.0005, score=0.6), None)   # 1.7" away: same source, refreshed
    c = store.upsert_candidate(_track(ra=100.01), "field A")            # 34" away: different source
    d = store.upsert_candidate(_track(series="D2"), "field A")          # other detector series
    assert a == b and len({a, c, d}) == 3
    got = store.get_candidate(a)
    assert got["anomaly_score"] == 0.6 and got["field_query"] == "field A"
    assert got["summary"]["detections"] == [{"frame_index": 0}]


def test_multiple_independent_classifications_and_agreement(db):
    cid = store.upsert_candidate(_track())
    store.classify(cid, "voter-aaaaaaa1", "moving")
    store.classify(cid, "voter-aaaaaaa2", "moving")
    agreement = store.classify(cid, "voter-aaaaaaa3", "artifact")
    assert agreement["n_voters"] == 3
    assert agreement["counts"]["moving"] == 2 and agreement["counts"]["artifact"] == 1
    assert agreement["majority_label"] == "moving"
    assert agreement["agreement_fraction"] == pytest.approx(2 / 3)


def test_revote_replaces_instead_of_double_counting(db):
    cid = store.upsert_candidate(_track())
    store.classify(cid, "voter-aaaaaaa1", "moving")
    agreement = store.classify(cid, "voter-aaaaaaa1", "stationary")
    assert agreement["n_voters"] == 1 and agreement["counts"]["stationary"] == 1 and agreement["counts"]["moving"] == 0
    assert store.get_candidate(cid, "voter-aaaaaaa1")["your_label"] == "stationary"


def test_rejects_bad_label_voter_and_unknown_candidate(db):
    cid = store.upsert_candidate(_track())
    with pytest.raises(ValueError):
        store.classify(cid, "voter-aaaaaaa1", "new planet")
    with pytest.raises(ValueError):
        store.classify(cid, "x", "moving")
    with pytest.raises(ValueError):
        store.classify(cid, "voter'; DROP TABLE--", "moving")
    with pytest.raises(KeyError):
        store.classify("nope", "voter-aaaaaaa1", "moving")
    assert store.get_candidate("nope") is None


def test_list_orders_by_score_and_no_votes_means_no_agreement(db):
    store.upsert_candidate(_track(ra=100.0, score=0.2))
    top = store.upsert_candidate(_track(ra=100.1, score=0.9))
    rows = store.list_candidates(limit=10)
    assert rows[0]["id"] == top and len(rows) == 2
    assert rows[0]["classifications"]["n_voters"] == 0
    assert rows[0]["classifications"]["agreement_fraction"] is None
    assert len(store.list_candidates(min_score=0.5)) == 1


def test_spectrotemporal_results_need_no_schema_change(db):
    """Phase 3: spectro-temporal results travel in the analysis record, not in the
    candidates table. Storing an analysis that contains them leaves the schema and
    the stored summaries unchanged (no migration; old rows stay readable)."""
    import sqlite3
    from app.main import _store_candidates
    old = store.upsert_candidate(_track(ra=101.0))                  # a row written before Phase 3
    analysis = {"series": [{"tracks": [_track(ra=100.0)]}],
                "spectrotemporal": {"sources": [{"id": "target", "measurements": [{"flux_mjy": 1.0}] * 50}]}}
    _store_candidates(analysis, "field")
    assert "store_error" not in analysis
    new_id = analysis["series"][0]["tracks"][0]["candidate_id"]
    got = store.get_candidate(new_id)
    assert "spectrotemporal" not in got["summary"] and "measurements" not in got["summary"]
    assert store.get_candidate(old)["anomaly_score"] == 0.4
    cols = {r[1] for r in sqlite3.connect(str(db / "test.sqlite")).execute("PRAGMA table_info(candidates)")}
    assert cols == {"id", "series", "ra_deg", "dec_deg", "field_query", "label", "anomaly_score", "reliability",
                    "summary_json", "created_at", "updated_at"}
