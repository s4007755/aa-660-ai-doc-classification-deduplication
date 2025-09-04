# src/persistence/state_store.py
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datetime import datetime

from src.learners.base import (
    LearnerState,
    serialize_state,
    deserialize_state,
)
from src.ensemble.arbiter import DecisionTrace
from src.storage.sqlite_store import get_conn

# Map final label to integer for DB
_LABEL_TO_INT = {"DUPLICATE": 1, "NON_DUPLICATE": 0, "UNCERTAIN": -1}
_INT_TO_LABEL = {v: k for k, v in _LABEL_TO_INT.items()}

# Create tables if they don't exist
def ensure_state_schema() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS learners (
            name TEXT PRIMARY KEY,
            state_json TEXT NOT NULL,
            version INTEGER DEFAULT 1,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_json TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            status TEXT,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS calibrations (
            run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            learner_name TEXT NOT NULL,
            method TEXT,
            params_json TEXT,
            reliability_json TEXT,
            PRIMARY KEY (run_id, learner_name)
        );

        CREATE TABLE IF NOT EXISTS decisions (
            run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            pair_key TEXT NOT NULL,
            doc1 TEXT NOT NULL,
            doc2 TEXT NOT NULL,
            final_label INTEGER NOT NULL,
            consensus TEXT,
            trace_json TEXT,
            PRIMARY KEY (run_id, pair_key)
        );

        CREATE INDEX IF NOT EXISTS idx_decisions_run ON decisions(run_id);
        CREATE INDEX IF NOT EXISTS idx_decisions_docs ON decisions(doc1, doc2);
        """
    )
    conn.commit()
    conn.close()

# Save a learner state (upsert)
def save_learner_state(name: str, state: LearnerState) -> None:
    ensure_state_schema()
    payload = serialize_state(state)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO learners (name, state_json, version, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(name) DO UPDATE SET
            state_json=excluded.state_json,
            version=excluded.version,
            updated_at=CURRENT_TIMESTAMP
        """,
        (name, payload, int(state.version)),
    )
    conn.commit()
    conn.close()

# Load a learner state
def load_learner_state(name: str) -> Optional[LearnerState]:
    ensure_state_schema()
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute("SELECT state_json FROM learners WHERE name=?", (name,)).fetchone()
    conn.close()
    if not row:
        return None
    try:
        return deserialize_state(row[0])
    except Exception:
        return None

# Start a new run and return run_id
def start_run(config_json: str, status: str = "running", notes: str = "") -> int:
    ensure_state_schema()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO runs (config_json, status, notes) VALUES (?, ?, ?)",
        (config_json, status, notes),
    )
    run_id = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return run_id

# Mark a run as ended with status
def end_run(run_id: int, status: str = "completed", notes: Optional[str] = None) -> None:
    ensure_state_schema()
    conn = get_conn()
    cur = conn.cursor()
    if notes is None:
        cur.execute(
            "UPDATE runs SET ended_at=CURRENT_TIMESTAMP, status=? WHERE run_id=?",
            (status, run_id),
        )
    else:
        cur.execute(
            "UPDATE runs SET ended_at=CURRENT_TIMESTAMP, status=?, notes=? WHERE run_id=?",
            (status, notes, run_id),
        )
    conn.commit()
    conn.close()

# Store a calibration snapshot for a learner in a run
def save_calibration(
    run_id: int,
    learner_name: str,
    method: str,
    params_json: str,
    reliability_json: str,
) -> None:
    ensure_state_schema()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO calibrations (run_id, learner_name, method, params_json, reliability_json)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(run_id, learner_name) DO UPDATE SET
            method=excluded.method,
            params_json=excluded.params_json,
            reliability_json=excluded.reliability_json
        """,
        (run_id, learner_name, method, params_json, reliability_json),
    )
    conn.commit()
    conn.close()

# Insert a single decision trace
def insert_decision(run_id: int, trace: DecisionTrace) -> None:
    ensure_state_schema()
    conn = get_conn()
    cur = conn.cursor()
    consensus = ",".join(trace.agreed_learners)
    label_int = _LABEL_TO_INT.get(trace.final_label, -1)
    trace_json = json.dumps(trace.as_dict(), ensure_ascii=False)
    cur.execute(
        """
        INSERT INTO decisions (run_id, pair_key, doc1, doc2, final_label, consensus, trace_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, pair_key) DO UPDATE SET
            final_label=excluded.final_label,
            consensus=excluded.consensus,
            trace_json=excluded.trace_json
        """,
        (run_id, trace.pair_key, trace.a_id, trace.b_id, label_int, consensus, trace_json),
    )
    conn.commit()
    conn.close()

# Bulk insert for many traces
def bulk_insert_decisions(run_id: int, traces: Iterable[DecisionTrace]) -> None:
    ensure_state_schema()
    conn = get_conn()
    cur = conn.cursor()
    rows = []
    for tr in traces:
        consensus = ",".join(tr.agreed_learners)
        label_int = _LABEL_TO_INT.get(tr.final_label, -1)
        trace_json = json.dumps(tr.as_dict(), ensure_ascii=False)
        rows.append((run_id, tr.pair_key, tr.a_id, tr.b_id, label_int, consensus, trace_json))
    cur.executemany(
        """
        INSERT INTO decisions (run_id, pair_key, doc1, doc2, final_label, consensus, trace_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, pair_key) DO UPDATE SET
            final_label=excluded.final_label,
            consensus=excluded.consensus,
            trace_json=excluded.trace_json
        """,
        rows,
    )
    conn.commit()
    conn.close()

# Get a run row
def get_run(run_id: int) -> Optional[Dict[str, Any]]:
    ensure_state_schema()
    conn = get_conn()
    conn.row_factory = lambda c, r: {
        "run_id": r[0],
        "config_json": r[1],
        "started_at": r[2],
        "ended_at": r[3],
        "status": r[4],
        "notes": r[5],
    }
    cur = conn.cursor()
    row = cur.execute("SELECT run_id, config_json, started_at, ended_at, status, notes FROM runs WHERE run_id=?", (run_id,)).fetchone()
    conn.close()
    return row

# List recent runs
def list_runs(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    ensure_state_schema()
    conn = get_conn()
    conn.row_factory = lambda c, r: {
        "run_id": r[0],
        "started_at": r[1],
        "ended_at": r[2],
        "status": r[3],
    }
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT run_id, started_at, ended_at, status FROM runs ORDER BY run_id DESC LIMIT ? OFFSET ?",
        (int(limit), int(offset)),
    ).fetchall()
    conn.close()
    return list(rows)

# Fetch calibrations for a run
def get_calibrations(run_id: int) -> List[Dict[str, Any]]:
    ensure_state_schema()
    conn = get_conn()
    conn.row_factory = lambda c, r: {
        "learner_name": r[0],
        "method": r[1],
        "params_json": r[2],
        "reliability_json": r[3],
    }
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT learner_name, method, params_json, reliability_json FROM calibrations WHERE run_id=?",
        (run_id,),
    ).fetchall()
    conn.close()
    return list(rows)

# Fetch decisions for a run
def get_decisions(
    run_id: int,
    *,
    label: Optional[str] = None,
    limit: int = 1000,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    ensure_state_schema()
    conn = get_conn()
    conn.row_factory = lambda c, r: {
        "pair_key": r[0],
        "doc1": r[1],
        "doc2": r[2],
        "final_label": _INT_TO_LABEL.get(r[3], "UNCERTAIN"),
        "consensus": r[4],
        "trace_json": r[5],
    }
    cur = conn.cursor()
    if label is None:
        rows = cur.execute(
            "SELECT pair_key, doc1, doc2, final_label, consensus, trace_json FROM decisions WHERE run_id=? LIMIT ? OFFSET ?",
            (run_id, int(limit), int(offset)),
        ).fetchall()
    else:
        lv = _LABEL_TO_INT.get(label, -1)
        rows = cur.execute(
            "SELECT pair_key, doc1, doc2, final_label, consensus, trace_json FROM decisions WHERE run_id=? AND final_label=? LIMIT ? OFFSET ?",
            (run_id, lv, int(limit), int(offset)),
        ).fetchall()
    conn.close()
    return list(rows)

# Fetch a single decision trace by pair
def get_decision_by_pair(run_id: int, doc1: str, doc2: str) -> Optional[Dict[str, Any]]:
    ensure_state_schema()
    conn = get_conn()
    conn.row_factory = lambda c, r: {
        "pair_key": r[0],
        "doc1": r[1],
        "doc2": r[2],
        "final_label": _INT_TO_LABEL.get(r[3], "UNCERTAIN"),
        "consensus": r[4],
        "trace_json": r[5],
    }
    cur = conn.cursor()
    key_ab = f"{min(doc1, doc2)}||{max(doc1, doc2)}"
    row = cur.execute(
        "SELECT pair_key, doc1, doc2, final_label, consensus, trace_json FROM decisions WHERE run_id=? AND pair_key=?",
        (run_id, key_ab),
    ).fetchone()
    conn.close()
    return row

# Compute quick run stats for GUI
def run_stats(run_id: int) -> Dict[str, Any]:
    ensure_state_schema()
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute(
        """
        SELECT
          SUM(CASE WHEN final_label=1 THEN 1 ELSE 0 END) AS dup_cnt,
          SUM(CASE WHEN final_label=0 THEN 1 ELSE 0 END) AS non_cnt,
          SUM(CASE WHEN final_label=-1 THEN 1 ELSE 0 END) AS unc_cnt,
          COUNT(*) AS total
        FROM decisions
        WHERE run_id=?
        """,
        (run_id,),
    ).fetchone()
    if not row:
        conn.close()
        return {"duplicates": 0, "non_duplicates": 0, "uncertain": 0, "total": 0, "consensus_rate": 0.0, "escalations_pct": 0.0}
    dup_cnt, non_cnt, unc_cnt, total = [int(x or 0) for x in row]
    consensus_rate = 0.0 if total == 0 else (dup_cnt + non_cnt) / total
    esc_row = cur.execute(
        "SELECT trace_json FROM decisions WHERE run_id=? LIMIT 10000",
        (run_id,),
    ).fetchall()
    esc_cnt = 0
    for (tj,) in esc_row:
        try:
            obj = json.loads(tj)
            if obj.get("escalation_steps"):
                if len(obj["escalation_steps"]) > 0:
                    esc_cnt += 1
        except Exception:
            pass
    escalations_pct = 0.0 if total == 0 else esc_cnt / total
    conn.close()
    return {
        "duplicates": dup_cnt,
        "non_duplicates": non_cnt,
        "uncertain": unc_cnt,
        "total": total,
        "consensus_rate": float(consensus_rate),
        "escalations_pct": float(escalations_pct),
    }

# compatibility + init
def init_db() -> None:
    ensure_state_schema()

def get_run_config(run_id: int) -> Optional[Dict[str, Any]]:
    row = get_run(run_id)
    if not row or not row.get("config_json"):
        return None
    try:
        return json.loads(row["config_json"])
    except Exception:
        return None

def get_calibrations_for_run(run_id: int) -> List[Dict[str, Any]]:
    rows = get_calibrations(run_id)
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({
            "learner_name": r["learner_name"],
            "method": r["method"],
            "params": json.loads(r.get("params_json") or "{}"),
            "reliability": json.loads(r.get("reliability_json") or "[]"),
        })
    return out

def list_decisions(run_id: int, limit: int = 500) -> List[Dict[str, Any]]:
    rows = get_decisions(run_id, limit=limit, offset=0)
    out: List[Dict[str, Any]] = []
    for r in rows:
        try:
            trace = json.loads(r["trace_json"]) if r.get("trace_json") else None
        except Exception:
            trace = None
        out.append({
            "pair_key": r["pair_key"],
            "doc1": r["doc1"],
            "doc2": r["doc2"],
            "final_label": r["final_label"],
            "consensus": r["consensus"] or "",
            "trace": trace,
        })
    return out

