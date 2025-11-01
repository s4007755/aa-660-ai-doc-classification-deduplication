# src/reporting/report_builder.py
from __future__ import annotations

import io
import json
import os
import base64
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import math

from src.persistence import state_store as store
from src.metrics.metrics import metrics_snapshot

try:
    from src.storage import sqlite_store
except Exception:
    sqlite_store = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

import numpy as np

# Small helpers
def _get(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def _pretty_json(s: Any) -> str:
    if isinstance(s, dict):
        try:
            return json.dumps(s, indent=2, ensure_ascii=False)
        except Exception:
            return str(s)
    try:
        return json.dumps(json.loads(s or "{}"), indent=2, ensure_ascii=False)
    except Exception:
        return str(s or "")

def _fmt_num(x: Any, nd: int = 3) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def _disp(mm: Optional[Dict[str, Any]]) -> str:
    if not mm:
        return ""
    mn = _fmt_num(_get(mm, "min"))
    mx = _fmt_num(_get(mm, "max"))
    return f"{mn}..{mx}"

def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# Public API
def generate_report(
    run_id: int,
    *,
    out_dir: str = "reports",
    fmt: str = "html",
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    # Run + decisions + calibrations from persistence
    run = store.get_run(run_id) or {}
    cal_rows = _safe_get_calibrations(run_id)
    dec_rows = _safe_get_decisions(run_id)

    # Parse decision traces from JSON
    traces_json: List[Dict[str, Any]] = []
    for r in dec_rows:
        try:
            traces_json.append(json.loads(r.get("trace_json") or "{}"))
        except Exception:
            continue

    # thresholds from any saved calibration, briers are shown only if present
    thresholds, briers = _collect_thresholds_briers(cal_rows)

    # build pseudo labels from final labels so charts and per-learner metrics have data
    pseudo = _pseudo_from_traces(traces_json)

    basics = _basics_from_traces(traces_json, thresholds, pseudo)

    # decide whether per-learner (AUC/Brier/Threshold) is shown
    has_calibration = any(isinstance(v, (int, float)) for v in thresholds.values()) \
                      or any((_get(row, "reliability_json") or _get(row, "reliability")) for row in cal_rows)


    # metrics snapshot
    snapshot = metrics_snapshot(
        _as_traces_with_thresholds(traces_json, thresholds),
        pseudo_labels=pseudo,
    )

    labels = _load_doc_labels()

    # Build sections for the report
    started = run.get("started_at")
    ended = run.get("ended_at")
    notes = run.get("notes") or ""
    cfg_str = _pretty_json(run.get("config_json", "{}"))
    cal_table = _calibration_table(cal_rows, thresholds, briers)
    run_table = _run_table(snapshot.get("run", {}), traces_json)
    per_learner_rows = _per_learner_rows(snapshot.get("per_learner", {}) or {}, thresholds)
    examples = _select_examples(traces_json, labels=labels)
    clusters = snapshot.get("clusters", []) or []

    # Charts per learner (reliability, ROC, PR, threshold sweep, histogram)
    charts = snapshot.get("charts", {}) or {}
    chart_imgs = _render_all_charts(charts)

    # Vote-rate and reasons tables
    votes = snapshot.get("votes", {}) or {}
    reasons = snapshot.get("reasons", {}) or {}

    vote_rows = [
        [name, str(v.get("support", 0)), f"{float(v.get('vote_rate', 0.0)):.3f}"]
        for name, v in sorted(votes.items())
    ]
    reasons_rows = [
        [reason, str(cnt)]
        for reason, cnt in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    # File name: stable
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    if fmt.lower() == "md":
        # Markdown path
        content = _render_markdown(
            run_id, started, ended, notes, cfg_str, cal_table, run_table,
            per_learner_rows, examples, clusters, labels
        )
        filename = f"run_{run_id}_{ts}.md"
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return os.path.abspath(path)
    else:
        # HTML path
        content = _render_html(
            run_id, started, ended, notes, cfg_str, cal_table, run_table,
            per_learner_rows if has_calibration else [],
            basics,
            examples, clusters, labels, chart_imgs, traces_json
        )
        filename = f"run_{run_id}_{ts}.html"
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return os.path.abspath(path)


# Persistence helpers

def _safe_get_calibrations(run_id: int) -> List[Dict[str, Any]]:
    try:
        if hasattr(store, "get_calibrations_for_run"):
            rows = store.get_calibrations_for_run(run_id)
            if rows:
                return rows
        if hasattr(store, "get_calibrations"):
            return store.get_calibrations(run_id)
    except Exception:
        pass
    return []

def _safe_get_decisions(run_id: int) -> List[Dict[str, Any]]:
    try:
        return store.get_decisions(run_id, limit=1_000_000)
    except Exception:
        return []

def _collect_thresholds_briers(cal_rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]]]:
    thresholds: Dict[str, Optional[float]] = {}
    briers: Dict[str, Optional[float]] = {}

    for row in cal_rows:
        name = row.get("learner_name") or row.get("learner") or ""
        if not name:
            continue

        thr = None
        br = None

        try:
            params = row.get("params_json") or row.get("params")
            if isinstance(params, str):
                params = json.loads(params)
            if isinstance(params, dict):
                thr = params.get("threshold", params.get("thr", params.get("cutoff")))
        except Exception:
            pass

        if thr is None and isinstance(row.get("threshold"), (int, float)):
            thr = float(row["threshold"])
        if br is None and isinstance(row.get("brier"), (int, float)):
            br = float(row["brier"])
        if br is None and isinstance(row.get("brier_score"), (int, float)):
            br = float(row["brier_score"])

        thresholds[name] = thr if isinstance(thr, (int, float)) else None
        briers[name] = br if isinstance(br, (int, float)) else None

    # Prefer values saved on learner state if present
    for row in cal_rows:
        name = row.get("learner_name") or row.get("learner") or ""
        if not name:
            continue
        try:
            st = store.load_learner_state(name)
        except Exception:
            st = None
        if st is None:
            continue
        cal = _get(st, "calibration")
        st_thr = _get(cal, "threshold") or _get(cal, "thr") or _get(cal, "cutoff")
        st_br = _get(cal, "brier_score", _get(cal, "brier"))
        if st_thr is None:
            st_thr = _get(st, "threshold")
        if st_br is None:
            st_br = _get(st, "brier_score", _get(st, "brier"))
        if isinstance(st_thr, (int, float)):
            thresholds[name] = float(st_thr)
        if isinstance(st_br, (int, float)):
            briers[name] = float(st_br)

    return thresholds, briers


# Snapshot helpers

def _pseudo_from_traces(traces_json: List[Dict[str, Any]]) -> Dict[str, int]:
    pseudo: Dict[str, int] = {}
    for t in traces_json:
        pk = t.get("pair_key")
        fl = (t.get("final_label") or "").upper()
        if pk and fl in ("DUPLICATE", "NON_DUPLICATE"):
            pseudo[pk] = 1 if fl == "DUPLICATE" else 0
    return pseudo

def _as_traces_with_thresholds(traces_json: List[Dict[str, Any]], thresholds: Dict[str, Optional[float]]):
    from types import SimpleNamespace
    out = []
    for obj in traces_json:
        ln: Dict[str, Any] = {}
        learners_map = obj.get("learners") or {}
        for k, v in learners_map.items():
            # pull both prob and raw_score, with safe fallbacks
            try:
                prob = float(v.get("prob", 0.0))
            except Exception:
                prob = 0.0
            try:
                raw = float(v.get("raw_score", prob))  # fallback to prob if raw_score missing
            except Exception:
                raw = prob

            thr = thresholds.get(k)
            ln[k] = SimpleNamespace(
                prob=prob,
                raw_score=raw,
                threshold=(float(thr) if isinstance(thr, (int, float)) else None),
            )
        out.append(
            SimpleNamespace(
                pair_key=obj.get("pair_key"),
                a_id=obj.get("a_id"),
                b_id=obj.get("b_id"),
                final_label=obj.get("final_label"),
                agreed_learners=obj.get("agreed_learners", []),
                escalation_steps=obj.get("escalation_steps", []),
                learner_outputs=ln,
            )
        )
    return out

def _basics_from_traces(
    traces_json: List[Dict[str, Any]],
    thresholds: Dict[str, Optional[float]],
    pseudo: Dict[str, int],
) -> Dict[str, Dict[str, Any]]:
    """
    Replicates the always-on Basics table shown in the GUI for each learner:
      n, near-dup/uncertain counts & rates, score min/max/mean/std,
      threshold presence, %>=thr, TP/FP/TN/FN at thr, Precision/Recall/F1@thr.
    """

    scores: Dict[str, List[float]] = {}
    labels: Dict[str, List[int]] = {}
    # collect scores and labels
    for t in traces_json:
        pk = t.get("pair_key")
        if pk not in pseudo:  # use only near-dup(1) vs uncertain(0)
            continue
        lrns = t.get("learners") or {}
        for name, v in lrns.items():
            try:
                sc = float(v.get("prob", 0.0))
            except Exception:
                continue
            scores.setdefault(name, []).append(sc)
            labels.setdefault(name, []).append(int(pseudo[pk]))

    out: Dict[str, Dict[str, Any]] = {}
    for name in sorted(set(scores.keys()) | set(labels.keys())):
        p = scores.get(name, [])
        y = labels.get(name, [])
        n = len(p)
        pos = sum(y)
        neg = n - pos
        th = thresholds.get(name)
        row: Dict[str, Any] = {
            "n": n,
            "near_dup_count": pos,
            "uncertain_count": neg,
            "near_dup_rate": (pos / n) if n else 0.0,
            "score_mean": (sum(p) / n) if n else None,
            "score_std": (float(np.std(np.asarray(p, dtype=np.float32))) if n else None),
            "score_min": (min(p) if n else None),
            "score_max": (max(p) if n else None),
            "has_threshold": isinstance(th, (int, float)),
            "threshold": (float(th) if isinstance(th, (int, float)) else None),
            "pct_ge_threshold": None,
            "tp": None, "fp": None, "tn": None, "fn": None,
            "precision_at_thr": None, "recall_at_thr": None, "f1_at_thr": None,
        }
        if n and isinstance(th, (int, float)):
            pred = [1 if v >= float(th) else 0 for v in p]
            tp = sum(1 for i in range(n) if pred[i] == 1 and y[i] == 1)
            fp = sum(1 for i in range(n) if pred[i] == 1 and y[i] == 0)
            tn = sum(1 for i in range(n) if pred[i] == 0 and y[i] == 0)
            fn = sum(1 for i in range(n) if pred[i] == 0 and y[i] == 1)
            row.update({
                "pct_ge_threshold": (sum(pred) / n) if n else 0.0,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            })
            prec = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0 if (tp + fp + fn) > 0 else None
            rec  = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0 if (tp + fp + fn) > 0 else None
            f1   = (2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and (prec + rec) > 0) else None
            row.update({
                "precision_at_thr": (None if prec is None else float(prec)),
                "recall_at_thr": (None if rec is None else float(rec)),
                "f1_at_thr": (None if f1 is None else float(f1)),
            })
        out[name] = row
    return out

def _fmt_basic(x: Any, nd: int = 3) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


# Table builders
def _calibration_table(cal_rows: List[Dict[str, Any]], thresholds: Dict[str, Any], briers: Dict[str, Any]):
    rows = []
    for r in cal_rows:
        name = r.get("learner_name") or r.get("learner") or "-"
        method = r.get("method") or "-"
        reliability = r.get("reliability_json") or r.get("reliability") or "[]"
        try:
            rb = json.loads(reliability) if isinstance(reliability, str) else reliability
            if not rb:
                # Skip empty calibration snapshots
                continue
            rb_str = ", ".join(
                f"{_get(x,'prob_center',0.0):.2f}:{_get(x,'observed_pos_rate',0.0):.2f}({_get(x,'count',0)})"
                for x in (rb or [])[:6]
            )
        except Exception:
            continue

        rows.append(
            {
                "learner": name,
                "method": method,
                "threshold": _fmt_num(thresholds.get(name)),
                "brier": _fmt_num(briers.get(name)),
                "reliability": rb_str,
            }
        )
    return rows


def _run_table(run: Dict[str, Any], traces_json: List[Dict[str, Any]]):
    pairs = len(traces_json)
    exact = 0
    near = 0
    unc = 0
    for t in traces_json:
        fl = (t.get("final_label") or "").upper()
        if fl == "DUPLICATE":
            if (t.get("dup_kind") or "").upper() == "EXACT":
                exact += 1
            else:
                near += 1
        elif fl == "UNCERTAIN":
            unc += 1

    cons = run.get("consensus_rate", run.get("consensus_pct", 0.0))
    esc = run.get("escalations_pct", run.get("escalations_rate", 0.0))
    return [
        {"metric": "Total pairs", "value": str(pairs or run.get("total_pairs") or run.get("pairs_scored") or run.get("pairs") or 0)},
        {"metric": "Exact duplicates", "value": str(exact)},
        {"metric": "Near duplicates", "value": str(near)},
        {"metric": "Duplicates (total)", "value": str(exact + near)},
        {"metric": "Uncertain", "value": str(unc)},
        {"metric": "Consensus rate", "value": f"{100.0 * float(cons or 0.0):.1f}%"},
        {"metric": "Escalations", "value": f"{100.0 * float(esc or 0.0):.1f}%"},
    ]


def _per_learner_rows(per_learner: Dict[str, Any], thresholds: Dict[str, Optional[float]]) -> List[List[str]]:
    rows: List[List[str]] = []
    if not isinstance(per_learner, dict):
        return rows
    for name, m in per_learner.items():
        rows.append(
            [
                str(name),
                str(_get(m, "n", "")),
                _fmt_num(_get(m, "pos_rate")),
                _fmt_num(_get(m, "auc")),
                _fmt_num(_get(m, "brier")),
                _fmt_num(thresholds.get(name)),
            ]
        )
    # Keep consistent order
    order = {"simhash": 0, "minhash": 1, "embedding": 2}
    rows.sort(key=lambda r: order.get(r[0].split()[0].lower(), 99))
    return rows


def _select_examples(
    traces: List[Dict[str, Any]],
    *,
    k_easy: int = 5,
    k_hard: int = 5,
    k_unc: int = 5,
    labels: Optional[Dict[str, str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    labels = labels or []
    easy: List[Dict[str, Any]] = []
    hard: List[Dict[str, Any]] = []
    uncertain: List[Dict[str, Any]] = []
    for tr in traces:
        label = tr.get("final_label")
        learners = tr.get("learners", {}) or {}
        probs = []
        for v in learners.values():
            try:
                probs.append(float(_get(v, "prob", 0.0)))
            except Exception:
                probs.append(0.0)
        maxp = max(probs) if probs else 0.0
        esc = len(tr.get("escalation_steps", [])) > 0
        if label == "DUPLICATE" and not esc and maxp >= 0.95:
            easy.append(tr)
        elif label == "DUPLICATE" and esc:
            hard.append(tr)
        elif label == "UNCERTAIN":
            uncertain.append(tr)
    easy = easy[:k_easy]
    hard = hard[:k_hard]
    uncertain = uncertain[:k_unc]
    for bucket in (easy, hard, uncertain):
        for tr in bucket:
            tr["a_name"] = _pretty_doc(tr.get("a_id"), labels)
            tr["b_name"] = _pretty_doc(tr.get("b_id"), labels)
    return {"easy": easy, "hard": hard, "uncertain": uncertain}


def _load_doc_labels() -> Dict[str, str]:
    labels: Dict[str, str] = {}
    try:
        if sqlite_store and hasattr(sqlite_store, "get_all_document_files"):
            rows = sqlite_store.get_all_document_files()
            for r in rows:
                did = r.get("doc_id")
                name = r.get("filename") or os.path.basename(r.get("filepath") or "") or None
                if did and name and did not in labels:
                    labels[did] = name
    except Exception:
        pass
    return labels


def _pretty_doc(doc_id: Optional[str], labels: Dict[str, str]) -> str:
    if not doc_id:
        return ""
    name = labels.get(doc_id)
    return f"{name} ({doc_id[:8]})" if name else doc_id


def _basics_html(basics: Dict[str, Dict[str, Any]]) -> str:
    if not basics:
        return "<p class='muted'>No per-learner basics available.</p>"
    parts = []
    order = {"simhash": 0, "minhash": 1, "embedding": 2}
    for learner in sorted(basics.keys(), key=lambda k: order.get(k.lower().split()[0], 99)):
        b = basics[learner]
        rows = [
            ["Pairs (n)", str(b.get("n", 0))],
            ["Near-dup count", str(b.get("near_dup_count", 0))],
            ["Uncertain count", str(b.get("uncertain_count", 0))],
            ["Near-dup rate", _fmt_basic(b.get("near_dup_rate"))],
            ["Score mean", _fmt_basic(b.get("score_mean"))],
            ["Score std", _fmt_basic(b.get("score_std"))],
            ["Score min", _fmt_basic(b.get("score_min"))],
            ["Score max", _fmt_basic(b.get("score_max"))],
            ["Has threshold", "yes" if b.get("has_threshold") else "no"],
            ["Threshold", _fmt_basic(b.get("threshold"))],
            ["% ≥ threshold", _fmt_basic(b.get("pct_ge_threshold"))],
            ["TP", str(b.get("tp") if b.get("tp") is not None else "")],
            ["FP", str(b.get("fp") if b.get("fp") is not None else "")],
            ["TN", str(b.get("tn") if b.get("tn") is not None else "")],
            ["FN", str(b.get("fn") if b.get("fn") is not None else "")],
            ["Precision@thr", _fmt_basic(b.get("precision_at_thr"))],
            ["Recall@thr", _fmt_basic(b.get("recall_at_thr"))],
            ["F1@thr", _fmt_basic(b.get("f1_at_thr"))],
        ]
        parts.append(f"<h3>{_esc(learner.capitalize())}</h3>")
        parts.append(_table_html(["Metric", "Value"], rows))
    return "\n".join(parts)


def _basics_inline_html(basics: Dict[str, Dict[str, Any]]) -> str:
    if not basics:
        return "<p class='muted'>No per-learner basics available.</p>"
    order = {"simhash": 0, "minhash": 1, "embedding": 2}
    cols: List[str] = []
    for learner in sorted(basics.keys(), key=lambda k: order.get(k.lower().split()[0], 99)):
        b = basics[learner]
        rows = [
            ["Pairs (n)", str(b.get("n", 0))],
            ["Near-dup count", str(b.get("near_dup_count", 0))],
            ["Uncertain count", str(b.get("uncertain_count", 0))],
            ["Near-dup rate", _fmt_basic(b.get("near_dup_rate"))],
            ["Score mean", _fmt_basic(b.get("score_mean"))],
            ["Score std", _fmt_basic(b.get("score_std"))],
            ["Score min", _fmt_basic(b.get("score_min"))],
            ["Score max", _fmt_basic(b.get("score_max"))],
            ["Has threshold", "yes" if b.get("has_threshold") else "no"],
            ["Threshold", _fmt_basic(b.get("threshold"))],
            ["% ≥ threshold", _fmt_basic(b.get("pct_ge_threshold"))],
            ["TP", str(b.get("tp") if b.get("tp") is not None else "")],
            ["FP", str(b.get("fp") if b.get("fp") is not None else "")],
            ["TN", str(b.get("tn") if b.get("tn") is not None else "")],
            ["FN", str(b.get("fn") if b.get("fn") is not None else "")],
            ["Precision@thr", _fmt_basic(b.get("precision_at_thr"))],
            ["Recall@thr", _fmt_basic(b.get("recall_at_thr"))],
            ["F1@thr", _fmt_basic(b.get("f1_at_thr"))],
        ]
        cols.append(
            f"<div class='card'><h3>{_esc(learner.capitalize())}</h3>"
            f"{_table_html(['Metric','Value'], rows)}</div>"
        )
    return "<div class='three-cols'>" + "".join(cols) + "</div>"


def _charts_inline_html(chart_imgs: Dict[str, Dict[str, str]]) -> str:
    if not chart_imgs:
        return "<p class='muted'>Charts unavailable.</p>"
    order = {"simhash": 0, "minhash": 1, "embedding": 2}
    learners = sorted(chart_imgs.keys(), key=lambda k: order.get(k.lower().split()[0], 99))
    cols: List[str] = []
    for learner in learners:
        imgs = chart_imgs.get(learner, {})
        tiles = []
        for key in ("reliability", "roc", "pr", "thr", "hist"):
            if imgs.get(key):
                tiles.append(f"<div><img class='chart' src='{imgs[key]}' alt='{_esc(learner)} {key}'></div>")
        content = "<div class='charts-grid'>" + "".join(tiles) + "</div>" if tiles else "<p class='muted'>No charts for this learner.</p>"
        cols.append(f"<div class='card'><h3>{_esc(learner.capitalize())}</h3>{content}</div>")
    return "<div class='three-cols'>" + "".join(cols) + "</div>"


# Chart helpers

def _render_all_charts(charts: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    if not _HAVE_MPL:
        return out
    for name, chart in (charts or {}).items():
        imgs: Dict[str, str] = {}

        # Reliability (only with bins)
        rel = chart.get("reliability") or []
        if isinstance(rel, list) and len(rel) >= 3:
            xs = [float(_get(r, "expected_pos_rate", 0.0)) for r in rel]
            ys = [float(_get(r, "observed_pos_rate", 0.0)) for r in rel]
            imgs["reliability"] = _line_to_data_uri(xs, ys, "Predicted probability", "Observed positive rate", "Reliability curve", diag=True)

        # ROC (need variation)
        roc = chart.get("roc") or {}
        fpr = list(map(float, roc.get("fpr") or [])); tpr = list(map(float, roc.get("tpr") or []))
        if len(set(fpr)) >= 3 and len(set(tpr)) >= 3:
            imgs["roc"] = _line_to_data_uri(fpr, tpr, "False Positive Rate", "True Positive Rate", f"ROC (AUC={_fmt_num(roc.get('auc'))})", diag=True)

        # PR (need variation)
        pr = chart.get("pr") or {}
        precision = list(map(float, pr.get("precision") or [])); recall = list(map(float, pr.get("recall") or []))
        if len(set(precision)) >= 3 and len(set(recall)) >= 3:
            imgs["pr"] = _line_to_data_uri(recall, precision, "Recall", "Precision", "Precision–Recall")

        # Threshold sweep (need multiple uniq thresholds and changing curves)
        ts = chart.get("thr_sweep") or {}
        ths = list(map(float, ts.get("thresholds") or []))
        prec_s = list(map(float, ts.get("precision") or []))
        rec_s  = list(map(float, ts.get("recall") or []))
        f1_s   = list(map(float, ts.get("f1") or []))
        if len(set(ths)) >= 5 and (len(set(prec_s)) > 1 or len(set(rec_s)) > 1 or len(set(f1_s)) > 1):
            imgs["thr"] = _multi_to_data_uri(ths, [prec_s, rec_s, f1_s], ["Precision","Recall","F1"], "Threshold", "Score", "Threshold sweep", ylim=(0.0, 1.05))

        # Histogram (render if any counts exist, label clearly)
        hist = chart.get("hist") or {}
        edges = hist.get("bin_edges") or []
        pos   = hist.get("pos") or []; neg = hist.get("neg") or []
        total_counts = (sum(pos) + sum(neg)) if edges else 0
        if len(edges) >= 2 and total_counts > 0:
            imgs["hist"] = _hist_to_data_uri(
                edges, pos, neg,
                "Score (0 = Uncertain, 1 = Near-duplicate)",
                "Number of pairs",
                "Learner score distribution (Near-duplicate vs Uncertain)"
            )


        if imgs:
            out[name] = imgs
    return out



def _line_to_data_uri(xs, ys, xlabel, ylabel, title, diag=False) -> str:
    # guard: need at least two points
    try:
        if not xs or not ys or len(xs) < 2 or len(ys) < 2:
            return ""
    except Exception:
        return ""

    fig, ax = plt.subplots(figsize=(5.0, 3.1), dpi=110)
    if diag:
        ax.plot([min(xs + [0]), max(xs + [1])], [min(ys + [0]), max(ys + [1])], linestyle="--", alpha=0.4)
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _multi_to_data_uri(xs, y_list, labels, xlabel, ylabel, title, ylim=None) -> str:
    # guard: thresholds and at least one non-empty series with >=2 points
    try:
        if not xs or len(xs) < 2:
            return ""
        ok_series = any(y and len(y) >= 2 for y in (y_list or []))
        if not ok_series:
            return ""
    except Exception:
        return ""

    fig, ax = plt.subplots(figsize=(5.0, 3.1), dpi=110)
    for y, lab in zip(y_list, labels):
        if y and len(y) >= 2:
            ax.plot(xs, y, label=lab)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _hist_to_data_uri(edges, pos, neg, xlabel, ylabel, title) -> str:
    try:
        edges = list(edges or [])
        pos = list(pos or [])
        neg = list(neg or [])
        if len(edges) < 2 or ((sum(pos) + sum(neg)) <= 0):
            return ""
    except Exception:
        return ""

    centers = [(edges[i] + edges[i+1]) / 2.0 for i in range(len(edges)-1)]
    width = (edges[1]-edges[0]) * 0.9
    fig, ax = plt.subplots(figsize=(5.0, 3.1), dpi=110)
    ax.bar(centers, neg[: len(centers)], width=width, alpha=0.6, label="Uncertain")
    ax.bar(centers, pos[: len(centers)], width=width, alpha=0.6, label="Near-duplicates")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# HTML/MD rendering
def _render_html(
    run_id: int,
    started: Optional[str],
    ended: Optional[str],
    notes: str,
    cfg: str,
    cal_table,
    run_table,
    per_learner_rows: List[List[str]],
    basics: Dict[str, Dict[str, Any]],
    examples: Dict[str, List[Dict[str, Any]]],
    clusters: List[Dict[str, Any]],
    labels: Dict[str, str],
    chart_imgs: Dict[str, Dict[str, str]],
    traces_json: List[Dict[str, Any]],
) -> str:
    css = """
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; padding: 24px; color: #1f2937; }
      h1 { font-size: 24px; margin: 0 0 12px 0; }
      h2 { font-size: 18px; margin-top: 24px; }
      h3 { font-size: 15px; margin-top: 16px; }
      code, pre { background: #f9fafb; padding: 8px; border-radius: 6px; display: block; overflow-x: auto; }
      table { border-collapse: collapse; width: 100%; margin-top: 8px; }
      th, td { border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; font-size: 13px; vertical-align: top; }
      th { background: #f3f4f6; }
      .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }
      .pill { display: inline-block; padding: 2px 8px; border-radius: 9999px; background:#eef2ff; color:#3730a3; font-size:12px; }
      .muted { color:#6b7280; }
      img.chart { width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 6px; background: #fff; }
      .learner-section { display:grid; grid-template-columns: minmax(260px, 360px) 1fr; gap:16px; align-items:start; margin-top:12px; }
      .learner-col { min-width:0; }
      .charts-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:12px; }
      .three-cols { display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:12px; align-items:start; }
      .charts-grid { display:grid; grid-template-columns: 1fr; gap:12px; }
      .card { background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:12px; }
      .card h3 { margin-top:0; }
    </style>
    """
    header = f"<h1>Duplicate Detection Report — Run {run_id}</h1>"
    meta = (
        f"<p><span class='pill'>Started</span> { _esc(started or '') } "
        f"&nbsp;&nbsp; <span class='pill'>Ended</span> { _esc(ended or '') }"
        f"{' &nbsp;&nbsp; <span class=\"pill\">Notes</span> ' + _esc(notes) if notes else ''}</p>"
    )

    # Run summary
    run_table_html = _table_html(["Metric", "Value"], [[r["metric"], r["value"]] for r in run_table])

    # Per-learner metrics
    pl_headers = ["Learner", "N", "PosRate", "AUC", "Brier", "Threshold"]
    pl_html = _table_html(pl_headers, per_learner_rows) if per_learner_rows else ""

    basics_html = _basics_html(basics)

    # Calibration snapshot
    cal_rows_render = []
    for r in cal_table or []:
        cal_rows_render.append([r.get("learner","-"), r.get("method","-"), r.get("threshold",""), r.get("brier",""), r.get("reliability","")])
    cal_section_html = ""
    if cal_rows_render:
        cal_html = _table_html(["Learner", "Method", "Threshold", "Brier", "Reliability (center:obs(count))"], cal_rows_render)
        cal_section_html = f"<h2>Calibration snapshot</h2>\n{cal_html}"

    # Charts
    charts_html = _charts_inline_html(chart_imgs)

    # Examples
    ex_html = _examples_html(examples)

    # Decision traces
    traces_html = _traces_html(traces_json, labels)

    # Clusters
    clusters_html = _clusters_html(clusters, labels)

    # Config
    cfg_html = f"<pre><code>{_esc(cfg)}</code></pre>"

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Run {run_id} Report</title>{css}</head>
<body>
{header}
{meta}

<h2>Run summary</h2>
{run_table_html}

{('<h2>Per-learner metrics</h2>' + pl_html) if pl_html else ''}

<h2>Learner basics</h2>
{_basics_inline_html(basics)}

<h2>Charts</h2>
{charts_html}

{cal_section_html}

<h2>Examples</h2>
{ex_html}

<h2>Decision traces</h2>
{traces_html}

<h2>Clusters (from DUPLICATE traces)</h2>
{clusters_html}

<h2>Config</h2>
{cfg_html}
</body></html>"""


def _render_markdown(
    run_id: int,
    started: Optional[str],
    ended: Optional[str],
    notes: str,
    cfg: str,
    cal_table,
    run_table,
    per_learner_rows: List[List[str]],
    examples: Dict[str, List[Dict[str, Any]]],
    clusters: List[Dict[str, Any]],
    labels: Dict[str, str],
) -> str:
    lines = []
    lines.append(f"# Duplicate Detection Report — Run {run_id}")
    lines.append("")
    if started:
        lines.append(f"- **Started:** {started}")
    if ended:
        lines.append(f"- **Ended:** {ended}")
    if notes:
        lines.append(f"- **Notes:** {notes}")
    lines.append("")
    lines.append("## Run summary")
    lines.append(_table_md(["Metric", "Value"], [[r["metric"], r["value"]] for r in run_table]))
    lines.append("")
    lines.append("## Per-learner metrics")
    if per_learner_rows:
        lines.append(
            _table_md(
                ["Learner", "N", "PosRate", "AUC", "Brier", "Threshold"],
                per_learner_rows,
            )
        )
    else:
        lines.append("_No per-learner metrics._")

    # Calibration snapshot
    cal_rows_md = [[r.get("learner","-"), r.get("method","-"), str(r.get("threshold","")), str(r.get("brier","")), r.get("reliability","")]
                   for r in (cal_table or [])]
    if cal_rows_md:
        lines.append("")
        lines.append("## Calibration snapshot")
        lines.append(_table_md(["Learner", "Method", "Threshold", "Brier", "Reliability (center:obs(count))"], cal_rows_md))

    lines.append("")
    lines.append("## Examples")
    lines.extend(_examples_md(examples))
    lines.append("")
    lines.append("## Clusters (from DUPLICATE traces)")
    lines.append(_clusters_md(clusters, labels))
    lines.append("")
    lines.append("## Config")
    lines.append("```json")
    lines.append(cfg)
    lines.append("```")
    return "\n".join(lines)



# HTML/MD section builders
def _derive_agreed_from_trace(tr: Dict[str, Any]) -> List[str]:
    agreed: List[str] = []
    for name, v in (tr.get("learners") or {}).items():
        try:
            thr = v.get("threshold")
            prob = float(_get(v, "prob", 0.0))
            if isinstance(thr, (int, float)) and prob >= float(thr):
                agreed.append(name)
        except Exception:
            pass
    return sorted(agreed)

def _examples_html(examples: Dict[str, List[Dict[str, Any]]]) -> str:
    out = []
    for title, rows in [
        ("Easy positives", examples.get("easy", [])),
        ("Hard but resolved (escalated)", examples.get("hard", [])),
        ("Uncertain", examples.get("uncertain", [])),
    ]:
        out.append(f"<h3>{_esc(title)}</h3>")
        if not rows:
            out.append("<p><i>None</i></p>")
            continue
        hdr = ["Doc A", "Doc B", "Final", "Agreed", "Escalation", "Per-learner probs"]
        body = []
        for tr in rows:
            learners = ", ".join(f"{_esc(k)}={float(_get(v,'prob',0.0)):.2f}" for k, v in (tr.get("learners") or {}).items())
            agreed = tr.get("agreed_learners") or _derive_agreed_from_trace(tr)
            escal = " → ".join(tr.get("escalation_steps", []))
            body.append(
                [
                    _esc(tr.get("a_name") or tr.get("a_id") or ""),
                    _esc(tr.get("b_name") or tr.get("b_id") or ""),
                    _esc(tr.get("final_label", "")),
                    _esc(", ".join(agreed)),
                    _esc(escal),
                    _esc(learners),
                ]
            )
        out.append(_table_html(hdr, body))
    return "\n".join(out)


def _examples_md(examples: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    lines = []
    for title, rows in [
        ("Easy positives", examples.get("easy", [])),
        ("Hard but resolved (escalated)", examples.get("hard", [])),
        ("Uncertain", examples.get("uncertain", [])),
    ]:
        lines.append(f"### {title}")
        if not rows:
            lines.append("_None_")
            lines.append("")
            continue
        hdr = ["Doc A", "Doc B", "Final", "Agreed", "Escalation", "Per-learner probs"]
        body = []
        for tr in rows:
            learners = ", ".join(f"{k}={float(_get(v,'prob',0.0)):.2f}" for k, v in (tr.get("learners") or {}).items())
            agreed = tr.get("agreed_learners") or _derive_agreed_from_trace(tr)
            escal = " → ".join(tr.get("escalation_steps", []))
            body.append(
                [
                    tr.get("a_name") or tr.get("a_id") or "",
                    tr.get("b_name") or tr.get("b_id") or "",
                    tr.get("final_label", ""),
                    ", ".join(agreed),
                    escal,
                    learners,
                ]
            )
        lines.append(_table_md(hdr, body))
        lines.append("")
    return lines


def _traces_html(traces: List[Dict[str, Any]], labels: Dict[str, str]) -> str:
    if not traces:
        return "<p class='muted'>No traces stored.</p>"
    hdr = ["Doc A", "Doc B", "Final", "Dup kind", "Agreed", "Escalation", "Per-learner (prob)"]
    body = []
    for t in traces[:1000]:
        a = _pretty_doc(t.get("a_id"), labels)
        b = _pretty_doc(t.get("b_id"), labels)
        agreed = t.get("agreed_learners") or _derive_agreed_from_trace(t)
        esc = " → ".join(t.get("escalation_steps", []))
        learners = t.get("learners") or {}
        plist = []
        for k, v in learners.items():
            try:
                plist.append(f"{k}:{float(_get(v,'prob',0.0)):.3f}")
            except Exception:
                plist.append(f"{k}:?")
        body.append([a, b, t.get("final_label",""), t.get("dup_kind",""), ", ".join(agreed), esc, ", ".join(plist)])
    return _table_html(hdr, body)


def _clusters_html(clusters: List[Dict[str, Any]], labels: Dict[str, str]) -> str:
    if not clusters:
        return "<p><i>No clusters found</i></p>"
    hdr = ["#", "Size", "Members", "Avg prob (simhash|minhash|embed)", "Dispersion (min..max)"]
    body = []
    for c in clusters:
        members = [_pretty_doc(m, labels) for m in c.get("members", [])]
        avg = f"{_fmt_num(c.get('avg_simhash_prob'))}|{_fmt_num(c.get('avg_minhash_prob'))}|{_fmt_num(c.get('avg_embedding_prob'))}"
        disp = f"{_disp(c.get('dispersion_simhash'))} | {_disp(c.get('dispersion_minhash'))} | {_disp(c.get('dispersion_embedding'))}"
        body.append([str(c.get("cluster_index")), str(c.get("size")), ", ".join(members), avg, disp])
    return _table_html(hdr, body)


def _table_html(headers: List[str], rows: List[List[str]]) -> str:
    thead = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    trs = []
    for r in rows:
        tds = "".join(f"<td>{_esc(str(x))}</td>" for x in r)
        trs.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def _table_md(headers: List[str], rows: List[List[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    body = "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in rows)
    return "\n".join([head, sep, body])
