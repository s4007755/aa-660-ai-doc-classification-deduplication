# src/metrics/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.ensemble.arbiter import DecisionTrace

# Public API

# Compute run-level summary for GUI
def summarize_run(traces: Iterable[DecisionTrace]) -> Dict[str, Any]:
    t = list(traces)
    total = len(t)
    dup = sum(1 for x in t if x.final_label == "DUPLICATE")
    non = sum(1 for x in t if x.final_label == "NON_DUPLICATE")
    unc = sum(1 for x in t if x.final_label == "UNCERTAIN")
    consensus_rate = 0.0 if total == 0 else (dup + non) / total
    escalations = sum(1 for x in t if len(x.escalation_steps) > 0)
    return {
        "total_pairs": total,
        "duplicates": dup,
        "non_duplicates": non,
        "uncertain": unc,
        "consensus_rate": float(consensus_rate),
        "escalations_pct": (0.0 if total == 0 else escalations / total),
    }

# Evaluate learners on pseudo labels
def per_learner_from_pseudo(
    traces: Iterable[DecisionTrace],
    pseudo_labels: Dict[str, int],
    *,
    reliability_bins: int = 10,
) -> Dict[str, Dict[str, Any]]:
    # collect probs per learner aligned to labels
    probs_by_learner: Dict[str, List[float]] = {}
    labels_by_learner: Dict[str, List[int]] = {}

    for tr in traces:
        key = tr.pair_key
        if key not in pseudo_labels:
            continue
        y = int(pseudo_labels[key])
        for name, out in tr.learner_outputs.items():
            if np.isnan(out.prob):
                continue
            probs_by_learner.setdefault(name, []).append(float(out.prob))
            labels_by_learner.setdefault(name, []).append(y)

    out: Dict[str, Dict[str, Any]] = {}
    for name in sorted(probs_by_learner.keys()):
        p = np.asarray(probs_by_learner[name], dtype=np.float32)
        y = np.asarray(labels_by_learner[name], dtype=np.int32)
        out[name] = {
            "n": int(p.shape[0]),
            "auc": _roc_auc(p, y),
            "brier": float(np.mean((p - y) ** 2)) if p.size else 0.0,
            "reliability": _reliability_bins(p, y, n_bins=reliability_bins),
            "pos_rate": (float(np.mean(y)) if y.size else 0.0),
        }
    return out

# Compare calibration snapshots to show drift
def compare_calibration_drift(
    prev: Dict[str, Dict[str, Any]],
    curr: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    learners = set(prev.keys()) | set(curr.keys())
    for name in learners:
        p = prev.get(name, {})
        c = curr.get(name, {})
        out[name] = {
            "threshold_prev": p.get("threshold"),
            "threshold_curr": c.get("threshold"),
            "threshold_delta": _delta_num(p.get("threshold"), c.get("threshold")),
            "brier_prev": p.get("brier"),
            "brier_curr": c.get("brier"),
            "brier_delta": _delta_num(p.get("brier"), c.get("brier")),
        }
    return out

# Build a lightweight cluster summary
def cluster_stats_from_traces(traces: Iterable[DecisionTrace]) -> List[Dict[str, Any]]:
    dup_tr = [tr for tr in traces if tr.final_label == "DUPLICATE"]
    # build adjacency
    adj: Dict[str, set] = {}
    for tr in dup_tr:
        adj.setdefault(tr.a_id, set()).add(tr.b_id)
        adj.setdefault(tr.b_id, set()).add(tr.a_id)
    # find components
    comps: List[List[str]] = []
    seen: set = set()
    for node in adj.keys():
        if node in seen:
            continue
        stack = [node]
        comp = []
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            comp.append(v)
            for nb in adj.get(v, []):
                if nb not in seen:
                    stack.append(nb)
        if len(comp) >= 2:
            comps.append(sorted(comp))

    # aggregate per component
    results: List[Dict[str, Any]] = []
    for idx, comp in enumerate(comps, 1):
        # collect pairwise probs where both docs in comp
        simhash_vals: List[float] = []
        minhash_vals: List[float] = []
        embed_vals: List[float] = []
        for tr in dup_tr:
            if tr.a_id in comp and tr.b_id in comp:
                if "simhash" in tr.learner_outputs:
                    simhash_vals.append(float(tr.learner_outputs["simhash"].prob))
                if "minhash" in tr.learner_outputs:
                    minhash_vals.append(float(tr.learner_outputs["minhash"].prob))
                if "embedding" in tr.learner_outputs:
                    embed_vals.append(float(tr.learner_outputs["embedding"].prob))
        results.append({
            "cluster_index": idx,
            "size": len(comp),
            "members": comp,
            "avg_simhash_prob": _safe_mean(simhash_vals),
            "avg_minhash_prob": _safe_mean(minhash_vals),
            "avg_embedding_prob": _safe_mean(embed_vals),
            "dispersion_simhash": _min_max(simhash_vals),
            "dispersion_minhash": _min_max(minhash_vals),
            "dispersion_embedding": _min_max(embed_vals),
        })
    return results

# Build a compact snapshot for the Metrics tab
def metrics_snapshot(
    traces: Iterable[DecisionTrace],
    pseudo_labels: Dict[str, int],
    *,
    reliability_bins: int = 10,
) -> Dict[str, Any]:
    t = list(traces)
    run = summarize_run(t)
    per_learner = per_learner_from_pseudo(t, pseudo_labels, reliability_bins=reliability_bins)
    clusters = cluster_stats_from_traces(t)
    return {
        "run": run,
        "per_learner": per_learner,
        "clusters": clusters,
    }

# Internals

def _safe_mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0

def _min_max(xs: List[float]) -> Dict[str, Optional[float]]:
    if not xs:
        return {"min": None, "max": None}
    return {"min": float(np.min(xs)), "max": float(np.max(xs))}

def _delta_num(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(b) - float(a)

# ROC AUC via Mann–Whitney
def _roc_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    if probs.size == 0:
        return 0.0
    y = labels.astype(np.int32)
    p = probs.astype(np.float32)
    pos = (y == 1)
    neg = (y == 0)
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # rank with average ties
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and p[order[j + 1]] == p[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    sum_pos = float(np.sum(ranks[pos]))
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

# Reliability bins centered at 0.05..0.95
def _reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> List[Dict[str, Any]]:
    centers = np.linspace(0.05, 0.95, n_bins, dtype=np.float32)
    rows: List[Dict[str, Any]] = []
    for c in centers:
        mask = (probs >= (c - 0.05)) & (probs < (c + 0.05))
        if not np.any(mask):
            rows.append({"prob_center": float(c), "expected_pos_rate": float(c), "observed_pos_rate": 0.0, "count": 0})
            continue
        obs = float(np.mean(labels[mask]))
        rows.append({"prob_center": float(c), "expected_pos_rate": float(c), "observed_pos_rate": obs, "count": int(np.sum(mask))})
    return rows
