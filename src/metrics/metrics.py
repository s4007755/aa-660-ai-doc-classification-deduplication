# src/metrics/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.ensemble.arbiter import DecisionTrace

# Public API

def summarize_run(traces: Iterable[DecisionTrace]) -> Dict[str, Any]:
    t = list(traces)
    total = len(t)
    dup = sum(1 for x in t if x.final_label == "DUPLICATE")
    near = sum(1 for x in t if (x.final_label or "") == "NEAR_DUPLICATE")
    non = sum(1 for x in t if x.final_label == "NON_DUPLICATE")
    unc = sum(1 for x in t if x.final_label == "UNCERTAIN")
    consensus_rate = 0.0 if total == 0 else (dup + non + near) / total
    escalations = sum(1 for x in t if len(x.escalation_steps) > 0)
    return {
        "total_pairs": total,
        "pairs_scored": total,
        "duplicates": dup,
        "near_duplicates": near,
        "non_duplicates": non,
        "uncertain": unc,
        "consensus_rate": float(consensus_rate),
        "escalations_pct": (0.0 if total == 0 else escalations / total),
    }


def per_learner_from_pseudo(
    traces: Iterable[DecisionTrace],
    pseudo_labels: Dict[str, int],
    *,
    reliability_bins: int = 10,
) -> Dict[str, Dict[str, Any]]:
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
        brier = float(np.mean((p - y) ** 2)) if p.size else 0.0
        rel = _reliability_bins(p, y, n_bins=reliability_bins)
        ece = _expected_calibration_error(rel)
        out[name] = {
            "n": int(p.shape[0]),
            "auc": _roc_auc(p, y),
            "brier": brier,
            "ece": ece,
            "reliability": rel,
            "pos_rate": (float(np.mean(y)) if y.size else 0.0),
        }
    return out


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
            "ece_prev": p.get("ece"),
            "ece_curr": c.get("ece"),
            "ece_delta": _delta_num(p.get("ece"), c.get("ece")),
        }
    return out


def cluster_stats_from_traces(traces: Iterable[DecisionTrace]) -> List[Dict[str, Any]]:
    dup_tr = [tr for tr in traces if tr.final_label == "DUPLICATE"]
    adj: Dict[str, set] = {}
    for tr in dup_tr:
        adj.setdefault(tr.a_id, set()).add(tr.b_id)
        adj.setdefault(tr.b_id, set()).add(tr.a_id)
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


def metrics_snapshot(
    traces: Iterable[DecisionTrace],
    pseudo_labels: Dict[str, int],
    *,
    reliability_bins: int = 10,
) -> Dict[str, Any]:
    """
    Rich snapshot for the Metrics tab.
    """
    t = list(traces)
    run = summarize_run(t)
    per_learner = per_learner_from_pseudo(t, pseudo_labels, reliability_bins=reliability_bins)
    clusters = cluster_stats_from_traces(t)

    charts = _chart_payloads_per_learner(t, pseudo_labels)
    thresholds = _threshold_report(t)
    consensus = _consensus_report(t)
    escalations = _escalation_report(t)

    # confusion and curated examples
    confusion, examples = _confusion_and_examples(t)

    return {
        "run": run,
        "per_learner": per_learner,
        "clusters": clusters,
        "charts": charts,
        "thresholds": thresholds,
        "consensus": consensus,
        "escalations": escalations,
        "confusion": confusion,
        "examples": examples,
    }

# Internals: reports

def _threshold_report(traces: List[DecisionTrace]) -> Dict[str, Any]:
    pseudo: Dict[str, int] = {}
    for tr in traces:
        if tr.final_label == "DUPLICATE":
            pseudo[tr.pair_key] = 1
        elif tr.final_label == "NON_DUPLICATE":
            pseudo[tr.pair_key] = 0

    out: Dict[str, Any] = {}
    for tr in traces:
        for name, lo in tr.learner_outputs.items():
            if name not in out:
                out[name] = {"scores": [], "labels": [], "threshold": lo.threshold, "near_band_share": 0.0}
            if tr.pair_key in pseudo:
                out[name]["scores"].append(float(lo.prob))
                out[name]["labels"].append(int(pseudo[tr.pair_key]))

    for name, data in out.items():
        s = np.asarray(data["scores"], dtype=np.float32)
        y = np.asarray(data["labels"], dtype=np.int32)
        th = data["threshold"]
        if th is None or s.size == 0:
            data.update({"precision": None, "recall": None, "f1": None, "support": int(s.size)})
            continue
        preds = (s >= float(th))
        tp = float(np.sum((preds == 1) & (y == 1)))
        fp = float(np.sum((preds == 1) & (y == 0)))
        fn = float(np.sum((preds == 0) & (y == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else None
        rec = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (2 * prec * rec / (prec + rec)) if prec is not None and rec is not None and (prec + rec) > 0 else None
        band = 0.05
        nb = np.mean((s >= (float(th) - band)) & (s < (float(th) + band))) if s.size else 0.0
        data.update({
            "precision": None if prec is None else float(prec),
            "recall": None if rec is None else float(rec),
            "f1": None if f1 is None else float(f1),
            "support": int(s.size),
            "near_band_share": float(nb),
        })
    return out


def _consensus_report(traces: List[DecisionTrace]) -> Dict[str, Any]:
    names = sorted({n for tr in traces for n in tr.learner_outputs.keys()})
    n = len(names)
    agree = np.zeros((n, n), dtype=np.float32)
    counts = np.zeros((n, n), dtype=np.float32)
    for tr in traces:
        votes = {}
        for nme, lo in tr.learner_outputs.items():
            th = lo.threshold
            v = None
            if th is not None:
                v = (float(lo.prob) >= float(th))
            votes[nme] = int(bool(v)) if v is not None else None
        for i in range(n):
            for j in range(n):
                a = names[i]; b = names[j]
                va = votes.get(a); vb = votes.get(b)
                if va is None or vb is None:
                    continue
                counts[i, j] += 1
                if va == vb:
                    agree[i, j] += 1
    mat = (agree / np.maximum(counts, 1e-9)).tolist() if n else []
    voter_share = {}
    for nme in names:
        pos = 0; tot = 0
        for tr in traces:
            lo = tr.learner_outputs.get(nme)
            if not lo or lo.threshold is None:
                continue
            tot += 1
            if float(lo.prob) >= float(lo.threshold):
                pos += 1
        voter_share[nme] = float(pos / tot) if tot else 0.0
    return {"learners": names, "agreement": mat, "voter_share": voter_share}


def _escalation_report(traces: List[DecisionTrace]) -> Dict[str, Any]:
    total = len(traces)
    step_counts: Dict[str, int] = {}
    escalated = 0
    for tr in traces:
        if tr.escalation_steps:
            escalated += 1
            for s in tr.escalation_steps:
                step_counts[s] = step_counts.get(s, 0) + 1
    return {
        "rate": (0.0 if total == 0 else escalated / total),
        "by_step": step_counts,
    }


def _confusion_and_examples(traces: List[DecisionTrace]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    confusion: Dict[str, Dict[str, Any]] = {}
    false_pos: List[Dict[str, Any]] = []
    false_neg: List[Dict[str, Any]] = []

    for tr in traces:
        if tr.final_label not in ("DUPLICATE", "NON_DUPLICATE"):
            continue
        y = 1 if tr.final_label == "DUPLICATE" else 0

        for learner, lo in tr.learner_outputs.items():
            th = lo.threshold
            if th is None or np.isnan(lo.prob):
                continue
            pred = 1 if float(lo.prob) >= float(th) else 0
            entry = confusion.setdefault(learner, {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
            if pred == 1 and y == 1:
                entry["tp"] += 1
            elif pred == 1 and y == 0:
                entry["fp"] += 1
                false_pos.append({
                    "learner": learner, "pair_key": tr.pair_key,
                    "a_id": tr.a_id, "b_id": tr.b_id,
                    "prob": float(lo.prob), "threshold": float(th),
                })
            elif pred == 0 and y == 0:
                entry["tn"] += 1
            elif pred == 0 and y == 1:
                entry["fn"] += 1
                false_neg.append({
                    "learner": learner, "pair_key": tr.pair_key,
                    "a_id": tr.a_id, "b_id": tr.b_id,
                    "prob": float(lo.prob), "threshold": float(th),
                })

    # derive PR, RC, F1
    for learner, e in confusion.items():
        tp = float(e["tp"]); fp = float(e["fp"]); fn = float(e["fn"])
        prec = tp / (tp + fp) if (tp + fp) > 0 else None
        rec  = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (2 * prec * rec / (prec + rec)) if prec is not None and rec is not None and (prec + rec) > 0 else None
        e["precision"] = None if prec is None else float(prec)
        e["recall"] = None if rec is None else float(rec)
        e["f1"] = None if f1 is None else float(f1)

    # curate examples
    false_pos.sort(key=lambda r: (-float(r["prob"]), r["pair_key"]))
    false_neg.sort(key=lambda r: (float(r["prob"]), r["pair_key"]))
    fp_show = false_pos[:50]
    fn_show = false_neg[:50]

    return confusion, {"false_positives": fp_show, "false_negatives": fn_show}


def _chart_payloads_per_learner(
    traces: List[DecisionTrace],
    pseudo_labels: Dict[str, int],
    bins: int = 20,
) -> Dict[str, Any]:
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

    charts: Dict[str, Any] = {}
    for name, p_list in probs_by_learner.items():
        p = np.asarray(p_list, dtype=np.float32)
        y = np.asarray(labels_by_learner[name], dtype=np.int32)
        rel = _reliability_bins(p, y, n_bins=10)
        fpr, tpr = _roc_curve(p, y, points=200)
        prec, rec = _pr_curve(p, y, points=200)
        edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float32)
        pos_hist, _ = np.histogram(p[y == 1], bins=edges)
        neg_hist, _ = np.histogram(p[y == 0], bins=edges)
        ths = np.linspace(0.0, 1.0, 101, dtype=np.float32)
        pr = np.zeros_like(ths)
        rc = np.zeros_like(ths)
        f1 = np.zeros_like(ths)
        for i, th in enumerate(ths):
            pred = (p >= th)
            tp = float(np.sum((pred == 1) & (y == 1)))
            fp = float(np.sum((pred == 1) & (y == 0)))
            fn = float(np.sum((pred == 0) & (y == 1)))
            pr[i] = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rc[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1[i] = (2 * pr[i] * rc[i] / (pr[i] + rc[i])) if (pr[i] + rc[i]) > 0 else 0.0

        charts[name] = {
            "reliability": rel,
            "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": _roc_auc(p, y)},
            "pr": {"precision": prec.tolist(), "recall": rec.tolist()},
            "hist": {
                "bin_edges": edges.tolist(),
                "pos": pos_hist.astype(int).tolist(),
                "neg": neg_hist.astype(int).tolist(),
            },
            "thr_sweep": {
                "thresholds": ths.tolist(),
                "precision": pr.tolist(),
                "recall": rc.tolist(),
                "f1": f1.tolist(),
            },
        }
    return charts

# Internals: math helpers

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
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and p[order[j + 1]] == p[order[i]]:  # tie handling
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    sum_pos = float(np.sum(ranks[pos]))
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def _roc_curve(probs: np.ndarray, labels: np.ndarray, points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    if probs.size == 0:
        return np.zeros(1), np.zeros(1)
    y = labels.astype(np.int32)
    p = probs.astype(np.float32)
    ths = np.linspace(0.0, 1.0, points, dtype=np.float32)
    tpr = np.zeros_like(ths)
    fpr = np.zeros_like(ths)
    for i, th in enumerate(ths):
        pred = (p >= th)
        tp = float(np.sum((pred == 1) & (y == 1)))
        fp = float(np.sum((pred == 1) & (y == 0)))
        fn = float(np.sum((pred == 0) & (y == 1)))
        tn = float(np.sum((pred == 0) & (y == 0)))
        tpr[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr[i] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return fpr, tpr

def _pr_curve(probs: np.ndarray, labels: np.ndarray, points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    if probs.size == 0:
        return np.zeros(1), np.zeros(1)
    y = labels.astype(np.int32)
    p = probs.astype(np.float32)
    ths = np.linspace(0.0, 1.0, points, dtype=np.float32)
    precision = np.zeros_like(ths)
    recall = np.zeros_like(ths)
    for i, th in enumerate(ths):
        pred = (p >= th)
        tp = float(np.sum((pred == 1) & (y == 1)))
        fp = float(np.sum((pred == 1) & (y == 0)))
        fn = float(np.sum((pred == 0) & (y == 1)))
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall

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

def _expected_calibration_error(rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    total = sum(r.get("count", 0) for r in rows)
    if total == 0:
        return 0.0
    acc = 0.0
    for r in rows:
        w = r.get("count", 0) / total
        acc += w * abs(float(r.get("observed_pos_rate", 0.0)) - float(r.get("expected_pos_rate", 0.0)))
    return float(acc)
