# src/metrics/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.ensemble.arbiter import DecisionTrace


# Public, top-level helpers
def summarize_run(traces: Iterable[DecisionTrace]) -> Dict[str, Any]:
    """
    Summarize a run using the Arbiter's semantics:
    - final_label = {"DUPLICATE", "NON_DUPLICATE", "UNCERTAIN"}
    - dup_kind = {"EXACT", "NEAR"} only when final_label == "DUPLICATE"

    We report:
      - duplicates (total), exact_duplicates, near_duplicates
      - non_duplicates, uncertain
      - consensus_rate = share of pairs that resolved (not UNCERTAIN)
      - escalations_pct = share of pairs that escalated at least once
    """
    t = list(traces)
    total = len(t)

    dup_total = 0
    exact = 0
    near = 0
    non = 0
    unc = 0
    escalations = 0

    for x in t:
        if x.final_label == "DUPLICATE":
            dup_total += 1
            kind = (getattr(x, "dup_kind", None) or "").upper()
            if kind == "EXACT":
                exact += 1
            else:
                near += 1
        elif x.final_label == "NON_DUPLICATE":
            non += 1
        else:
            # includes UNCERTAIN and any unexpected value
            unc += 1

        if getattr(x, "escalation_steps", None):
            if len(x.escalation_steps) > 0:
                escalations += 1

    # Resolved = anything that is not UNCERTAIN
    resolved = dup_total + non
    consensus_rate = (resolved / total) if total else 0.0
    escalations_pct = (escalations / total) if total else 0.0

    return {
        "total_pairs": total,
        "pairs_scored": total,
        "duplicates": dup_total,
        "exact_duplicates": exact,
        "near_duplicates": near,
        "non_duplicates": non,
        "uncertain": unc,
        "consensus_rate": float(consensus_rate),
        "escalations_pct": float(escalations_pct),
    }



def per_learner_from_pseudo(
    traces: Iterable[DecisionTrace],
    pseudo_labels: Dict[str, int],
    *,
    reliability_bins: int = 10,
    use_calibrated: bool = False,
) -> Dict[str, Dict[str, Any]]:
    scores_by_learner: Dict[str, List[float]] = {}
    labels_by_learner: Dict[str, List[int]] = {}
    calibrated_seen: Dict[str, bool] = {}

    for tr in traces:
        # Exclude exact matches from learner metrics
        if _is_exact_trace(tr):
            continue

        key = tr.pair_key
        if key not in pseudo_labels:
            continue
        y = int(pseudo_labels[key])

        for name, out in tr.learner_outputs.items():
            calibrated_seen[name] = calibrated_seen.get(name, False) or (out.threshold is not None)
            score = _pick_score(out, prefer_calibrated=use_calibrated)
            if score is None or np.isnan(score):
                continue
            scores_by_learner.setdefault(name, []).append(score)
            labels_by_learner.setdefault(name, []).append(y)

    out: Dict[str, Dict[str, Any]] = {}
    for name in sorted(scores_by_learner.keys()):
        p = np.asarray(scores_by_learner[name], dtype=np.float32)
        y = np.asarray(labels_by_learner[name], dtype=np.int32)
        is_calibrated = bool(calibrated_seen.get(name, False))

        # If no variation or only one class, report minimal info
        if not _has_two_classes(y) or not _has_variation(p):
            out[name] = {
                "n": int(p.shape[0]),
                "pos_rate": (float(np.mean(y)) if y.size else 0.0),
                "auc": None, "brier": None, "ece": None,
                "reliability": [],
                "is_calibrated": is_calibrated,
            }
            continue

        auc = _roc_auc(p, y)
        if is_calibrated:
            brier = float(np.mean((p - y) ** 2))
            rel = _reliability_bins(p, y, n_bins=reliability_bins)
            ece = _expected_calibration_error(rel)
        else:
            brier = None; rel = []; ece = None

        out[name] = {
            "n": int(p.shape[0]),
            "auc": auc, "brier": brier, "ece": ece,
            "reliability": rel, "pos_rate": (float(np.mean(y)) if y.size else 0.0),
            "is_calibrated": is_calibrated,
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

    # Aggregate per component
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
    use_calibrated: bool = False,
) -> Dict[str, Any]:
    t = list(traces)

    # Overall run summary
    run = summarize_run(t)

    # Per-learner aggregates and charts
    per_learner = per_learner_from_pseudo(
        t, pseudo_labels, reliability_bins=reliability_bins, use_calibrated=use_calibrated
    )
    clusters = cluster_stats_from_traces(t)

    # Outcome breakdown for tiny bar, Only Exact/Near/Uncertain
    dup_kinds = {"exact": 0, "near": 0, "uncertain": 0}
    for tr in t:
        if tr.final_label == "DUPLICATE":
            if (getattr(tr, "dup_kind", None) or "").upper() == "EXACT":
                dup_kinds["exact"] += 1
            else:
                dup_kinds["near"] += 1
        else:
            if tr.final_label == "UNCERTAIN":
                dup_kinds["uncertain"] += 1

    # Lightweight per-learner settings harvested from rationales
    def _collect_settings(traces_list: List[DecisionTrace]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for tr in traces_list:
            for name, lo in tr.learner_outputs.items():
                r = getattr(lo, "rationale", None)
                if isinstance(r, dict):
                    dest = out.setdefault(name, {})
                    for k, v in r.items():
                        if isinstance(v, (int, float, str, bool)) and (k not in dest):
                            dest[k] = v
        return out

    settings = _collect_settings(t)

    # Histograms below use Near-dup (1) vs Uncertain (0), NON_DUPLICATE pairs are ignored.
    charts = _chart_payloads_per_learner(t, pseudo_labels, use_calibrated=use_calibrated)

    basics = _basic_stats_per_learner(t, pseudo_labels)

    thresholds = _threshold_report(t, use_calibrated=use_calibrated)
    consensus = _consensus_report(t)
    escalations = _escalation_report(t)
    confusion, examples = _tp_tn_examples_only(t)

    return {
        "use_calibrated": bool(use_calibrated),
        "run": run,
        "dup_kinds": dup_kinds,
        "per_learner": per_learner,
        "settings": settings,
        "clusters": clusters,
        "charts": charts,
        "basics": basics,
        "thresholds": thresholds,
        "consensus": consensus,
        "escalations": escalations,
        "confusion": confusion,
        "examples": examples,
    }


def _is_exact_trace(tr: DecisionTrace) -> bool:
    kind = (getattr(tr, "dup_kind", None) or "").upper()
    return kind == "EXACT"

def _has_two_classes(y: np.ndarray) -> bool:
    return np.unique(y).size >= 2

def _has_variation(p: np.ndarray, min_unique: int = 3) -> bool:
    return np.unique(np.round(p, 6)).size >= min_unique

def _pick_score(out, prefer_calibrated: bool = False) -> Optional[float]:
    """
    Return a score in [0,1]. If prefer_calibrated=True, prefer `prob` (calibrated)
    and fall back to `raw_score`. Otherwise prefer `raw_score` and fall back to `prob`.
    """
    try:
        first, second = ("prob", "raw_score") if prefer_calibrated else ("raw_score", "prob")
        val = getattr(out, first, None)
        if val is None:
            val = getattr(out, second, None)
        if val is None:
            return None
        x = float(val)
        if x < 0.0: x = 0.0
        if x > 1.0: x = 1.0
        return x
    except Exception:
        return None



def _basic_stats_per_learner(
    traces: Iterable[DecisionTrace],
    pseudo_labels: Dict[str, int],
) -> Dict[str, Dict[str, Any]]:
    """
    Per-learner 'always-on' stats, aligned with Arbiter semantics:
      - labels: 1 = Near-duplicate, 0 = Uncertain
      - exact duplicates are excluded

    Reported per learner:
      n, near_count, uncertain_count, near_rate
      score_min/max/mean/std (overall)
      score_mean_near / score_mean_uncertain
      has_threshold, threshold, pct_at_or_above_threshold
      TP/FP/TN/FN at threshold (when both y and threshold exist)
      precision/recall/f1 at threshold (same condition)
    """
    scores_by_learner: Dict[str, List[float]] = {}
    labels_by_learner: Dict[str, List[int]] = {}
    threshold_seen: Dict[str, Optional[float]] = {}

    for tr in traces:
        if _is_exact_trace(tr):
            continue
        key = tr.pair_key
        if key not in pseudo_labels:
            continue
        y = int(pseudo_labels[key])

        for name, out in tr.learner_outputs.items():
            sc = _pick_score(out)
            if sc is None or np.isnan(sc):
                continue
            scores_by_learner.setdefault(name, []).append(float(sc))
            labels_by_learner.setdefault(name, []).append(y)
            if getattr(out, "threshold", None) is not None and threshold_seen.get(name) is None:
                try:
                    threshold_seen[name] = float(out.threshold)
                except Exception:
                    threshold_seen[name] = None

    basics: Dict[str, Dict[str, Any]] = {}
    for name in sorted(set(scores_by_learner.keys()) | set(labels_by_learner.keys())):
        p = np.asarray(scores_by_learner.get(name, []), dtype=np.float32)
        y = np.asarray(labels_by_learner.get(name, []), dtype=np.int32)

        n = int(p.size)
        near_count = int(np.sum(y == 1)) if y.size else 0
        unc_count  = int(np.sum(y == 0)) if y.size else 0
        near_rate  = (near_count / n) if n else 0.0

        row: Dict[str, Any] = {
            "n": n,
            "near_count": near_count,
            "uncertain_count": unc_count,
            "near_rate": float(near_rate),
            "score_min": float(np.min(p)) if n else None,
            "score_max": float(np.max(p)) if n else None,
            "score_mean": float(np.mean(p)) if n else None,
            "score_std": float(np.std(p)) if n else None,
            "score_mean_near": float(np.mean(p[y == 1])) if n and np.any(y == 1) else None,
            "score_mean_uncertain": float(np.mean(p[y == 0])) if n and np.any(y == 0) else None,
            "has_threshold": threshold_seen.get(name) is not None,
            "threshold": threshold_seen.get(name),
            "pct_ge_threshold": None,
            "tp": None, "fp": None, "tn": None, "fn": None,
            "precision_at_thr": None, "recall_at_thr": None, "f1_at_thr": None,
        }

        th = threshold_seen.get(name)
        if n and th is not None and y.size == n:
            pred = (p >= float(th))
            tp = int(np.sum((pred == 1) & (y == 1)))
            fp = int(np.sum((pred == 1) & (y == 0)))
            tn = int(np.sum((pred == 0) & (y == 0)))
            fn = int(np.sum((pred == 0) & (y == 1)))
            row.update({"tp": tp, "fp": fp, "tn": tn, "fn": fn})

            denom_at = float(n) if n else 1.0
            row["pct_ge_threshold"] = float(np.sum(pred)) / denom_at

            prec = (tp / (tp + fp)) if (tp + fp) > 0 else None
            rec  = (tp / (tp + fn)) if (tp + fn) > 0 else None
            f1 = ((2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and (prec + rec) > 0) else None)
            row.update({
                "precision_at_thr": (None if prec is None else float(prec)),
                "recall_at_thr": (None if rec is None else float(rec)),
                "f1_at_thr": (None if f1 is None else float(f1)),
            })

        basics[name] = row

    return basics


# Internals: reports
def _threshold_report(traces: List[DecisionTrace], *, use_calibrated: bool = False) -> Dict[str, Any]:
    if not use_calibrated:
        return {}

    pseudo: Dict[str, int] = {}
    for tr in traces:
        if _is_exact_trace(tr):
            continue
        if tr.final_label == "DUPLICATE":
            pseudo[tr.pair_key] = 1
        elif tr.final_label == "NON_DUPLICATE":
            pseudo[tr.pair_key] = 0

    out: Dict[str, Any] = {}
    for tr in traces:
        for name, lo in tr.learner_outputs.items():
            # Only include learners that actually have a threshold
            if lo.threshold is None:
                continue
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


def _tp_tn_examples_only(traces: List[DecisionTrace]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    true_pos: List[Dict[str, Any]] = []
    true_neg: List[Dict[str, Any]] = []

    for tr in traces:
        if tr.final_label not in ("DUPLICATE", "NON_DUPLICATE"):
            continue
        y = 1 if tr.final_label == "DUPLICATE" else 0

        for learner, lo in tr.learner_outputs.items():
            th = lo.threshold
            if th is None or np.isnan(lo.prob):
                continue
            pred = 1 if float(lo.prob) >= float(th) else 0
            row = {
                "learner": learner, "pair_key": tr.pair_key,
                "a_id": tr.a_id, "b_id": tr.b_id,
                "prob": float(lo.prob), "threshold": float(th),
            }
            if pred == 1 and y == 1:
                true_pos.append(row)
            elif pred == 0 and y == 0:
                true_neg.append(row)

    true_pos.sort(key=lambda r: (-float(r["prob"]), r["pair_key"]))
    true_neg.sort(key=lambda r: (float(r["prob"]), r["pair_key"]))
    return {}, {"true_positives": true_pos[:50], "true_negatives": true_neg[:50]}


def _chart_payloads_per_learner(
    traces: List[DecisionTrace],
    pseudo_labels: Dict[str, int],
    bins: int = 20,
    use_calibrated: bool = False,
) -> Dict[str, Any]:
    """
    Build plot payloads per learner.

    Semantics:
      - Labels come from Arbiter-derived pseudo labels Within the clustered world:
          1 = Near-duplicate
          0 = Uncertain

      - Histograms are produced even with a single class/degenerate scores.
      - ROC/PR/Threshold-sweep are only shown when we have both classes and score variation.
    """
    scores_by_learner: Dict[str, List[float]] = {}
    labels_by_learner: Dict[str, List[int]] = {}
    calibrated_seen: Dict[str, bool] = {}

    # Collect scores/labels
    for tr in traces:
        if _is_exact_trace(tr):
            continue
        key = tr.pair_key
        if key not in pseudo_labels:
            continue
        y = int(pseudo_labels[key])

        for name, out in tr.learner_outputs.items():
            calibrated_seen[name] = calibrated_seen.get(name, False) or (getattr(out, "threshold", None) is not None)
            score = _pick_score(out, prefer_calibrated=use_calibrated)
            if score is None or np.isnan(score):
                continue
            scores_by_learner.setdefault(name, []).append(float(score))
            labels_by_learner.setdefault(name, []).append(y)

    charts: Dict[str, Any] = {}

    for name, s_list in scores_by_learner.items():
        p = np.asarray(s_list, dtype=np.float32)
        y = np.asarray(labels_by_learner.get(name, []), dtype=np.int32)
        is_calibrated = bool(calibrated_seen.get(name, False))

        # Default payload
        payload: Dict[str, Any] = {
            "reliability": [],
            "roc": {},
            "pr": {},
            "hist": {},
            "thr_sweep": {},
            "flags": {
                "is_calibrated": is_calibrated,
                "roc_ok": False,
                "pr_ok": False,
                "thr_ok": False,
            },
        }

        # Always compute histogram
        edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float32)
        if p.size:
            pos_hist, _ = np.histogram(p[y == 1], bins=edges)  # Near-duplicates
            neg_hist, _ = np.histogram(p[y == 0], bins=edges)  # Uncertain
        else:
            pos_hist = np.zeros(bins, dtype=int)
            neg_hist = np.zeros(bins, dtype=int)

        if (pos_hist.sum() + neg_hist.sum()) > 0:
            payload["hist"] = {
                "bin_edges": edges.tolist(),
                "pos": pos_hist.astype(int).tolist(),  # Near-duplicates
                "neg": neg_hist.astype(int).tolist(),  # Uncertain
            }

            # Provide explicit labels so the UI can render an unambiguous chart
            title = (
                "Calibrated learner score distribution (Near-duplicate vs Uncertain)"
                if (use_calibrated or is_calibrated)
                else "Learner score distribution (Near-duplicate vs Uncertain)"
            )
            xlab = (
                "Calibrated probability (0 = Uncertain, 1 = Near-duplicate)"
                if (use_calibrated or is_calibrated)
                else "Score (0 = Uncertain, 1 = Near-duplicate)"
            )
            payload["hist_meta"] = {
                "title": title,
                "x_label": xlab,
                "y_label": "Number of pairs",
                "legend_pos": "Near-duplicates",
                "legend_neg": "Uncertain pairs",
            }

        # Curves only when we have both classes and score variation
        if p.size and _has_two_classes(y) and _has_variation(p):
            # ROC
            fpr, tpr = _roc_curve(p, y, points=200)
            if fpr.size >= 3 and tpr.size >= 3:
                auc = _roc_auc(p, y)
                payload["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc)}
                payload["flags"]["roc_ok"] = True

            # PR
            prec, rec = _pr_curve(p, y, points=200)
            if prec.size >= 3 and rec.size >= 3:
                payload["pr"] = {"precision": prec.tolist(), "recall": rec.tolist()}
                payload["flags"]["pr_ok"] = True

            # Threshold sweep
            ths = np.linspace(0.0, 1.0, 101, dtype=np.float32)
            pr_s = np.zeros_like(ths)
            rc_s = np.zeros_like(ths)
            f1_s = np.zeros_like(ths)
            for i, th in enumerate(ths):
                pred = (p >= th)
                tp = float(np.sum((pred == 1) & (y == 1)))
                fp = float(np.sum((pred == 1) & (y == 0)))
                fn = float(np.sum((pred == 0) & (y == 1)))
                pr_s[i] = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                rc_s[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_s[i] = (2 * pr_s[i] * rc_s[i] / (pr_s[i] + rc_s[i])) if (pr_s[i] + rc_s[i]) > 0 else 0.0

            if (np.unique(np.round(pr_s, 6)).size > 1 or
                np.unique(np.round(rc_s, 6)).size > 1 or
                np.unique(np.round(f1_s, 6)).size > 1):
                payload["thr_sweep"] = {
                    "thresholds": ths.tolist(),
                    "precision": pr_s.tolist(),
                    "recall": rc_s.tolist(),
                    "f1": f1_s.tolist(),
                }
                payload["flags"]["thr_ok"] = True

            # Reliability bins only for calibrated learners
            if is_calibrated:
                payload["reliability"] = _reliability_bins(p, y, n_bins=10)

        charts[name] = payload

    return charts


# Internals: maths bits
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

def _has_both_classes(labels: np.ndarray) -> bool:
    if labels.size == 0:
        return False
    y = labels.astype(np.int32)
    return (np.any(y == 0) and np.any(y == 1))

def _unique_count(x: np.ndarray) -> int:
    return int(np.unique(x).size)

def _roc_curve(probs: np.ndarray, labels: np.ndarray, points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    ROC using unique score thresholds (descending). Returns empty arrays if degenerate.
    """
    if probs.size == 0 or not _has_both_classes(labels):
        return np.array([]), np.array([])

    # Sort by score descending
    order = np.argsort(-probs)
    p = probs[order]
    y = labels[order].astype(np.int32)

    if _unique_count(p) < 3:
        return np.array([]), np.array([])

    # Unique thresholds
    thresholds, idx = np.unique(p, return_index=True)
    thresholds = thresholds[::-1] # descending
    idx = idx[::-1]

    # Cumulative TP/FP as we move threshold downward
    tp = 0.0; fp = 0.0
    P = float(np.sum(y == 1)); N = float(np.sum(y == 0))
    tpr = []; fpr = []
    j = 0
    for th in thresholds:
        # advance j to include all items >= th
        while j < len(p) and p[j] >= th:
            if y[j] == 1: tp += 1.0
            else: fp += 1.0
            j += 1
        tpr.append(tp / P if P > 0 else 0.0)
        fpr.append(fp / N if N > 0 else 0.0)

    return np.asarray(fpr, dtype=np.float32), np.asarray(tpr, dtype=np.float32)

def _pr_curve(probs: np.ndarray, labels: np.ndarray, points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precision–Recall using unique thresholds (descending). Returns empty arrays if degenerate.
    """
    if probs.size == 0 or not _has_both_classes(labels):
        return np.array([]), np.array([])

    order = np.argsort(-probs)
    p = probs[order]
    y = labels[order].astype(np.int32)

    if _unique_count(p) < 3:
        return np.array([]), np.array([])

    thresholds = np.unique(p)[::-1]
    precision = []; recall = []
    P = float(np.sum(y == 1))

    for th in thresholds:
        pred = (p >= th)
        tp = float(np.sum((pred == 1) & (y == 1)))
        fp = float(np.sum((pred == 1) & (y == 0)))
        fn = float(np.sum((pred == 0) & (y == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision.append(prec)
        recall.append(rec)

    return np.asarray(precision, dtype=np.float32), np.asarray(recall, dtype=np.float32)


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
        w = r.get("count", 0) / total if total else 0.0
        acc += w * abs(float(r.get("observed_pos_rate", 0.0)) - float(r.get("expected_pos_rate", 0.0)))
    return float(acc)
