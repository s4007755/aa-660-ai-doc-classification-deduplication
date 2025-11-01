from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.learners.base import (
    DocumentView,
    Pair,
    CalibrationParams,
)

# Public API

def calibrate_adaptive_and_select_threshold(
    raw_scores: Sequence[float],
    labels: Sequence[int],
    *,
    target_precision: float = 0.98,
    n_bins: int = 20,
    tiny_cutoff: int = 120,
    min_pos: int = 5,
    min_neg: int = 5,
    eps: float = 1e-6,
) -> Tuple[CalibrationParams, Dict[str, float], np.ndarray, np.ndarray]:
    """
    Calibrate scores and pick a decision threshold to meet a precision target.

    Strategy
    --------
    - For tiny or class-poor datasets: Platt scaling (logistic).
    - Otherwise: monotone, quantile-binned calibration with isotonic-like smoothing.
    - Threshold is chosen on calibrated probs to satisfy `target_precision` as
      tightly as possible under a maximize threshold policy.
    """
    scores = _as_np(raw_scores)
    y = _as_np(labels).astype(np.float32)
    n = int(scores.size)

    # Degenerate guard: missing data or single class = identity-ish mapping.
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if n == 0 or pos == 0 or neg == 0:
        th = 0.95
        params = CalibrationParams(
            method="none",
            params={},
            threshold=float(min(th, 1.0 - 1e-6)),
            brier_score=0.25,
            reliability_bins=_reliability_bins(scores, y, n_bins=10),
        )
        return params, {}, np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # Small-sample regime -> Platt scaling.
    if n < tiny_cutoff or pos < min_pos or neg < min_neg:
        a, b = _fit_platt_reg(scores, y, l2=1.0, iters=200, lr=0.2)
        cal = _sigmoid(a * scores + b)
        th = _choose_threshold_by_precision(cal, y, target_precision, mode="max")
        th = float(min(th, 1.0 - 1e-6))
        brier = float(np.mean((cal - y) ** 2))
        bins = _reliability_bins(cal, y, n_bins=10)
        params = CalibrationParams(
            method="platt",
            params={"a": float(a), "b": float(b)},
            threshold=th,
            brier_score=brier,
            reliability_bins=bins,
        )
        return params, {"a": float(a), "b": float(b)}, np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # Larger datasets -> quantile-binned calibration with monotonic smoothing.
    edges, probs = _fit_quantile_binned(scores, y, n_bins=max(6, n_bins))
    cal = np.array([_calibrated_prob_binned(s, edges, probs) for s in scores], dtype=np.float32)
    th = _choose_threshold_by_precision(cal, y, target_precision, mode="max")
    th = float(min(th, 1.0 - 1e-6))
    brier = float(np.mean((cal - y) ** 2))
    bins = _reliability_bins(cal, y, n_bins=10)
    params = CalibrationParams(method="isotonic", params={}, threshold=th, brier_score=brier, reliability_bins=bins)
    return params, {}, edges, probs


def apply_binning(raw_score: float, edges: np.ndarray, probs: np.ndarray) -> float:
    """Apply piecewise-constant with linear interpolation binning calibration to a single score."""
    return float(_calibrated_prob_binned(float(raw_score), edges, probs))


def apply_platt(raw_score: float, a: float, b: float) -> float:
    """Apply Platt scaling (sigmoid with parameters a, b) to a single score."""
    return float(_sigmoid(a * float(raw_score) + b))


def calibrate_binning_and_select_threshold(
    raw_scores: Sequence[float],
    labels: Sequence[int],
    *,
    target_precision: float = 0.98,
    n_bins: int = 20,
) -> Tuple[CalibrationParams, np.ndarray, np.ndarray]:
    """
    Quantile-binned calibration and precision-targeted threshold selection.
    Prefer `calibrate_adaptive_and_select_threshold` unless need binning only.
    """
    scores = _as_np(raw_scores)
    y = _as_np(labels).astype(np.float32)
    edges, probs = _fit_quantile_binned(scores, y, n_bins=n_bins)
    cal = np.array([_calibrated_prob_binned(s, edges, probs) for s in scores], dtype=np.float32)
    th = _choose_threshold_by_precision(cal, y, target_precision, mode="max")
    th = float(min(th, 1.0 - 1e-6))
    brier = float(np.mean((cal - y) ** 2))
    bins = _reliability_bins(cal, y, n_bins=10)
    params = CalibrationParams(method="isotonic", params={}, threshold=float(th), brier_score=brier, reliability_bins=bins)
    return params, edges, probs


def calibrate_platt_and_select_threshold(
    raw_scores: Sequence[float],
    labels: Sequence[int],
    *,
    target_precision: float = 0.98,
    lr: float = 0.1,
    iters: int = 200,
) -> Tuple[CalibrationParams, Tuple[float, float]]:
    """
    Platt calibration (logistic) and precision targeted threshold selection.
    Prefer `calibrate_adaptive_and_select_threshold` unless need Platt only.
    """
    scores = _as_np(raw_scores)
    y = _as_np(labels).astype(np.float32)
    a, b = _fit_platt_reg(scores, y, l2=1.0, iters=iters, lr=lr)
    cal = _sigmoid(a * scores + b)
    th = _choose_threshold_by_precision(cal, y, target_precision, mode="max")
    th = float(min(th, 1.0 - 1e-6))
    brier = float(np.mean((cal - y) ** 2))
    bins = _reliability_bins(cal, y, n_bins=10)
    params = CalibrationParams(method="platt", params={"a": float(a), "b": float(b)}, threshold=float(th), brier_score=brier, reliability_bins=bins)
    return params, (a, b)


def apply_binning_or_platt(raw_score: float, cal: Optional[CalibrationParams], edges: Optional[np.ndarray], probs: Optional[np.ndarray]) -> float:
    """
    Apply the appropriate calibration given `CalibrationParams` and optional bin specs.
    """
    if cal is not None and cal.method == "platt":
        a = float(cal.params.get("a", 0.0))
        b = float(cal.params.get("b", 0.0))
        return apply_platt(raw_score, a, b)
    if edges is not None and probs is not None and edges.size and probs.size:
        return apply_binning(raw_score, edges, probs)
    return float(max(0.0, min(1.0, raw_score)))


# Bootstrap builders

def build_bootstrap_from_exact_duplicates(
    docs: Dict[str, DocumentView],
    *,
    max_pos_pairs: int = 50_000,
    max_neg_pairs: int = 50_000,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Pair], List[Pair]]:
    """
    Construct a fast bootstrap set using exact-text groups.

    Positives
    All within-bucket pairs.

    Negatives
    Uniformly sampled cross-bucket pairs, with caps scaled to corpus size.
    """
    rng = rng or random.Random(13)

    # Group by normalized exact text.
    by_text: Dict[str, List[str]] = {}
    for did, dv in docs.items():
        key = (dv.text or "").strip()
        if key:
            by_text.setdefault(key, []).append(did)

    # Positives: within-group pairs.
    positives: List[Pair] = []
    for ids in by_text.values():
        k = len(ids)
        if k < 2:
            continue
        all_pairs = list(itertools.combinations(ids, 2))
        rng.shuffle(all_pairs)
        for a, b in all_pairs[: max_pos_pairs - len(positives)]:
            positives.append((docs[a], docs[b]))
        if len(positives) >= max_pos_pairs:
            break

    # Negatives: sample uniformly without replacement.
    ids = list(docs.keys())
    n = len(ids)
    if n < 2:
        return positives, []

    max_possible_pairs = n * (n - 1) // 2
    neg_cap = min(max_neg_pairs, max(50, 10 * n), max_possible_pairs)

    pos_set = {tuple(sorted((a.doc_id, b.doc_id))) for a, b in positives}

    negatives: List[Pair] = []
    tried = set()
    attempts = 0
    limit_attempts = neg_cap * 5  # soft bound to avoid long tails
    while len(negatives) < neg_cap and attempts < limit_attempts:
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            attempts += 1
            continue
        a = ids[i]; b = ids[j]
        key = tuple(sorted((a, b)))
        if key in tried or key in pos_set:
            attempts += 1
            continue
        tried.add(key)
        negatives.append((docs[a], docs[b]))
        attempts += 1

    return positives, negatives


def build_bootstrap_via_fetchers(
    fetch_all_docs: Callable[[], Dict[str, DocumentView]],
    *,
    max_pos_pairs: int = 50_000,
    max_neg_pairs: int = 50_000,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Pair], List[Pair]]:
    """
    Convenience wrapper: fetch documents and derive caps proportional to corpus size.
    """
    docs = fetch_all_docs()
    n = max(0, len(docs))
    max_pos_pairs = int(min(max_pos_pairs, max(20, n * 5)))
    max_neg_pairs = int(min(max_neg_pairs, max(50, n * 10)))
    return build_bootstrap_from_exact_duplicates(
        docs,
        max_pos_pairs=max_pos_pairs,
        max_neg_pairs=max_neg_pairs,
        rng=rng,
    )


def reliability_bins_from_probs(probs: Sequence[float], labels: Sequence[int], n_bins: int = 10):
    """Compute fixed-width reliability bins from already calibrated probabilities."""
    p = _as_np(probs).astype(np.float32)
    y = _as_np(labels).astype(np.float32)
    return _reliability_bins(p, y, n_bins=n_bins)


# Internals

def _as_np(x: Sequence[float]) -> np.ndarray:
    """Convert a Python sequence to a float32 NumPy array."""
    return np.asarray(list(x), dtype=np.float32)


def _fit_quantile_binned(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 20,
    use_fixed_edges: bool = False,  # Optional override for deterministic edges
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit quantile-based bins and per-bin positive rates with Laplacian smoothing.
    Enforces monotonicity across bin probabilities.
    """
    if scores.size == 0:
        edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
        probs = np.linspace(0.0, 1.0, n_bins, dtype=np.float32)
        return edges, probs

    # Choose edges: quantiles by default, fixed-width if requested.
    if use_fixed_edges:
        edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
    else:
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        try:
            edges = np.unique(np.quantile(scores, qs, method="linear")).astype(np.float32)
        except TypeError:
            edges = np.unique(np.quantile(scores, qs, interpolation="linear")).astype(np.float32)
        # Ensure full [0,1] coverage for interpolation safety.
        if edges[0] > 0.0:  edges = np.insert(edges, 0, 0.0).astype(np.float32)
        if edges[-1] < 1.0: edges = np.append(edges, 1.0).astype(np.float32)

    nb = max(1, edges.size - 1)
    idx = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, nb - 1)

    y = np.clip(labels.astype(np.float32), 0.0, 1.0)
    pos = np.bincount(idx, weights=y, minlength=nb).astype(np.float64)
    cnt = np.bincount(idx, minlength=nb).astype(np.float64)

    # Laplace smoothing: (pos+1)/(cnt+2)
    probs = (pos + 1.0) / (cnt + 2.0)

    # Enforce monotonicity (non-decreasing).
    for i in range(1, nb):
        if probs[i] < probs[i - 1]:
            probs[i] = probs[i - 1]

    return edges.astype(np.float32), probs.astype(np.float32)


def _calibrated_prob_binned(score: float, edges: np.ndarray, probs: np.ndarray) -> float:
    """
    Map a raw score into a calibrated probability using bin interpolation.
    Linear interpolation between adjacent bins reduces edge artifacts.
    """
    if edges.size == 0:
        return float(max(0.0, min(1.0, score)))
    nb = probs.shape[0]
    i = int(np.clip(np.searchsorted(edges, score, side="right") - 1, 0, nb - 1))
    left = edges[i]; right = edges[i + 1]
    p = probs[i]
    if right > left:
        t = (score - left) / (right - left)
        p_next = probs[min(i + 1, nb - 1)]
        return float((1 - t) * p + t * p_next)
    return float(p)


def _fit_platt_reg(scores: np.ndarray, labels: np.ndarray, *, l2: float = 1.0, iters: int = 25, lr: float = 1.0) -> Tuple[float, float]:
    """
    Fit logistic regression with L2 regularization via Newton/IRLS.
    """
    x = scores.astype(np.float32)
    y = np.clip(labels.astype(np.float32), 0.0, 1.0)

    # Add bias term: w = [a, b] s.t. p = sigmoid(a*x + b)
    X = np.stack([x, np.ones_like(x)], axis=1)
    w = np.zeros(2, dtype=np.float64)

    lam = float(l2)
    I = np.eye(2, dtype=np.float64)
    for _ in range(max(1, iters)):
        z = X @ w
        z = np.clip(z, -60.0, 60.0)
        p = 1.0 / (1.0 + np.exp(-z))

        W = p * (1.0 - p)  # per-sample curvature
        if np.all(W < 1e-12):
            break

        H = (X.T * W) @ X + lam * I
        g = X.T @ (p - y) + lam * w

        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            delta = g  # conservative fallback

        w -= lr * delta
        if np.linalg.norm(delta) < 1e-8:
            break

    a, b = float(w[0]), float(w[1])
    return a, b


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically-stable sigmoid with clipping."""
    z = np.clip(z, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _choose_threshold_by_precision(
    calibrated_probs: np.ndarray,
    labels: np.ndarray,
    target_precision: float,
    *,
    mode: str = "max",
) -> float:
    """
    Select a probability threshold meeting (or most nearly meeting) the precision target.

    Policy
    - Evaluate at unique score cut points (descending).
    - Choose the highest threshold that still meets the precision target (mode="max"),
      which generally minimizes recall loss while honoring precision.
    - If target is infeasible, back off to just below the max positive probability.
    """
    if calibrated_probs.size == 0:
        return 0.5

    y = labels.astype(np.int32)
    p = calibrated_probs.astype(np.float32)

    # Sort once by probability.
    order = np.argsort(-p)
    p = p[order]
    y = y[order]

    # Online precision curve: TP / (TP+FP) as we lower the threshold.
    cum_pos = np.cumsum(y == 1, dtype=np.int64)
    cum_tot = np.arange(1, p.size + 1, dtype=np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = cum_pos / cum_tot

    # Consider only distinct thresholds.
    change_idx = np.r_[0, np.nonzero(np.diff(p))[0] + 1]
    cut_prec = prec[change_idx]
    cut_thr  = p[change_idx]

    ok_mask = cut_prec >= float(target_precision)
    if not np.any(ok_mask):
        # Infeasible: push threshold near the max positive prob.
        pos_probs = calibrated_probs[labels == 1]
        if pos_probs.size:
            best = float(min(float(np.max(pos_probs)) - 1e-3, 0.999))
        else:
            best = 0.999
        return float(best)

    if mode == "max":
        th = float(np.max(cut_thr[ok_mask]))
    else:
        th = float(np.min(cut_thr[ok_mask]))

    return float(min(th, 1.0 - 1e-6))


def _reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> List[Dict[str, float]]:
    """
    Compute fixed-width reliability bins across [0,1].
    """
    if probs.size == 0:
        centers = np.linspace(0.05, 0.95, n_bins, dtype=np.float32)
        return [{"prob_center": float(c), "expected_pos_rate": float(c), "observed_pos_rate": 0.0, "count": 0} for c in centers]

    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
    bins = np.digitize(probs.astype(np.float32), edges, right=False) - 1
    bins = np.clip(bins, 0, n_bins - 1)

    cnt = np.bincount(bins, minlength=n_bins).astype(np.int64)
    pos = np.bincount(bins, weights=labels.astype(np.float32), minlength=n_bins).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        obs = np.where(cnt > 0, pos / cnt, 0.0)

    centers = (edges[:-1] + edges[1:]) * 0.5
    rows: List[Dict[str, float]] = []
    for i in range(n_bins):
        rows.append({
            "prob_center": float(centers[i]),
            "expected_pos_rate": float(centers[i]),
            "observed_pos_rate": float(obs[i]),
            "count": int(cnt[i]),
        })
    return rows
