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
    scores = _as_np(raw_scores)
    y = _as_np(labels).astype(np.float32)
    n = int(scores.size)

    # Guard: degenerate or missing classes
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if n == 0 or pos == 0 or neg == 0:
        # Identity-ish mapping with conservative threshold
        th = 0.95
        params = CalibrationParams(
            method="none",
            params={},
            threshold=float(min(th, 1.0 - 1e-6)),
            brier_score=0.25,
            reliability_bins=_reliability_bins(scores, y, n_bins=10),
        )
        return params, {}, np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # Tiny data -> Platt
    if n < tiny_cutoff or pos < min_pos or neg < min_neg:
        a, b = _fit_platt_reg(scores, y, l2=1.0, iters=200, lr=0.2)
        cal = _sigmoid(a * scores + b)
        th = _choose_threshold_by_precision(cal, y, target_precision, mode="max")
        th = float(min(th, 1.0 - 1e-6))
        brier = float(np.mean((cal - y) ** 2))
        bins = _reliability_bins(cal, y, n_bins=10)
        params = CalibrationParams(method="platt", params={"a": float(a), "b": float(b)}, threshold=th, brier_score=brier, reliability_bins=bins)
        return params, {"a": float(a), "b": float(b)}, np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # Larger data -> quantile-binned
    edges, probs = _fit_quantile_binned(scores, y, n_bins=max(6, n_bins))
    cal = np.array([_calibrated_prob_binned(s, edges, probs) for s in scores], dtype=np.float32)
    th = _choose_threshold_by_precision(cal, y, target_precision, mode="max")
    th = float(min(th, 1.0 - 1e-6))
    brier = float(np.mean((cal - y) ** 2))
    bins = _reliability_bins(cal, y, n_bins=10)
    params = CalibrationParams(method="isotonic", params={}, threshold=th, brier_score=brier, reliability_bins=bins)
    return params, {}, edges, probs


def apply_binning(raw_score: float, edges: np.ndarray, probs: np.ndarray) -> float:
    return float(_calibrated_prob_binned(float(raw_score), edges, probs))


def apply_platt(raw_score: float, a: float, b: float) -> float:
    return float(_sigmoid(a * float(raw_score) + b))


def calibrate_binning_and_select_threshold(
    raw_scores: Sequence[float],
    labels: Sequence[int],
    *,
    target_precision: float = 0.98,
    n_bins: int = 20,
) -> Tuple[CalibrationParams, np.ndarray, np.ndarray]:
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


def apply_binning_or_platt(raw_score: float, cal: CalibrationParams, edges: Optional[np.ndarray], probs: Optional[np.ndarray]) -> float:
    if cal.method == "platt":
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
    rng = rng or random.Random(13)

    # Group by normalized exact text
    by_text: Dict[str, List[str]] = {}
    for did, dv in docs.items():
        key = (dv.text or "").strip()
        if key:
            by_text.setdefault(key, []).append(did)

    # Positives = all pairs within identical-text buckets
    positives: List[Pair] = []
    for ids in by_text.values():
        k = len(ids)
        if k < 2:
            continue
        # All pairs if small, else sample up to cap
        all_pairs = list(itertools.combinations(ids, 2))
        rng.shuffle(all_pairs)
        for a, b in all_pairs[: max_pos_pairs - len(positives)]:
            positives.append((docs[a], docs[b]))
        if len(positives) >= max_pos_pairs:
            break

    ids = list(docs.keys())
    n = len(ids)
    if n < 2:
        return positives, []

    max_possible_pairs = n * (n - 1) // 2
    neg_cap = min(max_neg_pairs, max(50, 10 * n), max_possible_pairs)

    # Build a set of positive pairs to avoid overlap
    pos_set = set()
    for a, b in positives:
        pos_set.add(tuple(sorted((a.doc_id, b.doc_id))))

    # Sample neg pairs uniformly without replacement
    negatives: List[Pair] = []
    tried = set()
    attempts = 0
    limit_attempts = neg_cap * 5
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
    docs = fetch_all_docs()
    # auto-reduce caps relative to corpus size
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
    p = _as_np(probs).astype(np.float32)
    y = _as_np(labels).astype(np.float32)
    return _reliability_bins(p, y, n_bins=n_bins)


# Internals

def _as_np(x: Sequence[float]) -> np.ndarray:
    return np.asarray(list(x), dtype=np.float32)


def _fit_quantile_binned(scores: np.ndarray, labels: np.ndarray, n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    if scores.size == 0:
        edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
        probs = np.linspace(0.0, 1.0, n_bins, dtype=np.float32)
        return edges, probs
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(scores, qs, method="linear")).astype(np.float32)
    if edges[0] > 0.0:
        edges = np.insert(edges, 0, 0.0).astype(np.float32)
    if edges[-1] < 1.0:
        edges = np.append(edges, 1.0).astype(np.float32)
    nb = max(1, edges.size - 1)
    idx = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, nb - 1)
    pos = np.bincount(idx, weights=labels, minlength=nb).astype(np.float64)
    cnt = np.bincount(idx, minlength=nb).astype(np.float64)
    probs = (pos + 1.0) / (cnt + 2.0)
    for i in range(1, nb):
        if probs[i] < probs[i - 1]:
            probs[i] = probs[i - 1]
    return edges.astype(np.float32), probs.astype(np.float32)


def _calibrated_prob_binned(score: float, edges: np.ndarray, probs: np.ndarray) -> float:
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


def _fit_platt_reg(scores: np.ndarray, labels: np.ndarray, *, l2: float = 1.0, iters: int = 200, lr: float = 0.1) -> Tuple[float, float]:
    a = 0.0
    b = 0.0
    x = scores.astype(np.float32)
    y = labels.astype(np.float32)
    for _ in range(max(1, iters)):
        z = a * x + b
        p = _sigmoid(z)
        # gradients of log loss and L2
        grad_a = float(np.mean((p - y) * x) + l2 * a)
        grad_b = float(np.mean(p - y) + l2 * b * 0.01)
        a -= lr * grad_a
        b -= lr * grad_b
    return float(a), float(b)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _choose_threshold_by_precision(
    calibrated_probs: np.ndarray,
    labels: np.ndarray,
    target_precision: float,
    *,
    mode: str = "max"
) -> float:
    if calibrated_probs.size == 0:
        return 0.5
    cand = np.linspace(0.0, 1.0, 201, dtype=np.float32)
    y = labels.astype(np.float32)
    ok: List[float] = []
    for th in cand:
        preds = (calibrated_probs >= th)
        tp = float(np.sum((preds == 1) & (y == 1)))
        fp = float(np.sum((preds == 1) & (y == 0)))
        if tp + fp == 0:
            continue
        prec = tp / (tp + fp)
        if prec >= target_precision:
            ok.append(float(th))
    if not ok:
        pos_probs = calibrated_probs[labels == 1]
        if pos_probs.size:
            best = float(min(np.max(pos_probs) - 1e-3, 0.999))
        else:
            best = 0.999
    else:
        best = max(ok) if mode == "max" else min(ok)
    return float(best)


def _reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> List[Dict[str, float]]:
    centers = np.linspace(0.05, 0.95, n_bins, dtype=np.float32)
    rows: List[Dict[str, float]] = []
    for c in centers:
        mask = (probs >= (c - 0.05)) & (probs < (c + 0.05))
        if not np.any(mask):
            rows.append({"prob_center": float(c), "expected_pos_rate": float(c), "observed_pos_rate": 0.0, "count": 0})
            continue
        obs = float(np.mean(labels[mask]))
        rows.append({"prob_center": float(c), "expected_pos_rate": float(c), "observed_pos_rate": obs, "count": int(np.sum(mask))})
    return rows
