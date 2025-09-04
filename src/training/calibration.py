# src/training/calibration.py
from __future__ import annotations

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

# Fit a simple isotonic-like binned calibrator and pick a threshold to hit target precision
def calibrate_binning_and_select_threshold(
    raw_scores: Sequence[float],
    labels: Sequence[int],
    *,
    target_precision: float = 0.98,
    n_bins: int = 20,
) -> Tuple[CalibrationParams, np.ndarray, np.ndarray]:
    scores = _as_np(raw_scores)
    y = _as_np(labels).astype(np.float32)
    edges, probs = _fit_binned_calibration(scores, y, n_bins=n_bins)
    cal = np.array([_calibrated_prob_binned(s, edges, probs) for s in scores], dtype=np.float32)
    th = _choose_threshold_by_precision(cal, y, target_precision)
    brier = float(np.mean((cal - y) ** 2))
    bins = _reliability_bins(cal, y, n_bins=10)
    params = CalibrationParams(method="isotonic", params={}, threshold=float(th), brier_score=brier, reliability_bins=bins)
    return params, edges, probs

# Fit Platt scaling and pick a threshold to hit target precision
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
    a, b = _fit_platt(scores, y, lr=lr, iters=iters)
    cal = _sigmoid(a * scores + b)
    th = _choose_threshold_by_precision(cal, y, target_precision)
    brier = float(np.mean((cal - y) ** 2))
    bins = _reliability_bins(cal, y, n_bins=10)
    params = CalibrationParams(method="platt", params={"a": float(a), "b": float(b)}, threshold=float(th), brier_score=brier, reliability_bins=bins)
    return params, (a, b)

# Apply a previously fit calibrator (binning)
def apply_binning(raw_score: float, edges: np.ndarray, probs: np.ndarray) -> float:
    return float(_calibrated_prob_binned(float(raw_score), edges, probs))

# Apply a previously fit Platt scaler
def apply_platt(raw_score: float, a: float, b: float) -> float:
    return float(_sigmoid(a * float(raw_score) + b))

# Build bootstrap positives/negatives from exact duplicates in memory
def build_bootstrap_from_exact_duplicates(
    docs: Dict[str, DocumentView],
    *,
    max_pos_pairs: int = 50_000,
    max_neg_pairs: int = 50_000,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Pair], List[Pair]]:
    rng = rng or random.Random(13)
    by_text: Dict[str, List[str]] = {}
    for did, dv in docs.items():
        key = (dv.text or "").strip()
        if not key:
            continue
        by_text.setdefault(key, []).append(did)

    positives: List[Pair] = []
    for ids in by_text.values():
        if len(ids) < 2:
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                positives.append((docs[ids[i]], docs[ids[j]]))
                if len(positives) >= max_pos_pairs:
                    break
            if len(positives) >= max_pos_pairs:
                break
        if len(positives) >= max_pos_pairs:
            break

    ids = list(docs.keys())
    n = len(ids)
    negatives: List[Pair] = []
    attempts = 0
    cap_attempts = max_neg_pairs * 20
    while len(negatives) < max_neg_pairs and attempts < cap_attempts and n >= 2:
        a = ids[rng.randrange(n)]
        b = ids[rng.randrange(n)]
        if a == b:
            attempts += 1
            continue
        if _cheap_far_apart(docs[a].text, docs[b].text):
            negatives.append((docs[a], docs[b]))
        attempts += 1

    return positives, negatives

# Build bootstrap sets using callables to fetch doc views
def build_bootstrap_via_fetchers(
    fetch_all_docs: Callable[[], Dict[str, DocumentView]],
    *,
    max_pos_pairs: int = 50_000,
    max_neg_pairs: int = 50_000,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Pair], List[Pair]]:
    docs = fetch_all_docs()
    return build_bootstrap_from_exact_duplicates(
        docs,
        max_pos_pairs=max_pos_pairs,
        max_neg_pairs=max_neg_pairs,
        rng=rng,
    )

# Reliability bins for GUI/report using calibrated probabilities
def reliability_bins_from_probs(probs: Sequence[float], labels: Sequence[int], n_bins: int = 10):
    p = _as_np(probs).astype(np.float32)
    y = _as_np(labels).astype(np.float32)
    return _reliability_bins(p, y, n_bins=n_bins)

# Internals
def _as_np(x: Sequence[float]) -> np.ndarray:
    return np.asarray(list(x), dtype=np.float32)

# Binned calibration
def _fit_binned_calibration(scores: np.ndarray, labels: np.ndarray, n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    if scores.size == 0:
        edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
        probs = np.linspace(0.0, 1.0, n_bins, dtype=np.float32)
        return edges, probs
    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
    idx = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, n_bins - 1)
    pos = np.bincount(idx, weights=labels, minlength=n_bins).astype(np.float64)
    cnt = np.bincount(idx, minlength=n_bins).astype(np.float64)
    probs = (pos + 1.0) / (cnt + 2.0)  # Laplace smoothing
    for i in range(1, n_bins):
        if probs[i] < probs[i - 1]:
            probs[i] = probs[i - 1]
    return edges, probs.astype(np.float32)

def _calibrated_prob_binned(score: float, edges: np.ndarray, probs: np.ndarray) -> float:
    if edges.size == 0:
        return float(max(0.0, min(1.0, score)))
    n_bins = probs.shape[0]
    i = int(np.clip(np.searchsorted(edges, score, side="right") - 1, 0, n_bins - 1))
    left = edges[i]; right = edges[i + 1]
    p = probs[i]
    if right > left:
        t = (score - left) / (right - left)
        p_next = probs[min(i + 1, n_bins - 1)]
        return float((1 - t) * p + t * p_next)
    return float(p)

# Platt scaling via simple gradient steps
def _fit_platt(scores: np.ndarray, labels: np.ndarray, lr: float = 0.1, iters: int = 200) -> Tuple[float, float]:
    a = 0.0
    b = 0.0
    x = scores.astype(np.float32)
    y = labels.astype(np.float32)
    for _ in range(max(1, iters)):
        z = a * x + b
        p = _sigmoid(z)
        # gradients of log loss
        grad_a = float(np.mean((p - y) * x))
        grad_b = float(np.mean(p - y))
        a -= lr * grad_a
        b -= lr * grad_b
    return float(a), float(b)

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

# Choose smallest threshold meeting target precision
def _choose_threshold_by_precision(calibrated_probs: np.ndarray, labels: np.ndarray, target_precision: float) -> float:
    if calibrated_probs.size == 0:
        return 0.5
    cand = np.linspace(0.0, 1.0, 201, dtype=np.float32)
    best = 1.0
    y = labels.astype(np.float32)
    for th in cand:
        preds = (calibrated_probs >= th)
        tp = float(np.sum((preds == 1) & (y == 1)))
        fp = float(np.sum((preds == 1) & (y == 0)))
        if tp + fp == 0:
            continue
        prec = tp / (tp + fp)
        if prec >= target_precision:
            best = min(best, float(th))
    return float(best)

# Compute reliability table (expected vs observed) over probabilities
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

# Cheap negative screen: very small token overlap and length ratio outside range
def _cheap_far_apart(text_a: str, text_b: str) -> bool:
    ta = set((text_a or "").lower().split())
    tb = set((text_b or "").lower().split())
    if not ta or not tb:
        return True
    inter = len(ta & tb)
    uni = len(ta | tb)
    j = 0.0 if uni == 0 else inter / uni
    if j > 0.02:
        return False
    la = max(1, len((text_a or "").split()))
    lb = max(1, len((text_b or "").split()))
    lr = la / lb if la >= lb else lb / la
    return lr >= 2.5
