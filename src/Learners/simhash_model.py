# src/learners/simhash_model.py
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
try:
    from simhash import Simhash
except Exception:
    Simhash = None

from src.learners.base import (
    DocumentView,
    Pair,
    ILearner,
    LearnerOutput,
    LearnerConfig,
    LearnerState,
    PairLabel,
    CorpusStats,
    CalibrationParams,
    make_fresh_state,
)

# Small stopword seed
_DEFAULT_SW = {
    "the","a","an","and","or","for","of","to","in","on","at","by","with","from","as",
    "is","are","was","were","be","been","it","this","that","these","those","you","your",
}

# Helper: safe bitcount-based Hamming distance
def _hamming64(a: int, b: int) -> int:
    return int((a ^ b).bit_count())

# Helper: tokenize with optional strict cleanup
def _tokenize(s: str, min_len: int, stopwords: set[str], strict: bool, strip_ids: bool) -> List[str]:
    if not s:
        return []
    s = s.lower()
    if strict:
        s = re.sub(r"[^\w\s]", " ", s)  # strip punctuation
    toks = s.split()
    out: List[str] = []
    for t in toks:
        if strip_ids and (re.fullmatch(r"\d{2,}", t) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", t)):
            continue
        if len(t) < min_len:
            continue
        if t in stopwords:
            continue
        out.append(t)
    return out

# Helper: build a 64-bit simhash value from tokens->weights
def _simhash_from_tokens(tokens: List[str], max_weight: int) -> int:
    if not tokens:
        return 0
    if Simhash is not None:
        counts = Counter(tokens)
        feats = [(t, min(c, max_weight)) for t, c in counts.items()]
        return int(Simhash(feats, f=64).value)
    v = [0] * 64
    for t in tokens:
        h = hash(t) & ((1 << 64) - 1)
        for i in range(64):
            v[i] += 1 if ((h >> i) & 1) else -1
    x = 0
    for i in range(64):
        if v[i] >= 0:
            x |= (1 << i)
    return int(x)

# Helper: piecewise isotonic-like calibration via binning
def _fit_binned_calibration(scores: np.ndarray, labels: np.ndarray, n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    if scores.size == 0:
        edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
        probs = np.linspace(0.0, 1.0, n_bins, dtype=np.float32)
        return edges, probs
    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
    idx = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, n_bins - 1)
    bin_pos = np.bincount(idx, weights=labels, minlength=n_bins).astype(np.float64)
    bin_cnt = np.bincount(idx, minlength=n_bins).astype(np.float64)
    # Laplace smoothing
    probs = (bin_pos + 1.0) / (bin_cnt + 2.0)
    # enforce monotonicity
    for i in range(1, n_bins):
        if probs[i] < probs[i - 1]:
            probs[i] = probs[i - 1]
    return edges, probs.astype(np.float32)

# Helper: apply binned calibration
def _calibrated_prob(score: float, edges: np.ndarray, probs: np.ndarray) -> float:
    if not (0.0 <= score <= 1.0) or edges.size == 0:
        return max(0.0, min(1.0, score))
    n_bins = probs.shape[0]
    i = int(np.clip(np.searchsorted(edges, score, side="right") - 1, 0, n_bins - 1))
    # interpolate within bin using bin center and neighbors
    left = edges[i]
    right = edges[i + 1]
    p = probs[i]
    if right > left:
        t = (score - left) / (right - left)
        p_next = probs[min(i + 1, n_bins - 1)]
        return float((1 - t) * p + t * p_next)
    return float(p)

# Helper: choose threshold to reach target precision on a validation set
def _choose_threshold(scores: np.ndarray, labels: np.ndarray, edges: np.ndarray, probs: np.ndarray, target_precision: float) -> float:
    if scores.size == 0:
        return 0.5
    cand = np.linspace(0.0, 1.0, 201, dtype=np.float32)
    best = 1.0
    for th in cand:
        preds = (np.array([_calibrated_prob(s, edges, probs) for s in scores]) >= th)
        tp = float(np.sum((preds == 1) & (labels == 1)))
        fp = float(np.sum((preds == 1) & (labels == 0)))
        if tp + fp == 0:
            continue
        prec = tp / (tp + fp)
        if prec >= target_precision:
            best = min(best, float(th))
    return float(best)

@dataclass
class _State:
    # compact in-memory state
    edges: Optional[np.ndarray] = None
    probs: Optional[np.ndarray] = None

class SimHashLearner(ILearner):
    @property
    def name(self) -> str:
        return "simhash"

    # Init with default config and fresh state
    def __init__(self, config: Optional[LearnerConfig] = None):
        self._config: LearnerConfig = config or LearnerConfig()
        self._state: LearnerState = make_fresh_state("simhash init")
        self._istate = _State(edges=None, probs=None)
        self._stopwords = set(_DEFAULT_SW)

    # Active configuration
    @property
    def config(self) -> LearnerConfig:
        return self._config

    # Apply configuration
    def configure(self, config: LearnerConfig) -> None:
        self._config = config
        extras = config.extras or {}
        sw_extra = set(map(str.lower, extras.get("stopwords", [])))
        self._stopwords = set(_DEFAULT_SW) | sw_extra

    # Load persisted learner state
    def load_state(self, state: Optional[LearnerState]) -> None:
        self._state = state or make_fresh_state("simhash fresh")
        lp = self._state.learned_params or {}
        edges = np.array(lp.get("bin_edges", []), dtype=np.float32)
        probs = np.array(lp.get("bin_probs", []), dtype=np.float32)
        self._istate.edges = edges if edges.size else None
        self._istate.probs = probs if probs.size else None

    # Return a snapshot of current state
    def get_state(self) -> LearnerState:
        return self._state

    def prepare(self, corpus_stats: Optional[CorpusStats] = None) -> None:
        pass

    # Score a pair and return calibrated probability and rationale
    def score_pair(self, a: DocumentView, b: DocumentView) -> LearnerOutput:
        strict = bool(self._config.extras.get("normalize_strict", False))
        strip_ids = bool(self._config.extras.get("strip_dates_ids", False))
        min_len = int(self._config.extras.get("min_token_len", 2))
        max_w = int(self._config.extras.get("max_token_weight", 255))

        tok_a = _tokenize(a.text, min_len, self._stopwords, strict, strip_ids)
        tok_b = _tokenize(b.text, min_len, self._stopwords, strict, strip_ids)

        h1 = _simhash_from_tokens(tok_a, max_w)
        h2 = _simhash_from_tokens(tok_b, max_w)
        hd = _hamming64(h1, h2)
        sim = max(0.0, 1.0 - hd / 64.0)

        edges, probs = self._istate.edges, self._istate.probs
        prob = float(sim) if edges is None or probs is None else _calibrated_prob(sim, edges, probs)
        th = self._state.calibration.threshold

        # simple overlap rationale
        ov = list((Counter(tok_a) & Counter(tok_b)).elements())
        top_shared = [t for t, _ in Counter(ov).most_common(5)]

        rationale = {
            "hamming_distance": int(hd),
            "similarity_est": float(sim),
            "top_stable_tokens": top_shared,
            "threshold_used": None if th is None else float(th),
        }
        return LearnerOutput(
            raw_score=float(sim),
            prob=float(prob),
            threshold=th,
            rationale=rationale,
            warnings=[],
            internals=None,
        )

    # Batch scoring
    def batch_score(self, pairs: Iterable[Pair]) -> List[LearnerOutput]:
        strict = bool(self._config.extras.get("normalize_strict", False))
        strip_ids = bool(self._config.extras.get("strip_dates_ids", False))
        min_len = int(self._config.extras.get("min_token_len", 2))
        max_w = int(self._config.extras.get("max_token_weight", 255))

        edges, probs = self._istate.edges, self._istate.probs
        th = self._state.calibration.threshold

        cache_tokens: Dict[str, List[str]] = {}
        cache_hash: Dict[str, int] = {}

        def get_hash(doc: DocumentView) -> int:
            if doc.doc_id in cache_hash:
                return cache_hash[doc.doc_id]
            toks = cache_tokens.get(doc.doc_id)
            if toks is None:
                toks = _tokenize(doc.text, min_len, self._stopwords, strict, strip_ids)
                cache_tokens[doc.doc_id] = toks
            h = _simhash_from_tokens(toks, max_w)
            cache_hash[doc.doc_id] = h
            return h

        outs: List[LearnerOutput] = []
        for a, b in pairs:
            h1 = get_hash(a)
            h2 = get_hash(b)
            hd = _hamming64(h1, h2)
            sim = max(0.0, 1.0 - hd / 64.0)
            prob = float(sim) if edges is None or probs is None else _calibrated_prob(sim, edges, probs)

            rationale = {
                "hamming_distance": int(hd),
                "similarity_est": float(sim),
                "threshold_used": None if th is None else float(th),
            }
            outs.append(LearnerOutput(raw_score=float(sim), prob=float(prob), threshold=th, rationale=rationale))
        return outs

    # Fit/refresh calibration on bootstrap pairs and set threshold for target precision
    def fit_calibration(self, positives: Iterable[Pair], negatives: Iterable[Pair]) -> LearnerState:
        pos_pairs = list(positives)
        neg_pairs = list(negatives)
        if not pos_pairs and not neg_pairs:
            return self._state

        # score bootstrap pairs
        pos_scores = np.array([self._raw_score(a, b) for a, b in pos_pairs], dtype=np.float32)
        neg_scores = np.array([self._raw_score(a, b) for a, b in neg_pairs], dtype=np.float32)
        scores = np.concatenate([pos_scores, neg_scores], axis=0)
        labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)], axis=0)

        edges, probs = _fit_binned_calibration(scores, labels, n_bins=int(self._config.extras.get("n_bins", 20)))
        th = _choose_threshold(scores, labels, edges, probs, target_precision=float(self._config.target_precision))

        # compute a simple Brier score on bootstrap
        cal_probs = np.array([_calibrated_prob(s, edges, probs) for s in scores], dtype=np.float32)
        brier = float(np.mean((cal_probs - labels) ** 2))

        # save to state
        self._istate.edges = edges
        self._istate.probs = probs

        self._state.calibration = CalibrationParams(
            method="isotonic",
            params={},
            threshold=float(th),
            brier_score=brier,
            reliability_bins=self._reliability_bins(scores, labels, edges, probs),
        )
        self._state.learned_params = {
            "bin_edges": edges.tolist(),
            "bin_probs": probs.tolist(),
            "stopwords": sorted(self._stopwords),
            "min_token_len": int(self._config.extras.get("min_token_len", 2)),
            "max_token_weight": int(self._config.extras.get("max_token_weight", 255)),
        }
        return self._state

    def self_train(self, pseudo_labels: Iterable[PairLabel]) -> LearnerState:
        return self._state

    # Compute raw similarity in [0,1] for a pair
    def _raw_score(self, a: DocumentView, b: DocumentView) -> float:
        strict = bool(self._config.extras.get("normalize_strict", False))
        strip_ids = bool(self._config.extras.get("strip_dates_ids", False))
        min_len = int(self._config.extras.get("min_token_len", 2))
        max_w = int(self._config.extras.get("max_token_weight", 255))
        tok_a = _tokenize(a.text, min_len, self._stopwords, strict, strip_ids)
        tok_b = _tokenize(b.text, min_len, self._stopwords, strict, strip_ids)
        h1 = _simhash_from_tokens(tok_a, max_w)
        h2 = _simhash_from_tokens(tok_b, max_w)
        hd = _hamming64(h1, h2)
        return max(0.0, 1.0 - hd / 64.0)

    # Build coarse reliability table for GUI/report
    def _reliability_bins(self, scores: np.ndarray, labels: np.ndarray, edges: np.ndarray, probs: np.ndarray, n_bins: int = 10):
        centers = np.linspace(0.05, 0.95, n_bins, dtype=np.float32)
        rows: List[Dict[str, Any]] = []
        for c in centers:
            cal = np.array([_calibrated_prob(s, edges, probs) for s in scores], dtype=np.float32)
            mask = (cal >= (c - 0.05)) & (cal < (c + 0.05))
            if not np.any(mask):
                rows.append({"prob_center": float(c), "expected_pos_rate": float(c), "observed_pos_rate": 0.0, "count": 0})
                continue
            obs = float(np.mean(labels[mask]))
            rows.append({"prob_center": float(c), "expected_pos_rate": float(c), "observed_pos_rate": obs, "count": int(np.sum(mask))})
        return rows
