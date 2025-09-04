# src/learners/minhash_model.py
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from datasketch import MinHash
except Exception:
    MinHash = None

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

# Tokenize words
def _tokenize_words(s: str, *, strict: bool, min_len: int, stopwords: set[str], remove_stopwords: bool) -> List[str]:
    if not s:
        return []
    s = s.lower()
    if strict:
        s = re.sub(r"[^\w\s]", " ", s)
    toks = s.split()
    out: List[str] = []
    for t in toks:
        if len(t) < min_len:
            continue
        if remove_stopwords and t in stopwords:
            continue
        out.append(t)
    return out

# Word n-gram shingles
def _word_shingles(tokens: List[str], k: int) -> List[str]:
    if k <= 1:
        return tokens[:]
    if not tokens or len(tokens) < k:
        return []
    return [" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)]

# Char n-gram shingles
def _char_shingles(s: str, k: int) -> List[str]:
    if not s or k <= 0 or len(s) < k:
        return []
    return [s[i : i + k] for i in range(len(s) - k + 1)]

# Exact Jaccard on two shingle lists
def _jaccard_from_sets(a: List[str], b: List[str]) -> float:
    if not a and not b:
        return 1.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return float(inter) / float(union)

# Simple binned calibration
def _fit_binned_calibration(scores: np.ndarray, labels: np.ndarray, n_bins: int = 20):
    if scores.size == 0:
        edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
        probs = np.linspace(0.0, 1.0, n_bins, dtype=np.float32)
        return edges, probs
    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
    idx = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, n_bins - 1)
    bin_pos = np.bincount(idx, weights=labels, minlength=n_bins).astype(np.float64)
    bin_cnt = np.bincount(idx, minlength=n_bins).astype(np.float64)
    probs = (bin_pos + 1.0) / (bin_cnt + 2.0)  # Laplace smoothing
    for i in range(1, n_bins):
        if probs[i] < probs[i - 1]:
            probs[i] = probs[i - 1]
    return edges, probs.astype(np.float32)

# Apply binned calibration
def _calibrated_prob(score: float, edges: np.ndarray, probs: np.ndarray) -> float:
    if not (0.0 <= score <= 1.0) or edges.size == 0:
        return max(0.0, min(1.0, score))
    n_bins = probs.shape[0]
    i = int(np.clip(np.searchsorted(edges, score, side="right") - 1, 0, n_bins - 1))
    left = edges[i]; right = edges[i + 1]
    p = probs[i]
    if right > left:
        t = (score - left) / (right - left)
        p_next = probs[min(i + 1, n_bins - 1)]
        return float((1 - t) * p + t * p_next)
    return float(p)

# Threshold by target precision
def _choose_threshold(scores: np.ndarray, labels: np.ndarray, edges: np.ndarray, probs: np.ndarray, target_precision: float) -> float:
    if scores.size == 0:
        return 0.5
    cand = np.linspace(0.0, 1.0, 201, dtype=np.float32)
    best = 1.0
    cal = np.array([_calibrated_prob(s, edges, probs) for s in scores], dtype=np.float32)
    for th in cand:
        preds = (cal >= th)
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
    edges: Optional[np.ndarray] = None
    probs: Optional[np.ndarray] = None

class MinHashLearner(ILearner):
    @property
    def name(self) -> str:
        return "minhash"

    # Init with default config and fresh state
    def __init__(self, config: Optional[LearnerConfig] = None):
        self._config: LearnerConfig = config or LearnerConfig()
        self._state: LearnerState = make_fresh_state("minhash init")
        self._istate = _State(edges=None, probs=None)
        self._stopwords = set(_DEFAULT_SW)

    # Active configuration
    @property
    def config(self) -> LearnerConfig:
        return self._config

    # Apply configuration and propagate extras
    def configure(self, config: LearnerConfig) -> None:
        self._config = config
        extras = config.extras or {}
        sw_extra = set(map(str.lower, extras.get("stopwords", [])))
        self._stopwords = set(_DEFAULT_SW) | sw_extra

    # Load persisted learner state
    def load_state(self, state: Optional[LearnerState]) -> None:
        self._state = state or make_fresh_state("minhash fresh")
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
        jaccard, sample_shared, overlap_count, universe_size = self._score_and_sample(a, b)
        edges, probs = self._istate.edges, self._istate.probs
        prob = float(jaccard) if edges is None or probs is None else _calibrated_prob(jaccard, edges, probs)
        th = self._state.calibration.threshold

        rationale = {
            "jaccard": float(jaccard),
            "shared_top_shingles": sample_shared,
            "overlap_count": int(overlap_count),
            "universe_size": int(universe_size),
            "shingle_size": int(self._get_shingle_size()),
            "tokenizer_mode": str(self._get_tokenizer_mode()),
            "threshold_used": None if th is None else float(th),
        }
        return LearnerOutput(
            raw_score=float(jaccard),
            prob=float(prob),
            threshold=th,
            rationale=rationale,
            warnings=[],
            internals=None,
        )

    # Batch scoring with caches
    def batch_score(self, pairs: Iterable[Pair]) -> List[LearnerOutput]:
        edges, probs = self._istate.edges, self._istate.probs
        th = self._state.calibration.threshold

        cache_shingles: Dict[str, List[str]] = {}
        cache_minhash: Dict[str, Any] = {}

        use_minhash = bool(self._config.extras.get("use_minhash", True) and MinHash is not None)
        num_perm = int(self._config.extras.get("num_perm", 64))

        outs: List[LearnerOutput] = []
        for a, b in pairs:
            if use_minhash:
                ja = self._jaccard_minhash(a, b, cache_shingles, cache_minhash, num_perm)
                jaccard = ja
            else:
                sh_a = self._get_shingles_cached(a, cache_shingles)
                sh_b = self._get_shingles_cached(b, cache_shingles)
                jaccard = _jaccard_from_sets(sh_a, sh_b)

            prob = float(jaccard) if edges is None or probs is None else _calibrated_prob(jaccard, edges, probs)
            rationale = {
                "jaccard": float(jaccard),
                "threshold_used": None if th is None else float(th),
            }
            outs.append(LearnerOutput(raw_score=float(jaccard), prob=float(prob), threshold=th, rationale=rationale))
        return outs

    # Fit calibration on bootstrap and choose threshold for target precision
    def fit_calibration(self, positives: Iterable[Pair], negatives: Iterable[Pair]) -> LearnerState:
        pos_pairs = list(positives)
        neg_pairs = list(negatives)
        if not pos_pairs and not neg_pairs:
            return self._state

        pos_scores = np.array([self._raw_score(a, b) for a, b in pos_pairs], dtype=np.float32)
        neg_scores = np.array([self._raw_score(a, b) for a, b in neg_pairs], dtype=np.float32)
        scores = np.concatenate([pos_scores, neg_scores], axis=0)
        labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)], axis=0)

        edges, probs = _fit_binned_calibration(scores, labels, n_bins=int(self._config.extras.get("n_bins", 20)))
        th = _choose_threshold(scores, labels, edges, probs, target_precision=float(self._config.target_precision))

        cal_probs = np.array([_calibrated_prob(s, edges, probs) for s in scores], dtype=np.float32)
        brier = float(np.mean((cal_probs - labels) ** 2))

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
            "shingle_size": int(self._get_shingle_size()),
            "tokenizer_mode": str(self._get_tokenizer_mode()),
            "use_minhash": bool(self._config.extras.get("use_minhash", True) and MinHash is not None),
            "num_perm": int(self._config.extras.get("num_perm", 64)),
        }
        return self._state

    def self_train(self, pseudo_labels: Iterable[PairLabel]) -> LearnerState:
        return self._state

    # internals

    def _get_tokenizer_mode(self) -> str:
        return str(self._config.extras.get("tokenizer_mode", "word"))

    def _get_shingle_size(self) -> int:
        return int(self._config.extras.get("shingle_size", 3))

    def _get_min_token_len(self) -> int:
        return int(self._config.extras.get("min_token_len", 2))

    def _remove_stopwords(self) -> bool:
        return bool(self._config.extras.get("remove_stopwords", True))

    def _normalize_strict(self) -> bool:
        return bool(self._config.extras.get("normalize_strict", False))

    def _strip_dates_ids(self) -> bool:
        return bool(self._config.extras.get("strip_dates_ids", False))

    # Compute raw Jaccard on shingles
    def _raw_score(self, a: DocumentView, b: DocumentView) -> float:
        sh_a = self._get_shingles(a)
        sh_b = self._get_shingles(b)
        return _jaccard_from_sets(sh_a, sh_b)

    # Score and pick a few shared shingles for rationale
    def _score_and_sample(self, a: DocumentView, b: DocumentView) -> Tuple[float, List[str], int, int]:
        sh_a = self._get_shingles(a)
        sh_b = self._get_shingles(b)
        sa, sb = set(sh_a), set(sh_b)
        inter = sa & sb
        union = sa | sb
        j = 0.0 if not union else float(len(inter)) / float(len(union))
        sample_shared = [t for t, _ in Counter(list(inter)).most_common(5)]
        return j, sample_shared, len(inter), len(union)

    # Get shingles with caching dict
    def _get_shingles_cached(self, d: DocumentView, cache: Dict[str, List[str]]) -> List[str]:
        if d.doc_id in cache:
            return cache[d.doc_id]
        sh = self._get_shingles(d)
        cache[d.doc_id] = sh
        return sh

    # Build shingles from document view based on mode/size
    def _get_shingles(self, d: DocumentView) -> List[str]:
        mode = self._get_tokenizer_mode()
        k = self._get_shingle_size()
        strict = self._normalize_strict()
        min_len = self._get_min_token_len()
        remove_sw = self._remove_stopwords()

        if mode == "char":
            s = d.text.lower() if d.text else ""
            if strict:
                s = re.sub(r"\s+", " ", s)
                s = re.sub(r"[^\w\s]", " ", s)
            if self._strip_dates_ids():
                s = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", s)
                s = re.sub(r"\b\d{6,}\b", " ", s)
            return _char_shingles(s, k)

        toks = d.tokens if d.tokens is not None else _tokenize_words(
            d.text or "", strict=strict, min_len=min_len, stopwords=self._stopwords, remove_stopwords=remove_sw
        )
        if self._strip_dates_ids():
            toks = [t for t in toks if not re.fullmatch(r"\d{6,}", t) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", t)]
        return _word_shingles(toks, k)

    # MinHash-based Jaccard estimate using datasketch
    def _jaccard_minhash(
        self,
        a: DocumentView,
        b: DocumentView,
        cache_shingles: Dict[str, List[str]],
        cache_minhash: Dict[str, Any],
        num_perm: int,
    ) -> float:
        if MinHash is None:
            sh_a = self._get_shingles_cached(a, cache_shingles)
            sh_b = self._get_shingles_cached(b, cache_shingles)
            return _jaccard_from_sets(sh_a, sh_b)

        def get_mh(doc: DocumentView) -> Any:
            if doc.doc_id in cache_minhash:
                return cache_minhash[doc.doc_id]
            sh = self._get_shingles_cached(doc, cache_shingles)
            mh = MinHash(num_perm=num_perm)
            for s in sh:
                mh.update(s.encode("utf-8", errors="ignore"))
            cache_minhash[doc.doc_id] = mh
            return mh

        mh1 = get_mh(a)
        mh2 = get_mh(b)
        try:
            return float(mh1.jaccard(mh2))
        except Exception:
            sh_a = self._get_shingles_cached(a, cache_shingles)
            sh_b = self._get_shingles_cached(b, cache_shingles)
            return _jaccard_from_sets(sh_a, sh_b)

    # Build a coarse reliability table for GUI/report
    def _reliability_bins(self, scores: np.ndarray, labels: np.ndarray, edges: np.ndarray, probs: np.ndarray, n_bins: int = 10):
        centers = np.linspace(0.05, 0.95, n_bins, dtype=np.float32)
        rows: List[Dict[str, Any]] = []
        cal = np.array([_calibrated_prob(s, edges, probs) for s in scores], dtype=np.float32)
        for c in centers:
            mask = (cal >= (c - 0.05)) & (cal < (c + 0.05))
            if not np.any(mask):
                rows.append({"prob_center": float(c), "expected_pos_rate": float(c), "observed_pos_rate": 0.0, "count": 0})
                continue
            obs = float(np.mean(labels[mask]))
            rows.append({"prob_center": float(c), "expected_pos_rate": float(c), "observed_pos_rate": obs, "count": int(np.sum(mask))})
        return rows
