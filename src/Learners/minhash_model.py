from __future__ import annotations

import re
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

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

# Unified calibration utilities
from src.training.calibration import (
    calibrate_adaptive_and_select_threshold,
    apply_binning_or_platt,
)

# Tokenization / Jaccard utils

_DEFAULT_SW = {
    "the","a","an","and","or","for","of","to","in","on","at","by","with","from","as",
    "is","are","was","were","be","been","it","this","that","these","those","you","your",
}

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

def _word_shingles(tokens: List[str], k: int) -> List[str]:
    if k <= 1:
        return tokens[:]
    if not tokens or len(tokens) < k:
        return []
    return [" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)]

def _char_shingles(s: str, k: int) -> List[str]:
    if not s or k <= 0 or len(s) < k:
        return []
    return [s[i : i + k] for i in range(len(s) - k + 1)]

def _jaccard_from_sets(a: List[str], b: List[str]) -> float:
    if not a and not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return float(inter) / float(union)

# Internal state

@dataclass
class _State:
    edges: Optional[np.ndarray] = None
    probs: Optional[np.ndarray] = None
    platt_a: Optional[float] = None
    platt_b: Optional[float] = None

# Learner

class MinHashLearner(ILearner):
    @property
    def name(self) -> str:
        return "minhash"

    def __init__(self, config: Optional[LearnerConfig] = None):
        self._config: LearnerConfig = config or LearnerConfig()
        self._state: LearnerState = make_fresh_state("minhash init")
        self._istate = _State(edges=None, probs=None, platt_a=None, platt_b=None)
        self._stopwords = set(_DEFAULT_SW)

    # config/state

    @property
    def config(self) -> LearnerConfig:
        return self._config

    def configure(self, config: LearnerConfig) -> None:
        self._config = config
        extras = config.extras or {}
        sw_extra = set(map(str.lower, extras.get("stopwords", [])))
        self._stopwords = set(_DEFAULT_SW) | sw_extra

    def load_state(self, state: Optional[LearnerState]) -> None:
        self._state = state or make_fresh_state("minhash fresh")
        lp = self._state.learned_params or {}
        edges = np.array(lp.get("bin_edges", []), dtype=np.float32)
        probs = np.array(lp.get("bin_probs", []), dtype=np.float32)
        self._istate.edges = edges if edges.size else None
        self._istate.probs = probs if probs.size else None
        self._istate.platt_a = lp.get("platt_a")
        self._istate.platt_b = lp.get("platt_b")

    def get_state(self) -> LearnerState:
        return self._state

    def prepare(self, corpus_stats: Optional[CorpusStats] = None) -> None:
        pass

    # scoring

    def score_pair(self, a: DocumentView, b: DocumentView) -> LearnerOutput:
        jaccard, sample_shared, overlap_count, universe_size = self._score_and_sample(a, b)
        prob = apply_binning_or_platt(jaccard, self._state.calibration, self._istate.edges, self._istate.probs)
        th = self._state.calibration.threshold

        warns: List[str] = []
        if universe_size == 0:
            warns.append("Both sides produced no shingles after preprocessing")

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
            prob=float(min(prob, 1.0 - 1e-9)),
            threshold=th,
            rationale=rationale,
            warnings=warns,
            internals=None,
        )


    def _score_one(
        self,
        a: DocumentView,
        b: DocumentView,
        edges: Optional[np.ndarray],
        probs: Optional[np.ndarray],
        use_minhash: bool,
        cache_shingles: Dict[str, List[str]],
        cache_minhash: Dict[str, Any],
        num_perm: int,
    ) -> LearnerOutput:
        """Compute score and probability for a single pair."""
        if use_minhash:
            jaccard = self._jaccard_minhash(a, b, cache_shingles, cache_minhash, num_perm)
        else:
            sh_a = self._get_shingles_cached(a, cache_shingles)
            sh_b = self._get_shingles_cached(b, cache_shingles)
            jaccard = _jaccard_from_sets(sh_a, sh_b)

        prob = apply_binning_or_platt(jaccard, self._state.calibration, edges, probs)
        warns: List[str] = []
        if not self._get_shingles_cached(a, cache_shingles) and not self._get_shingles_cached(b, cache_shingles):
            warns.append("Both sides produced no shingles after preprocessing")

        rationale = {
            "jaccard": float(jaccard),
            "threshold_used": None if self._state.calibration.threshold is None else float(self._state.calibration.threshold),
        }

        return LearnerOutput(
            raw_score=float(jaccard),
            prob=float(min(prob, 1.0 - 1e-9)),
            threshold=self._state.calibration.threshold,
            rationale=rationale,
            warnings=warns,
        )


    def batch_score(
        self,
        pairs: Iterable[Pair],
        use_parallel: bool = True,
        max_workers: int = os.cpu_count() or 4
    ) -> List[LearnerOutput]:
        """Compute scores for multiple pairs, optionally in parallel."""
        pairs_list = list(pairs)
        if not pairs_list:
            return []

        edges, probs = self._istate.edges, self._istate.probs
        use_minhash = bool(self._config.extras.get("use_minhash", True) and MinHash is not None)
        num_perm = int(self._config.extras.get("num_perm", 64))
        cache_shingles: Dict[str, List[str]] = {}
        cache_minhash: Dict[str, Any] = {}

        # Sequential fallback
        if not use_parallel or len(pairs_list) < 4:
            return [
                self._score_one(a, b, edges, probs, use_minhash, cache_shingles, cache_minhash, num_perm)
                for a, b in pairs_list
            ]

        # Parallel execution
        outs: List[LearnerOutput] = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._score_one, a, b, edges, probs, use_minhash, cache_shingles, cache_minhash, num_perm): (a, b)
                for a, b in pairs_list
            }
            for future in as_completed(futures):
                try:
                    outs.append(future.result())
                except Exception as e:
                    a, b = futures[future]
                    print(f"[!] Error scoring pair {a.doc_id}/{b.doc_id}: {e}")

        return outs


    # training

    def fit_calibration(self, positives: Iterable[Pair], negatives: Iterable[Pair]) -> LearnerState:
        pos_pairs = list(positives)
        neg_pairs = list(negatives)
        if not pos_pairs and not neg_pairs:
            return self._state
        
        # Precompute unique docs
        all_pairs = pos_pairs + neg_pairs
        unique_docs = {d.doc_id: d for a, b in all_pairs for d in (a, b)}
        shingles_map = {doc_id: self._get_shingles(d) for doc_id, d in unique_docs.items()}

        # Vectorized Jaccard for all pairs
        def jaccard_from_ids(a: DocumentView, b: DocumentView) -> float:
            sa, sb = set(shingles_map[a.doc_id]), set(shingles_map[b.doc_id])
            inter = len(sa & sb)
            union = len(sa | sb)
            return 0.0 if union == 0 else float(inter) / union

        scores = np.array([jaccard_from_ids(a, b) for a, b in all_pairs], dtype=np.float32)

        # Split pos/neg scores
        pos_scores = scores[:len(pos_pairs)]
        neg_scores = scores[len(pos_pairs):]

        scores = np.concatenate([pos_scores, neg_scores], axis=0)
        labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)], axis=0)

        cal, platt_params, edges, probs = calibrate_adaptive_and_select_threshold(
            scores, labels,
            target_precision=float(self._config.target_precision),
            n_bins=int(self._config.extras.get("n_bins", 20)),
        )

        self._state.calibration = cal
        self._istate.edges = edges if edges.size else None
        self._istate.probs = probs if probs.size else None
        self._istate.platt_a = platt_params.get("a")
        self._istate.platt_b = platt_params.get("b")

        self._state.learned_params = {
            "bin_edges": [] if self._istate.edges is None else self._istate.edges.tolist(),
            "bin_probs": [] if self._istate.probs is None else self._istate.probs.tolist(),
            "platt_a": self._istate.platt_a,
            "platt_b": self._istate.platt_b,
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

    def _raw_score(self, a: DocumentView, b: DocumentView) -> float:
        sh_a = self._get_shingles(a)
        sh_b = self._get_shingles(b)
        return _jaccard_from_sets(sh_a, sh_b)

    def _score_and_sample(self, a: DocumentView, b: DocumentView) -> Tuple[float, List[str], int, int]:
        sh_a = self._get_shingles(a)
        sh_b = self._get_shingles(b)
        sa, sb = set(sh_a), set(sh_b)
        inter = sa & sb
        union = sa | sb
        j = 0.0 if not union else float(len(inter)) / float(len(union))
        sample_shared = [t for t, _ in Counter(list(inter)).most_common(5)]
        return j, sample_shared, len(inter), len(union)

    def _get_shingles_cached(self, d: DocumentView, cache: Dict[str, List[str]]) -> List[str]:
        if d.doc_id in cache:
            return cache[d.doc_id]
        sh = self._get_shingles(d)
        cache[d.doc_id] = sh
        return sh

    def _get_shingles(self, d: DocumentView) -> List[str]:
        mode = self._get_tokenizer_mode()
        k = self._get_shingle_size()
        strict = self._normalize_strict()
        min_len = self._get_min_token_len()
        remove_sw = self._remove_stopwords()

        if mode == "char":
            s = (d.text or "").lower()
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
