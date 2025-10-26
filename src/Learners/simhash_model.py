# src/learners/simhash_model.py
from __future__ import annotations

import re
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

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

# Unified calibration utilities
from src.training.calibration import (
    calibrate_adaptive_and_select_threshold,
    apply_binning_or_platt,
)

# Tokenization / SimHash utils

_DEFAULT_SW: set[str] = set()

def _hamming(a: int, b: int, bits: int) -> int:
    mask = (1 << bits) - 1
    return int(((a ^ b) & mask).bit_count())

def _tokenize(s: str, min_len: int, stopwords: set[str], strict: bool, strip_ids: bool) -> List[str]:
    if not s:
        return []
    s = s.lower()
    if strict:
        s = re.sub(r"[^\w\s]", " ", s)
    toks = s.split()
    out: List[str] = []
    for t in toks:
        if strip_ids and (re.fullmatch(r"\d{6,}", t) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", t)):
            continue
        if len(t) < min_len:
            continue
        if t in stopwords:
            continue
        out.append(t)
    return out

def _word_shingles(tokens: List[str], k: int) -> List[str]:
    if k <= 1:
        return tokens
    joiner = "\u241F"  # Symbol for unit separator
    return [joiner.join(tokens[i : i + k]) for i in range(0, len(tokens) - k + 1)]

def _char_ngrams(s: str, n: int, strict: bool) -> List[str]:
    s = s.lower()
    if strict:
        s = re.sub(r"[^\w\s]", " ", s)
    if n <= 1:
        return list(s)
    if len(s) < n:
        return []
    return [s[i : i + n] for i in range(0, len(s) - n + 1)]

def _simhash_from_tokens(tokens: List[str], max_weight: int, bits: int) -> int:
    if not tokens:
        return 0
    if Simhash is not None:
        counts = Counter(tokens)
        feats = [(t, min(c, max_weight)) for t, c in counts.items()]
        return int(Simhash(feats, f=bits).value)
    """v = [0] * bits
    mask = (1 << bits) - 1
    for t in tokens:
        h = hash(t) & mask
        for i in range(bits):
            v[i] += 1 if ((h >> i) & 1) else -1
    x = 0
    for i in range(bits):
        if v[i] >= 0:
            x |= (1 << i)
    """
    v = np.zeros(bits, dtype=int)
    hashes = np.array([hash(t) & ((1 << bits) - 1) for t in tokens], dtype=np.uint64)
    for i in range(bits):
        v[i] = np.sum((hashes >> i) & 1) * 2 - len(tokens)  # +1/-1 logic
    x = np.packbits(v >= 0)[0]  # simplified conversion back to int (needs care for >64 bits)

        
    return int(x)

# Internal state

@dataclass
class _State:
    edges: Optional[np.ndarray] = None
    probs: Optional[np.ndarray] = None
    platt_a: Optional[float] = None
    platt_b: Optional[float] = None

# Learner

class SimHashLearner(ILearner):
    @property
    def name(self) -> str:
        return "simhash"

    def __init__(self, config: Optional[LearnerConfig] = None):
        self._config: LearnerConfig = config or LearnerConfig()
        self._state: LearnerState = make_fresh_state("simhash init")
        self._istate = _State()
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
        self._state = state or make_fresh_state("simhash fresh")
        lp = self._state.learned_params or {}
        edges = np.array(lp.get("bin_edges", []), dtype=np.float32)
        probs = np.array(lp.get("bin_probs", []), dtype=np.float32)
        self._istate.edges = edges if edges.size else None
        self._istate.probs = probs if probs.size else None
        self._istate.platt_a = lp.get("platt_a")
        self._istate.platt_b = lp.get("platt_b")

        # Restore stopwords if persisted
        sw = lp.get("stopwords")
        if isinstance(sw, list):
            self._stopwords = set(map(str.lower, sw))

    def get_state(self) -> LearnerState:
        return self._state

    def prepare(self, corpus_stats: Optional[CorpusStats] = None) -> None:
        pass

    # hashing

    def _hash(self, dv: DocumentView) -> int:
        extras = self._config.extras or {}
        strict = bool(extras.get("normalize_strict", False))
        strip_ids = bool(extras.get("strip_dates_ids", False))
        min_len = int(extras.get("min_token_len", 2))
        max_w = int(extras.get("max_token_weight", 255))
        bits = int(extras.get("hash_bits", 128))

        # base tokens with order preserved
        base_tokens = _tokenize(dv.text or "", min_len, self._stopwords, strict, strip_ids)

        # optional light positional signal (bucket index appended)
        pos_bucket = int(extras.get("pos_bucket", 0) or 0)
        if pos_bucket > 0:
            base_tokens = [f"{tok}@{i//pos_bucket}" for i, tok in enumerate(base_tokens)]

        mode = (extras.get("simhash_mode") or "unigram").lower()

        if mode == "wshingle":
            k = int(extras.get("shingle_size", 3))
            feats = _word_shingles(base_tokens, k)
            return _simhash_from_tokens(feats, max_w, bits)

        if mode == "cngram":
            n = int(extras.get("char_ngram", 5))
            feats = _char_ngrams(dv.text or "", n, strict=False)
            return _simhash_from_tokens(feats, max_w, bits)

        # default: unigram tokens
        return _simhash_from_tokens(base_tokens, max_w, bits)

    # scoring

    def score_pair(self, a: DocumentView, b: DocumentView) -> LearnerOutput:
        extras = self._config.extras or {}
        bits = int(extras.get("hash_bits", 128))

        h1 = self._hash(a)
        h2 = self._hash(b)
        hd = _hamming(h1, h2, bits)
        sim = max(0.0, 1.0 - hd / float(bits))

        prob = apply_binning_or_platt(
            sim,
            self._state.calibration or CalibrationParams(method="none", params={}, threshold=None, brier_score=None, reliability_bins=[]),
            self._istate.edges,
            self._istate.probs,
        )
        th = self._state.calibration.threshold

        # Show top overlapping tokens
        ov = list((Counter(_tokenize(a.text or "", 2, self._stopwords, False, False)) &
                   Counter(_tokenize(b.text or "", 2, self._stopwords, False, False))).elements())
        top_shared = [t for t, _ in Counter(ov).most_common(5)]

        rationale = {
            "mode": (extras.get("simhash_mode") or "unigram"),
            "hash_bits": bits,
            "hamming_distance": int(hd),
            "similarity_est": float(sim),
            "top_stable_tokens": top_shared,
            "threshold_used": None if th is None else float(th),
        }
        return LearnerOutput(
            raw_score=float(sim),
            prob=float(min(prob, 1.0 - 1e-9)),
            threshold=th,
            rationale=rationale,
            warnings=[],
            internals=None,
        )

    # inside SimHashLearner class

    def _score_one(
        self,
        a: DocumentView,
        b: DocumentView,
        edges: Optional[np.ndarray],
        probs: Optional[np.ndarray],
        bits: int
    ) -> LearnerOutput:
        """Compute similarity and calibrated probability for a single pair."""
        h1 = self._hash(a)
        h2 = self._hash(b)
        hd = _hamming(h1, h2, bits)
        sim = max(0.0, 1.0 - hd / float(bits))
        prob = apply_binning_or_platt(sim, self._state.calibration, edges, probs)
        th = self._state.calibration.threshold

        extras = self._config.extras or {}
        # Show top overlapping tokens
        ov = list((Counter(_tokenize(a.text or "", 2, self._stopwords, False, False)) &
                Counter(_tokenize(b.text or "", 2, self._stopwords, False, False))).elements())
        top_shared = [t for t, _ in Counter(ov).most_common(5)]

        rationale = {
            "mode": (extras.get("simhash_mode") or "unigram"),
            "hash_bits": bits,
            "hamming_distance": int(hd),
            "similarity_est": float(sim),
            "top_stable_tokens": top_shared,
            "threshold_used": None if th is None else float(th),
        }

        return LearnerOutput(
            raw_score=float(sim),
            prob=float(min(prob, 1.0 - 1e-9)),
            threshold=th,
            rationale=rationale,
            warnings=[],
            internals=None,
        )


    def batch_score(
        self,
        pairs: Iterable[Pair],
        use_parallel: bool = True,
        max_workers: int = os.cpu_count() or 4
    ) -> List[LearnerOutput]:
        pairs_list = list(pairs)
        if not pairs_list:
            return []

        extras = self._config.extras or {}
        bits = int(extras.get("hash_bits", 128))
        edges, probs = self._istate.edges, self._istate.probs

        # Sequential fallback for few pairs or disabled parallel
        if not use_parallel or len(pairs_list) < 4:
            return [self._score_one(a, b, edges, probs, bits) for a, b in pairs_list]

        # Parallel execution (CPU-bound => ProcessPoolExecutor)
        outs: List[LearnerOutput] = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._score_one, a, b, edges, probs, bits): (a, b)
                for a, b in pairs_list
            }
            for future in as_completed(futures):
                try:
                    outs.append(future.result())
                except Exception as e:
                    a, b = futures[future]
                    print(f"[!] Error scoring pair {a.doc_id} / {b.doc_id}: {e}")

        return outs

    # training

    def fit_calibration(self, positives: Iterable[Pair], negatives: Iterable[Pair]) -> LearnerState:
        pos_pairs = list(positives); neg_pairs = list(negatives)
        if not pos_pairs and not neg_pairs:
            return self._state

        extras = self._config.extras or {}
        bits = int(extras.get("hash_bits", 128))

        def _raw(a: DocumentView, b: DocumentView) -> float:
            h1 = self._hash(a)
            h2 = self._hash(b)
            hd = _hamming(h1, h2, bits)
            return max(0.0, 1.0 - hd / float(bits))

        pos_scores = np.array([_raw(a, b) for a, b in pos_pairs], dtype=np.float32)
        neg_scores = np.array([_raw(a, b) for a, b in neg_pairs], dtype=np.float32)
        scores = np.concatenate([pos_scores, neg_scores], axis=0)
        labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)], axis=0)

        cal, platt_params, edges, probs = calibrate_adaptive_and_select_threshold(
            scores, labels,
            target_precision=float(self._config.target_precision),
            n_bins=int(extras.get("n_bins", 20)),
        )

        # Store calibration + params + tokenization knobs for reproducibility
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
            "stopwords": sorted(self._stopwords),
            "min_token_len": int(extras.get("min_token_len", 2)),
            "max_token_weight": int(extras.get("max_token_weight", 255)),
            "hash_bits": bits,
            "simhash_mode": (extras.get("simhash_mode") or "unigram"),
            "shingle_size": int(extras.get("shingle_size", 3)),
            "char_ngram": int(extras.get("char_ngram", 5)),
            "pos_bucket": int(extras.get("pos_bucket", 0) or 0),
        }
        return self._state


    def self_train(self, pseudo_labels: Iterable[PairLabel]) -> LearnerState:
        return self._state
