from __future__ import annotations

"""
SimHash-based near-duplicate learner.

Overview
* Tokenization -> (optional) word shingles / char n-grams -> SimHash.
* Similarity ~= 1 - HammingDistance(bits)/bits.
* Adaptive calibration (binning/Platt) to convert similarity to probability.
"""

import re
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    """Hamming distance between two bitstrings."""
    mask = (1 << bits) - 1
    return int(((a ^ b) & mask).bit_count())

def _tokenize(s: str, min_len: int, stopwords: set[str], strict: bool, strip_ids: bool) -> List[str]:
    """
    Simple whitespace tokenizer with optional punctuation stripping and
    removal of long numeric IDs or YYYY-MM-DD dates.
    """
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
    """k-gram word shingles, uses a visible separator to preserve boundaries."""
    if k <= 1:
        return tokens
    joiner = "\u241F"  # Symbol for unit separator
    return [joiner.join(tokens[i : i + k]) for i in range(0, len(tokens) - k + 1)]

def _char_ngrams(s: str, n: int, strict: bool) -> List[str]:
    """Overlapping character n-grams with optional punctuation stripping."""
    s = s.lower()
    if strict:
        s = re.sub(r"[^\w\s]", " ", s)
    if n <= 1:
        return list(s)
    if len(s) < n:
        return []
    return [s[i : i + n] for i in range(0, len(s) - n + 1)]

def _simhash_from_tokens(tokens: List[str], max_weight: int, bits: int) -> int:
    """
    Compute SimHash from tokens with capped token weights.
    Uses simhash.Simhash if available, otherwise falls back to a deterministic
    manual implementation using blake2b for stable hashing.
    """
    if not tokens:
        return 0
    if Simhash is not None:
        counts = Counter(tokens)
        feats = [(t, min(c, max_weight)) for t, c in counts.items()]
        return int(Simhash(feats, f=bits).value)

    # Fallback: manual simhash
    v = np.zeros(bits, dtype=np.int64)
    mask = (1 << bits) - 1
    import hashlib  # stable per-token hash
    for t, c in Counter(tokens).items():
        h = int(hashlib.blake2b(t.encode("utf-8"), digest_size=16).hexdigest(), 16) & mask
        w = min(int(c), max_weight)
        for i in range(bits):
            v[i] += w if ((h >> i) & 1) else -w
    x = 0
    for i in range(bits):
        if v[i] >= 0:
            x |= (1 << i)
    return x


# Internal state

@dataclass
class _State:
    """Transient calibration artifacts stored outside LearnerState."""
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
        """
        Construct SimHash learner with default state.
        Stopwords can be extended via config.extras["stopwords"].
        """
        self._config: LearnerConfig = config or LearnerConfig()
        self._state: LearnerState = make_fresh_state("simhash init")
        self._istate = _State()
        self._stopwords = set(_DEFAULT_SW)


    # Config/state

    @property
    def config(self) -> LearnerConfig:
        return self._config

    def configure(self, config: LearnerConfig) -> None:
        """Update config, refresh stopwords and threshold from extras."""
        self._config = config
        extras = config.extras or {}
        sw_extra = set(map(str.lower, extras.get("stopwords", [])))
        self._stopwords = set(_DEFAULT_SW) | sw_extra
        self._ensure_threshold_from_config()

    def load_state(self, state: Optional[LearnerState]) -> None:
        """
        Load persisted calibration/bins/Platt parameters, also apply any
        threshold override from current config extras.
        """
        self._state = state or make_fresh_state("simhash fresh")
        lp = self._state.learned_params or {}
        edges = np.array(lp.get("bin_edges", []), dtype=np.float32)
        probs = np.array(lp.get("bin_probs", []), dtype=np.float32)
        self._istate.edges = edges if edges.size else None
        self._istate.probs = probs if probs.size else None
        self._istate.platt_a = lp.get("platt_a")
        self._istate.platt_b = lp.get("platt_b")
        self._ensure_threshold_from_config()

        # Restore stopwords if persisted
        sw = lp.get("stopwords")
        if isinstance(sw, list):
            self._stopwords = set(map(str.lower, sw))

    def get_state(self) -> LearnerState:
        return self._state

    def prepare(self, corpus_stats: Optional[CorpusStats] = None) -> None:
        """No heavy model to load, ensure threshold reflects config extras."""
        self._ensure_threshold_from_config()
        pass


    # Hashing

    def _hash(self, dv: DocumentView) -> int:
        """
        Tokenize and featureize according to mode, then produce a SimHash.
        Modes:
          - "unigram": raw token unigrams (default)
          - "wshingle": word shingles of size k
          - "cngram": character n-grams
        """
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


    # Scoring

    def score_pair(self, a: DocumentView, b: DocumentView) -> LearnerOutput:
        """
        Score a single pair:
        1) SimHash each side, compute similarity from Hamming distance.
        2) Calibrate similarity to probability via learned bins/Platt.
        3) Provide brief rationale.
        """
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
        cal = self._state.calibration or CalibrationParams(
            method="none", params={}, threshold=None, brier_score=None, reliability_bins=[]
        )
        prob = apply_binning_or_platt(sim, cal, self._istate.edges, self._istate.probs)
        th = cal.threshold

        extras = self._config.extras or {}
        # Top overlapping tokens for quick interpretability
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
        """
        Score multiple pairs with optional parallelization.
        """
        pairs_list = list(pairs)
        if not pairs_list:
            return []

        extras = self._config.extras or {}
        bits = int(extras.get("hash_bits", 128))
        edges, probs = self._istate.edges, self._istate.probs

        # Sequential fallback for few pairs or disabled parallel
        if not use_parallel or len(pairs_list) < 4:
            return [self._score_one(a, b, edges, probs, bits) for a, b in pairs_list]

        # Parallel execution
        outs: List[LearnerOutput] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
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


    # Training

    def fit_calibration(self, positives: Iterable[Pair], negatives: Iterable[Pair]) -> LearnerState:
        """
        Fit adaptive calibration and select an operating threshold.

        Steps
        1) Compute raw SimHash similarity for labeled positives/negatives.
        2) Calibrate similarity -> probability (bins/Platt) against labels.
        3) Persist learned artifacts and tokenization settings.
        """
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

        # Store calibration, params and tokenization knobs for reproducibility
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
        """Placeholder"""
        return self._state
    
    def _ensure_threshold_from_config(self) -> None:
        """
        Sync a threshold override from config extras into the active calibration.

        Priority (first non-None wins):
          decision_threshold -> threshold -> cosine_threshold (legacy) -> config.threshold

        Behaviour
        - If `force_threshold` is True, set method='none' and adopt the provided
          threshold unconditionally.
        - Otherwise, override only default-looking calibrations (method='none'
          and threshold element of {None, 0.5, 0.75} and no learned_params) or when the
          current threshold is None.
        """
        ex = self._config.extras or {}
        # Primary: decision_threshold (simhash)
        thr = (
            ex.get("decision_threshold")    # primary for simhash
            or ex.get("threshold")
            or ex.get("cosine_threshold")
            or getattr(self._config, "threshold", None)
        )
        if thr is None:
            return

        force = bool(ex.get("force_threshold", False))

        cal = self._state.calibration
        if cal is None:
            self._state.calibration = CalibrationParams(
                method="none", params={}, threshold=float(thr),
                brier_score=None, reliability_bins=[]
            )
            return

        if force:
            cal.method = "none"
            cal.threshold = float(thr)
            return

        looks_default = (
            (cal.method == "none")
            and (cal.threshold in (None, 0.5, 0.75))
            and not self._state.learned_params
        )
        if looks_default or cal.threshold is None:
            cal.threshold = float(thr)
