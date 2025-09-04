# src/learners/embed_model.py
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

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

# simple sentence splitter
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# unitize cosine
def _cos_to_unit(x: float) -> float:
    return float((x + 1.0) * 0.5)

# binned calibration (isotonic-like)
def _fit_binned_calibration(scores: np.ndarray, labels: np.ndarray, n_bins: int = 20):
    if scores.size == 0:
        edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
        probs = np.linspace(0.0, 1.0, n_bins, dtype=np.float32)
        return edges, probs
    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
    idx = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, n_bins - 1)
    bin_pos = np.bincount(idx, weights=labels, minlength=n_bins).astype(np.float64)
    bin_cnt = np.bincount(idx, minlength=n_bins).astype(np.float64)
    probs = (bin_pos + 1.0) / (bin_cnt + 2.0)
    for i in range(1, n_bins):
        if probs[i] < probs[i - 1]:
            probs[i] = probs[i - 1]
    return edges, probs.astype(np.float32)

# apply binned calibration
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

# threshold by target precision
def _choose_threshold(scores: np.ndarray, labels: np.ndarray, edges: np.ndarray, probs: np.ndarray, target_precision: float) -> float:
    if scores.size == 0:
        return 0.5
    cand = np.linspace(0.0, 1.0, 201, dtype=np.float32)
    cal = np.array([_calibrated_prob(s, edges, probs) for s in scores], dtype=np.float32)
    best = 1.0
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

# cheap hashing embed fallback
def _cheap_embed(texts: List[str], dim: int = 384) -> np.ndarray:
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    vecs = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        h = 0
        for j, ch in enumerate(t):
            h = (h * 1315423911 + ord(ch) + j) & 0xFFFFFFFFFFFFFFFF
            idx = h % dim
            vecs[i, idx] += 1.0
        n = np.linalg.norm(vecs[i])
        if n > 0:
            vecs[i] /= n
    return vecs

@dataclass
class _State:
    edges: Optional[np.ndarray] = None
    probs: Optional[np.ndarray] = None
    mean: Optional[np.ndarray] = None
    top_pc: Optional[np.ndarray] = None

class EmbeddingLearner(ILearner):
    # name
    @property
    def name(self) -> str:
        return "embedding"

    # init
    def __init__(self, config: Optional[LearnerConfig] = None):
        self._config: LearnerConfig = config or LearnerConfig()
        self._state: LearnerState = make_fresh_state("embedding init")
        self._istate = _State()
        self._model: Optional[Any] = None
        self._using_fallback = False

    # config
    @property
    def config(self) -> LearnerConfig:
        return self._config

    def configure(self, config: LearnerConfig) -> None:
        self._config = config

    # state
    def load_state(self, state: Optional[LearnerState]) -> None:
        self._state = state or make_fresh_state("embedding fresh")
        lp = self._state.learned_params or {}
        self._istate.edges = np.array(lp.get("bin_edges", []), dtype=np.float32) or None
        self._istate.probs = np.array(lp.get("bin_probs", []), dtype=np.float32) or None
        mean = np.array(lp.get("domain_mean", []), dtype=np.float32)
        self._istate.mean = mean if mean.size else None
        top_pc = np.array(lp.get("domain_top_pc", []), dtype=np.float32)
        self._istate.top_pc = top_pc if top_pc.size else None

    def get_state(self) -> LearnerState:
        return self._state

    # prepare
    def prepare(self, corpus_stats: Optional[CorpusStats] = None) -> None:
        if self._model is not None or self._using_fallback:
            return
        model_name = str(self._config.extras.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
        try:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers not available")
            self._model = SentenceTransformer(model_name)
            _ = int(self._config.extras.get("batch_size", 64))
        except Exception:
            self._model = None
            self._using_fallback = True

    # score single pair
    def score_pair(self, a: DocumentView, b: DocumentView) -> LearnerOutput:
        e1 = self._embed_texts([a.text])[0]
        e2 = self._embed_texts([b.text])[0]
        e1 = self._apply_whiten(e1)
        e2 = self._apply_whiten(e2)
        cos = float(np.dot(e1, e2))
        score = _cos_to_unit(cos)

        edges, probs = self._istate.edges, self._istate.probs
        prob = score if edges is None or probs is None else _calibrated_prob(score, edges, probs)
        th = self._state.calibration.threshold

        top_pairs = self._top_sentence_pairs(a, b, top_k=int(self._config.extras.get("topk_sentences", 3)))
        rationale = {
            "cosine": float(cos),
            "top_matching_sentences": top_pairs,
            "threshold_used": None if th is None else float(th),
            "whitening": {
                "enabled": bool(self._config.extras.get("whiten", False)),
                "remove_top_pc": bool(self._config.extras.get("remove_top_pc", False)),
                "has_domain_mean": bool(self._istate.mean is not None),
                "has_top_pc": bool(self._istate.top_pc is not None),
            },
            "model_fallback": bool(self._using_fallback),
        }
        return LearnerOutput(
            raw_score=float(score),
            prob=float(prob),
            threshold=th,
            rationale=rationale,
            warnings=[],
            internals=None,
        )

    # batch score
    def batch_score(self, pairs: Iterable[Pair]) -> List[LearnerOutput]:
        pairs_list = list(pairs)
        if not pairs_list:
            return []
        # embed all unique docs once
        unique: Dict[str, str] = {}
        for a, b in pairs_list:
            unique.setdefault(a.doc_id, a.text)
            unique.setdefault(b.doc_id, b.text)
        ids = list(unique.keys())
        texts = [unique[i] for i in ids]
        embs = self._embed_texts(texts)
        embs = np.vstack([self._apply_whiten(e) for e in embs])
        emb_map = {ids[i]: embs[i] for i in range(len(ids))}

        edges, probs = self._istate.edges, self._istate.probs
        th = self._state.calibration.threshold

        outs: List[LearnerOutput] = []
        for a, b in pairs_list:
            e1 = emb_map[a.doc_id]
            e2 = emb_map[b.doc_id]
            cos = float(np.dot(e1, e2))
            score = _cos_to_unit(cos)
            prob = score if edges is None or probs is None else _calibrated_prob(score, edges, probs)
            rationale = {"cosine": float(cos), "threshold_used": None if th is None else float(th), "model_fallback": bool(self._using_fallback)}
            outs.append(LearnerOutput(raw_score=float(score), prob=float(prob), threshold=th, rationale=rationale))
        return outs

    # fit calibration from bootstrap
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
            "domain_mean": None if self._istate.mean is None else self._istate.mean.tolist(),
            "domain_top_pc": None if self._istate.top_pc is None else self._istate.top_pc.tolist(),
            "model_name": str(self._config.extras.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")),
            "batch_size": int(self._config.extras.get("batch_size", 64)),
        }
        return self._state
    def self_train(self, pseudo_labels: Iterable[PairLabel]) -> LearnerState:
        return self._state

    # internals

    # raw score in [0,1] from cosine
    def _raw_score(self, a: DocumentView, b: DocumentView) -> float:
        e1 = self._apply_whiten(self._embed_texts([a.text])[0])
        e2 = self._apply_whiten(self._embed_texts([b.text])[0])
        cos = float(np.dot(e1, e2))
        return _cos_to_unit(cos)

    # embed many texts with model or fallback
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        if self._model is None and not self._using_fallback:
            self.prepare(None)
        if self._model is None:
            return _cheap_embed(texts, dim=int(self._config.extras.get("fallback_dim", 384)))
        try:
            arr = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=int(self._config.extras.get("batch_size", 64)),
                show_progress_bar=False,
            ).astype(np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr
        except Exception:
            self._using_fallback = True
            return _cheap_embed(texts, dim=int(self._config.extras.get("fallback_dim", 384)))

    def _apply_whiten(self, v: np.ndarray) -> np.ndarray:
        v = v.astype(np.float32, copy=True)
        if bool(self._config.extras.get("whiten", False)):
            if self._istate.mean is not None and self._istate.mean.shape == v.shape:
                v = v - self._istate.mean
            if bool(self._config.extras.get("remove_top_pc", False)) and self._istate.top_pc is not None and self._istate.top_pc.shape == v.shape:
                pc = self._istate.top_pc
                v = v - float(np.dot(v, pc)) * pc
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        return v

    # top-K matching sentence pairs for rationale
    def _top_sentence_pairs(self, a: DocumentView, b: DocumentView, top_k: int = 3) -> List[Dict[str, Any]]:
        sa = a.sentences if a.sentences else [s.strip() for s in _SENT_SPLIT.split(a.text or "") if s.strip()]
        sb = b.sentences if b.sentences else [s.strip() for s in _SENT_SPLIT.split(b.text or "") if s.strip()]
        if not sa or not sb:
            return []

        max_sents = int(self._config.extras.get("max_sentences_explain", 20))
        sa = sa[:max_sents]; sb = sb[:max_sents]

        ea = self._embed_texts(sa)
        eb = self._embed_texts(sb)
        ea = np.vstack([self._apply_whiten(x) for x in ea])
        eb = np.vstack([self._apply_whiten(x) for x in eb])

        sim = ea @ eb.T  # cosine since vectors are normalized
        pairs: List[Tuple[int, int, float]] = []
        for i in range(sim.shape[0]):
            for j in range(sim.shape[1]):
                pairs.append((i, j, float(sim[i, j])))
        pairs.sort(key=lambda x: -x[2])
        out: List[Dict[str, Any]] = []
        used_a: set = set()
        used_b: set = set()
        for i, j, c in pairs:
            if i in used_a or j in used_b:
                continue
            out.append({"a": sa[i], "b": sb[j], "cosine": float(c)})
            used_a.add(i); used_b.add(j)
            if len(out) >= top_k:
                break
        return out

    # reliability bins for report/GUI
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
