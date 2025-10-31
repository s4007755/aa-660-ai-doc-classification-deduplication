from __future__ import annotations

"""
Embedding-based learner with optional SentenceTransformer backend and a fast
fallback embedder.

Highlights
* Two embedding modes:
  - Real model via sentence-transformers
  - Lightweight deterministic fallback (_cheap_embed) with parallel chunking
• Whitening options (domain mean subtraction / top-PC removal)
• Adaptive calibration support: binning/Platt with threshold selection
• Batch scoring caches doc embeddings and parallelizes pair scoring safely
• Sentence-pair rationale extraction for UI "Decision Traces"

Design choices
- ThreadPoolExecutor is used (not ProcessPool) to avoid pickling heavy models,
  and because NumPy dot-products release the GIL.
- Cosine similarity is mapped from [-1,1] to [0,1] for unified thresholding.
- `prob` is capped just below 1.0 (1 - 1e-9) to avoid displaying exactly 1.0
  except for explicit exact-duplicate short-circuits handled elsewhere.
"""

import re
import os
import concurrent.futures
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# Unified calibration utils
from src.training.calibration import (
    calibrate_adaptive_and_select_threshold,
    apply_binning_or_platt,
)


# Helpers

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

def _cos_to_unit(x: float) -> float:
    # Map cosine from [-1, 1] into [0, 1] = uniform handling for thresholds.
    return float((x + 1.0) * 0.5)

def _cheap_embed(texts: List[str], dim: int = 384) -> np.ndarray:
    """
    Deterministic, allocation friendly embedding fallback.

    Very simple character hashing into a fixed-length vector, then L2-normalize.
    This is not semantic but gives a quick, stable signal.
    """
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    vecs = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        if not t:
            continue
        h = 0
        for j, ch in enumerate(t):
            h = (h * 1315423911 + ord(ch) + j) & 0xFFFFFFFFFFFFFFFF
            idx = h % dim
            vecs[i, idx] += 1.0
        n = np.linalg.norm(vecs[i])
        if n > 0:
            vecs[i] /= n
    return vecs


# Internal state

@dataclass
class _State:
    """
    Internal runtime state.

    edges/probs : calibration binning arrays
    mean/top_pc : whitening vectors aligned to embedding dimension
    platt_a/b   : Platt parameters
    """
    edges: Optional[np.ndarray] = None
    probs: Optional[np.ndarray] = None
    mean: Optional[np.ndarray] = None
    top_pc: Optional[np.ndarray] = None
    platt_a: Optional[float] = None
    platt_b: Optional[float] = None


# Learner

class EmbeddingLearner(ILearner):
    @property
    def name(self) -> str:
        return "embedding"

    def __init__(self, config: Optional[LearnerConfig] = None):
        """
        Construct with optional LearnerConfig.

        Config extras of interest (non-exhaustive):
          * model_name: str | "fallback"
          * batch_size: int (for SentenceTransformer)
          * max_workers: int (parallel embedding/scoring)
          * whiten: bool, remove_top_pc: bool
          * fallback_dim: int (cheap embed size)
          * topk_sentences: int (for rationale)
          * max_sentences_explain: int (cap per-side for rationale)
          * cosine_threshold / decision_threshold / threshold
          * force_threshold: bool (override calibrated threshold if True)
        """
        self._config: LearnerConfig = config or LearnerConfig()
        self._state: LearnerState = make_fresh_state("embedding init")
        self._istate = _State()
        self._model: Optional[Any] = None
        self._using_fallback = False
        self._max_workers: int = int(self._config.extras.get("max_workers", os.cpu_count() or 4))


    # Config/state

    @property
    def config(self) -> LearnerConfig:
        return self._config

    def configure(self, config: LearnerConfig) -> None:
        """
        Update config on the fly. Also re computes internal thresholds derived
        from extras when present.
        """
        self._config = config
        self._max_workers = int(self._config.extras.get("max_workers", os.cpu_count() or 4))
        self._ensure_threshold_from_config()

    def load_state(self, state: Optional[LearnerState]) -> None:
        """
        Load persisted state. Rehydrates calibration and whitening artifacts.
        """
        self._state = state or make_fresh_state("embedding fresh")
        lp = self._state.learned_params or {}

        edges_arr = np.array(lp.get("bin_edges", []), dtype=np.float32)
        probs_arr = np.array(lp.get("bin_probs", []), dtype=np.float32)
        self._istate.edges = edges_arr if edges_arr.size else None
        self._istate.probs = probs_arr if probs_arr.size else None

        self._istate.platt_a = lp.get("platt_a")
        self._istate.platt_b = lp.get("platt_b")

        mean = np.array(lp.get("domain_mean", []), dtype=np.float32)
        self._istate.mean = mean if mean.size else None
        top_pc = np.array(lp.get("domain_top_pc", []), dtype=np.float32)
        self._istate.top_pc = top_pc if top_pc.size else None
        self._ensure_threshold_from_config()   # NEW

    def get_state(self) -> LearnerState:
        return self._state

    def prepare(self, corpus_stats: Optional[CorpusStats] = None) -> None:
        """
        Lazily load the SentenceTransformer model if requested, otherwise
        enable fallback. Called once per run or when needed.
        """
        if self._model is not None or self._using_fallback:
            return
        model_name = str(self._config.extras.get("model_name", "fallback")).strip().lower()

        if model_name in ("fallback", "none", "off"):
            self._model = None
            self._using_fallback = True
            return

        try:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers not available")
            self._model = SentenceTransformer(model_name)
            _ = int(self._config.extras.get("batch_size", 64))
        except Exception:
            # Silent fallback to cheap embedder for robustness.
            self._model = None
            self._using_fallback = True
        self._ensure_threshold_from_config()


    # Scoring

    def score_pair(self, a: DocumentView, b: DocumentView) -> LearnerOutput:
        """
        Score a single pair. Embeds each text once, applies whitening, computes
        cosine to [0,1] score, calibrates to prob and returns a structured
        LearnerOutput with rationale including top sentence matches.
        """
        t1 = a.text or ""
        t2 = b.text or ""

        e1 = self._embed_texts([t1])[0]
        e2 = self._embed_texts([t2])[0]
        e1 = self._apply_whiten(e1)
        e2 = self._apply_whiten(e2)

        n1 = float(np.linalg.norm(e1))
        n2 = float(np.linalg.norm(e2))

        if n1 == 0.0 and n2 == 0.0:
            cos = 0.0
        else:
            cos = float(np.dot(e1, e2))

        score = _cos_to_unit(cos)

        prob = apply_binning_or_platt(
            score,
            self._state.calibration,
            self._istate.edges,
            self._istate.probs,
        )
        th = self._state.calibration.threshold

        top_pairs = self._top_sentence_pairs(a, b, top_k=int(self._config.extras.get("topk_sentences", 3)))
        warnings: List[str] = []
        if (not t1.strip()) and (not t2.strip()):
            warnings.append("Both texts are empty after basic extraction.")
        elif n1 == 0.0 and n2 == 0.0:
            warnings.append("Both embeddings are zero after preprocessing/whitening (no evidence).")

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
            prob=float(min(prob, 1.0 - 1e-9)),
            threshold=th,
            rationale=rationale,
            warnings=warnings,
            internals=None,
        )

    def batch_score(
        self,
        pairs: Iterable[Pair],
        use_parallel: bool = True,
        max_workers: int = os.cpu_count() or 4
    ) -> List[LearnerOutput]:
        """
        Batch score with single-pass embedding cache.

        Steps
        1) Collect unique docs across pairs and embed once.
        2) Apply whitening to cached vectors.
        3) Compute cosine, convert to [0,1], calibrate to prob.
        4) Optionally parallelize pair-computation.
        """
        pairs_list = list(pairs)
        if not pairs_list:
            return []

        # Embed each unique document once
        unique: Dict[str, str] = {}
        for a, b in pairs_list:
            unique.setdefault(a.doc_id, a.text or "")
            unique.setdefault(b.doc_id, b.text or "")
        ids = list(unique.keys())
        texts = [unique[i] for i in ids]
        embs = self._embed_texts(texts).astype(np.float32, copy=True)
        for i in range(embs.shape[0]):
            embs[i] = self._apply_whiten(embs[i])
        emb_map = {ids[i]: embs[i] for i in range(len(ids))}

        th = self._state.calibration.threshold
        outs: List[LearnerOutput] = []

        if not use_parallel or len(pairs_list) < 4:
            # Sequential fast path
            for a, b in pairs_list:
                v1 = emb_map[a.doc_id]
                v2 = emb_map[b.doc_id]
                n1 = float(np.linalg.norm(v1))
                n2 = float(np.linalg.norm(v2))
                cos = 0.0 if (n1 == 0.0 and n2 == 0.0) else float(np.dot(v1, v2))
                score = _cos_to_unit(cos)
                prob = apply_binning_or_platt(score, self._state.calibration, self._istate.edges, self._istate.probs)

                top_pairs = self._top_sentence_pairs(a, b, top_k=int(self._config.extras.get("topk_sentences", 3)))
                warnings: List[str] = []
                if (not (a.text or "").strip()) and (not (b.text or "").strip()):
                    warnings.append("Both texts are empty after basic extraction.")
                elif n1 == 0.0 and n2 == 0.0:
                    warnings.append("Both embeddings are zero after preprocessing/whitening (no evidence).")

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

                outs.append(LearnerOutput(
                    raw_score=float(score),
                    prob=float(min(prob, 1.0 - 1e-9)),
                    threshold=th,
                    rationale=rationale,
                    warnings=warnings,
                    internals=None,
                ))
            return outs

        # Parallel execution
        def _compute_pair(pair: Pair) -> LearnerOutput:
            a, b = pair
            v1 = emb_map[a.doc_id]
            v2 = emb_map[b.doc_id]
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            cos = 0.0 if (n1 == 0.0 and n2 == 0.0) else float(np.dot(v1, v2))
            score = _cos_to_unit(cos)
            prob = apply_binning_or_platt(score, self._state.calibration, self._istate.edges, self._istate.probs)

            top_pairs = self._top_sentence_pairs(a, b, top_k=int(self._config.extras.get("topk_sentences", 3)))
            warnings: List[str] = []
            if (not (a.text or "").strip()) and (not (b.text or "").strip()):
                warnings.append("Both texts are empty after basic extraction.")
            elif n1 == 0.0 and n2 == 0.0:
                warnings.append("Both embeddings are zero after preprocessing/whitening (no evidence).")

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
                prob=float(min(prob, 1.0 - 1e-9)),
                threshold=th,
                rationale=rationale,
                warnings=warnings,
                internals=None,
            )

        # Parallelize (threads avoid pickling issues and NumPy releases GIL on dot-products)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_compute_pair, p): p for p in pairs_list}
            for future in as_completed(futures):
                try:
                    outs.append(future.result())
                except Exception as e:
                    a, b = futures[future]
                    print(f"Error scoring pair {a.doc_id} / {b.doc_id}: {e}")

        return outs


    # Training

    def fit_calibration(self, positives: Iterable[Pair], negatives: Iterable[Pair]) -> LearnerState:
        """
        Fit calibration on bootstrap labels and select an operating threshold.

        - Embeds unique docs once
        - Computes [0,1] scores from cosine
        - Uses `calibrate_adaptive_and_select_threshold` to produce:
            * CalibrationParams
            * Optional Platt parameters
            * Optional binning arrays
        - Persists artifacts in state.learned_params for reuse
        """
        pos_pairs = list(positives)
        neg_pairs = list(negatives)
        if not pos_pairs and not neg_pairs:
            return self._state

        # 1) Embed each unique doc once
        uniq: Dict[str, DocumentView] = {}
        for a, b in pos_pairs:
            uniq.setdefault(a.doc_id, a)
            uniq.setdefault(b.doc_id, b)
        for a, b in neg_pairs:
            uniq.setdefault(a.doc_id, a)
            uniq.setdefault(b.doc_id, b)

        ids = list(uniq.keys())
        texts = [uniq[i].text or "" for i in ids]
        embs = self._embed_texts(texts)
        embs = np.vstack([self._apply_whiten(e) for e in embs])
        emb_map = {ids[i]: embs[i] for i in range(len(ids))}

        def _pair_score(a: DocumentView, b: DocumentView) -> float:
            v1 = emb_map[a.doc_id]; v2 = emb_map[b.doc_id]
            n1 = float(np.linalg.norm(v1)); n2 = float(np.linalg.norm(v2))
            if n1 == 0.0 and n2 == 0.0:
                c = 0.0
            else:
                c = float(np.dot(v1, v2))
            return _cos_to_unit(c)

        # 2) Compute scores using cached embeddings
        pos_scores = np.array([_pair_score(a, b) for a, b in pos_pairs], dtype=np.float32)
        neg_scores = np.array([_pair_score(a, b) for a, b in neg_pairs], dtype=np.float32)
        scores = np.concatenate([pos_scores, neg_scores], axis=0)
        labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)], axis=0)

        # 3) Smart calibration and pick threshold
        cal, platt_params, edges, probs = calibrate_adaptive_and_select_threshold(
            scores, labels,
            target_precision=float(self._config.target_precision),
            n_bins=int(self._config.extras.get("n_bins", 20)),
        )

        # Store
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
            "domain_mean": None if self._istate.mean is None else self._istate.mean.tolist(),
            "domain_top_pc": None if self._istate.top_pc is None else self._istate.top_pc.tolist(),
            "model_name": str(self._config.extras.get("model_name", "fallback")),
            "batch_size": int(self._config.extras.get("batch_size", 64)),
        }
        return self._state

    def self_train(self, pseudo_labels: Iterable[PairLabel]) -> LearnerState:
        # Placeholder
        return self._state


    # Internals

    def _raw_score(self, a: DocumentView, b: DocumentView) -> float:
        """
        Compute uncalibrated [0,1] similarity for a single pair.
        """
        v1 = self._apply_whiten(self._embed_texts([a.text or ""])[0])
        v2 = self._apply_whiten(self._embed_texts([b.text or ""])[0])
        n1 = float(np.linalg.norm(v1)); n2 = float(np.linalg.norm(v2))
        cos = 0.0 if (n1 == 0.0 and n2 == 0.0) else float(np.dot(v1, v2))
        return _cos_to_unit(cos)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts, optionally in parallel for speed."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        if self._model is None and not self._using_fallback:
            self.prepare(None)

        # If using fallback mode, can parallelize _cheap_embed
        if self._using_fallback or self._model is None:
            dim = int(self._config.extras.get("fallback_dim", 384))

            # Split into chunks for workers
            if len(texts) < 100 or self._max_workers <= 1:
                return _cheap_embed(texts, dim=dim)

            chunk_size = max(1, len(texts) // self._max_workers)
            chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]

            results: List[np.ndarray] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = [executor.submit(_cheap_embed, ch, dim) for ch in chunks]
                for f in concurrent.futures.as_completed(futures):
                    results.append(f.result())
            return np.vstack(results)

        # If using SentenceTransformer, parallelize in chunks
        try:
            batch_size = int(self._config.extras.get("batch_size", 64))
            if len(texts) <= batch_size or self._max_workers <= 1:
                arr = self._model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    batch_size=batch_size,
                    show_progress_bar=False,
                ).astype(np.float32)
                return arr if arr.ndim > 1 else arr.reshape(1, -1)

            # Split texts into chunks
            chunk_size = max(1, len(texts) // self._max_workers)
            chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]

            def _encode_chunk(chunk):
                return self._model.encode(
                    chunk,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    batch_size=batch_size,
                    show_progress_bar=False,
                ).astype(np.float32)

            results: List[np.ndarray] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = [executor.submit(_encode_chunk, ch) for ch in chunks]
                for f in concurrent.futures.as_completed(futures):
                    results.append(f.result())
            arr = np.vstack(results)
            return arr

        except Exception:
            # Fall back to cheap embedding if model fails mid-run.
            self._using_fallback = True
            return _cheap_embed(texts, dim=int(self._config.extras.get("fallback_dim", 384)))

    def _apply_whiten(self, v: np.ndarray) -> np.ndarray:
        """
        Apply optional domain whitening and re-normalize:
          - subtract domain mean if available
          - remove projection on top principal component if requested
        """
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

    def _top_sentence_pairs(self, a: DocumentView, b: DocumentView, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Build a small rationale: take up to N sentences from each doc, embed,
        compute cross-similarity and greedily pick top non-overlapping pairs.
        """
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
        sim = ea @ eb.T
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

    def _reliability_bins(self, scores: np.ndarray, labels: np.ndarray, edges: np.ndarray, probs: np.ndarray, n_bins: int = 10):
        """
        Produce coarse reliability points around fixed centers for plotting
        Falls back to zeros when no support.
        """
        centers = np.linspace(0.05, 0.95, n_bins, dtype=np.float32)
        rows: List[Dict[str, Any]] = []
        cal = np.array([apply_binning_or_platt(s, self._state.calibration, edges, probs) for s in scores], dtype=np.float32)
        for c in centers:
            mask = (cal >= (c - 0.05)) & (cal < (c + 0.05))
            if not np.any(mask):
                rows.append({"prob_center": float(c), "expected_pos_rate": float(c), "observed_pos_rate": 0.0, "count": 0})
                continue
            obs = float(np.mean(labels[mask]))
            rows.append({"prob_center": float(c), "expected_pos_rate": float(c), "observed_pos_rate": obs, "count": int(np.sum(mask))})
        return rows

    def _ensure_threshold_from_config(self) -> None:
        """
        Sync a threshold override from config extras into the active calibration.

        Priority (first non-None wins):
          cosine_threshold -> decision_threshold -> threshold -> config.threshold

        Behavior
        - If `force_threshold` is True in extras, we set method='none' and
          adopt the provided threshold unconditionally.
        - Otherwise, we only override "default-looking" calibrations (method='none'
          and threshold element of {None, 0.5, 0.75} and no learned_params present) or
          when current threshold is None.
        """
        ex = self._config.extras or {}
        # Primary: cosine_threshold
        thr = (
            ex.get("cosine_threshold")      # primary for embedding
            or ex.get("decision_threshold") # legacy
            or ex.get("threshold")          # generic
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

        # UI requested a hard override -> always set and stop
        if force:
            cal.method = "none"
            cal.threshold = float(thr)
            return

        # Default-looking states are safe to override
        looks_default = (
            (cal.method == "none")
            and (cal.threshold in (None, 0.5, 0.75))   # 0.75 = default-look
            and not self._state.learned_params
        )
        if looks_default or cal.threshold is None:
            cal.threshold = float(thr)
