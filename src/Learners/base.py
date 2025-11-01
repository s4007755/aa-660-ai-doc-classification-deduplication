# src/learners/base.py
from __future__ import annotations

"""
Core learner interfaces and lightweight data containers.

This module defines:
- DocumentView/Pair: the minimal, immutable view over a document used by learners
- CalibrationParams/LearnerState/LearnerConfig: common config/state payloads
- LearnerOutput/PairLabel/CorpusStats: IO and dataset metadata structures
- ILearner: the protocol all learners must implement
- Stable JSON (de)serialization helpers for LearnerState
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Protocol, Tuple, TypedDict, runtime_checkable
from datetime import datetime
import math
import json
from typing import Mapping, Tuple
from types import MappingProxyType


@dataclass(frozen=True)
class DocumentView:
    """
    Immutable, minimal view over a document's text and metadata.

    Attributes
    doc_id : str
        Stable unique identifier for the document (used in keys/joins).
    text : str
        The normalized/plaintext body used by learners.
    language : Optional[str]
        ISO-ish language tag.
    tokens : Optional[List[str]]
        Pre-tokenized form when upstream provides it.
    sentences : Optional[List[str]]
        Pre-sentence-split form.
    meta : Mapping[str, Any]
        Read-only metadata mapping.
    """
    doc_id: str
    text: str
    language: Optional[str] = None
    tokens: Optional[List[str]] = None
    sentences: Optional[List[str]] = None
    meta: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


# Pair of documents fed to learners
Pair = Tuple[DocumentView, DocumentView]


def stable_pair_key(a: str, b: str) -> str:
    """
    Produce an order-invariant key for a pair of document IDs.
    Ensures (a,b) and (b,a) collide to the same key.
    """
    x, y = (a, b) if a <= b else (b, a)
    return f"{x}||{y}"


class ReliabilityBin(TypedDict):
    """
    One reliability bin used in calibration reliability diagrams.

    prob_center: midpoint of the predicted probability bucket
    expected_pos_rate: the model's expected positive rate in this bin
    observed_pos_rate: empirical positive rate measured in this bin
    count: number of samples in the bin
    """
    prob_center: float
    expected_pos_rate: float
    observed_pos_rate: float
    count: int


# Calibration parameters and chosen threshold for a learner
@dataclass
class CalibrationParams:
    """
    Calibration metadata and operating point for a learner.
    """
    method: Literal["isotonic", "platt", "none"] = "none"
    params: Dict[str, Any] = field(default_factory=dict)
    threshold: Optional[float] = None
    brier_score: Optional[float] = None
    reliability_bins: List[ReliabilityBin] = field(default_factory=list)

    # Serialize to JSON
    def to_json(self) -> str:
        """
        Serialize the calibration parameters to a JSON string.
        """
        return json.dumps({
            "method": self.method,
            "params": self.params,
            "threshold": self.threshold,
            "brier_score": self.brier_score,
            "reliability_bins": self.reliability_bins,
        }, ensure_ascii=False)

    # Deserialize from JSON
    @staticmethod
    def from_json(s: str) -> CalibrationParams:
        """
        Parse a CalibrationParams instance from a JSON string.
        """
        obj = json.loads(s)
        t = obj.get("threshold", None)
        thr = None if t is None else float(t)
        return CalibrationParams(
            method=obj.get("method", "none"),
            params=obj.get("params", {}),
            threshold=thr,
            brier_score=obj.get("brier_score"),
            reliability_bins=list(obj.get("reliability_bins", [])),
        )


# Generic per-learner configuration
@dataclass
class LearnerConfig:
    """
    Configuration inputs that can be set by presets/GUI for each learner.

    enabled : bool
        When False, the learner is skipped entirely.
    target_precision : float
        Desired precision target used by calibration/threshold selection.
    min_confidence : Optional[float]
        Hard lower bound in probability space for a positive vote. If set,
        learner will not vote DUPLICATE below this probability even if its
        threshold would otherwise allow it.
    max_pairs_per_epoch : int
        Backpressure for self-training/epoch processing.
    random_state : int
        Seed for any randomized components.
    extras : Dict[str, Any]
        Learner-specific extra options.
    """
    enabled: bool = True
    target_precision: float = 0.98
    min_confidence: Optional[float] = None
    max_pairs_per_epoch: int = 50_000
    random_state: int = 13
    extras: Dict[str, Any] = field(default_factory=dict)


# Persisted state for a learner
@dataclass
class LearnerState:
    """
    Opaque, persisted learner state.

    version : int
        Schema version for forwards/backwards compatibility.
    last_updated : str
        ISO8601 timestamp with 'Z' UTC suffix.
    calibration : CalibrationParams
        The currently active calibration for the learner.
    learned_params : Dict[str, Any]
        Learner-specific trained weights/parameters.
    notes : str
        Freeform note field for audit/debugging.
    """
    version: int = 1
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")
    calibration: CalibrationParams = field(default_factory=CalibrationParams)
    learned_params: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


# Output of scoring a single pair by one learner
@dataclass
class LearnerOutput:
    """
    Result of scoring a (doc_a, doc_b) pair by a learner.

    raw_score : float
        Model-native score (e.g. cosine similarity mapped to [0,1], Jaccard, etc.).
    prob : float
        Calibrated probability of "duplicate" in [0,1]. May be identical to
        raw_score if calibration is disabled.
    threshold : Optional[float]
        Operating threshold in probability space. If None, this learner does
        not cast a positive vote.
    rationale : Dict[str, Any]
        Short structured explanation/debug info.
    warnings : List[str]
        Non-fatal issues encountered while scoring.
    internals : Optional[Dict[str, Any]]
        Optional detailed internals for downstream debugging.
    """
    raw_score: float # model-native score
    prob: float # calibrated probability
    threshold: Optional[float]  # current operating threshold
    rationale: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    internals: Optional[Dict[str, Any]] = None

    # True if learner votes duplicate
    def agrees(self) -> bool:
        """
        Return True iff this learner votes DUPLICATE under its current threshold.

        Defensive behavior:
        - If threshold is None, returns False (no vote).
        - If prob is NaN, returns False (no vote).
        """
        if self.threshold is None or math.isnan(self.prob):
            return False
        return self.prob >= float(self.threshold)


# Labeled pair used for calibration/self-training
@dataclass(frozen=True)
class PairLabel:
    """
    Weak label container for pseudo-/supervised learning.

    a_id, b_id : str
        Document IDs composing the pair.
    label : {0,1}
        1 = duplicate, 0 = non-duplicate.
    weight : float
        Optional importance weight. Downstream learners may
        interpret this as a sample weight.
    """
    a_id: str
    b_id: str
    label: Literal[0, 1] # 1 = duplicate, 0 = non-duplicate
    weight: float = 1.0

    @property
    def key(self) -> str:
        """Order-invariant key to de-duplicate PairLabels."""
        return stable_pair_key(self.a_id, self.b_id)


# Aggregate stats to allow learners to adapt
@dataclass
class CorpusStats:
    """
    High-level corpus summary available to learners during `prepare()`.

    doc_count : int
        Number of documents in the working set.
    avg_doc_len : float
        Mean document length.
    lang_counts : Dict[str, int]
        Language distribution.
    extras : Dict[str, Any]
        Freeform additional stats.
    """
    doc_count: int
    avg_doc_len: float
    lang_counts: Dict[str, int] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


# Interface every learner must implement
@runtime_checkable
class ILearner(Protocol):
    """
    Minimal protocol implemented by all learners.

    Implementations should be pure with respect to their inputs: no global
    state, side effects limited to `load_state/configure` and deterministic
    where `random_state` is honored.
    """

    @property
    def name(self) -> str: ...

    @property
    def config(self) -> LearnerConfig: ...

    def configure(self, config: LearnerConfig) -> None: ...

    def load_state(self, state: Optional[LearnerState]) -> None: ...

    def get_state(self) -> LearnerState: ...

    def prepare(self, corpus_stats: Optional[CorpusStats] = None) -> None: ...

    def score_pair(self, a: DocumentView, b: DocumentView) -> LearnerOutput: ...

    def batch_score(
            self,
            pairs: Iterable[Pair],
            *args: Any,
            **kwargs: Any,
        ) -> List[LearnerOutput]: ...
    
    def fit_calibration(self, positives: Iterable[Pair], negatives: Iterable[Pair]) -> LearnerState: ...

    def self_train(self, pseudo_labels: Iterable[PairLabel]) -> LearnerState: ...


# Create a new learner state with default calibration
def make_fresh_state(notes: str = "") -> LearnerState:
    """
    Factory: produce a clean LearnerState with no calibration/threshold.

    Useful for initializing brand-new learners or resetting state between runs.
    """
    return LearnerState(
        version=1,
        calibration=CalibrationParams(method="none", params={}, threshold=None),
        notes=notes,
    )


# JSON-serialize a LearnerState for DB persistence
def serialize_state(state: LearnerState) -> str:
    """
    Convert a LearnerState to a JSON string suitable for DB storage.
    """
    payload = {
        "version": state.version,
        "last_updated": state.last_updated,
        "calibration": json.loads(state.calibration.to_json()),
        "learned_params": state.learned_params,
        "notes": state.notes,
    }
    return json.dumps(payload, ensure_ascii=False)


# Deserialize a LearnerState from JSON
def deserialize_state(s: str) -> LearnerState:
    """
    Parse a LearnerState from its JSON representation.
    """
    obj = json.loads(s)
    return LearnerState(
        version=int(obj.get("version", 1)),
        last_updated=obj.get("last_updated") or (datetime.utcnow().isoformat(timespec="seconds") + "Z"),
        calibration=CalibrationParams.from_json(json.dumps(obj.get("calibration", {}))),
        learned_params=obj.get("learned_params", {}),
        notes=obj.get("notes", ""),
    )
