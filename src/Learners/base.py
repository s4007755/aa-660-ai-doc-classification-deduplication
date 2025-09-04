# src/learners/base.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Literal, Optional, Protocol, Tuple, TypedDict, runtime_checkable
from datetime import datetime
import math
import json

@dataclass(frozen=True)
class DocumentView:
    doc_id: str
    text: str
    language: Optional[str] = None
    tokens: Optional[List[str]] = None 
    sentences: Optional[List[str]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

# Pair of documents
Pair = Tuple[DocumentView, DocumentView]

def stable_pair_key(a: str, b: str) -> str:
    x, y = (a, b) if a <= b else (b, a)
    return f"{x}||{y}"

class ReliabilityBin(TypedDict):
    prob_center: float
    expected_pos_rate: float
    observed_pos_rate: float
    count: int

# Calibration parameters and chosen threshold for a learner
@dataclass
class CalibrationParams:
    method: Literal["isotonic", "platt", "none"] = "none"
    params: Dict[str, Any] = field(default_factory=dict)
    threshold: float = 0.5
    brier_score: Optional[float] = None
    reliability_bins: List[ReliabilityBin] = field(default_factory=list)

    # Serialize to JSON
    def to_json(self) -> str:
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
        obj = json.loads(s)
        return CalibrationParams(
            method=obj.get("method", "none"),
            params=obj.get("params", {}),
            threshold=float(obj.get("threshold", 0.5)),
            brier_score=obj.get("brier_score"),
            reliability_bins=list(obj.get("reliability_bins", [])),
        )

# Generic per-learner configuration
@dataclass
class LearnerConfig:
    enabled: bool = True
    target_precision: float = 0.98
    min_confidence: Optional[float] = None
    max_pairs_per_epoch: int = 50_000
    random_state: int = 13
    extras: Dict[str, Any] = field(default_factory=dict)

# Persisted state for a learner
@dataclass
class LearnerState:
    version: int = 1
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")
    calibration: CalibrationParams = field(default_factory=CalibrationParams)
    learned_params: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

# Output of scoring a single pair by one learner
@dataclass
class LearnerOutput:
    raw_score: float               # model-native score
    prob: float                    # calibrated probability
    threshold: Optional[float]     # current operating threshold
    rationale: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    internals: Optional[Dict[str, Any]] = None

    # True if learner votes duplicate
    def agrees(self) -> bool:
        if self.threshold is None or math.isnan(self.prob):
            return False
        return self.prob >= float(self.threshold)

# Labeled pair used for calibration/self-training
@dataclass(frozen=True)
class PairLabel:
    a_id: str
    b_id: str
    label: Literal[0, 1]           # 1 = duplicate, 0 = non-duplicate
    weight: float = 1.0

    @property
    def key(self) -> str:
        return stable_pair_key(self.a_id, self.b_id)

# Aggregate stats to allow learners to adapt
@dataclass
class CorpusStats:
    doc_count: int
    avg_doc_len: float
    lang_counts: Dict[str, int] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

# Interface every learner must implement
@runtime_checkable
class ILearner(Protocol):

    @property
    def name(self) -> str: ...

    @property
    def config(self) -> LearnerConfig: ...

    def configure(self, config: LearnerConfig) -> None: ...

    def load_state(self, state: Optional[LearnerState]) -> None: ...

    def get_state(self) -> LearnerState: ...

    def prepare(self, corpus_stats: Optional[CorpusStats] = None) -> None: ...

    def score_pair(self, a: DocumentView, b: DocumentView) -> LearnerOutput: ...

    def batch_score(self, pairs: Iterable[Pair]) -> List[LearnerOutput]: ...

    def fit_calibration(self, positives: Iterable[Pair], negatives: Iterable[Pair]) -> LearnerState: ...

    def self_train(self, pseudo_labels: Iterable[PairLabel]) -> LearnerState: ...

# Create a new learner state with default calibration
def make_fresh_state(notes: str = "") -> LearnerState:
    return LearnerState(
        version=1,
        calibration=CalibrationParams(method="none", params={}, threshold=0.5),
        notes=notes,
    )

# JSON-serialize a LearnerState for DB persistence
def serialize_state(state: LearnerState) -> str:
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
    obj = json.loads(s)
    return LearnerState(
        version=int(obj.get("version", 1)),
        last_updated=obj.get("last_updated") or (datetime.utcnow().isoformat(timespec="seconds") + "Z"),
        calibration=CalibrationParams.from_json(json.dumps(obj.get("calibration", {}))),
        learned_params=obj.get("learned_params", {}),
        notes=obj.get("notes", ""),
    )
