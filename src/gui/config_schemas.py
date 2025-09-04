# src/gui/config_schemas.py
from __future__ import annotations

from dataclasses import dataclass, asdict, replace, fields, is_dataclass
from typing import Any, Dict, Optional, List

import json

try:
    import yaml
except Exception:
    yaml = None

from src.learners.base import LearnerConfig
from src.ensemble.arbiter import ArbiterConfig
from src.pipelines.near_duplicate import (
    PipelineConfig,
    CandidateConfig,
    BootstrapConfig,
    SelfLearningConfig,
)

# helpers: dataclass reconstruction
def _dc_from_dict(dc_type, data: Dict[str, Any]):
    if not is_dataclass(dc_type):
        return data
    kwargs = {}
    for f in fields(dc_type):
        if f.name not in data:
            continue
        val = data[f.name]
        if is_dataclass(f.type) and isinstance(val, dict):
            kwargs[f.name] = _dc_from_dict(f.type, val)
        else:
            kwargs[f.name] = val
    return dc_type(**kwargs)

def _deep_asdict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _deep_asdict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [ _deep_asdict(x) for x in obj ]
    if isinstance(obj, dict):
        return {k: _deep_asdict(v) for k, v in obj.items()}
    return obj

# profiles
PROFILE_BALANCED = "Balanced"
PROFILE_HIGH_PREC = "High Precision"
PROFILE_RECALL = "Recall-Heavy"
PROFILE_CUSTOM = "Custom"

def make_profile(name: str) -> PipelineConfig:
    n = name.strip().lower()
    if n in ("balanced",):
        return PipelineConfig(
            simhash=LearnerConfig(enabled=True, target_precision=0.98),
            minhash=LearnerConfig(enabled=True, target_precision=0.98),
            embedding=LearnerConfig(enabled=True, target_precision=0.98),
            arbiter=ArbiterConfig(require_agreement=2, gray_zone_margin=0.05, max_self_train_epochs=2),
            candidates=CandidateConfig(use_lsh=True, shingle_size=3, num_perm=64, lsh_threshold=0.6),
            bootstrap=BootstrapConfig(max_pos_pairs=50_000, max_neg_pairs=50_000),
            self_learning=SelfLearningConfig(enabled=True, epochs=2),
            persist=True,
        )
    if n in ("high precision", "high-precision", "precision"):
        return PipelineConfig(
            simhash=LearnerConfig(enabled=True, target_precision=0.995, min_confidence=0.60),
            minhash=LearnerConfig(enabled=True, target_precision=0.995, min_confidence=0.60),
            embedding=LearnerConfig(enabled=True, target_precision=0.995, min_confidence=0.60),
            arbiter=ArbiterConfig(require_agreement=2, gray_zone_margin=0.04, max_self_train_epochs=2),
            candidates=CandidateConfig(use_lsh=True, shingle_size=3, num_perm=64, lsh_threshold=0.65),
            bootstrap=BootstrapConfig(max_pos_pairs=60_000, max_neg_pairs=80_000),
            self_learning=SelfLearningConfig(enabled=True, epochs=1),
            persist=True,
        )
    if n in ("recall-heavy", "recall", "high recall"):
        return PipelineConfig(
            simhash=LearnerConfig(enabled=True, target_precision=0.95),
            minhash=LearnerConfig(enabled=True, target_precision=0.95),
            embedding=LearnerConfig(enabled=True, target_precision=0.95),
            arbiter=ArbiterConfig(require_agreement=2, gray_zone_margin=0.08, max_self_train_epochs=3),
            candidates=CandidateConfig(use_lsh=True, shingle_size=2, num_perm=64, lsh_threshold=0.5, max_candidates_per_doc=4000),
            bootstrap=BootstrapConfig(max_pos_pairs=70_000, max_neg_pairs=70_000),
            self_learning=SelfLearningConfig(enabled=True, epochs=3),
            persist=True,
        )
    return make_profile(PROFILE_BALANCED)

def list_profiles() -> List[str]:
    return [PROFILE_BALANCED, PROFILE_HIGH_PREC, PROFILE_RECALL, PROFILE_CUSTOM]

# serialization
def to_dict(cfg: PipelineConfig) -> Dict[str, Any]:
    return _deep_asdict(cfg)

def from_dict(d: Dict[str, Any]) -> PipelineConfig:
    return _dc_from_dict(PipelineConfig, d)

def to_json(cfg: PipelineConfig, *, indent: Optional[int] = 2) -> str:
    return json.dumps(to_dict(cfg), ensure_ascii=False, indent=indent)

def from_json(s: str) -> PipelineConfig:
    return from_dict(json.loads(s))

def to_yaml(cfg: PipelineConfig) -> str:
    if yaml is None:
        raise RuntimeError("pyyaml not installed")
    return yaml.safe_dump(to_dict(cfg), sort_keys=False)

def from_yaml(s: str) -> PipelineConfig:
    if yaml is None:
        raise RuntimeError("pyyaml not installed")
    return from_dict(yaml.safe_load(s))

# overrides
def apply_overrides(cfg: PipelineConfig, overrides: Dict[str, Any]) -> PipelineConfig:
    base = to_dict(cfg)
    _deep_update(base, overrides)
    out = from_dict(base)
    return validate(out)

def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v

# validation
def validate(cfg: PipelineConfig) -> PipelineConfig:
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))
    # learner configs
    for lc in (cfg.simhash, cfg.minhash, cfg.embedding):
        lc.target_precision = _clip01(lc.target_precision)
        if lc.min_confidence is not None:
            lc.min_confidence = _clip01(lc.min_confidence)
        lc.max_pairs_per_epoch = max(1, int(lc.max_pairs_per_epoch))
    # arbiter
    cfg.arbiter.require_agreement = max(1, int(cfg.arbiter.require_agreement))
    cfg.arbiter.gray_zone_margin = max(0.0, float(cfg.arbiter.gray_zone_margin))
    cfg.arbiter.max_escalation_steps = max(0, int(cfg.arbiter.max_escalation_steps))
    cfg.arbiter.max_self_train_epochs = max(0, int(cfg.arbiter.max_self_train_epochs))
    cfg.arbiter.strong_margin = max(0.0, float(cfg.arbiter.strong_margin))
    # candidates
    cfg.candidates.shingle_size = max(1, int(cfg.candidates.shingle_size))
    cfg.candidates.num_perm = max(8, int(cfg.candidates.num_perm))
    cfg.candidates.lsh_threshold = _clip01(cfg.candidates.lsh_threshold)
    cfg.candidates.max_candidates_per_doc = max(1, int(cfg.candidates.max_candidates_per_doc))
    if cfg.candidates.max_total_candidates is not None:
        cfg.candidates.max_total_candidates = max(1, int(cfg.candidates.max_total_candidates))
    # bootstrap
    cfg.bootstrap.max_pos_pairs = max(1, int(cfg.bootstrap.max_pos_pairs))
    cfg.bootstrap.max_neg_pairs = max(1, int(cfg.bootstrap.max_neg_pairs))
    # self-learning
    cfg.self_learning.epochs = max(0, int(cfg.self_learning.epochs))
    return cfg

# helpers for GUI
def profile_config(profile_name: str) -> PipelineConfig:
    return validate(make_profile(profile_name))

def config_summary(cfg: PipelineConfig) -> Dict[str, Any]:
    return {
        "profile_like": _guess_profile(cfg),
        "require_agreement": cfg.arbiter.require_agreement,
        "gray_zone": cfg.arbiter.gray_zone_margin,
        "targets": {
            "simhash": cfg.simhash.target_precision,
            "minhash": cfg.minhash.target_precision,
            "embedding": cfg.embedding.target_precision,
        },
        "candidates": {
            "shingle": cfg.candidates.shingle_size,
            "lsh_threshold": cfg.candidates.lsh_threshold,
        },
        "self_learning_epochs": cfg.self_learning.epochs,
        "persist": cfg.persist,
    }

def _guess_profile(cfg: PipelineConfig) -> str:
    tp = (cfg.simhash.target_precision, cfg.minhash.target_precision, cfg.embedding.target_precision)
    if all(x >= 0.994 for x in tp) and cfg.arbiter.gray_zone_margin <= 0.05:
        return PROFILE_HIGH_PREC
    if all(x <= 0.955 for x in tp) and cfg.candidates.shingle_size <= 2:
        return PROFILE_RECALL
    return PROFILE_BALANCED
