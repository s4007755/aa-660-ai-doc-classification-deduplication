# src/ensemble/arbiter.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from src.learners.base import (
    DocumentView,
    Pair,
    ILearner,
    LearnerOutput,
    LearnerConfig,
    LearnerState,
    PairLabel,
    CorpusStats,
    stable_pair_key,
)

# Final labels assigned by consensus/escalation
FinalLabel = Literal["DUPLICATE", "NON_DUPLICATE", "UNCERTAIN"]

# Arbiter configuration
@dataclass
class ArbiterConfig:
    require_agreement: int = 2            # how many learners must agree
    gray_zone_margin: float = 0.05        # margin around thresholds
    max_escalation_steps: int = 3         # cap on escalation attempts
    escalation_order: List[str] = field(default_factory=lambda: [
        "normalize_strict",
        "minhash_alt_shingle",
        "embed_whiten",
    ])
    max_self_train_epochs: int = 2        # number of self-training rounds
    strong_margin: float = 0.07           # margin used to create strong pseudo-labels
    random_state: int = 13                # for any sampling choices

# Single decision trace for a pair
@dataclass
class DecisionTrace:
    pair_key: str
    a_id: str
    b_id: str
    learner_outputs: Dict[str, LearnerOutput]
    agreed_learners: List[str]
    final_label: FinalLabel
    reason: str
    escalation_steps: List[str] = field(default_factory=list)

    # Convenience for GUI
    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "pair_key": self.pair_key,
            "a_id": self.a_id,
            "b_id": self.b_id,
            "agreed_learners": list(self.agreed_learners),
            "final_label": self.final_label,
            "reason": self.reason,
            "escalation_steps": list(self.escalation_steps),
            "learners": {},
        }
        for k, v in self.learner_outputs.items():
            out["learners"][k] = {
                "raw_score": float(v.raw_score),
                "prob": float(v.prob),
                "threshold": None if v.threshold is None else float(v.threshold),
                "rationale": v.rationale,
                "warnings": v.warnings,
            }
        return out

# Summary of what changed in a self-learning round
@dataclass
class LearningUpdate:
    learner_name: str
    updated_state: LearnerState
    notes: str = ""

class Arbiter:
    # Build with a list of learners and an arbiter config
    def __init__(self, learners: List[ILearner], config: Optional[ArbiterConfig] = None):
        if not learners:
            raise ValueError("Arbiter requires at least one learner")
        self.learners: List[ILearner] = [ln for ln in learners if ln.config.enabled]
        if not self.learners:
            raise ValueError("All learners are disabled")
        self.config: ArbiterConfig = config or ArbiterConfig()

    # Prepare learners with corpus stats
    def prepare(self, corpus_stats: Optional[CorpusStats] = None) -> None:
        for ln in self.learners:
            ln.prepare(corpus_stats)

    # Calibrate each learner from bootstrap positives/negatives
    def calibrate_from_bootstrap(self, positives: Iterable[Pair], negatives: Iterable[Pair]) -> List[LearningUpdate]:
        updates: List[LearningUpdate] = []
        for ln in self.learners:
            st = ln.fit_calibration(positives, negatives)
            updates.append(LearningUpdate(learner_name=ln.name, updated_state=st, notes="bootstrap calibration"))
        return updates

    # Score a single pair with consensus and escalation if needed
    def score_pair(self, a: DocumentView, b: DocumentView) -> DecisionTrace:
        base_outputs = self._score_all(a, b)
        agreed = [n for n, o in base_outputs.items() if self._votes(o, self._min_confidence_of(n))]
        if len(agreed) >= self.config.require_agreement:
            return self._make_trace(a, b, base_outputs, agreed, "DUPLICATE", "consensus")

        if self._all_clearly_below(base_outputs):
            return self._make_trace(a, b, base_outputs, [], "NON_DUPLICATE", "all_below")

        if not self._in_gray_zone(base_outputs):
            return self._make_trace(a, b, base_outputs, [], "NON_DUPLICATE", "no_consensus_not_gray")

        # Try escalation steps
        outputs, agreed_after, steps_done = self._escalate(a, b, base_outputs)
        if len(agreed_after) >= self.config.require_agreement:
            return self._make_trace(a, b, outputs, agreed_after, "DUPLICATE", "consensus_after_escalation", steps_done)

        if self._all_clearly_below(outputs):
            return self._make_trace(a, b, outputs, [], "NON_DUPLICATE", "all_below_after_escalation", steps_done)

        return self._make_trace(a, b, outputs, [], "UNCERTAIN", "unresolved_gray_zone", steps_done)

    # Score many pairs
    def batch_score(self, pairs: Iterable[Pair]) -> List[DecisionTrace]:
        traces: List[DecisionTrace] = []
        for a, b in pairs:
            traces.append(self.score_pair(a, b))
        return traces

    # Build pseudo-labels from decision traces
    def build_pseudo_labels(self, traces: Iterable[DecisionTrace]) -> List[PairLabel]:
        labels: List[PairLabel] = []
        t_strong = self.config.strong_margin
        for tr in traces:
            outs = tr.learner_outputs
            pos_votes = 0
            strong_pos = 0
            strong_neg = 0
            for name, out in outs.items():
                th = self._threshold_of(name)
                if th is None:
                    continue
                if out.prob >= th:
                    pos_votes += 1
                if out.prob >= (th + t_strong):
                    strong_pos += 1
                if out.prob <= (th - t_strong):
                    strong_neg += 1

            if strong_pos >= self.config.require_agreement:
                labels.append(PairLabel(a_id=tr.a_id, b_id=tr.b_id, label=1, weight=1.0))
            elif strong_neg == len(outs):
                labels.append(PairLabel(a_id=tr.a_id, b_id=tr.b_id, label=0, weight=1.0))
        return labels

    # One self-training round per learner
    def self_training_round(self, pseudo_labels: Iterable[PairLabel]) -> List[LearningUpdate]:
        updates: List[LearningUpdate] = []
        pls = list(pseudo_labels)
        if not pls:
            return updates
        for ln in self.learners:
            st = ln.self_train(pls)
            updates.append(LearningUpdate(learner_name=ln.name, updated_state=st, notes="self_train"))
        return updates

    # Utility: rescore a list of pairs to generate pseudo-labels and iterate
    def run_self_learning_loop(self, pairs: Iterable[Pair], epochs: Optional[int] = None) -> Dict[str, Any]:
        ep = epochs if epochs is not None else self.config.max_self_train_epochs
        history: List[Dict[str, Any]] = []
        for e in range(ep):
            traces = self.batch_score(pairs)
            pseudo = self.build_pseudo_labels(traces)
            updates = self.self_training_round(pseudo)
            history.append({
                "epoch": e + 1,
                "pairs_scored": len(traces),
                "pseudo_labels": len(pseudo),
                "updates": [u.learner_name for u in updates],
            })
            if not pseudo:
                break
        return {"epochs_run": len(history), "history": history}

    # Internal helpers
    def _score_all(self, a: DocumentView, b: DocumentView) -> Dict[str, LearnerOutput]:
        outputs: Dict[str, LearnerOutput] = {}
        for ln in self.learners:
            outputs[ln.name] = ln.score_pair(a, b)
        return outputs

    def _votes(self, out: LearnerOutput, min_conf: Optional[float]) -> bool:
        if out.threshold is None:
            return False
        if min_conf is not None and out.prob < min_conf:
            return False
        return out.prob >= out.threshold

    def _min_confidence_of(self, learner_name: str) -> Optional[float]:
        for ln in self.learners:
            if ln.name == learner_name:
                return ln.config.min_confidence
        return None

    def _threshold_of(self, learner_name: str) -> Optional[float]:
        for ln in self.learners:
            if ln.name == learner_name:
                return ln.get_state().calibration.threshold
        return None

    def _all_clearly_below(self, outputs: Dict[str, LearnerOutput]) -> bool:
        for name, out in outputs.items():
            th = self._threshold_of(name)
            if th is None:
                continue
            if out.prob >= (th - self.config.gray_zone_margin):
                return False
        return True

    def _in_gray_zone(self, outputs: Dict[str, LearnerOutput]) -> bool:
        for name, out in outputs.items():
            th = self._threshold_of(name)
            if th is None:
                continue
            if abs(out.prob - th) > self.config.gray_zone_margin:
                return False
        return True

    def _make_trace(
        self,
        a: DocumentView,
        b: DocumentView,
        outputs: Dict[str, LearnerOutput],
        agreed: List[str],
        label: FinalLabel,
        reason: str,
        steps: Optional[List[str]] = None,
    ) -> DecisionTrace:
        return DecisionTrace(
            pair_key=stable_pair_key(a.doc_id, b.doc_id),
            a_id=a.doc_id,
            b_id=b.doc_id,
            learner_outputs=outputs,
            agreed_learners=agreed,
            final_label=label,
            reason=reason,
            escalation_steps=list(steps or []),
        )

    def _escalate(
        self,
        a: DocumentView,
        b: DocumentView,
        initial_outputs: Dict[str, LearnerOutput],
    ) -> Tuple[Dict[str, LearnerOutput], List[str], List[str]]:
        originals: Dict[str, LearnerConfig] = {ln.name: ln.config for ln in self.learners}
        steps_done: List[str] = []
        current_outputs = dict(initial_outputs)

        for step_i, step in enumerate(self.config.escalation_order[: self.config.max_escalation_steps], start=1):
            self._apply_escalation_step(step)
            rescored = self._score_all(a, b)
            agreed = [n for n, o in rescored.items() if self._votes(o, self._min_confidence_of(n))]

            steps_done.append(step)
            current_outputs = rescored

            if len(agreed) >= self.config.require_agreement:
                self._restore_configs(originals)
                return current_outputs, agreed, steps_done

            if self._all_clearly_below(rescored):
                self._restore_configs(originals)
                return current_outputs, [], steps_done

        # Restore configs if unresolved
        self._restore_configs(originals)
        final_agreed = [n for n, o in current_outputs.items() if self._votes(o, self._min_confidence_of(n))]
        return current_outputs, final_agreed, steps_done

    def _apply_escalation_step(self, step: str) -> None:
        # Mutate learner configs in-memory;
        for ln in self.learners:
            cfg = LearnerConfig(
                enabled=ln.config.enabled,
                target_precision=ln.config.target_precision,
                min_confidence=ln.config.min_confidence,
                max_pairs_per_epoch=ln.config.max_pairs_per_epoch,
                random_state=ln.config.random_state,
                extras=dict(ln.config.extras),
            )
            if step == "normalize_strict":
                cfg.extras["normalize_strict"] = True
                cfg.extras["strip_dates_ids"] = True
            elif step == "minhash_alt_shingle" and ln.name.lower().startswith("minhash"):
                # Flip between common shingle sizes 2 and 3
                prev = int(cfg.extras.get("shingle_size", 3))
                cfg.extras["shingle_size"] = 2 if prev == 3 else 3
            elif step == "embed_whiten" and ln.name.lower().startswith("embedding"):
                cfg.extras["whiten"] = True
                cfg.extras["remove_top_pc"] = True
            ln.configure(cfg)
            # Allow learner to re-prepare if needed
            ln.prepare(None)

    def _restore_configs(self, originals: Dict[str, LearnerConfig]) -> None:
        for ln in self.learners:
            if ln.name in originals:
                ln.configure(originals[ln.name])
                ln.prepare(None)
