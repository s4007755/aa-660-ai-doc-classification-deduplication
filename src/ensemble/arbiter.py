# src/ensemble/arbiter.py
from __future__ import annotations

import hashlib
import unicodedata
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

"""
Arbiter

Fusion layer that combines multiple near-duplicate learners (SimHash, MinHash, Embedding)
and produces a final decision per-doc pair:

    - DUPLICATE
    - NON_DUPLICATE
    - UNCERTAIN   (falls in the gray zone, eligible for escalation)

Key responsibilities:
* Run all learners and apply voting/consensus.
* Consider exact duplicate short-circuit (hash-based, canonicalized).
* Handle gray zone vs. confident decisions using per-learner thresholds.
* Optionally escalate (normalize stricter, alter MinHash shingle size, whiten embeddings).
* Provide a decision trace for explainability (raw scores, calibrated probs, thresholds, steps).
* Support pseudo-label generation for self-learning rounds.
"""

FinalLabel = Literal["DUPLICATE", "NON_DUPLICATE", "UNCERTAIN"]
DupKind = Literal["EXACT", "NEAR"]

@dataclass
class ArbiterConfig:
    require_agreement: int = 2
    gray_zone_margin: float = 0.05
    max_escalation_steps: int = 3
    escalation_order: List[str] = field(default_factory=lambda: [
        "normalize_strict", # tighten normalization, strip IDs/dates
        "minhash_alt_shingle", # try alternate shingle size on MinHash to escape false ties
        "embed_whiten", # whiten embeddings / remove top PCs to reduce topical bias
    ])
    max_self_train_epochs: int = 2
    strong_margin: float = 0.07
    random_state: int = 13
    near_support_band_mult: float = 2.0
    near_support_min_band: float = 0.02
    enable_exact_check: bool = True
    exact_normalization: Literal["none", "line_endings", "unicode_lines"] = "unicode_lines"
    use_calibrated: bool = True

@dataclass
class DecisionTrace:
    """
    For each pair, record everything needed to explain the result:
    - learner raw scores and probabilities (calibrated or raw per config)
    - threshold values used for comparison (if available)
    - final label, reason, escalation steps taken and which learners agreed
    - duplicate kind (EXACT vs NEAR) when applicable
    """
    pair_key: str
    a_id: str
    b_id: str
    learner_outputs: Dict[str, LearnerOutput]
    final_label: FinalLabel
    reason: str
    escalation_steps: List[str] = field(default_factory=list)
    dup_kind: Optional[DupKind] = None
    exact_voters: List[str] = field(default_factory=list)
    near_voters: List[str] = field(default_factory=list)
    agreed_learners: List[str] = field(default_factory=list)

    use_calibrated: bool = True

    def as_dict(self) -> Dict[str, Any]:
        """
        Produce a JSON-serializable dict that’s safe to store in run history
        and render in decision trace UIs.
        """
        out: Dict[str, Any] = {
            "pair_key": self.pair_key,
            "a_id": self.a_id,
            "b_id": self.b_id,
            "final_label": self.final_label,
            "reason": self.reason,
            "dup_kind": self.dup_kind,
            "escalation_steps": list(self.escalation_steps),
            "exact_voters": list(self.exact_voters),
            "near_voters": list(self.near_voters),
            "agreed_learners": list(self.agreed_learners),
            "learners": {},
            "use_calibrated": bool(self.use_calibrated),
        }
        def _disp3(x: float) -> float:
            """
            Display helper:
            - round down to 3 decimals (floor) to avoid overconfidence in UI
            - clamp < 1.0 for non-exact reasons
            """
            from math import floor
            try:
                x = float(x)
            except Exception:
                return x
            if self.reason != "exact_content_match":
                x = min(x, 0.999999)
            return floor(x * 1000.0) / 1000.0
        for k, v in self.learner_outputs.items():
            # Compare/display probability depends on calibration switch
            p_display = v.prob if self.use_calibrated else v.raw_score
            thr_display = None if v.threshold is None else float(v.threshold)


            entry = {
                "raw_score": _disp3(v.raw_score),
                "prob": _disp3(p_display),
                "threshold": thr_display,
                "rationale": v.rationale,
                "warnings": v.warnings,
            }
            # Convenience: if the rationale contains an internal similarity, provide a display copy
            try:
                if isinstance(v.rationale, dict) and "similarity_est" in v.rationale:
                    rr = dict(v.rationale)
                    rr["similarity_est_display"] = _disp3(float(v.rationale["similarity_est"]))
                    entry["rationale"] = rr
            except Exception:
                pass
            out["learners"][k] = entry
        return out


@dataclass
class LearningUpdate:
    """
    Container for reporting learner state updates.
    """
    learner_name: str
    updated_state: LearnerState
    notes: str = ""


class Arbiter:
    """
    Orchestrates the multi-learner decision pipeline for near-duplicate detection.
    """
    def __init__(self, learners: List[ILearner], config: Optional[ArbiterConfig] = None):
        """
        Initialize the Arbiter with a list of learners and a configuration.
        """
        if not learners:
            raise ValueError("Arbiter requires at least one learner")
        self.learners: List[ILearner] = [ln for ln in learners if ln.config.enabled]
        if not self.learners:
            raise ValueError("All learners are disabled")
        self.config: ArbiterConfig = config or ArbiterConfig()

    def prepare(self, corpus_stats: Optional[CorpusStats] = None) -> None:
        """
        Allow learners to pre-compute any corpus-dependent state.
        """
        for ln in self.learners:
            ln.prepare(corpus_stats)

    def calibrate_from_bootstrap(self, positives: Iterable[Pair], negatives: Iterable[Pair]) -> List[LearningUpdate]:
        """
        Ask each learner to fit its calibration (probability mapping and threshold) from bootstrap sets.
        """
        updates: List[LearningUpdate] = []
        for ln in self.learners:
            st = ln.fit_calibration(positives, negatives)
            updates.append(LearningUpdate(learner_name=ln.name, updated_state=st, notes="bootstrap calibration"))
        return updates


    # Exact duplicate helpers

    def _doc_text(self, dv: DocumentView) -> str:
        """
        Safely extract the textual payload from a DocumentView variant.
        """        
        if hasattr(dv, "text") and getattr(dv, "text") is not None:
            return dv.text
        if hasattr(dv, "content") and getattr(dv, "content") is not None:
            return dv.content
        return ""

    def _canonicalize_for_exact(self, s: str) -> str:
        """
        Canonicalize text before hashing for exact-duplicate detection according to config.
        """        
        mode = self.config.exact_normalization
        if mode in ("line_endings", "unicode_lines"):
            s = s.replace("\r\n", "\n").replace("\r", "\n")
        if mode == "unicode_lines":
            # Remove BOM if present, NFC to coalesce canonically equivalent sequences
            if s.startswith("\ufeff"):
                s = s.lstrip("\ufeff")
            s = unicodedata.normalize("NFC", s)
        return s

    def _exact_hash(self, dv: DocumentView) -> str:
        """
        Hash (SHA-256) of canonicalized text for exact duplicate detection.
        """        
        text = self._doc_text(dv)
        text = self._canonicalize_for_exact(text)
        return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()

    def _is_exact_duplicate(self, a: DocumentView, b: DocumentView) -> bool:
        """
        Constant-time equivalent of exact text equality under chosen canonicalization.
        """
        if not self.config.enable_exact_check:
            return False
        return self._exact_hash(a) == self._exact_hash(b)


    # Voting helpers

    def _learner_by_name(self, name: str) -> Optional[ILearner]:
        """
        Retrieve a learner instance by its name.
        """
        for ln in self.learners:
            if ln.name == name:
                return ln
        return None

    def _threshold_of(self, learner_name: str) -> Optional[float]:
        """
        Return the learner's calibrated threshold.
        """
        if not getattr(self.config, "use_calibrated", True):
            return None
        ln = self._learner_by_name(learner_name)
        if not ln:
            return None
        st = ln.get_state()
        cal = getattr(st, "calibration", None)
        return getattr(cal, "threshold", None) if cal is not None else None


    def _required_votes(self) -> int:
        """
        The effective number of learners needed to agree, bounded by the number of active learners.
        """        
        return max(1, min(self.config.require_agreement, len(self.learners)))

    def _votes(self, learner_name: str, out: LearnerOutput) -> bool:
        """
        Decide whether a single learner considers a pair a duplicate.

        Voting logic:
        1) Embedding backstop cosine threshold (robust guardrail for high cosine).
        2) If calibrated: compare calibrated prob vs. calibrated threshold, also respect min_confidence.
        3) If uncalibrated: consult per-learner explicit thresholds in config.extras, else min_confidence.
        """
        ln = self._learner_by_name(learner_name)
        min_conf = ln.config.min_confidence if ln else None

        # Embedding backstop
        try:
            if ln and ln.name.lower().startswith("embedding"):
                # Strong cosine floor to capture near-identical texts even if calibration is off/noisy
                cos_floor = float((ln.config.extras or {}).get("cosine_threshold", 0.97))
                cos_val = None
                if isinstance(out.rationale, dict) and "cosine" in out.rationale:
                    try: cos_val = float(out.rationale["cosine"])
                    except Exception: cos_val = None
                if cos_val is None and out.raw_score is not None:
                    # Convert prob-like raw_score back to cosine if needed
                    try: cos_val = 2.0 * float(out.raw_score) - 1.0
                    except Exception: cos_val = None
                if cos_val is not None and cos_val >= cos_floor:
                    return True
        except Exception:
            pass

        use_cal = getattr(self.config, "use_calibrated", True)

        # Choose the comparison value
        p = out.prob if use_cal else out.raw_score
        if p is None:
            return False

        if use_cal and out.threshold is not None:
            # Respect per-learner min_conf as an additional floor for certainty     
            if min_conf is not None and p < min_conf:
                return False
            return p >= out.threshold

        # Uncalibrated path:
        thr = None
        if ln:
            ex = ln.config.extras or {}
            if ln.name.lower().startswith("embedding"):
                thr = ex.get("cosine_threshold")
            else:
                thr = ex.get("decision_threshold")
            try:
                thr = float(thr) if thr is not None else None
            except Exception:
                thr = None

        if thr is not None:
            return p >= thr
        if min_conf is not None:
            return p >= min_conf
        return False



    def _all_clearly_below(self, outputs: Dict[str, LearnerOutput]) -> bool:
        """
        True if all learners' probabilities are comfortably below their thresholds
        by at least gray_zone_margin. Only applies when calibrated thresholds exist.
        """
        saw_any = False
        for name, out in outputs.items():
            th = self._threshold_of(name)
            if th is None or out.prob is None:
                continue
            saw_any = True
            if out.prob >= (th - self.config.gray_zone_margin):
                return False
        return saw_any


    def _in_gray_zone(self, outputs: Dict[str, LearnerOutput]) -> bool:
        """
        True if at least one calibrated learner is near its threshold, within gray zone,
        and none are confidently above/below.
        """        
        saw_any = False
        for name, out in outputs.items():
            th = self._threshold_of(name)
            if th is None or out.prob is None:
                continue
            saw_any = True
            if abs(out.prob - th) > self.config.gray_zone_margin:
                return False
        return saw_any


    # Main scoring

    def score_pair(self, a: DocumentView, b: DocumentView) -> DecisionTrace:
        """
        Score a single document pair via:
        exact duplicate short-circuit -> base scoring -> gray-zone checks -> escalation if needed.
        """
        need = self._required_votes()
        uc = bool(self.config.use_calibrated)

        # 0) exact short circuit
        if self._is_exact_duplicate(a, b):
            synthetic: Dict[str, LearnerOutput] = {}
            exact_voters = []
            for ln in self.learners:
                th = self._threshold_of(ln.name)
                synthetic[ln.name] = LearnerOutput(
                    raw_score=1.0,
                    prob=1.0,
                    threshold=(0.999 if th is None else float(th)),
                    rationale={"rule": "exact_content_match", "canonicalization": str(self.config.exact_normalization)},
                    warnings=[],
                    internals=None,
                )
                exact_voters.append(ln.name)
            return DecisionTrace(
                pair_key=stable_pair_key(a.doc_id, b.doc_id),
                a_id=a.doc_id,
                b_id=b.doc_id,
                learner_outputs=synthetic,
                final_label="DUPLICATE",
                reason="exact_content_match",
                escalation_steps=[],
                dup_kind="EXACT",
                exact_voters=exact_voters,
                near_voters=[],
                agreed_learners=exact_voters[:],
                use_calibrated=uc,
            )

        # 1) base scoring
        base_outputs = self._score_all(a, b)
        agreed = [n for n, o in base_outputs.items() if self._votes(n, o)]
        if len(agreed) >= need:
            return DecisionTrace(
                pair_key=stable_pair_key(a.doc_id, b.doc_id),
                a_id=a.doc_id,
                b_id=b.doc_id,
                learner_outputs=base_outputs,
                final_label="DUPLICATE",
                reason="duplicate_by_consensus",
                escalation_steps=[],
                dup_kind="NEAR",
                exact_voters=[],
                near_voters=sorted(agreed),
                agreed_learners=sorted(agreed),
                use_calibrated=uc,
            )

        # 2) clearly below
        if self._all_clearly_below(base_outputs):
            return DecisionTrace(
                pair_key=stable_pair_key(a.doc_id, b.doc_id),
                a_id=a.doc_id,
                b_id=b.doc_id,
                learner_outputs=base_outputs,
                final_label="NON_DUPLICATE",
                reason="below_thresholds",
                escalation_steps=[],
                dup_kind=None,
                exact_voters=[],
                near_voters=[],
                agreed_learners=[],
                use_calibrated=uc,
            )

        # 3) confident non-duplicate
        if not self._in_gray_zone(base_outputs):
            return DecisionTrace(
                pair_key=stable_pair_key(a.doc_id, b.doc_id),
                a_id=a.doc_id,
                b_id=b.doc_id,
                learner_outputs=base_outputs,
                final_label="NON_DUPLICATE",
                reason="confident_non_duplicate",
                escalation_steps=[],
                dup_kind=None,
                exact_voters=[],
                near_voters=[],
                agreed_learners=[],
                use_calibrated=uc,
            )

        # 4) escalation loop
        outputs, agreed_after, steps_done = self._escalate(a, b, base_outputs)
        if len(agreed_after) >= need:
            return DecisionTrace(
                pair_key=stable_pair_key(a.doc_id, b.doc_id),
                a_id=a.doc_id,
                b_id=b.doc_id,
                learner_outputs=outputs,
                final_label="DUPLICATE",
                reason="consensus_after_escalation",
                escalation_steps=steps_done,
                dup_kind="NEAR",
                exact_voters=[],
                near_voters=sorted(agreed_after),
                agreed_learners=sorted(agreed_after),
                use_calibrated=uc,
            )

        if self._all_clearly_below(outputs):
            return DecisionTrace(
                pair_key=stable_pair_key(a.doc_id, b.doc_id),
                a_id=a.doc_id,
                b_id=b.doc_id,
                learner_outputs=outputs,
                final_label="NON_DUPLICATE",
                reason="below_thresholds_after_escalation",
                escalation_steps=steps_done,
                dup_kind=None,
                exact_voters=[],
                near_voters=[],
                agreed_learners=[],
                use_calibrated=uc,
            )

        # 5) unresolved
        return DecisionTrace(
            pair_key=stable_pair_key(a.doc_id, b.doc_id),
            a_id=a.doc_id,
            b_id=b.doc_id,
            learner_outputs=outputs,
            final_label="UNCERTAIN",
            reason="uncertain_in_gray_zone",
            escalation_steps=steps_done,
            dup_kind=None,
            exact_voters=[],
            near_voters=[],
            agreed_learners=[],
            use_calibrated=uc,
        )

    def batch_score(self, pairs: Iterable[Pair]) -> List[DecisionTrace]:
        """
        Convenience: score a batch of pairs, preserving order.
        """        
        traces: List[DecisionTrace] = []
        for a, b in pairs:
            traces.append(self.score_pair(a, b))
        return traces

    def build_pseudo_labels(self, traces: Iterable[DecisionTrace]) -> List[PairLabel]:
        """
        Build high-confidence pseudo-labels from traces for self-training.

        - Positive label if >= required votes are strongly above threshold by strong_margin.
        - Negative label if all learners are strongly below threshold by strong_margin.
        """        
        labels: List[PairLabel] = []
        t_strong = self.config.strong_margin
        need = self._required_votes()
        for tr in traces:
            outs = tr.learner_outputs
            strong_pos = 0
            strong_neg = 0
            for name, out in outs.items():
                th = self._threshold_of(name)
                if th is None:
                    continue
                if out.prob >= (th + t_strong):
                    strong_pos += 1
                if out.prob <= (th - t_strong):
                    strong_neg += 1
            if strong_pos >= need:
                labels.append(PairLabel(a_id=tr.a_id, b_id=tr.b_id, label=1, weight=1.0))
            elif strong_neg == len(outs):
                labels.append(PairLabel(a_id=tr.a_id, b_id=tr.b_id, label=0, weight=1.0))
        return labels

    def self_training_round(self, pseudo_labels: Iterable[PairLabel]) -> List[LearningUpdate]:
        """
        Execute one round of self-training across learners with provided pseudo-labels.
        """        
        updates: List[LearningUpdate] = []
        pls = list(pseudo_labels)
        if not pls:
            return updates
        for ln in self.learners:
            st = ln.self_train(pls)
            updates.append(LearningUpdate(learner_name=ln.name, updated_state=st, notes="self_train"))
        return updates

    def run_self_learning_loop(self, pairs: Iterable[Pair], epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        - score current pairs,
        - extract confident pseudo-labels,
        - self-train learners,
        - record epoch summary.

        Stops early if no pseudo labels are produced.
        """
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


    # Internals

    def _score_all(self, a: DocumentView, b: DocumentView) -> Dict[str, LearnerOutput]:
        """
        Run all learners on a single pair and collect outputs keyed by learner name.
        """        
        outputs: Dict[str, LearnerOutput] = {}
        for ln in self.learners:
            outputs[ln.name] = ln.score_pair(a, b)
        return outputs

    def _escalate(
        self,
        a: DocumentView,
        b: DocumentView,
        initial_outputs: Dict[str, LearnerOutput],
    ) -> Tuple[Dict[str, LearnerOutput], List[str], List[str]]:
        """
        Apply escalation steps in order, attempting to move an in-gray-zone pair to a confident decision.

        """
        originals: Dict[str, LearnerConfig] = {ln.name: ln.config for ln in self.learners}
        steps_done: List[str] = []
        current_outputs = dict(initial_outputs)
        need = self._required_votes()

        for step_i, step in enumerate(self.config.escalation_order[: self.config.max_escalation_steps], start=1):
            self._apply_escalation_step(step)
            rescored = self._score_all(a, b)
            agreed = [n for n, o in rescored.items() if self._votes(n, o)]

            steps_done.append(step)
            current_outputs = rescored

            if len(agreed) >= need:
                self._restore_configs(originals)
                return current_outputs, agreed, steps_done

            if self._all_clearly_below(rescored):
                self._restore_configs(originals)
                return current_outputs, [], steps_done

        self._restore_configs(originals)
        final_agreed = [n for n, o in current_outputs.items() if self._votes(n, o)]
        return current_outputs, final_agreed, steps_done

    def _apply_escalation_step(self, step: str) -> None:
        """
        Mutate learner configurations for a single escalation action, then re-prepare learners.
        """
        for ln in self.learners:
            cfg = LearnerConfig(
                enabled=ln.config.enabled,
                target_precision=ln.config.target_precision,
                min_confidence=ln.config.min_confidence,
                max_pairs_per_epoch=ln.config.max_pairs_per_epoch,
                random_state=ln.config.random_state,
                extras=dict(ln.config.extras or {})
            )
            if step == "normalize_strict":
                cfg.extras["normalize_strict"] = True
                cfg.extras["strip_dates_ids"] = True
            elif step == "minhash_alt_shingle" and ln.name.lower().startswith("minhash"):
                prev = int(cfg.extras.get("shingle_size", 3))
                cfg.extras["shingle_size"] = 2 if prev == 3 else 3
            elif step == "embed_whiten" and ln.name.lower().startswith("embedding"):
                cfg.extras["whiten"] = True
                cfg.extras["remove_top_pc"] = True
            ln.configure(cfg)
            ln.prepare(None)

    def _restore_configs(self, originals: Dict[str, LearnerConfig]) -> None:
        """
        Restore original learner configs after escalation attempts.
        """        
        for ln in self.learners:
            if ln.name in originals:
                ln.configure(originals[ln.name])
                ln.prepare(None)
