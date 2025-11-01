# src/pipelines/near_duplicate.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

try:
    from datasketch import MinHash, MinHashLSH
except Exception:
    MinHash = None
    MinHashLSH = None

from src.learners.base import DocumentView, LearnerConfig, CorpusStats
from src.learners.simhash_model import SimHashLearner as SimhashLearner
from src.learners.minhash_model import MinHashLearner
from src.learners.embed_model import EmbeddingLearner
from src.ensemble.arbiter import Arbiter, ArbiterConfig, DecisionTrace
from src.features.text_preproc import compute_corpus_stats, tokenize_words
from src.training.calibration import build_bootstrap_from_exact_duplicates
from src.persistence import state_store as store
from src.metrics.metrics import summarize_run, metrics_snapshot

ProgressCB = Optional[Callable[[int, int, str], None]]


@dataclass
class CandidateConfig:
    """
    Candidate generation parameters.
    """
    use_lsh: bool = True
    shingle_size: int = 3
    num_perm: int = 64
    lsh_threshold: float = 0.6
    max_candidates_per_doc: int = 2000
    max_total_candidates: Optional[int] = None


@dataclass
class BootstrapConfig:
    """
    Bootstrap sampling for calibration.
    """
    max_pos_pairs: int = 50_000
    max_neg_pairs: int = 50_000


@dataclass
class SelfLearningConfig:
    """
    Self-training stage configuration.
    """
    enabled: bool = True
    epochs: int = 2


@dataclass
class PipelineConfig:
    """
    Full pipeline configuration bundle.
    """
    simhash: LearnerConfig = field(default_factory=LearnerConfig)
    minhash: LearnerConfig = field(default_factory=LearnerConfig)
    embedding: LearnerConfig = field(default_factory=LearnerConfig)
    arbiter: ArbiterConfig = field(default_factory=ArbiterConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    self_learning: SelfLearningConfig = field(default_factory=SelfLearningConfig)
    persist: bool = True
    disable_calibration: bool = False


def run_intelligent_pipeline(
    docs: Iterable[DocumentView],
    *,
    config: Optional[PipelineConfig] = None,
    run_notes: str = "",
    progress_cb: ProgressCB = None,
) -> Dict[str, Any]:
    """
    End-to-end near-duplicate detection pipeline.

    Stages
    1) Configure learners and Arbiter.
    2) Compute corpus stats and prepare learners.
    3) Optional calibration via exact-dup bootstrap (per-learner and arbiter).
    4) Candidate generation (LSH or bounded sampler).
    5) Scoring through Arbiter to produce DecisionTrace objects.
    6) Optional self-learning loop.
    7) Metrics snapshot and clustering.
    8) Optional persistence of states, decisions and summaries.
    """
    def report(phase: str, done: int, total: int):
        if progress_cb:
            try:
                progress_cb(done, total, phase)
            except Exception:
                pass

    cfg = config or PipelineConfig()
    # Filter out empty-text docs up front, everything else flows from here.
    docs_list = [d for d in docs if (d.text or "").strip()]
    id_map = {d.doc_id: d for d in docs_list}

    use_cal = not cfg.disable_calibration

    # ArbiterConfig: prefer a copy to avoid mutating shared configs.
    try:
        from dataclasses import replace
        arb_cfg = replace(cfg.arbiter, use_calibrated=use_cal)
    except Exception:
        cfg.arbiter.use_calibrated = use_cal
        arb_cfg = cfg.arbiter

    # Instantiate learners and restore prior state.
    sim = SimhashLearner(cfg.simhash); _maybe_load(sim, use_calibration=use_cal)
    mnh = MinHashLearner(cfg.minhash); _maybe_load(mnh, use_calibration=use_cal)
    emb = EmbeddingLearner(cfg.embedding); _maybe_load(emb, use_calibration=use_cal)
    learners = [ln for ln in (sim, mnh, emb) if ln.config.enabled]
    if not learners:
        raise ValueError("no learners enabled")

    arb = Arbiter(learners, arb_cfg)

    # Global corpus statistics.
    stats: CorpusStats = compute_corpus_stats(docs_list)
    arb.prepare(stats)

    # Optional: calibration
    pos: List[Tuple[DocumentView, DocumentView]] = []
    neg: List[Tuple[DocumentView, DocumentView]] = []
    if not cfg.disable_calibration:
        report("calibration/bootstrap/prepare", 0, 1)
        pos, neg = _build_easy_bootstrap(
            docs_list,
            max_pos=cfg.bootstrap.max_pos_pairs,
            max_neg=cfg.bootstrap.max_neg_pairs,
            progress=lambda d, t, p: report(f"calibration/bootstrap/{p}", d, t),
        )
        report("calibration/bootstrap/done", 1, 1)

        # Per-learner calibration.
        for ln in learners:
            phase = f"calibration/{ln.name}"
            report(phase, 0, 1)
            try:
                ln.fit_calibration(pos, neg)
                store.save_learner_state(ln.name, ln.get_state())
            finally:
                report(phase, 1, 1)

        # Arbiter-level calibration if supported.
        try:
            _ = arb.calibrate_from_bootstrap(pos, neg)
        except TypeError:
            pass
    else:
        report("calibration/disabled", 1, 1)

    # Candidate generation.
    report("candidates", 0, 1)
    cands = generate_candidates(docs_list, cfg.candidates)
    report("candidates", 1, 1)

    # Persist start of run.
    run_cfg_json = json.dumps(_export_run_config(cfg, stats), ensure_ascii=False)
    run_id = store.start_run(run_cfg_json, status="running", notes=run_notes) if cfg.persist else -1

    # Score each candidate via the Arbiter.
    traces: List[DecisionTrace] = []
    total = max(1, len(cands))
    for i, (a_id, b_id) in enumerate(cands, start=1):
        tr = arb.score_pair(id_map[a_id], id_map[b_id])
        traces.append(tr)
        if (i % 25) == 0 or i == total:
            report("scoring", i, total)

    # Optional self-training on pseudo labels.
    if cfg.self_learning.enabled and cfg.self_learning.epochs > 0:
        epochs = int(cfg.self_learning.epochs)
        for e in range(epochs):
            report("self-training (pseudo-labels)", e, epochs)
            _ = arb.run_self_learning_loop(((id_map[a], id_map[b]) for (a, b) in cands), epochs=1)
            for ln in learners:
                store.save_learner_state(ln.name, ln.get_state())
        report("self-learning (pseudo-labels)", epochs, epochs)

    # Build pseudo labels from final Arbiter decisions.
    pseudo_labels: Dict[str, int] = {}
    for tr in traces:
        if tr.final_label == "DUPLICATE":
            pseudo_labels[tr.pair_key] = 1
        elif tr.final_label == "NON_DUPLICATE":
            pseudo_labels[tr.pair_key] = 0

    # Metrics snapshot for UI/report surfaces.
    snapshot = metrics_snapshot(traces, pseudo_labels=pseudo_labels)

    # Enrich per-learner view with thresholds and targets for display.
    per_learner = snapshot.setdefault("per_learner", {})
    for ln in learners:
        info = per_learner.setdefault(ln.name, {})
        st = ln.get_state()
        cal = getattr(st, "calibration", None)
        if use_cal and cal and cal.threshold is not None:
            info["threshold"] = float(cal.threshold)
        info["target_precision"] = float(ln.config.target_precision)

    # Compact calibration summary for headers/tooltips.
    calibration_snapshot: Dict[str, Dict[str, Any]] = {}
    for ln in learners:
        st = ln.get_state()
        cal = getattr(st, "calibration", None)
        calibration_snapshot[ln.name] = {
            "method": getattr(cal, "method", None) if use_cal else None,
            "threshold": getattr(cal, "threshold", None) if use_cal else None,
            "brier": getattr(cal, "brier_score", None) if use_cal else None,
        }

    # Clusters and run summary.
    clusters = build_clusters_from_traces(traces)
    run_summary = summarize_run(traces) or {}
    run_summary_out = dict(run_summary)
    try:
        run_summary_out["clusters"] = len(clusters)
    except Exception:
        pass

    # Persistence layer: calibrations, decisions, rollup summaries and end-run.
    if cfg.persist and run_id != -1:
        report("persisting", 0, 1)
        if use_cal:
            _persist_calibrations(run_id, learners)
        store.bulk_insert_decisions(run_id, traces)

        # Optional extras
        try:
            store.save_run_summary(run_id, run_summary_out)
        except Exception:
            pass
        try:
            store.save_metrics_snapshot(run_id, snapshot)
        except Exception:
            pass

        # Finalize the run row with scalars.
        try:
            store.end_run(
                run_id,
                status="completed",
                pairs=run_summary.get("total_pairs", run_summary.get("pairs_scored")),
                duplicates=run_summary.get("duplicates"),
                non_duplicates=run_summary.get("non_duplicates"),
                uncertain=run_summary.get("uncertain"),
                consensus_rate=run_summary.get("consensus_rate", run_summary.get("consensus_pct")),
                escalations_rate=run_summary.get("escalations_pct", run_summary.get("escalations_rate")),
                clusters=len(clusters) if isinstance(clusters, list) else None,
            )
        except TypeError:
            store.end_run(run_id, status="completed")

        report("persisting", 1, 1)

    return {
        "run_id": run_id,
        "pairs_scored": len(traces),
        "clusters": clusters,
        "traces": traces,
        "run_summary": run_summary_out,
        "metrics_snapshot": snapshot,
        "calibration_snapshot": calibration_snapshot,
    }


# Bootstrap construction
def _build_easy_bootstrap(
    docs: List[DocumentView],
    *,
    max_pos: int,
    max_neg: int,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[List[Tuple[DocumentView, DocumentView]], List[Tuple[DocumentView, DocumentView]]]:
    """
    Build a simple bootstrap set:
    - Positives: all pairs of documents that share identical normalized text.
    - Negatives: round robin pairs across groups.

    This is intentionally fast and conservative, it seeds calibration only.
    """
    def p(d, t, s):
        if progress:
            try:
                progress(d, t, s)
            except Exception:
                pass

    # Group by exact normalized text.
    p(0, 3, "group")
    groups: Dict[str, List[DocumentView]] = {}
    for d in docs:
        groups.setdefault(d.text or "", []).append(d)
    p(1, 3, "group")

    # Positives: all unique pairs within each text group, capped.
    pos: List[Tuple[DocumentView, DocumentView]] = []
    for g in groups.values():
        n = len(g)
        if n >= 2:
            for i in range(n):
                for j in range(i + 1, n):
                    pos.append((g[i], g[j]))
                    if len(pos) >= max_pos:
                        break
                if len(pos) >= max_pos:
                    break
    p(2, 3, "positives")

    # Negatives: easy reps across groups, capped.
    neg: List[Tuple[DocumentView, DocumentView]] = []
    reps = [v[0] for v in groups.values() if v]
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            neg.append((reps[i], reps[j]))
            if len(neg) >= max_neg:
                break
        if len(neg) >= max_neg:
            break
    p(3, 3, "negatives")

    return pos, neg


# Clustering from DecisionTraces
def build_clusters_from_traces(traces: Iterable[DecisionTrace]) -> List[List[str]]:
    """
    Build connected components from final duplicate decisions using DSU/Union–Find.

    Notes
    - Only edges with final_label == "DUPLICATE" are joined.
    - Singletons are omitted, clusters must have >= 2 members.
    """
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for tr in traces:
        if tr.final_label == "DUPLICATE":
            union(tr.a_id, tr.b_id)

    comp: Dict[str, List[str]] = {}
    for tr in traces:
        for did in (tr.a_id, tr.b_id):
            root = find(did)
            comp.setdefault(root, []).append(did)

    out: List[List[str]] = []
    for members in comp.values():
        uniq = sorted(list(set(members)))
        if len(uniq) >= 2:
            out.append(uniq)
    # Sort by size desc, then lexicographically for stability.
    out.sort(key=lambda c: (-len(c), c[0]))
    return out


# Candidate generation
def generate_candidates(docs: List[DocumentView], ccfg: CandidateConfig) -> List[Tuple[str, str]]:
    """
    Generate candidate pairs using MinHash LSH when available, otherwise
    fall back to a bounded local pairing sampler.
    """
    tok_map: Dict[str, List[str]] = {d.doc_id: (d.tokens if d.tokens is not None else tokenize_words(d.text)) for d in docs}
    ids = [d.doc_id for d in docs]

    # Preferred path: LSH-based neighbor search.
    if ccfg.use_lsh and MinHash is not None and MinHashLSH is not None:
        def shingles(tokens: List[str], k: int):
            if k <= 1:
                for t in tokens:
                    yield t
            else:
                L = max(0, len(tokens) - k + 1)
                for i in range(L):
                    yield " ".join(tokens[i : i + k])

        mhs: Dict[str, Any] = {}
        lsh = MinHashLSH(threshold=float(ccfg.lsh_threshold), num_perm=int(ccfg.num_perm))
        for did in ids:
            mh = MinHash(num_perm=int(ccfg.num_perm))
            for s in shingles(tok_map[did], int(ccfg.shingle_size)):
                mh.update(s.encode("utf-8", errors="ignore"))
            mhs[did] = mh
            # Avoid duplicate-checking overhead
            lsh.insert(did, mh, check_duplication=False)

        pairs: Set[Tuple[str, str]] = set()
        for did in ids:
            cands = lsh.query(mhs[did])
            # Trim self and apply per-doc cap for safety.
            cands = [c for c in cands if c != did][: int(ccfg.max_candidates_per_doc)]
            for c in cands:
                a, b = (did, c) if did < c else (c, did)
                pairs.add((a, b))
            if ccfg.max_total_candidates and len(pairs) >= ccfg.max_total_candidates:
                break

        out = sorted(list(pairs))
        if ccfg.max_total_candidates and len(out) > ccfg.max_total_candidates:
            out = out[: int(ccfg.max_total_candidates)]
        return out

    # Fallback: bounded local pairing.
    pairs: List[Tuple[str, str]] = []
    n = len(ids)
    cap = ccfg.max_total_candidates or (n * min(n - 1, 20))
    # Step narrows the local window to respect the global cap.
    step = max(1, n // max(1, (cap // max(1, n))))
    for i in range(0, n, step):
        for j in range(i + 1, min(n, i + 1 + step)):
            pairs.append((ids[i], ids[j]))
            if len(pairs) >= cap:
                return pairs
    return pairs


# State handling helpers
def _maybe_load(learner, *, use_calibration: bool = True) -> None:
    """
    Load a persisted learner state when available. If calibration is disabled
    for this run, strip calibration from the state before loading to ensure
    raw scoring paths are used consistently.
    """
    st = store.load_learner_state(learner.name)

    # If calibration is disabled, remove from state before handing to learner.
    if not use_calibration and st is not None:
        try:
            if hasattr(st, "calibration"):
                setattr(st, "calibration", None)
        except Exception:
            pass
        try:
            if hasattr(st, "extras") and isinstance(st.extras, dict):
                st.extras.pop("calibration", None)
        except Exception:
            pass

    learner.load_state(st)

    # Add explicit runtime hints to force raw paths and a deterministic threshold.
    if not use_calibration:
        ex = dict(getattr(learner, "config", None).extras or {})
        ex["use_calibrated"] = False
        ex["force_threshold"] = True
        learner.config.extras = ex
        # Belt-and-braces: clear any internal calibrator fields if exposed.
        for attr in ("calibrator", "calibration", "calib", "_calibrator"):
            if hasattr(learner, attr):
                try:
                    setattr(learner, attr, None)
                except Exception:
                    pass


def _persist_calibrations(run_id: int, learners: List[Any]) -> None:
    """
    Persist per-learner calibration artifacts for the given run.
    """
    for ln in learners:
        st = ln.get_state()
        cal = st.calibration
        method = cal.method or "none"
        params_json = json.dumps(cal.params or {}, ensure_ascii=False)
        reliability_json = json.dumps(cal.reliability_bins or [], ensure_ascii=False)
        store.save_calibration(run_id, ln.name, method, params_json, reliability_json)


def _export_run_config(cfg: PipelineConfig, stats: CorpusStats) -> Dict[str, Any]:
    """
    Build a compact, JSON-serializable configuration snapshot for persistence.
    """
    return {
        "pipeline": {
            "arbiter": cfg.arbiter.__dict__,
            "candidates": cfg.candidates.__dict__,
            "bootstrap": cfg.bootstrap.__dict__,
            "self_learning": cfg.self_learning.__dict__,
            "persist": cfg.persist,
            "disable_calibration": cfg.disable_calibration,
        },
        "learners": {
            "simhash": cfg.simhash.__dict__,
            "minhash": cfg.minhash.__dict__,
            "embedding": cfg.embedding.__dict__,
        },
        "corpus_stats": {
            "doc_count": stats.doc_count,
            "avg_doc_len": stats.avg_doc_len,
            "lang_counts": stats.lang_counts,
            "extras": stats.extras,
        },
    }
