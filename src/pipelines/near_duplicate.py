# src/pipelines/near_duplicate.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    from datasketch import MinHash, MinHashLSH
except Exception:
    MinHash = None
    MinHashLSH = None

from src.learners.base import (
    DocumentView,
    LearnerConfig,
    CorpusStats,
)
from src.learners.simhash_model import SimHashLearner as SimhashLearner
from src.learners.minhash_model import MinHashLearner
from src.learners.embed_model import EmbeddingLearner
from src.ensemble.arbiter import Arbiter, ArbiterConfig, DecisionTrace
from src.features.text_preproc import compute_corpus_stats, tokenize_words
from src.training.calibration import build_bootstrap_from_exact_duplicates
from src.persistence import state_store as store
from src.metrics.metrics import summarize_run, metrics_snapshot

# pipeline knobs
@dataclass
class CandidateConfig:
    use_lsh: bool = True
    shingle_size: int = 3
    num_perm: int = 64
    lsh_threshold: float = 0.6
    max_candidates_per_doc: int = 2000
    max_total_candidates: Optional[int] = None

@dataclass
class BootstrapConfig:
    max_pos_pairs: int = 50_000
    max_neg_pairs: int = 50_000

@dataclass
class SelfLearningConfig:
    enabled: bool = True
    epochs: int = 2

@dataclass
class PipelineConfig:
    simhash: LearnerConfig = field(default_factory=LearnerConfig)
    minhash: LearnerConfig = field(default_factory=LearnerConfig)
    embedding: LearnerConfig = field(default_factory=LearnerConfig)
    arbiter: ArbiterConfig = field(default_factory=ArbiterConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    self_learning: SelfLearningConfig = field(default_factory=SelfLearningConfig)
    persist: bool = True

# run intelligent mode on a set of documents
def run_intelligent_pipeline(
    docs: Iterable[DocumentView],
    *,
    config: Optional[PipelineConfig] = None,
    run_notes: str = "",
) -> Dict[str, Any]:
    cfg = config or PipelineConfig()
    docs_list = [d for d in docs if (d.text or "").strip()]
    id_map = {d.doc_id: d for d in docs_list}

    # learners + arbiter
    sim = SimhashLearner(cfg.simhash); _maybe_load(sim)
    mnh = MinHashLearner(cfg.minhash); _maybe_load(mnh)
    emb = EmbeddingLearner(cfg.embedding); _maybe_load(emb)
    learners = [ln for ln in (sim, mnh, emb) if ln.config.enabled]
    if not learners:
        raise ValueError("no learners enabled")
    arb = Arbiter(learners, cfg.arbiter)

    # corpus stats
    stats: CorpusStats = compute_corpus_stats(docs_list)
    arb.prepare(stats)

    # bootstrap calibration
    pos, neg = build_bootstrap_from_exact_duplicates(
        id_map,
        max_pos_pairs=cfg.bootstrap.max_pos_pairs,
        max_neg_pairs=cfg.bootstrap.max_neg_pairs,
    )
    _ = arb.calibrate_from_bootstrap(pos, neg)
    for ln in learners:
        store.save_learner_state(ln.name, ln.get_state())

    # candidates
    cands = generate_candidates(docs_list, cfg.candidates)

    # start run
    run_cfg_json = json.dumps(_export_run_config(cfg, stats), ensure_ascii=False)
    run_id = store.start_run(run_cfg_json, status="running", notes=run_notes) if cfg.persist else -1

    # score
    traces: List[DecisionTrace] = []
    for (a_id, b_id) in cands:
        tr = arb.score_pair(id_map[a_id], id_map[b_id])
        traces.append(tr)

    # self-learning
    if cfg.self_learning.enabled and cfg.self_learning.epochs > 0:
        _ = arb.run_self_learning_loop(((id_map[a], id_map[b]) for (a, b) in cands), epochs=cfg.self_learning.epochs)
        for ln in learners:
            store.save_learner_state(ln.name, ln.get_state())

    # persist
    if cfg.persist and run_id != -1:
        _persist_calibrations(run_id, learners)
        store.bulk_insert_decisions(run_id, traces)
        store.end_run(run_id, status="completed")

    # clusters + metrics
    clusters = build_clusters_from_traces(traces)
    run_summary = summarize_run(traces)
    snapshot = metrics_snapshot(traces, pseudo_labels={})

    return {
        "run_id": run_id,
        "pairs_scored": len(traces),
        "clusters": clusters,
        "traces": traces,
        "run_summary": run_summary,
        "metrics_snapshot": snapshot,
    }

# build clusters from duplicate traces
def build_clusters_from_traces(traces: Iterable[DecisionTrace]) -> List[List[str]]:
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
    out = []
    for members in comp.values():
        uniq = sorted(list(set(members)))
        if len(uniq) >= 2:
            out.append(uniq)
    out.sort(key=lambda c: (-len(c), c[0]))
    return out

# generate candidates via MinHash LSH
def generate_candidates(
    docs: List[DocumentView],
    ccfg: CandidateConfig,
) -> List[Tuple[str, str]]:
    tok_map: Dict[str, List[str]] = {d.doc_id: (d.tokens if d.tokens is not None else tokenize_words(d.text)) for d in docs}
    ids = [d.doc_id for d in docs]

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
            lsh.insert(did, mh, check_duplication=False)

        pairs: Set[Tuple[str, str]] = set()
        for did in ids:
            cands = lsh.query(mhs[did])
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

    # fallback: dense windowing sample
    pairs: List[Tuple[str, str]] = []
    n = len(ids)
    cap = ccfg.max_total_candidates or (n * min(n - 1, 20))
    step = max(1, n // max(1, (cap // max(1, n))))
    for i in range(0, n, 1):
        for j in range(i + 1, min(n, i + 1 + step)):
            pairs.append((ids[i], ids[j]))
            if len(pairs) >= cap:
                return pairs
    return pairs

# load prior learner state if present
def _maybe_load(learner) -> None:
    st = store.load_learner_state(learner.name)
    learner.load_state(st)

# persist per-learner calibration snapshot
def _persist_calibrations(run_id: int, learners: List[Any]) -> None:
    for ln in learners:
        st = ln.get_state()
        cal = st.calibration
        method = cal.method or "none"
        params_json = json.dumps(cal.params or {}, ensure_ascii=False)
        reliability_json = json.dumps(cal.reliability_bins or [], ensure_ascii=False)
        store.save_calibration(run_id, ln.name, method, params_json, reliability_json)

# export config for persistence
def _export_run_config(cfg: PipelineConfig, stats: CorpusStats) -> Dict[str, Any]:
    return {
        "pipeline": {
            "arbiter": cfg.arbiter.__dict__,
            "candidates": cfg.candidates.__dict__,
            "bootstrap": cfg.bootstrap.__dict__,
            "self_learning": cfg.self_learning.__dict__,
            "persist": cfg.persist,
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
