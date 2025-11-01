# src/cli_nd.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# Hardware and perf helpers
import psutil
try:
    import pynvml
    _NVML_OK = True
except Exception:
    _NVML_OK = False

# Core pipeline and configs
from src.features.text_preproc import normalize_text, build_document_view
from src.learners.base import DocumentView, LearnerConfig
from src.gui.config_schemas import profile_config
from src.pipelines.near_duplicate import (
    run_intelligent_pipeline,
    PipelineConfig,
    CandidateConfig,
    BootstrapConfig,
    SelfLearningConfig,
)

# DB and state
from src.storage import sqlite_store
from src.persistence import state_store

# Optional ingestion module
_ingestion_mod = None
try:
    from src.pipelines import ingestion as _ingestion_mod
except Exception:
    try:
        import ingestion as _ingestion_mod
    except Exception:
        _ingestion_mod = None

# Optional requests for API mode
try:
    import requests
except Exception:
    requests = None


def _eprint(*a, **k):
    print(*a, file=sys.stderr, **k)


# Hardware helpers
def _safe_nvml_snapshot(min_vram_gb: float = 6.0) -> Tuple[bool, float]:
    """
    Returns (gpu_ok: bool, total_vram_gb: float).
    """
    if not _NVML_OK:
        return False, 0.0
    try:
        pynvml.nvmlInit()
        cnt = pynvml.nvmlDeviceGetCount()
        meets = False
        total = 0
        for i in range(cnt):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_total = pynvml.nvmlDeviceGetMemoryInfo(h).total
            total += mem_total
            if mem_total / (1024 ** 3) >= min_vram_gb:
                meets = True
        total_gb = total / (1024 ** 3)
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return (cnt > 0 and meets), total_gb
    except Exception:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return False, 0.0



def _set_threading_env(workers: int, torch_intra: Optional[int] = None, torch_inter: int = 1) -> None:
    try:
        os.environ["OMP_NUM_THREADS"] = str(max(1, workers))
        os.environ["OPENBLAS_NUM_THREADS"] = str(max(1, workers))
        os.environ["MKL_NUM_THREADS"] = str(max(1, workers))
        os.environ["NUMEXPR_NUM_THREADS"] = str(max(1, workers))
    except Exception:
        pass
    try:
        import torch
        if torch_intra is None:
            torch_intra = max(1, (psutil.cpu_count(logical=False) or 1) // 2) if workers > 1 else 1
        torch.set_num_threads(max(1, torch_intra))
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, torch_inter))
    except Exception:
        pass


# Loaders (folder/API/JSON/DB)
def _load_docs_from_folder(folder: str) -> List[DocumentView]:
    if _ingestion_mod is None:
        raise RuntimeError("ingestion.py not found (required for --docs / run-folder).")

    exts = {".pdf", ".docx", ".txt"}
    paths: List[str] = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(root, f))

    if not paths:
        _eprint("No supported files found (.pdf/.docx/.txt).")
        return []

    docs: List[DocumentView] = []
    for p in paths:
        try:
            data = _ingestion_mod.extract_document(p)
            raw_text = data.get("raw_text") or ""
            if not raw_text.strip():
                continue
            norm = normalize_text(raw_text)

            # reproducible doc_id from absolute path and mtime
            abs_path = os.path.abspath(p)
            try:
                mtime_ns = os.stat(p).st_mtime_ns
            except Exception:
                mtime_ns = 0
            doc_id = hashlib.sha1(f"{abs_path}|{mtime_ns}".encode("utf-8", errors="ignore")).hexdigest()

            docs.append(build_document_view(doc_id=doc_id, text=norm, language=None, meta={"path": p}))
        except Exception as ex:
            _eprint(f"[skip] {p}: {ex}")
    return docs


def _docs_from_api_items(items: List[Dict[str, Any]]) -> List[DocumentView]:
    out: List[DocumentView] = []
    for it in items:
        try:
            doc_id = it.get("doc_id") or it.get("id")
            text = it.get("text") if "text" in it else it.get("raw_text")
            if not doc_id or not isinstance(text, str) or not text.strip():
                raise ValueError("Each item needs 'doc_id' (or 'id') and 'text' (or 'raw_text') string.")
            norm = normalize_text(text)
            out.append(build_document_view(doc_id=str(doc_id), text=norm, language=None, meta={"source": "api"}))
        except Exception as ex:
            _eprint(f"[api item skipped] {ex}")
    return out


def _load_docs_from_api(url: str, headers: Optional[Dict[str, str]]) -> List[DocumentView]:
    if requests is None:
        raise RuntimeError("The 'requests' package is required for --api-url (pip install requests).")
    r = requests.get(url, headers=headers or {})
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        items = data["items"]
    elif isinstance(data, list):
        items = data
    else:
        raise RuntimeError("API must return a JSON list or a dict with an 'items' list.")
    return _docs_from_api_items(items)


def _load_docs_from_json_file(path: str) -> List[DocumentView]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        items = data["items"]
    elif isinstance(data, list):
        items = data
    else:
        raise RuntimeError("JSON file must be a list or a dict with an 'items' list.")
    return _docs_from_api_items(items)


def _load_docs_from_db() -> List[DocumentView]:
    pairs = sqlite_store.get_docs_text(include_dirty=False)
    docs = [
        build_document_view(doc_id=did, text=(txt or ""), language=None, meta={})
        for (did, txt) in pairs
        if (txt or "").strip()
    ]
    return docs


# Reporting
def _generate_report(run_id: int, out_dir: str) -> Optional[str]:
    try:
        from src.reporting.report_builder import generate_report as _gen
    except Exception:
        try:
            from reporting.report_builder import generate_report as _gen
        except Exception:
            return None

    try:
        path = _gen(run_id, out_dir=out_dir, fmt="html")
        try:
            state_store.save_report_path(run_id, path)
        except Exception:
            pass
        return path
    except Exception:
        return None


# Summaries
def _pretty_rate(x: Any) -> str:
    try:
        return f"{float(x) * 100.0:.1f}%"
    except Exception:
        return "—"


def _summarize_from_traces(traces: List[Any]) -> Tuple[int, int]:
    exact = 0
    near = 0
    for tr in traces or []:
        try:
            t = tr.as_dict() if hasattr(tr, "as_dict") else tr
            if (t.get("final_label") or "").upper() == "DUPLICATE":
                if (t.get("dup_kind") or "").upper() == "EXACT":
                    exact += 1
                else:
                    near += 1
        except Exception:
            pass
    return exact, near


def _print_run_summary(res: Dict[str, Any]) -> None:
    rs = res.get("run_summary") or res.get("metrics_snapshot", {}).get("run") or {}
    traces = res.get("traces") or []
    exact, near = _summarize_from_traces(traces)
    pairs = rs.get("pairs_scored") or rs.get("total_pairs") or rs.get("pairs")

    print("\n=== Run Summary ===")
    print(f"Run ID: {res.get('run_id', '—')}")
    print(f"Pairs: {pairs if pairs is not None else '—'}")
    print(f"Exact duplicates: {exact}")
    print(f"Near duplicates:  {near}")
    try:
        print(f"Duplicates (total): {int(exact) + int(near)}")
    except Exception:
        pass
    if "uncertain" in rs:
        print(f"Uncertain: {rs.get('uncertain')}")
    if "consensus_rate" in rs:
        print(f"Consensus: {_pretty_rate(rs.get('consensus_rate'))}")
    if "escalations_rate" in rs or "escalations_pct" in rs:
        er = rs.get("escalations_rate", rs.get("escalations_pct"))
        print(f"Escalations: {_pretty_rate(er)}")

    clusters = res.get("clusters") or []
    if clusters:
        print(f"Clusters: {len(clusters)} (showing sizes of up to 5)")
        for i, cl in enumerate(clusters[:5], start=1):
            try:
                size = len(cl)
            except Exception:
                size = "?"
            print(f"  - Cluster {i}: {size} docs")


# Performance preset logic
def _compute_perf_overrides(n_docs: int, preset: str) -> Dict[str, Any]:
    """
    Returns an overrides dict that mirrors app.py's _apply_perf_preset().
    """
    preset_norm = (preset or "auto").strip().lower()
    Pphys = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 4
    RAM_GB = (psutil.virtual_memory().total or 0) / (1024**3)

    gpu_ok, VRAM_GB = _safe_nvml_snapshot(min_vram_gb=6.0)

    bucket = "small" if n_docs < 5_000 else ("medium" if n_docs < 50_000 else "large")

    # Defaults
    workers   = min(Pphys, 8)
    emb_batch = 64
    mh_perm   = 128
    cand_doc  = 2000
    cand_total= None
    model     = "fallback"

    # SimHash defaults
    sim_bits       = 192 if bucket == "large" else 128
    sim_mode       = "unigram"
    sim_wshingle   = 3
    sim_cngram     = 5
    sim_posbucket  = 0
    sim_minlen     = 2
    sim_norm_strict= False
    sim_strip_ids  = False
    sim_maxw       = 255

    if preset_norm.startswith("auto"):
        workers   = max(2, min(12, min(Pphys, 2 + (n_docs // 10_000))))
        if gpu_ok:
            emb_batch = 128 + (64 if VRAM_GB >= 12 else 0)
            model     = "all-MiniLM-L6-v2"
        else:
            emb_batch = 32 + (32 if RAM_GB >= 16 else 0) + (32 if RAM_GB >= 32 else 0)
            emb_batch = min(128, emb_batch)
            model     = "fallback"

        mh_perm    = 64 if bucket == "small" else (128 if bucket == "medium" else 192)
        cand_doc   = 1000 if bucket == "small" else (3000 if bucket == "medium" else 5000)
        cand_total = n_docs * cand_doc

        sim_bits     = 192 if bucket == "large" else 128
        sim_mode     = "unigram" if bucket == "small" else "wshingle"
        sim_wshingle = 3 if bucket != "large" else 4
        _set_threading_env(workers, torch_intra=max(1, Pphys // 2 if gpu_ok else workers), torch_inter=1)

    elif preset_norm.startswith("light"):
        workers   = min(Pphys, 4)
        emb_batch = 48 if RAM_GB >= 16 else 32
        mh_perm   = 64
        cand_doc  = 1000
        cand_total= min(100_000, 20 * n_docs)
        model     = "fallback"
        sim_bits  = 128
        sim_mode  = "unigram"
        _set_threading_env(workers, torch_intra=1, torch_inter=1)

    elif preset_norm.startswith("medium"):
        workers   = min(Pphys, 8)
        if gpu_ok:
            model = "all-MiniLM-L6-v2"; emb_batch = 128
        else:
            model = "fallback"; emb_batch = 64 if RAM_GB < 16 else 96
        mh_perm   = 128
        cand_doc  = 2000 if bucket == "small" else (3000 if bucket == "medium" else 4000)
        cand_total= n_docs * 2000
        sim_bits  = 128 if bucket != "large" else 192
        sim_mode  = "unigram"
        _set_threading_env(workers, torch_intra=max(1, Pphys // 2), torch_inter=1)

    elif preset_norm.startswith("high-throughput"):
        workers   = min(Pphys, 12)
        model     = "fallback"
        emb_batch = 64 if RAM_GB < 16 else (96 if RAM_GB < 32 else 128)
        mh_perm   = 128 if bucket != "large" else 192
        cand_doc  = 3000 if bucket == "small" else (4000 if bucket == "medium" else 5000)
        cand_total= n_docs * 3000
        sim_bits  = 128 if bucket != "large" else 192
        sim_mode  = "wshingle" if bucket != "small" else "unigram"
        sim_wshingle = 3 if bucket != "large" else 4
        _set_threading_env(workers, torch_intra=Pphys, torch_inter=1)

    elif preset_norm.startswith("high-recall"):
        workers   = min(Pphys, 6)
        model     = "all-MiniLM-L6-v2" if gpu_ok else "fallback"
        emb_batch = 128 if (gpu_ok and VRAM_GB < 12) else (192 if gpu_ok else 64)
        mh_perm   = 192 if bucket == "large" else 128
        cand_doc  = 4000 if bucket != "large" else 6000
        cand_total= n_docs * 4000
        sim_bits  = 192 if bucket == "large" else 128
        sim_mode  = "wshingle" if bucket != "small" else "unigram"
        sim_wshingle = 3 if bucket != "large" else 4
        _set_threading_env(workers, torch_intra=max(1, Pphys // 2), torch_inter=1)

    else:
        _set_threading_env(workers)

    return {
        "max_workers": workers,
        "emb_batch": emb_batch,
        "use_minhash": True,
        "minhash_num_perm": mh_perm,
        "cand_per_doc": cand_doc,
        "cand_total": cand_total,
        "model_name": model,
        "simhash_bits": sim_bits,
        "simhash_mode": sim_mode,
        "simhash_wshingle": sim_wshingle,
        "simhash_cngram": sim_cngram,
        "simhash_posbucket": sim_posbucket,
        "simhash_minlen": sim_minlen,
        "simhash_norm_strict": sim_norm_strict,
        "simhash_strip_ids": sim_strip_ids,
        "simhash_maxw": sim_maxw,
    }


def _merge_cli_perf_overrides(base: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Apply explicit CLI overrides on top of computed preset.
    """
    out = dict(base or {})

    def _maybe(name: str, cast, key: str):
        v = getattr(args, name, None)
        if v is None:
            return
        try:
            out[key] = cast(v)
        except Exception:
            pass

    _maybe("workers", int, "max_workers")
    _maybe("emb_batch", int, "emb_batch")
    _maybe("minhash_perm", int, "minhash_num_perm")
    _maybe("cand_per_doc", int, "cand_per_doc")
    if args.cand_total is not None:
        try:
            out["cand_total"] = int(args.cand_total)
        except Exception:
            out["cand_total"] = None
    if args.model_name:
        out["model_name"] = str(args.model_name)

    _maybe("simhash_bits", int, "simhash_bits")
    if args.simhash_mode:
        out["simhash_mode"] = str(args.simhash_mode)
    _maybe("simhash_wshingle", int, "simhash_wshingle")
    _maybe("simhash_cngram", int, "simhash_cngram")
    _maybe("simhash_posbucket", int, "simhash_posbucket")
    _maybe("simhash_minlen", int, "simhash_minlen")
    if args.simhash_norm_strict is not None:
        out["simhash_norm_strict"] = bool(args.simhash_norm_strict)
    if args.simhash_strip_ids is not None:
        out["simhash_strip_ids"] = bool(args.simhash_strip_ids)
    _maybe("simhash_maxw", int, "simhash_maxw")

    return out


def _apply_perf_overrides_to_config(cfg: PipelineConfig,
                                    overrides: Dict[str, Any],
                                    calibration_enabled: bool,
                                    lsh_threshold: Optional[float],
                                    shingle_size: Optional[int]) -> PipelineConfig:
    """
    Mutate a PipelineConfig (from profile_config) to inject performance overrides
    and mirror the GUI's _make_pipeline_config() behavior, including force_threshold.
    """
    # Candidate config
    per_doc_default = int(overrides.get("cand_per_doc", 2000))
    total_default = overrides.get("cand_total", None)
    try:
        shingle = int(shingle_size) if shingle_size is not None else 3
    except Exception:
        shingle = 3
    try:
        lsh_thr = float(lsh_threshold) if lsh_threshold is not None else 0.60
    except Exception:
        lsh_thr = 0.60

    cfg.candidates = CandidateConfig(
        use_lsh=True,
        shingle_size=shingle,
        num_perm=int(overrides.get("minhash_num_perm", 128)),
        lsh_threshold=lsh_thr,
        max_candidates_per_doc=per_doc_default,
        max_total_candidates=total_default,
    )

    # Embedding extras
    emb_ex = dict(getattr(cfg.embedding, "extras", {}) or {})
    if "max_workers" in overrides: emb_ex["max_workers"] = int(overrides["max_workers"])
    if "emb_batch" in overrides:  emb_ex["batch_size"]  = int(overrides["emb_batch"])
    if "model_name" in overrides and (overrides["model_name"] or "").strip():
        emb_ex["model_name"] = str(overrides["model_name"]).strip()
    if not calibration_enabled:
        emb_ex["force_threshold"] = True
    cfg.embedding.extras = emb_ex

    # MinHash extras
    min_ex = dict(getattr(cfg.minhash, "extras", {}) or {})
    if "use_minhash" in overrides:      min_ex["use_minhash"] = bool(overrides["use_minhash"])
    if "minhash_num_perm" in overrides: min_ex["num_perm"] = int(overrides["minhash_num_perm"])
    if not calibration_enabled:
        min_ex["force_threshold"] = True
    cfg.minhash.extras = min_ex

    # SimHash extras
    sim_ex = dict(getattr(cfg.simhash, "extras", {}) or {})
    if "simhash_bits" in overrides:          sim_ex["hash_bits"]        = int(overrides["simhash_bits"])
    if "simhash_mode" in overrides:          sim_ex["simhash_mode"]     = str(overrides["simhash_mode"])
    if "simhash_wshingle" in overrides:      sim_ex["shingle_size"]     = int(overrides["simhash_wshingle"])
    if "simhash_cngram" in overrides:        sim_ex["char_ngram"]       = int(overrides["simhash_cngram"])
    if "simhash_posbucket" in overrides:     sim_ex["pos_bucket"]       = int(overrides["simhash_posbucket"])
    if "simhash_minlen" in overrides:        sim_ex["min_token_len"]    = int(overrides["simhash_minlen"])
    if "simhash_norm_strict" in overrides:   sim_ex["normalize_strict"] = bool(overrides["simhash_norm_strict"])
    if "simhash_strip_ids" in overrides:     sim_ex["strip_dates_ids"]  = bool(overrides["simhash_strip_ids"])
    if "simhash_maxw" in overrides:          sim_ex["max_token_weight"] = int(overrides["simhash_maxw"])
    if not calibration_enabled:
        sim_ex["force_threshold"] = True
    cfg.simhash.extras = sim_ex

    return cfg


# Ingestion helpers (DB)
def _ingest_paths_batch(paths: Iterable[str]) -> Tuple[int, int]:
    """Batch ingest like app.py. Returns (ok, fail)."""
    if _ingestion_mod is None:
        raise RuntimeError("ingestion.py not found (required for ingest).")

    ps = list(paths)
    docs_batch = []
    mappings_batch = []
    ok = 0
    fail = 0

    for p in ps:
        try:
            data = _ingestion_mod.extract_document(p)
            raw_text = data.get("raw_text") or ""
            meta = data.get("metadata") or {}

            abs_path = os.path.abspath(p)
            try:
                mtime_ns = os.stat(p).st_mtime_ns
            except Exception:
                mtime_ns = 0

            doc_id = hashlib.sha1(f"{abs_path}|{mtime_ns}".encode("utf-8", errors="ignore")).hexdigest()
            norm = normalize_text(raw_text)

            doc_entry = (
                doc_id,
                raw_text,
                norm,
                json.dumps({
                    "language": meta.get("language"),
                    "filesize": meta.get("filesize") or 0,
                }),
            )
            mapping_entry = (doc_id, abs_path, mtime_ns)

            docs_batch.append(doc_entry)
            mappings_batch.append(mapping_entry)
            ok += 1
        except Exception:
            fail += 1

    if docs_batch:
        if hasattr(sqlite_store, "batch_upsert_documents"):
            sqlite_store.batch_upsert_documents(docs_batch)
        else:
            for (doc_id, raw, norm, meta_s) in docs_batch:
                # meta_s is JSON from json.dumps above
                sqlite_store.upsert_document(doc_id, raw, norm, json.loads(meta_s))

    if mappings_batch:
        if hasattr(sqlite_store, "batch_add_file_mappings"):
            sqlite_store.batch_add_file_mappings(mappings_batch)
        else:
            for (doc_id, abs_path, mtime_ns) in mappings_batch:
                sqlite_store.add_file_mapping(doc_id, abs_path, mtime_ns)


    return ok, fail


def _apply_cli_profile_preset(cfg: PipelineConfig, preset_key: str) -> PipelineConfig:
    """
    Mirror app.py's _apply_preset(): assign target precision and baseline
    decision thresholds into learner extras so the pipeline has concrete
    cutoffs even when calibration is disabled.
    """
    key = (preset_key or "balanced").strip().lower()

    if key.startswith("balanced"):
        target = 0.98
        thr_sim = 0.75
        thr_min = 0.75
        thr_emb = 0.988
    elif key.startswith("high"):
        target = 0.995
        thr_sim = 0.88
        thr_min = 0.88
        thr_emb = 0.994
    elif key.startswith("recall"):
        target = 0.95
        thr_sim = 0.60
        thr_min = 0.60
        thr_emb = 0.975
    else:
        target = 0.98
        thr_sim = 0.75
        thr_min = 0.75
        thr_emb = 0.988

    # Helper to safely write into extras
    def _bump_extras(lc: Any, kv: dict):
        if lc is None:
            return
        ex = dict(getattr(lc, "extras", {}) or {})
        ex.update(kv)
        lc.extras = ex
        try:
            lc.target_precision = target
        except Exception:
            pass

    # Push thresholds exactly like the GUI does:
    _bump_extras(getattr(cfg, "simhash", None),   {"decision_threshold": float(thr_sim)})
    _bump_extras(getattr(cfg, "minhash", None),   {"decision_threshold": float(thr_min)})
    _bump_extras(getattr(cfg, "embedding", None), {"cosine_threshold":   float(thr_emb)})

    return cfg



# Sub-commands
def cmd_run(args: argparse.Namespace) -> int:
    """
    Run the near-duplicate pipeline from CLI.

    - Loads sources (folder/json/api or DB)
    - Builds PipelineConfig from the selected profile
    - Applies GUI-equivalent profile thresholds (via _apply_cli_profile_preset)
    - Applies performance presets/overrides
    - Runs the pipeline and writes an HTML report
    """
    t0 = time.time()

    sqlite_store.init_db()
    state_store.ensure_state_schema()

    # Build input set
    docs: List[DocumentView] = []

    if getattr(args, "docs", None):
        _eprint(f"Loading from folder: {args.docs}")
        docs.extend(_load_docs_from_folder(args.docs))

    if getattr(args, "json_file", None):
        _eprint(f"Loading from JSON file: {args.json_file}")
        docs.extend(_load_docs_from_json_file(args.json_file))

    if getattr(args, "api_url", None):
        headers = None
        if getattr(args, "api_headers", None):
            try:
                headers = json.loads(args.api_headers)
            except Exception:
                _eprint("Warning: --api-headers is not valid JSON; ignoring.")
        _eprint(f"Loading from API: {args.api_url}")
        docs.extend(_load_docs_from_api(args.api_url, headers))

    # If explicitly requested, only use DB when no other source is provided
    if getattr(args, "use_db", False) and not (
        getattr(args, "docs", None)
        or getattr(args, "json_file", None)
        or getattr(args, "api_url", None)
    ):
        _eprint("Loading from existing DB (--use-db).")
        docs = _load_docs_from_db()

    if len(docs) < 2:
        _eprint("Need at least 2 documents. Use --docs / --api-url / --json-file, or --use-db.")
        return 2

    # Build PipelineConfig from profile preset
    preset_norm = (getattr(args, "preset", "balanced") or "balanced").strip().lower()
    mapping = {
        "balanced": "Balanced",
        "high": "High Precision",
        "recall": "Recall-Heavy",
    }
    label = mapping.get(preset_norm, "Balanced")
    cfg: PipelineConfig = profile_config(label)

    # Align with GUI: inject baseline thresholds (decision/cosine) + targets
    cfg = _apply_cli_profile_preset(cfg, preset_norm)

    # Honor calibration toggle
    cfg.disable_calibration = not bool(getattr(args, "calibration", False))

    # Performance preset and explicit overrides
    perf_over = _compute_perf_overrides(
        n_docs=len(docs),
        preset=getattr(args, "perf_preset", "auto")
    )
    perf_over = _merge_cli_perf_overrides(perf_over, args)

    # Candidate knobs exposed on CLI
    lsh_thr = getattr(args, "lsh_threshold", None)
    shingle_size = getattr(args, "shingle_size", None)

    cfg = _apply_perf_overrides_to_config(
        cfg=cfg,
        overrides=perf_over,
        calibration_enabled=bool(getattr(args, "calibration", False)),
        lsh_threshold=lsh_thr,
        shingle_size=shingle_size,
    )

    # If calibration is off, nudge learners to honor static thresholds
    if cfg.disable_calibration:
        for name in ("embedding", "minhash", "simhash"):
            lc = getattr(cfg, name, None)
            if lc is None:
                continue
            ex = dict(getattr(lc, "extras", {}) or {})
            ex.setdefault("force_threshold", True)
            lc.extras = ex

    # Self-learning and bootstrap
    if not isinstance(getattr(cfg, "bootstrap", None), BootstrapConfig):
        cfg.bootstrap = BootstrapConfig(max_pos_pairs=50_000, max_neg_pairs=50_000)

    epochs = getattr(args, "sl_epochs", None)
    epochs = 2 if epochs is None else int(epochs)
    if getattr(cfg, "self_learning", None) is None:
        cfg.self_learning = SelfLearningConfig(
            enabled=not bool(getattr(args, "no_self_training", False)),
            epochs=epochs
        )
    else:
        cfg.self_learning.enabled = not bool(getattr(args, "no_self_training", False))
        cfg.self_learning.epochs = epochs

    # Run
    _eprint(f"Running pipeline with preset: {label}  | calibration={'ON' if not cfg.disable_calibration else 'OFF'}")
    _eprint(f"Performance preset: {getattr(args, 'perf_preset', 'auto')}  | overrides: {json.dumps(perf_over)}")

    def _progress(done: int, total: int, phase: str):
        pct = (done / total * 100.0) if total else 0.0
        _eprint(f"{phase}… {done}/{total} ({pct:.1f}%)")

    result = run_intelligent_pipeline(
        docs,
        config=cfg,
        run_notes=f"{label} | perf={getattr(args, 'perf_preset', 'auto')}",
        progress_cb=_progress,
    )

    _print_run_summary(result)

    # Report
    run_id = int(result.get("run_id", 0) or 0)
    report_dir = getattr(args, "report_dir", "reports") or "reports"
    try:
        os.makedirs(report_dir, exist_ok=True)
    except Exception:
        pass

    path = _generate_report(run_id, out_dir=report_dir) if run_id else None
    if path:
        print(f"\nReport written to: {path}")
    else:
        print("\nReport builder not available; skipped report generation.")

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s")
    return 0



def cmd_run_db(args: argparse.Namespace) -> int:
    """Convenience wrapper: run directly from the DB."""
    args = argparse.Namespace(**vars(args))
    args.use_db = True
    args.docs = None
    args.json_file = None
    args.api_url = None
    return cmd_run(args)


def cmd_run_folder(args: argparse.Namespace) -> int:
    """Convenience wrapper: run directly from a folder of docs."""
    if not args.folder:
        _eprint("run-folder requires a folder path.")
        return 2

    args = argparse.Namespace(**vars(args))

    args.docs = args.folder
    args.use_db = False

    if not hasattr(args, "json_file"):
        args.json_file = None
    if not hasattr(args, "api_url"):
        args.api_url = None
    if not hasattr(args, "api_headers"):
        args.api_headers = None

    if not hasattr(args, "lsh_threshold"):
        args.lsh_threshold = None
    if not hasattr(args, "shingle_size"):
        args.shingle_size = None

    return cmd_run(args)



def cmd_ingest(args: argparse.Namespace) -> int:
    """Ingest a folder (pdf/docx/txt) into the SQLite DB only (no run)."""
    if _ingestion_mod is None:
        _eprint("ingestion.py not found.")
        return 2

    sqlite_store.init_db()
    folder = args.folder

    exts = {".pdf", ".docx", ".txt"}
    paths: List[str] = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(root, f))

    if not paths:
        _eprint("No supported files found (.pdf/.docx/.txt).")
        return 2

    _eprint(f"Ingesting {len(paths)} files …")
    ok, fail = _ingest_paths_batch(paths)
    print(f"Done. ok={ok}, fail={fail}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List documents currently in the DB."""
    sqlite_store.init_db()
    try:
        rows = sqlite_store.get_all_document_files()
    except Exception as e:
        _eprint(f"DB error: {e}")
        return 1

    if not rows:
        print("No documents.")
        return 0

    uniq = len({r.get("doc_id") for r in rows})
    print(f"{len(rows)} files across {uniq} docs\n")
    for r in rows[: min(len(rows), args.max)]:
        doc_id = r.get("doc_id", "")
        lang = r.get("language") or "—"
        size_kb = int((r.get("filesize") or 0) / 1024)
        fname = r.get("filename") or "—"
        fpath = r.get("filepath") or ""
        print(f"{doc_id[:12]}…  {fname}  ({lang}, {size_kb} KB)  {fpath}")
    if len(rows) > args.max:
        print(f"\n… and {len(rows) - args.max} more.")
    return 0


def cmd_wipe(args: argparse.Namespace) -> int:
    """Delete ALL documents from the DB."""
    if not args.yes:
        _eprint("This will DELETE ALL documents from the DB.")
        _eprint("Re-run with --yes to confirm.")
        return 2

    sqlite_store.init_db()
    try:
        rows = sqlite_store.get_all_document_files()
        doc_ids = sorted({r.get("doc_id") for r in rows if r.get("doc_id")})
        if doc_ids:
            sqlite_store.delete_documents(doc_ids)
    except Exception as e:
        _eprint(f"Delete failed: {e}")
        return 1

    print("All documents deleted.")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Generate (or re-generate) an HTML report for an existing run_id."""
    sqlite_store.init_db()
    state_store.ensure_state_schema()

    run_id = int(args.run_id)
    os.makedirs(args.out_dir, exist_ok=True)
    path = _generate_report(run_id, out_dir=args.out_dir)
    if path:
        print(f"Report written to: {path}")
        return 0
    else:
        _eprint("Report builder not available or failed.")
        return 1


def cmd_import_csv(args: argparse.Namespace) -> int:
    """
    Import up to N rows from a CSV with a 'text' column into the DB.
    Mimics app.import_rows_from_csv but parameterized.
    """
    csv_path = args.path
    limit = int(args.limit)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        _eprint(f"Error reading CSV: {e}")
        return 1

    if "text" not in df.columns:
        _eprint("CSV must have a 'text' column.")
        _eprint(f"Found columns: {df.columns.tolist()}")
        return 2

    if len(df) > limit:
        df = df.sample(n=limit, random_state=1234)

    sqlite_store.init_db()

    fake_path = os.path.abspath(csv_path)
    for i, row in df.iterrows():
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        doc_id = hashlib.sha1(f"csv_row_{i}_{time.time_ns()}".encode()).hexdigest()
        norm = normalize_text(text)
        meta = {"language": "en", "filesize": len(text.encode("utf-8"))}

        sqlite_store.upsert_document(
            doc_id=doc_id,
            raw_text=text,
            normalized_text=norm,
            meta=meta,
        )
        mtime_ns = time.time_ns()
        sqlite_store.add_file_mapping(doc_id, fake_path, mtime_ns)

    print(f"Imported up to {limit} CSV rows into database.")
    return 0


# CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="duplicate-finder",
        description="Near-duplicate detection CLI (DB tools + pipeline runner).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # run
    pr = sub.add_parser("run", help="Run the pipeline on sources (or --use-db).")
    pr.add_argument("--preset", choices=["balanced", "high", "recall"], default="balanced",
                    help="Profile preset to use (balanced | high | recall).")
    src = pr.add_argument_group("Input sources (in-memory unless --use-db)")
    src.add_argument("--docs", metavar="FOLDER", help="Folder of .pdf/.docx/.txt (recursive).")
    src.add_argument("--api-url", metavar="URL", help="HTTP endpoint returning JSON list or {items:[...]}.")
    src.add_argument("--api-headers", metavar="JSON", help="Optional JSON headers for --api-url.")
    src.add_argument("--json-file", metavar="PATH", help="Local JSON file (list or {items:[...]}) of {doc_id,text} rows.")
    pr.add_argument("--use-db", action="store_true",
                    help="Load docs from existing SQLite DB (ignored if other sources provided).")
    pr.add_argument("--calibration", action="store_true", default=False,
                    help="Enable calibration (OFF by default).")
    pr.add_argument("--report-dir", default="reports", help="Directory to write the HTML report (default: reports).")

    # Performance presets and overrides
    perf = pr.add_argument_group("Performance profile (like app.py)")
    perf.add_argument("--perf-preset",
                      choices=["auto", "high-end", "medium", "light", "high-throughput", "high-recall", "custom"],
                      default="auto",
                      help="Performance preset to apply before overrides.")
    perf.add_argument("--workers", type=int, help="Max CPU workers.")
    perf.add_argument("--emb-batch", type=int, help="Embedding batch size.")
    perf.add_argument("--minhash-perm", type=int, help="MinHash permutations.")
    perf.add_argument("--cand-per-doc", type=int, help="Max candidates per doc.")
    perf.add_argument("--cand-total", type=int, help="Max total candidates.")
    perf.add_argument("--model-name", help="Embedding model name (e.g., all-MiniLM-L6-v2 | fallback).")

    sim = pr.add_argument_group("SimHash overrides")
    sim.add_argument("--simhash-bits", type=int, help="SimHash bits (e.g., 128, 192).")
    sim.add_argument("--simhash-mode", choices=["unigram", "wshingle", "cngram"], help="SimHash token mode.")
    sim.add_argument("--simhash-wshingle", type=int, help="Word shingle size (mode=wshingle).")
    sim.add_argument("--simhash-cngram", type=int, help="Char n-gram (mode=cngram).")
    sim.add_argument("--simhash-posbucket", type=int, help="Positional bucket.")
    sim.add_argument("--simhash-minlen", type=int, help="Min token length.")
    sim.add_argument("--simhash-norm-strict", action="store_true", help="Normalize strictly for SimHash.")
    sim.add_argument("--no-simhash-norm-strict", dest="simhash_norm_strict", action="store_false")
    sim.set_defaults(simhash_norm_strict=None)
    sim.add_argument("--simhash-strip-ids", action="store_true", help="Strip dates/IDs for SimHash.")
    sim.add_argument("--no-simhash-strip-ids", dest="simhash_strip_ids", action="store_false")
    sim.set_defaults(simhash_strip_ids=None)
    sim.add_argument("--simhash-maxw", type=int, help="Max token weight for SimHash.")

    # Candidate generator knobs
    cand = pr.add_argument_group("Candidate generation")
    cand.add_argument("--lsh-threshold", type=float, help="LSH threshold (default 0.60).")
    cand.add_argument("--shingle-size", type=int, help="Shingle size (default 3).")

    # Self-training
    sl = pr.add_argument_group("Self-training")
    sl.add_argument("--sl-epochs", type=int, default=2, help="Self-learning epochs (default: 2).")
    sl.add_argument("--no-self-training", action="store_true", help="Disable self-training.")

    pr.set_defaults(func=cmd_run)

    # run-db
    prdb = sub.add_parser("run-db", help="Run the pipeline using documents already in the DB.")
    prdb.add_argument("--preset", choices=["balanced", "high", "recall"], default="balanced",
                      help="Profile preset to use (balanced | high | recall).")
    prdb.add_argument("--calibration", action="store_true", default=False,
                      help="Enable calibration (OFF by default).")
    prdb.add_argument("--report-dir", default="reports", help="Directory to write the HTML report (default: reports).")
    # Same perf groups for parity
    perf_db = prdb.add_argument_group("Performance profile (like app.py)")
    perf_db.add_argument("--perf-preset",
                         choices=["auto", "high-end", "medium", "light", "high-throughput", "high-recall", "custom"],
                         default="auto")
    perf_db.add_argument("--workers", type=int)
    perf_db.add_argument("--emb-batch", type=int)
    perf_db.add_argument("--minhash-perm", type=int)
    perf_db.add_argument("--cand-per-doc", type=int)
    perf_db.add_argument("--cand-total", type=int)
    perf_db.add_argument("--model-name")
    sim_db = prdb.add_argument_group("SimHash overrides")
    sim_db.add_argument("--simhash-bits", type=int)
    sim_db.add_argument("--simhash-mode", choices=["unigram", "wshingle", "cngram"])
    sim_db.add_argument("--simhash-wshingle", type=int)
    sim_db.add_argument("--simhash-cngram", type=int)
    sim_db.add_argument("--simhash-posbucket", type=int)
    sim_db.add_argument("--simhash-minlen", type=int)
    sim_db.add_argument("--simhash-norm-strict", action="store_true")
    sim_db.add_argument("--no-simhash-norm-strict", dest="simhash_norm_strict", action="store_false")
    prdb.set_defaults(simhash_norm_strict=None)
    sim_db.add_argument("--simhash-strip-ids", action="store_true")
    sim_db.add_argument("--no-simhash-strip-ids", dest="simhash_strip_ids", action="store_false")
    prdb.set_defaults(simhash_strip_ids=None)
    sim_db.add_argument("--simhash-maxw", type=int)
    cand_db = prdb.add_argument_group("Candidate generation")
    cand_db.add_argument("--lsh-threshold", type=float)
    cand_db.add_argument("--shingle-size", type=int)
    sl_db = prdb.add_argument_group("Self-training")
    sl_db.add_argument("--sl-epochs", type=int, default=2)
    sl_db.add_argument("--no-self-training", action="store_true")
    prdb.set_defaults(func=cmd_run_db)

    # run-folder
    prf = sub.add_parser("run-folder", help="Run the pipeline on a folder of .pdf/.docx/.txt.")
    prf.add_argument("folder", help="Folder of docs (recursive).")
    prf.add_argument("--preset", choices=["balanced", "high", "recall"], default="balanced",
                     help="Profile preset to use (balanced | high | recall).")
    prf.add_argument("--calibration", action="store_true", default=False,
                     help="Enable calibration (OFF by default).")
    prf.add_argument("--report-dir", default="reports", help="Directory to write the HTML report (default: reports).")
    # Same perf groups for parity
    perf_f = prf.add_argument_group("Performance profile (like app.py)")
    perf_f.add_argument("--perf-preset",
                        choices=["auto", "high-end", "medium", "light", "high-throughput", "high-recall", "custom"],
                        default="auto")
    perf_f.add_argument("--workers", type=int)
    perf_f.add_argument("--emb-batch", type=int)
    perf_f.add_argument("--minhash-perm", type=int)
    perf_f.add_argument("--cand-per-doc", type=int)
    perf_f.add_argument("--cand-total", type=int)
    perf_f.add_argument("--model-name")
    sim_f = prf.add_argument_group("SimHash overrides")
    sim_f.add_argument("--simhash-bits", type=int)
    sim_f.add_argument("--simhash-mode", choices=["unigram", "wshingle", "cngram"])
    sim_f.add_argument("--simhash-wshingle", type=int)
    sim_f.add_argument("--simhash-cngram", type=int)
    sim_f.add_argument("--simhash-posbucket", type=int)
    sim_f.add_argument("--simhash-minlen", type=int)
    sim_f.add_argument("--simhash-norm-strict", action="store_true")
    sim_f.add_argument("--no-simhash-norm-strict", dest="simhash_norm_strict", action="store_false")
    prf.set_defaults(simhash_norm_strict=None)
    sim_f.add_argument("--simhash-strip-ids", action="store_true")
    sim_f.add_argument("--no-simhash-strip-ids", dest="simhash_strip_ids", action="store_false")
    prf.set_defaults(simhash_strip_ids=None)
    sim_f.add_argument("--simhash-maxw", type=int)
    cand_f = prf.add_argument_group("Candidate generation")
    cand_f.add_argument("--lsh-threshold", type=float)
    cand_f.add_argument("--shingle-size", type=int)
    sl_f = prf.add_argument_group("Self-training")
    sl_f.add_argument("--sl-epochs", type=int, default=2)
    sl_f.add_argument("--no-self-training", action="store_true")
    prf.set_defaults(func=cmd_run_folder)

    # ingest
    pi = sub.add_parser("ingest", help="Add an entire folder (pdf/docx/txt) into the SQLite DB.")
    pi.add_argument("folder", help="Folder to ingest (recursively).")
    pi.set_defaults(func=cmd_ingest)

    # list
    pl = sub.add_parser("list", help="List documents currently in the DB.")
    pl.add_argument("--max", type=int, default=50, help="Show at most this many rows (default: 50).")
    pl.set_defaults(func=cmd_list)

    # wipe
    pw = sub.add_parser("wipe", help="Delete ALL documents from the DB (dangerous).")
    pw.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    pw.set_defaults(func=cmd_wipe)

    # report
    prt = sub.add_parser("report", help="Generate an HTML report for an existing run_id.")
    prt.add_argument("run_id", type=int, help="Run ID to render a report for.")
    prt.add_argument("--out-dir", default="reports", help="Directory to write the HTML report (default: reports).")
    prt.set_defaults(func=cmd_report)

    # import-csv
    pcsv = sub.add_parser("import-csv", help="Import up to N rows from a CSV with a 'text' column into the DB.")
    pcsv.add_argument("path", help="CSV path")
    pcsv.add_argument("--limit", type=int, default=2000, help="Max rows to import (default: 2000).")
    pcsv.set_defaults(func=cmd_import_csv)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    sqlite_store.init_db()
    state_store.ensure_state_schema()

    try:
        return args.func(args)
    except KeyboardInterrupt:
        _eprint("\nInterrupted.")
        return 130
    except Exception as ex:
        _eprint(f"Error: {ex}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
