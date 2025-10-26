# src/cli_nd.py
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from src.features.text_preproc import normalize_text, build_document_view
from src.learners.base import DocumentView
from src.gui.config_schemas import profile_config, list_profiles
from src.pipelines.near_duplicate import run_intelligent_pipeline, PipelineConfig

# DB pieces
from src.storage import sqlite_store
from src.persistence import state_store

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

# Source loaders
def _load_docs_from_folder(folder: str) -> List[DocumentView]:

    if _ingestion_mod is None:
        raise RuntimeError("ingestion.py not found (needed for --docs).")

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

            # Build a reproducible doc_id from absolute path and mtime_ns
            import hashlib
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


# Printing
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


def _print_summary(res: Dict[str, Any]) -> None:
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


# Runner
def run_cli(
    preset: str,
    docs_folder: Optional[str],
    api_url: Optional[str],
    api_headers: Optional[str],
    json_file: Optional[str],
    report_dir: str,
    use_db: bool,
) -> int:
    t0 = time.time()

    sqlite_store.init_db()
    state_store.ensure_state_schema()

    # Build the in memory document set from requested sources
    docs: List[DocumentView] = []

    if docs_folder:
        _eprint(f"Loading from folder: {docs_folder}")
        docs.extend(_load_docs_from_folder(docs_folder))

    if json_file:
        _eprint(f"Loading from JSON file: {json_file}")
        docs.extend(_load_docs_from_json_file(json_file))

    if api_url:
        headers = None
        if api_headers:
            try:
                headers = json.loads(api_headers)
            except Exception:
                _eprint("Warning: --api-headers is not valid JSON; ignoring.")
        _eprint(f"Loading from API: {api_url}")
        docs.extend(_load_docs_from_api(api_url, headers))

    if use_db and not (docs_folder or json_file or api_url):
        _eprint("Loading from existing DB (--use-db).")
        docs = _load_docs_from_db()

    if len(docs) < 2:
        _eprint("Need at least 2 documents. Provide --docs / --api-url / --json-file, "
                "or pass --use-db to run on documents already in the DB.")
        return 2

    # Config from preset
    preset_norm = preset.strip().lower()
    mapping = {
        "balanced": "Balanced",
        "high": "High Precision",
        "high precision": "High Precision",
        "recall": "Recall-Heavy",
        "recall-heavy": "Recall-Heavy",
    }
    label = mapping.get(preset_norm, "Balanced")
    cfg: PipelineConfig = profile_config(label)

    _eprint(f"Running pipeline with preset: {label}")
    def _progress(done: int, total: int, phase: str):
        pct = (done / total * 100.0) if total else 0.0
        _eprint(f"{phase}… {done}/{total} ({pct:.1f}%)")

    result = run_intelligent_pipeline(
        docs,
        config=cfg,
        run_notes=label,
        progress_cb=_progress,
    )

    # Summary
    _print_summary(result)

    # Report
    run_id = int(result.get("run_id", 0) or 0)
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


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="duplicate-finder-cli",
        description="Run the near-duplicate pipeline from the command line (docs/API/JSON, no DB by default)."
    )
    p.add_argument(
        "--preset",
        choices=["balanced", "high", "recall"],
        default="balanced",
        help="Preset to use (balanced | high | recall).",
    )
    src = p.add_argument_group("Input sources (in-memory; NOT written to DB)")
    src.add_argument("--docs", metavar="FOLDER", help="Folder of .pdf/.docx/.txt (recursive).")
    src.add_argument("--api-url", metavar="URL", help="HTTP endpoint returning JSON list or {items:[...]}.")
    src.add_argument("--api-headers", metavar="JSON", help="Optional JSON headers (e.g. '{\"Authorization\":\"Bearer ...\"}').")
    src.add_argument("--json-file", metavar="PATH", help="Local JSON file with a list or {items:[...]} of {doc_id,text} rows.")
    p.add_argument("--report-dir", default="reports", help="Directory to write the HTML report (default: reports).")

    p.add_argument("--use-db", action="store_true",
                   help="Load docs from existing SQLite DB (ignored if --docs/--api-url/--json-file are provided).")

    args = p.parse_args(argv)

    if not any([args.docs, args.api_url, args.json_file, args.use_db]):
        _eprint("No input source provided (and --use-db not set).")
        return 2

    try:
        return run_cli(
            preset=args.preset,
            docs_folder=args.docs,
            api_url=args.api_url,
            api_headers=args.api_headers,
            json_file=args.json_file,
            report_dir=args.report_dir,
            use_db=args.use_db,
        )
    except KeyboardInterrupt:
        _eprint("\nInterrupted.")
        return 130
    except Exception as ex:
        _eprint(f"Error: {ex}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
