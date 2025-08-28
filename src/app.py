# src/app.py
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

from .pipelines.ingestion import extract_document
from .pipelines.normalization import normalize_text
from .pipelines.near_duplicate import detect_near_duplicates, Document

SUPPORTED = {".pdf", ".docx", ".txt"}

@dataclass
class ProcResult:
    doc_id: str
    filepath: str
    raw_text: str
    normalized_text: str
    metadata: Dict[str, Any]

def iter_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            yield p

def process_one(path: Path) -> ProcResult:
    doc = extract_document(str(path))
    normalized = normalize_text(doc["raw_text"])
    doc_id = doc["metadata"]["hash"]
    return ProcResult(
        doc_id=doc_id,
        filepath=str(path),
        raw_text=doc["raw_text"],
        normalized_text=normalized,
        metadata=doc["metadata"],
    )

def write_outputs(results: List[ProcResult], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"

    with index_path.open("w", encoding="utf-8") as index_f:
        for r in results:
            single_out = out_dir / (Path(r.filepath).stem + ".json")
            with single_out.open("w", encoding="utf-8") as sf:
                json.dump({
                    "raw_text": r.raw_text,
                    "normalized_text": r.normalized_text,
                    "metadata": r.metadata,
                }, sf, ensure_ascii=False, indent=2)

            index_f.write(json.dumps({
                "doc_id": r.doc_id,
                "filepath": r.filepath,
                "hash": r.metadata["hash"],
                "language": r.metadata.get("language"),
                "normalized_text": r.normalized_text,
            }, ensure_ascii=False) + "\n")

def load_docs_for_dedupe(index_file: Path) -> List[Document]:
    docs: List[Document] = []
    with index_file.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = (rec.get("normalized_text") or "").strip()
            if not text:
                continue
            docs.append(Document(
                doc_id=rec.get("doc_id") or rec.get("hash") or rec.get("filepath"),
                normalized_text=text
            ))
    return docs

def choose_canonicals(clusters: List[List[str]], index_file: Path) -> Dict[str, Dict[str, Any]]:
    # Map doc_id -> record for quick lookup
    by_id: Dict[str, Dict[str, Any]] = {}
    with index_file.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_id[rec["doc_id"]] = rec

    canonicals: Dict[str, Dict[str, Any]] = {}
    for cluster in clusters:
        # choose longest normalized_text as canonical
        best = None
        best_len = -1
        for did in cluster:
            rec = by_id.get(did)
            if not rec:
                continue
            L = len(rec.get("normalized_text") or "")
            if L > best_len:
                best = rec
                best_len = L
        if best:
            canonicals[best["doc_id"]] = best
    return canonicals

def main():
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    out_dir = project_root / "data" / "processed"
    index_path = out_dir / "index.jsonl"

    if not raw_dir.exists():
        raise SystemExit(f"Input folder not found: {raw_dir}")

    # Step 1: Ingest + Normalize
    results: List[ProcResult] = []
    files = list(iter_files(raw_dir))
    if not files:
        print(f"No input files found under {raw_dir}. Supported: {', '.join(SUPPORTED)}")
        return

    print(f"Processing {len(files)} documents...")
    for f in files:
        try:
            r = process_one(f)
            results.append(r)
            print(f"✓ {f}")
        except Exception as e:
            print(f"✗ {f}  ({e})")

    write_outputs(results, out_dir)
    print(f"\nSaved per-file JSON to: {out_dir}\\*.json")
    print(f"Saved combined index to: {index_path}")

    # Step 2: Near-duplicate detection
    print("\nBuilding near-duplicate clusters (SimHash, MinHash+LSH, Embeddings)...")
    docs = load_docs_for_dedupe(index_path)
    if not docs:
        print("No documents with non-empty normalized text to cluster.")
        return

    clusters = detect_near_duplicates(docs)

    # Pretty print clusters with filenames
    # Build a quick lookup: doc_id -> filepath
    id_to_path = {r.doc_id: r.filepath for r in results}
    print("\nDuplicate clusters:")
    kept = 0
    for i, cluster in enumerate(clusters, 1):
        if len(cluster) == 1:
            continue
        print(f"\nCluster {i} ({len(cluster)} docs):")
        for did in cluster:
            print(f"  - {did} :: {id_to_path.get(did, '(unknown path)')}")

    # Canonicals
    canonicals = choose_canonicals(clusters, index_path)
    if canonicals:
        print("\nCanonical representatives:")
        for did, rec in canonicals.items():
            print(f"  - {did} :: {rec.get('filepath')}")

if __name__ == "__main__":
    main()
