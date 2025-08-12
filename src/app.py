import os
import json
from pathlib import Path
from .pipelines.ingestion import extract_document
from .pipelines.normalization import normalize_text

SUPPORTED = {".pdf", ".docx", ".txt"}

def iter_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            yield p

def process_one(path: Path) -> dict:
    doc = extract_document(str(path))
    normalized = normalize_text(doc["raw_text"])
    return {
        "raw_text": doc["raw_text"],
        "normalized_text": normalized,
        "metadata": doc["metadata"],
    }

def main():
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"

    if not raw_dir.exists():
        raise SystemExit(f"Input folder not found: {raw_dir}")

    count = 0
    with index_path.open("w", encoding="utf-8") as index_f:
        for f in iter_files(raw_dir):
            result = process_one(f)
            # write a per-file JSON artifact (handy for debugging)
            single_out = out_dir / (f.stem + ".json")
            with single_out.open("w", encoding="utf-8") as sf:
                json.dump(result, sf, ensure_ascii=False, indent=2)

            # also append to a single JSONL index for later pipelines
            index_f.write(json.dumps({
                "filepath": str(f),
                "hash": result["metadata"]["hash"],
                "language": result["metadata"]["language"],
                "normalized_text": result["normalized_text"],
            }, ensure_ascii=False) + "\n")
            count += 1

    print(f"Processed {count} document(s).")
    print(f"- Per-file JSON: {out_dir}/*.json")
    print(f"- Combined index: {index_path}")

if __name__ == "__main__":
    main()
