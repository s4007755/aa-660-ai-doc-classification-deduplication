# src/pipelines/ingestion.py
import os
import hashlib
from typing import Dict, Any, Optional

import docx
from PyPDF2 import PdfReader
try:
    from langdetect import detect
except Exception:
    detect = None

from src.features.text_preproc import filename_tokens

# compute SHA256 of the file contents
def compute_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

# extract visible text from PDF
def extract_text_pdf(file_path: str) -> str:
    text = []
    reader = PdfReader(file_path)
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)

# extract visible text from DOCX
def extract_text_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs)

# extract visible text from TXT
def extract_text_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# internal: try to detect language safely
def _safe_detect_language(text: str) -> Optional[str]:
    try:
        if detect is None or not text or not text.strip():
            return None
        return detect(text)
    except Exception:
        return None

# internal: pull lightweight metadata if present
def _pdf_core_meta(file_path: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    try:
        reader = PdfReader(file_path)
        info = getattr(reader, "metadata", None) or {}
        def _get(key: str) -> Optional[str]:
            for k, v in info.items():
                if isinstance(k, str) and k.strip("/").lower() == key:
                    return str(v)
            return None
        meta["title"] = _get("title")
        meta["author"] = _get("author")
        meta["created"] = _get("creationdate") or _get("created")
    except Exception:
        pass
    return meta

def _docx_core_meta(file_path: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    try:
        doc = docx.Document(file_path)
        cp = doc.core_properties
        meta["title"] = cp.title or None
        meta["author"] = cp.author or None
        meta["created"] = cp.created.isoformat() if getattr(cp, "created", None) else None
    except Exception:
        pass
    return meta

# main: extract text and metadata
def extract_document(file_path: str) -> Dict[str, Any]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        raw_text = extract_text_pdf(file_path)
        core_meta = _pdf_core_meta(file_path)
        mime = "application/pdf"
    elif ext == ".docx":
        raw_text = extract_text_docx(file_path)
        core_meta = _docx_core_meta(file_path)
        mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif ext == ".txt":
        raw_text = extract_text_txt(file_path)
        core_meta = {}
        mime = "text/plain"
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    lang = _safe_detect_language(raw_text)
    fname = os.path.basename(file_path)

    metadata: Dict[str, Any] = {
        "filename": fname,
        "filename_tokens": filename_tokens(fname),
        "filepath": file_path,
        "hash": compute_file_hash(file_path),
        "filesize": os.path.getsize(file_path),
        "language": lang,
        "mime_type": mime,
        "title": core_meta.get("title"),
        "author": core_meta.get("author"),
        "created": core_meta.get("created"),
    }

    return {"raw_text": raw_text, "metadata": metadata}
