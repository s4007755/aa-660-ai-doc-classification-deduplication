import os
import hashlib
import docx
from PyPDF2 import PdfReader
from langdetect import detect
from typing import Dict, Any

"""Compute SHA256 hash of file."""
def compute_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

"""Extract text from PDF."""
def extract_text_pdf(file_path: str) -> str:
    text = []
    reader = PdfReader(file_path)
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)

"""Extract text from DOCX."""
def extract_text_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs)

"""Read plain text files."""
def extract_text_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

"""Extract text and metadata from a file."""
def extract_document(file_path: str) -> Dict[str, Any]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        raw_text = extract_text_pdf(file_path)
    elif ext == ".docx":
        raw_text = extract_text_docx(file_path)
    elif ext == ".txt":
        raw_text = extract_text_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    metadata = {
        "filename": os.path.basename(file_path),
        "filepath": file_path,
        "hash": compute_file_hash(file_path),
        "filesize": os.path.getsize(file_path),
        "language": detect(raw_text) if raw_text.strip() else None,
    }

    return {
        "raw_text": raw_text,
        "metadata": metadata
    }
