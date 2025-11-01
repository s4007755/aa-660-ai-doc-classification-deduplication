from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.learners.base import DocumentView, CorpusStats

# Minimal default stopwords
_DEFAULT_SW = {
    "the","a","an","and","or","for","of","to","in","on","at","by","with","from","as",
    "is","are","was","were","be","been","it","this","that","these","those","you","your",
}

# Split/cleanup regexes
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# quick header/footer/page-number patterns
_RE_PAGE_NUM = re.compile(r"\bpage\s+\d+(\s+of\s+\d+)?\b", re.IGNORECASE)
_RE_HR = re.compile(r"\n-{2,}\n")
_RE_SOFT_HYPHEN = re.compile(r"-\n")
_RE_MULTI_SPACE = re.compile(r"\s+")

# digits/IDs/dates for strict cleanup
_RE_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_RE_LONG_ID = re.compile(r"\b\d{6,}\b")

# More robust normalizers
_RE_SOFT_HYPHEN_CHAR = re.compile(u"\u00AD") # discretionary hyphen char
_RE_NBSP = re.compile(u"\u00A0") # non-breaking space
# Hyphenation across EOL:
_RE_LINE_HYPHEN = re.compile(r"(?<=\w)-\s*(?:\r?\n|\r)\s*(?=\w)")
# Collapse multiple blank lines
_RE_MULTINL = re.compile(r"(?:\r?\n){2,}")
# Strip common bullet prefixes at line starts
_RE_BULLET_PREFIX = re.compile(r"^[\u2022\u2023\u25E6\-\*\·]\s+", re.MULTILINE)

# Export surface
__all__ = [
    "normalize_text",
    "tokenize_words",
    "sentence_split",
    "filename_tokens",
    "build_document_view",
    "compute_corpus_stats",
    "drop_repeating_lines",
    "content_hash",
]

# Repeating header/footer removal
def drop_repeating_lines(text: str, *, min_count: int = 3, max_len: int = 120) -> str:
    """
    Remove lines that repeat many times.
    Keep it conservative by requiring a minimum count and a maximum line length.
    """
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines()]
    counts = Counter(ln for ln in lines if ln)
    repetitive = {ln for ln, c in counts.items() if c >= min_count and len(ln) <= max_len}
    if not repetitive:
        return "\n".join(lines)
    return "\n".join(ln for ln in lines if ln not in repetitive)

# Normalize unicode, lower, remove boilerplate
def normalize_text(
    text: str,
    *,
    strict: bool = True,
    strip_dates_ids: bool = True,
) -> str:
    """
    Aggressive, dedup friendly normalization that aims to make
    DOCX to PDF conversions converge to the same representation.
    - Unicode normalization
    - Remove discretionary hyphens and hyphenation at EOL
    - Normalize NBSPs, quotes and dashes
    - Strip obvious page labels and horizontal rules
    - Optionally strip dates/long IDs
    - Lowercase and collapse whitespace
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)

    # Unicode artifacts
    t = _RE_SOFT_HYPHEN_CHAR.sub("", t) # discretionary hyphen char
    t = _RE_NBSP.sub(" ", t) # NBSP -> space
    t = t.replace("\u2013", "-").replace("\u2014", "-")  # en/em dashes -> hyphen
    t = t.replace("\u2018", "'").replace("\u2019", "'")  # smart single quotes
    t = t.replace("\u201C", '"').replace("\u201D", '"')  # smart double quotes

    # Legacy removal and typical boilerplate
    t = _RE_PAGE_NUM.sub(" ", t)
    t = _RE_HR.sub("\n", t)

    # Join hyphenated line wraps (PDFs)
    t = _RE_LINE_HYPHEN.sub("", t)
    # Handle literal "-\n" pattern
    t = _RE_SOFT_HYPHEN.sub("", t)

    # Drop bullet prefixes per line (helps PDF outlines)
    t = _RE_BULLET_PREFIX.sub("", t)

    # Normalize newline noise before punctuation stripping
    t = _RE_MULTINL.sub("\n", t)

    if strict:
        # Drop most punctuation to stabilize dedup signatures
        t = re.sub(r"[^\w\s]", " ", t)

    if strip_dates_ids:
        t = _RE_DATE.sub(" ", t)
        t = _RE_LONG_ID.sub(" ", t)

    # Whitespace and lowercasing at the end
    t = _RE_MULTI_SPACE.sub(" ", t).strip().lower()
    return t


# Simple word tokenizer
def tokenize_words(
    text: str,
    *,
    min_len: int = 2,
    remove_stopwords: bool = True,
    stopwords: Optional[set] = None,
    strict: bool = True,
    strip_dates_ids: bool = True,
    assume_normalized: bool = False,
) -> List[str]:
    """
    Tokenize text into words after normalization.
    """
    sw = stopwords if stopwords is not None else _DEFAULT_SW
    t = text if assume_normalized else normalize_text(
        text, strict=strict, strip_dates_ids=strip_dates_ids
    )
    toks = t.split()
    out: List[str] = []
    for tok in toks:
        if len(tok) < min_len:
            continue
        if remove_stopwords and tok in sw:
            continue
        out.append(tok)
    return out

# Split text into sentences
def sentence_split(text: str, *, max_sentences: Optional[int] = None) -> List[str]:
    if not text:
        return []
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    if max_sentences is not None and len(sents) > max_sentences:
        return sents[:max_sentences]
    return sents

# Tokenize filename
def filename_tokens(filename: str) -> List[str]:
    if not filename:
        return []
    name = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    name = re.sub(r"\.[A-Za-z0-9]{1,6}$", "", name)  # drop extension
    name = re.sub(r"[^\w]+", " ", name).strip().lower()
    toks = [t for t in name.split() if t and not t.isdigit() and len(t) >= 2]
    return toks[:20]

# DocumentView builder
def build_document_view(
    *,
    doc_id: str,
    text: str,
    language: Optional[str] = None,
    meta: Optional[Dict[str, object]] = None,
    return_tokens: bool = True,
    return_sentences: bool = True,
    strict_norm: bool = True,
    strip_dates_ids: bool = True,
    min_token_len: int = 2,
    remove_stopwords: bool = True,
    stopwords: Optional[set] = None,
) -> DocumentView:
    """
    Build a DocumentView with strong normalization defaults so that
    DOCX and PDF versions converge to the same representation.
    """
    norm = normalize_text(text, strict=strict_norm, strip_dates_ids=strip_dates_ids)
    toks = (
        tokenize_words(
            norm,
            min_len=min_token_len,
            remove_stopwords=remove_stopwords,
            stopwords=stopwords or _DEFAULT_SW,
            strict=False,
            strip_dates_ids=False,
            assume_normalized=True,
        )
        if return_tokens
        else None
    )
    sents = sentence_split(text) if return_sentences else None
    return DocumentView(
        doc_id=doc_id,
        text=norm,
        language=language,
        tokens=toks,
        sentences=sents,
        meta=meta or {},
    )


# Compute simple corpus stats for learners.prepare()
def compute_corpus_stats(docs: Iterable[DocumentView]) -> CorpusStats:
    n_docs = 0
    total_len = 0
    lang_counts: Dict[str, int] = defaultdict(int)
    token_counts: Counter = Counter()

    for dv in docs:
        n_docs += 1
        total_len += len((dv.text or "").split())
        if dv.language:
            lang_counts[str(dv.language)] += 1
        if dv.tokens:
            token_counts.update(dv.tokens)

    avg_len = (total_len / n_docs) if n_docs > 0 else 0.0

    # top tokens for debugging
    top_tokens = token_counts.most_common(50)
    extras = {
        "top_tokens": top_tokens,
        "vocab_size": int(len(token_counts)),
    }
    return CorpusStats(
        doc_count=n_docs,
        avg_doc_len=float(avg_len),
        lang_counts=dict(lang_counts),
        extras=extras,
    )


# Content hash helper
def content_hash(normalized_text: str) -> str:
    """
    Stable hash of fully normalized content.
    """
    import hashlib
    return hashlib.sha256((normalized_text or "").encode("utf-8")).hexdigest()
