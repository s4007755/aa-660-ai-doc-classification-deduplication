# src/features/text_preproc.py
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.learners.base import DocumentView, CorpusStats

# minimal default stopwords
_DEFAULT_SW = {
    "the","a","an","and","or","for","of","to","in","on","at","by","with","from","as",
    "is","are","was","were","be","been","it","this","that","these","those","you","your",
}

# sentence splitter
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# quick header/footer/page-number patterns
_RE_PAGE_NUM = re.compile(r"\bpage\s+\d+(\s+of\s+\d+)?\b", re.IGNORECASE)
_RE_HR = re.compile(r"\n-{2,}\n")
_RE_SOFT_HYPHEN = re.compile(r"-\n")
_RE_MULTI_SPACE = re.compile(r"\s+")

# digits/IDs/dates for strict cleanup
_RE_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_RE_LONG_ID = re.compile(r"\b\d{6,}\b")

# export surface
__all__ = [
    "normalize_text",
    "tokenize_words",
    "sentence_split",
    "filename_tokens",
    "build_document_view",
    "compute_corpus_stats",
]

# normalize unicode, lower, remove boilerplate
def normalize_text(
    text: str,
    *,
    strict: bool = False,
    strip_dates_ids: bool = False,
) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)

    # remove typical page labels and horizontal rules
    t = _RE_PAGE_NUM.sub(" ", t)
    t = _RE_HR.sub("\n", t)

    # remove soft hyphenation at line breaks
    t = _RE_SOFT_HYPHEN.sub("", t)

    if strict:
        # drop most punctuation
        t = re.sub(r"[^\w\s]", " ", t)

    if strip_dates_ids:
        t = _RE_DATE.sub(" ", t)
        t = _RE_LONG_ID.sub(" ", t)

    # collapse whitespace and lowercase
    t = _RE_MULTI_SPACE.sub(" ", t).strip().lower()
    return t

# simple word tokenizer
def tokenize_words(
    text: str,
    *,
    min_len: int = 2,
    remove_stopwords: bool = True,
    stopwords: Optional[set] = None,
    strict: bool = False,
    strip_dates_ids: bool = False,
) -> List[str]:
    sw = stopwords if stopwords is not None else _DEFAULT_SW
    t = normalize_text(text, strict=strict, strip_dates_ids=strip_dates_ids)
    toks = t.split()
    out: List[str] = []
    for tok in toks:
        if len(tok) < min_len:
            continue
        if remove_stopwords and tok in sw:
            continue
        out.append(tok)
    return out

# split text into sentences
def sentence_split(text: str, *, max_sentences: Optional[int] = None) -> List[str]:
    if not text:
        return []
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    if max_sentences is not None and len(sents) > max_sentences:
        return sents[:max_sentences]
    return sents

# tokenize filename
def filename_tokens(filename: str) -> List[str]:
    if not filename:
        return []
    name = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    name = re.sub(r"\.[A-Za-z0-9]{1,6}$", "", name)  # drop extension
    name = re.sub(r"[^\w]+", " ", name).strip().lower()
    toks = [t for t in name.split() if t and not t.isdigit() and len(t) >= 2]
    return toks[:20]

# build a DocumentView from raw pieces
def build_document_view(
    *,
    doc_id: str,
    text: str,
    language: Optional[str] = None,
    meta: Optional[Dict[str, object]] = None,
    return_tokens: bool = True,
    return_sentences: bool = True,
    strict_norm: bool = False,
    strip_dates_ids: bool = False,
    min_token_len: int = 2,
    remove_stopwords: bool = True,
    stopwords: Optional[set] = None,
) -> DocumentView:
    norm = normalize_text(text, strict=strict_norm, strip_dates_ids=strip_dates_ids)
    toks = tokenize_words(
        norm,
        min_len=min_token_len,
        remove_stopwords=remove_stopwords,
        stopwords=stopwords or _DEFAULT_SW,
        strict=False,
        strip_dates_ids=False,
    ) if return_tokens else None
    sents = sentence_split(text) if return_sentences else None
    return DocumentView(
        doc_id=doc_id,
        text=norm,
        language=language,
        tokens=toks,
        sentences=sents,
        meta=meta or {},
    )

# compute simple corpus stats for learners.prepare()
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
    return CorpusStats(doc_count=n_docs, avg_doc_len=float(avg_len), lang_counts=dict(lang_counts), extras=extras)
