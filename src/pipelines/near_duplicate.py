from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
from simhash import Simhash
from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer

# Thresholds
SIMHASH_THRESHOLD = 5                 # max Hamming distance (64-bit)
SIMHASH_MAX_WEIGHT = 255

MINHASH_SHINGLE_SIZE = 3              # n-gram token shingle size
MINHASH_THRESHOLD = 0.8               # Jaccard threshold for LSH
MINHASH_NUM_PERM = 64

EMBED_COSINE_THRESHOLD = 0.92         # cosine similarity (normalized embeddings)
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 64                 # SBERT encode batch size

@dataclass
class Document:
    doc_id: str
    normalized_text: str

@dataclass
class ClusterMetric:
    avg: float
    min: float
    max: float
    n_pairs: int

class UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        p = self.parent.get(x, x)
        if p != x:
            p = self.find(p)
            self.parent[x] = p
        return p

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)

# Helpers

def _agg(values: List[float]) -> ClusterMetric:
    if not values:
        return ClusterMetric(avg=1.0, min=1.0, max=1.0, n_pairs=0)
    return ClusterMetric(
        avg=float(np.mean(values)),
        min=float(np.min(values)),
        max=float(np.max(values)),
        n_pairs=len(values),
    )

def compute_simhash(text: str) -> int:
    if not text:
        return 0
    tokens = text.lower().split()
    counts = Counter(tokens)
    features = [(tok, min(cnt, SIMHASH_MAX_WEIGHT)) for tok, cnt in counts.items()]
    return Simhash(features, f=64).value

def _simhash_similarity(h1: int, h2: int) -> float:
    dist = (h1 ^ h2).bit_count()
    return max(0.0, 1.0 - dist / 64.0)

def compute_minhash(text: str, shingle_size: int) -> MinHash:
    m = MinHash(num_perm=MINHASH_NUM_PERM)
    tokens = text.split()
    if shingle_size <= 1:
        shingles = tokens
    else:
        L = len(tokens) - shingle_size + 1
        shingles = (" ".join(tokens[i:i + shingle_size]) for i in range(max(L, 0)))
    for s in shingles:
        m.update(s.encode("utf-8", errors="ignore"))
    return m

def _minhash_jaccard(m1: MinHash, m2: MinHash) -> float:
    return float(m1.jaccard(m2))

def _ensure_2d(a: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        return a.reshape(1, -1)
    return a

def compute_embeddings(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    if not texts:
        try:
            dim = model.get_sentence_embedding_dimension()
        except Exception:
            dim = 384
        return np.zeros((0, dim), dtype=np.float32)

    emb = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
    )
    return _ensure_2d(emb)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def _cosine_to_unit(cosine: float) -> float:
    return float((cosine + 1.0) / 2.0)

def _text_hash(norm: str) -> str:
    return hashlib.sha256((norm or "").encode("utf-8", errors="ignore")).hexdigest()

# Main APIs

def detect_near_duplicates_with_scores(
    docs: List[Document],
    *,
    enable_exact_text_union: bool = True,
    enable_simhash: bool = True,
    enable_minhash: bool = True,
    enable_embeddings: bool = True,
    simhash_threshold: int = SIMHASH_THRESHOLD,
    minhash_shingle_size: int = MINHASH_SHINGLE_SIZE,
    minhash_threshold: float = MINHASH_THRESHOLD,
    embed_cosine_threshold: float = EMBED_COSINE_THRESHOLD,
    embed_model_name: str = DEFAULT_EMBED_MODEL,
) -> Tuple[
    List[Dict[str, Any]],
    List[Tuple[str, str, Optional[float], Optional[float], Optional[float]]],
]:

    if not docs:
        return [], []

    # Dedup by doc_id
    uniq_docs = list({d.doc_id: d for d in docs}.values())
    doc_ids = [d.doc_id for d in uniq_docs]
    n = len(doc_ids)

    uf = UnionFind()

    # Stage 0: exact-text union (identical normalized text)
    if enable_exact_text_union:
        by_text: Dict[str, List[str]] = defaultdict(list)
        for d in uniq_docs:
            by_text[_text_hash(d.normalized_text)].append(d.doc_id)
        for ids in by_text.values():
            if len(ids) > 1:
                base = ids[0]
                for other in ids[1:]:
                    uf.union(base, other)

    # Stage 1: SimHash (union by threshold)
    simhashes: Dict[str, int] = {}
    if enable_simhash:
        for d in uniq_docs:
            simhashes[d.doc_id] = compute_simhash(d.normalized_text)
        for i in range(n):
            h1 = simhashes[doc_ids[i]]
            for j in range(i + 1, n):
                if ((h1 ^ simhashes[doc_ids[j]]).bit_count() <= simhash_threshold):
                    uf.union(doc_ids[i], doc_ids[j])

    # Stage 2: MinHash + LSH (union candidates)
    minhashes: Dict[str, MinHash] = {}
    if enable_minhash:
        lsh = MinHashLSH(threshold=minhash_threshold, num_perm=MINHASH_NUM_PERM)
        for d in uniq_docs:
            mh = compute_minhash(d.normalized_text, shingle_size=minhash_shingle_size)
            minhashes[d.doc_id] = mh
            lsh.insert(d.doc_id, mh, check_duplication=False)
        for did in doc_ids:
            for cand in lsh.query(minhashes[did]):
                if cand != did:
                    uf.union(did, cand)

    # Stage 3: Embeddings (union by cosine)
    embeddings: Dict[str, np.ndarray] = {}
    if enable_embeddings:
        model = SentenceTransformer(embed_model_name)
        emb_matrix = compute_embeddings(model, [d.normalized_text for d in uniq_docs])
        for idx, did in enumerate(doc_ids):
            embeddings[did] = emb_matrix[idx]
        for i in range(n):
            e1 = embeddings[doc_ids[i]]
            for j in range(i + 1, n):
                if uf.find(doc_ids[i]) == uf.find(doc_ids[j]):
                    continue
                if cosine_similarity(e1, embeddings[doc_ids[j]]) >= embed_cosine_threshold:
                    uf.union(doc_ids[i], doc_ids[j])

    # Build clusters
    clusters: Dict[str, List[str]] = defaultdict(list)
    for did in doc_ids:
        clusters[uf.find(did)].append(did)

    # Compute metrics & pairwise scores
    results: List[Dict[str, Any]] = []
    pairs_out: List[Tuple[str, str, Optional[float], Optional[float], Optional[float]]] = []

    for members in clusters.values():
        members.sort()
        sim_vals: List[float] = []
        min_vals: List[float] = []
        emb_vals: List[float] = []

        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]

                sim_s = _simhash_similarity(simhashes[a], simhashes[b]) if enable_simhash else None
                jac = _minhash_jaccard(minhashes[a], minhashes[b]) if enable_minhash else None
                cos = cosine_similarity(embeddings[a], embeddings[b]) if enable_embeddings else None

                if sim_s is not None:
                    sim_vals.append(sim_s)
                if jac is not None:
                    min_vals.append(jac)
                if cos is not None:
                    emb_vals.append(_cosine_to_unit(cos))

                a1, b1 = (a, b) if a < b else (b, a)
                pairs_out.append((a1, b1, sim_s, jac, cos))

        thresholds = {
            "simhash_hamming_max": simhash_threshold,
            "simhash_similarity_min": max(0.0, 1.0 - simhash_threshold / 64.0),
            "minhash_jaccard_min": minhash_threshold,
            "minhash_shingle_size": minhash_shingle_size,
            "embedding_cosine_min": embed_cosine_threshold,
            "embedding_unit_min": _cosine_to_unit(embed_cosine_threshold),
        }

        results.append({
            "doc_ids": members,
            "metrics": {
                "simhash": _agg(sim_vals).__dict__ if sim_vals else _agg([]).__dict__,
                "minhash": _agg(min_vals).__dict__ if min_vals else _agg([]).__dict__,
                "embedding": _agg(emb_vals).__dict__ if emb_vals else _agg([]).__dict__,
            },
            "thresholds": thresholds,
            "config": {
                "enable_exact_text_union": enable_exact_text_union,
                "enable_simhash": enable_simhash,
                "enable_minhash": enable_minhash,
                "enable_embeddings": enable_embeddings,
                "embed_model": embed_model_name,
            },
        })

    results.sort(key=lambda r: (-len(r["doc_ids"]), r["doc_ids"][0]))
    return results, pairs_out


def detect_near_duplicates(docs: List[Document]) -> List[List[str]]:
    infos, _pairs = detect_near_duplicates_with_scores(
        docs,
        enable_exact_text_union=True,
        enable_simhash=True,
        enable_minhash=True,
        enable_embeddings=True,
        simhash_threshold=SIMHASH_THRESHOLD,
        minhash_shingle_size=MINHASH_SHINGLE_SIZE,
        minhash_threshold=MINHASH_THRESHOLD,
        embed_cosine_threshold=EMBED_COSINE_THRESHOLD,
        embed_model_name=DEFAULT_EMBED_MODEL,
    )
    clusters = [ci["doc_ids"] for ci in infos]
    clusters.sort(key=lambda c: (-len(c), c[0]))
    return clusters
