import sqlite3
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from typing import Iterable, Sequence, Union

DB_PATH = Path("data/db/docstore.sqlite")


def get_conn() -> sqlite3.Connection:
    # Open DB and enable FK
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    # Create tables and indexes
    with get_conn() as conn:
        cur = conn.cursor()
        cur.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA foreign_keys=ON;

            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                raw_text TEXT,
                normalized_text TEXT,
                language TEXT,
                filesize INTEGER,
                is_dirty INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS document_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                filepath TEXT NOT NULL,
                mtime_ns INTEGER,
                UNIQUE(doc_id, filepath)
            );

            CREATE TABLE IF NOT EXISTS clusters (
                cluster_id INTEGER PRIMARY KEY AUTOINCREMENT,
                config TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS cluster_members (
                cluster_id INTEGER NOT NULL REFERENCES clusters(cluster_id) ON DELETE CASCADE,
                doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                PRIMARY KEY (cluster_id, doc_id)
            );

            CREATE TABLE IF NOT EXISTS pairwise_scores (
                cluster_id INTEGER NOT NULL REFERENCES clusters(cluster_id) ON DELETE CASCADE,
                doc1 TEXT NOT NULL,
                doc2 TEXT NOT NULL,
                simhash REAL,
                minhash REAL,
                embedding REAL,
                PRIMARY KEY (cluster_id, doc1, doc2)
            );

            CREATE INDEX IF NOT EXISTS idx_docfiles_doc ON document_files(doc_id);
            CREATE INDEX IF NOT EXISTS idx_docfiles_path ON document_files(filepath);
            CREATE INDEX IF NOT EXISTS idx_members_cluster ON cluster_members(cluster_id);
            CREATE INDEX IF NOT EXISTS idx_pairs_cluster ON pairwise_scores(cluster_id);
            """
        )


def upsert_document(doc_id: str, raw_text: str, normalized_text: str, meta: Dict[str, Any]):
    # Insert/update a document row
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO documents (doc_id, raw_text, normalized_text, language, filesize, is_dirty, updated_at)
            VALUES (?, ?, ?, ?, ?, 0, CURRENT_TIMESTAMP)
            ON CONFLICT(doc_id) DO UPDATE SET
                raw_text=excluded.raw_text,
                normalized_text=excluded.normalized_text,
                language=excluded.language,
                filesize=excluded.filesize,
                is_dirty=0,
                updated_at=CURRENT_TIMESTAMP
            """,
            (doc_id, raw_text, normalized_text, meta.get("language"), meta.get("filesize", 0)),
        )

def batch_upsert_documents(
    docs: Iterable[Sequence[Union[str, int, dict]]]
):
    """
    Accepts items shaped like:
      (doc_id, raw_text, normalized_text)
      (doc_id, raw_text, normalized_text, language)
      (doc_id, raw_text, normalized_text, filesize)
      (doc_id, raw_text, normalized_text, meta_dict)
      (doc_id, raw_text, normalized_text, language, filesize)
    Normalizes to 5-tuple: (doc_id, raw_text, normalized_text, language, filesize)
    """
    normalized: List[Tuple[str, str, str, Optional[str], Optional[int]]] = []

    for item in docs:
        if not item:
            continue
        # Unpack safely
        if len(item) == 3:
            doc_id, raw_text, normalized_text = item
            language, filesize = None, None
        elif len(item) == 4:
            doc_id, raw_text, normalized_text, fourth = item
            language, filesize = None, None
            if isinstance(fourth, dict):
                language = fourth.get("language")
                filesize = fourth.get("filesize", 0)
            elif isinstance(fourth, str):
                language = fourth
            elif isinstance(fourth, int):
                filesize = fourth
            else:
                pass
        else:
            # 5 or more: take first five in order
            doc_id, raw_text, normalized_text, language, filesize = item[:5]

        normalized.append((doc_id, raw_text, normalized_text, language, filesize))

    if not normalized:
        return

    with get_conn() as conn:
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO documents (doc_id, raw_text, normalized_text, language, filesize, is_dirty, updated_at)
            VALUES (?, ?, ?, ?, ?, 0, CURRENT_TIMESTAMP)
            ON CONFLICT(doc_id) DO UPDATE SET
                raw_text=excluded.raw_text,
                normalized_text=excluded.normalized_text,
                language=excluded.language,
                filesize=excluded.filesize,
                is_dirty=0,
                updated_at=CURRENT_TIMESTAMP
            """,
            normalized,
        )


def batch_add_file_mappings(mappings: List[Tuple[str, str, Optional[int]]]):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO document_files (doc_id, filepath, mtime_ns)
            VALUES (?, ?, ?)
            ON CONFLICT(doc_id, filepath) DO UPDATE SET
                mtime_ns=excluded.mtime_ns
            """,
            mappings,
        )


def add_file_mapping(doc_id: str, filepath: str, mtime_ns: Optional[int]):
    # Map a file path to a document
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO document_files (doc_id, filepath, mtime_ns)
            VALUES (?, ?, ?)
            ON CONFLICT(doc_id, filepath) DO UPDATE SET
                mtime_ns=excluded.mtime_ns
            """,
            (doc_id, filepath, mtime_ns),
        )


def mark_dirty(doc_ids: List[str]):
    # Mark documents as needing re-normalization
    if not doc_ids:
        return
    with get_conn() as conn:
        cur = conn.cursor()
        cur.executemany(
            "UPDATE documents SET is_dirty=1, updated_at=CURRENT_TIMESTAMP WHERE doc_id=?",
            [(d,) for d in doc_ids],
        )


def delete_documents(doc_ids: List[str]):
    # Delete documents and cascading file mappings
    if not doc_ids:
        return
    with get_conn() as conn:
        cur = conn.cursor()
        cur.executemany("DELETE FROM documents WHERE doc_id=?", [(d,) for d in doc_ids])


def insert_cluster(config: str, members: List[str], pairwise: List[Tuple[str, str, float, float, float]]) -> int:
    # Insert a cluster with members and pairwise scores
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO clusters (config) VALUES (?)", (config,))
        cluster_id = cur.lastrowid

        cur.executemany(
            "INSERT INTO cluster_members (cluster_id, doc_id) VALUES (?, ?)",
            [(cluster_id, d) for d in members],
        )

        cur.executemany(
            "INSERT INTO pairwise_scores (cluster_id, doc1, doc2, simhash, minhash, embedding) VALUES (?, ?, ?, ?, ?, ?)",
            [(cluster_id, a, b, s, m, e) for (a, b, s, m, e) in pairwise],
        )

        return cluster_id


def get_all_documents() -> List[Dict[str, Any]]:
    # Return one row per document with aggregated filepaths
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        rows = cur.execute(
            "SELECT d.doc_id, d.language, d.filesize, d.is_dirty, f.filepath "
            "FROM documents d LEFT JOIN document_files f ON d.doc_id=f.doc_id "
            "ORDER BY d.updated_at DESC"
        ).fetchall()

    results: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        doc_id = r["doc_id"]
        if doc_id not in results:
            results[doc_id] = {
                "doc_id": doc_id,
                "language": r["language"],
                "filesize": r["filesize"],
                "is_dirty": r["is_dirty"],
                "filepaths": [],
            }
        if r["filepath"]:
            results[doc_id]["filepaths"].append(r["filepath"])
    return list(results.values())


def get_docs_text(include_dirty: bool = False) -> List[Tuple[str, str]]:
    # Return (doc_id, normalized_text) for all docs
    with get_conn() as conn:
        cur = conn.cursor()
        if include_dirty:
            rows = cur.execute(
                "SELECT doc_id, normalized_text FROM documents "
                "WHERE normalized_text IS NOT NULL AND LENGTH(normalized_text) > 0"
            ).fetchall()
        else:
            rows = cur.execute(
                "SELECT doc_id, normalized_text FROM documents "
                "WHERE is_dirty=0 AND normalized_text IS NOT NULL AND LENGTH(normalized_text) > 0"
            ).fetchall()
    return [(r[0], r[1]) for r in rows]


def get_docs_text_by_ids(doc_ids: List[str], include_dirty: bool = False) -> List[Tuple[str, str]]:
    # Return (doc_id, normalized_text) for specific doc_ids
    if not doc_ids:
        return []
    with get_conn() as conn:
        cur = conn.cursor()
        placeholders = ",".join("?" for _ in doc_ids)
        if include_dirty:
            sql = f"""
                SELECT doc_id, normalized_text
                FROM documents
                WHERE doc_id IN ({placeholders})
                  AND normalized_text IS NOT NULL AND LENGTH(normalized_text) > 0
            """
        else:
            sql = f"""
                SELECT doc_id, normalized_text
                FROM documents
                WHERE is_dirty=0 AND doc_id IN ({placeholders})
                  AND normalized_text IS NOT NULL AND LENGTH(normalized_text) > 0
            """
        rows = cur.execute(sql, doc_ids).fetchall()
    return [(r[0], r[1]) for r in rows]


def get_all_document_files() -> List[Dict[str, Any]]:
    # Return one row per mapped file (for the Documents tab)
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        rows = cur.execute(
            """
            SELECT
                d.doc_id,
                d.language,
                d.filesize,
                f.filepath,
                f.mtime_ns
            FROM documents d
            JOIN document_files f ON d.doc_id = f.doc_id
            ORDER BY d.updated_at DESC, f.filepath ASC
            """
        ).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        fp = r["filepath"] or ""
        out.append({
            "doc_id": r["doc_id"],
            "language": r["language"],
            "filesize": r["filesize"],
            "filepath": fp,
            "filename": os.path.basename(fp) if fp else "",
            "mtime_ns": r["mtime_ns"],
        })
    return out
