import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

DB_PATH = Path("data/db/docstore.sqlite")


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_conn()
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
    conn.commit()
    conn.close()


def upsert_document(doc_id: str, raw_text: str, normalized_text: str, meta: Dict[str, Any]):
    conn = get_conn()
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
    conn.commit()
    conn.close()


def add_file_mapping(doc_id: str, filepath: str, mtime_ns: Optional[int]):
    conn = get_conn()
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
    conn.commit()
    conn.close()


def mark_dirty(doc_ids: List[str]):
    if not doc_ids:
        return
    conn = get_conn()
    cur = conn.cursor()
    cur.executemany(
        "UPDATE documents SET is_dirty=1, updated_at=CURRENT_TIMESTAMP WHERE doc_id=?",
        [(d,) for d in doc_ids],
    )
    conn.commit()
    conn.close()


def delete_documents(doc_ids: List[str]):
    if not doc_ids:
        return
    conn = get_conn()
    cur = conn.cursor()
    cur.executemany("DELETE FROM documents WHERE doc_id=?", [(d,) for d in doc_ids])
    conn.commit()
    conn.close()


def insert_cluster(config: str, members: List[str], pairwise: List[Tuple[str, str, float, float, float]]) -> int:
    conn = get_conn()
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

    conn.commit()
    conn.close()
    return cluster_id


def get_all_documents() -> List[Dict[str, Any]]:
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT d.doc_id, d.language, d.filesize, d.is_dirty, f.filepath "
        "FROM documents d LEFT JOIN document_files f ON d.doc_id=f.doc_id "
        "ORDER BY d.updated_at DESC"
    ).fetchall()
    conn.close()

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

# Returns (doc_id, normalized_text) for all docs.
def get_docs_text(include_dirty: bool = False) -> List[Tuple[str, str]]:
    conn = get_conn()
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
    conn.close()
    return [(r[0], r[1]) for r in rows]

# Returns (doc_id, normalized_text) for provided doc_ids.
def get_docs_text_by_ids(doc_ids: List[str], include_dirty: bool = False) -> List[Tuple[str, str]]:
    if not doc_ids:
        return []
    conn = get_conn()
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
    conn.close()
    return [(r[0], r[1]) for r in rows]
