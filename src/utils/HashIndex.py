import sqlite3
from pathlib import Path

class HashIndex:
    def __init__(self, db_path="hash_index.db"):
        self.db_path = Path(db_path)

        # ensure parent directory exists
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # connect (creates DB file if not present)
        self.conn = sqlite3.connect(self.db_path)
        self.cur = self.conn.cursor()
        self.initialized = set()

    def ensure_table(self, collection: str):
        """Create per-collection hash table once."""
        if collection in self.initialized:
            return
        self.cur.execute(
            f"CREATE TABLE IF NOT EXISTS {collection} (hash TEXT PRIMARY KEY)"
        )
        self.conn.commit()
        self.initialized.add(collection)

    def add(self, collection: str, hash_value: str):
        self.ensure_table(collection)
        self.cur.execute(
            f"INSERT OR IGNORE INTO {collection} (hash) VALUES (?)", (hash_value,)
        )
        self.conn.commit()

    def exists(self, collection: str, hash_value: str) -> bool:
        self.ensure_table(collection)
        self.cur.execute(
            f"SELECT 1 FROM {collection} WHERE hash=? LIMIT 1", (hash_value,)
        )
        return self.cur.fetchone() is not None

    def bulk_add(self, collection: str, hashes: list[str]):
        self.ensure_table(collection)
        self.cur.executemany(
            f"INSERT OR IGNORE INTO {collection} (hash) VALUES (?)",
            [(h,) for h in hashes],
        )
        self.conn.commit()

    def load_all(self, collection: str) -> set[str]:
        self.ensure_table(collection)
        self.cur.execute(f"SELECT hash FROM {collection}")
        return {row[0] for row in self.cur.fetchall()}

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()
