import tempfile
from pathlib import Path

from src.services.sqlite_service import SQLiteService


def test_sqlite_service_dedup(tmp_path):
    db = tmp_path / "hash.db"
    svc = SQLiteService(db_path=str(db), log_function=lambda *_: None)

    added = svc.mark_as_processed("c1", ["h1", "h2", "h3"])
    assert added == 3
    assert svc.is_duplicate("c1", "h1")
    assert not svc.is_duplicate("c1", "hX")

    # add duplicates + new
    added2 = svc.mark_as_processed("c1", ["h2", "h4"])
    assert added2 == 1

    hs = svc.get_processed_hashes("c1")
    assert "h1" in hs and "h4" in hs

    stats = svc.get_stats("c1")
    assert stats["table_exists"] is True
    assert stats["hash_count"] == 4

    cols = svc.get_all_collections()
    assert "c1" in cols

    info = svc.get_database_info()
    assert info["table_count"] >= 1

    assert svc.clear_collection("c1")
    stats2 = svc.get_stats("c1")
    assert stats2["hash_count"] == 0

