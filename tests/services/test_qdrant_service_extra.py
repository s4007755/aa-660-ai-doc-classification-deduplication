import pytest
from unittest.mock import Mock
from src.services.qdrant_service import QdrantService


class TestQdrantReserveAndRetries:
    def test_reserve_id_block_increments(self, monkeypatch):
        svc = QdrantService(log_function=Mock())
        svc.connected = True
        mock_client = Mock()
        svc.client = mock_client
        # Simulate empty collection first
        mock_client.get_collection.return_value.points_count = 0
        start1 = svc._reserve_id_block('c', 3)
        start2 = svc._reserve_id_block('c', 2)
        assert start1 == 1
        assert start2 == 4

    def test_insert_vectors_retry_and_sqlite_mark(self, monkeypatch):
        svc = QdrantService(log_function=Mock())
        svc.connected = True
        mock_client = Mock()
        svc.client = mock_client
        # Make first upsert raise timeout, second succeed
        calls = {'n': 0}
        def upsert_side_effect(**kwargs):
            calls['n'] += 1
            if calls['n'] == 1:
                raise Exception('timed out')
            return None
        mock_client.get_collection.return_value.points_count = 0
        mock_client.upsert.side_effect = upsert_side_effect
        sqlite = Mock()
        sqlite.is_duplicate.return_value = False
        vectors = [[0.1], [0.2]]
        payloads = [{"hash":"h1"}, {"hash":"h2"}]
        ok, inserted, skipped = svc.insert_vectors('c', vectors, payloads, sqlite_service=sqlite)
        assert ok is True and inserted == 2
        assert sqlite.mark_as_processed.called

    def test_get_collection_metadata_fallback_creation(self, monkeypatch):
        svc = QdrantService(log_function=Mock())
        svc.connected = True
        mock_client = Mock()
        svc.client = mock_client
        # No metadata point exists initially, then creation path
        mock_client.get_collection.return_value.config.params.vectors.size = 1536
        mock_client.get_collection.return_value.points_count = 0
        ok = svc.update_collection_metadata('c', {"a":1})
        assert ok is True


