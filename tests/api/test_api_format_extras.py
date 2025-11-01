import pytest
from unittest.mock import patch
from api_server import app
from fastapi.testclient import TestClient

client = TestClient(app)


class TestResponseFormatExtras:
    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_labels_per_page_includes_total_labels(self, mock_cache, mock_qdrant):
        mock_cache.return_value = {"test_collection": {"size": 1536}}
        # Force non-metadata path
        mock_qdrant.get_collection_metadata.return_value = {}
        mock_qdrant.scroll_vectors.return_value = (
            [
                {"id": 1, "payload": {"predicted_label": "A"}},
                {"id": 2, "payload": {"predicted_label": "A"}},
                {"id": 3, "payload": {"predicted_label": "B"}},
            ],
            None
        )
        resp = client.get("/collections/test_collection/labels?use_cache=false")
        assert resp.status_code == 200
        data = resp.json()
        assert "labels" in data
        assert "total_labels" in data
        assert data["total_labels"] == 2

    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_clusters_per_page_includes_total_clusters(self, mock_cache, mock_qdrant):
        mock_cache.return_value = {"test_collection": {"size": 1536}}
        mock_qdrant.get_collection_metadata.return_value = {}
        mock_qdrant.scroll_vectors.return_value = (
            [
                {"id": 1, "payload": {"cluster_id": 0, "cluster_name": "C0"}},
                {"id": 2, "payload": {"cluster_id": 1, "cluster_name": "C1"}},
                {"id": 3, "payload": {"cluster_id": 0, "cluster_name": "C0"}},
            ],
            None
        )
        resp = client.get("/collections/test_collection/clusters?use_cache=false")
        assert resp.status_code == 200
        data = resp.json()
        assert "clusters" in data
        assert "total_clusters" in data
        assert data["total_clusters"] == 2


