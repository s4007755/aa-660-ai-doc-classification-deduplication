from unittest.mock import patch
from api_server import app
from fastapi.testclient import TestClient

client = TestClient(app)


@patch('api_server.get_cached_collections')
@patch('api_server.qdrant_service')
def test_api_collection_info_guard(mock_q, mock_cache):
    mock_cache.return_value = {"col": {"size": 1536}}
    mock_q.get_collection_info.return_value = None
    resp = client.get("/collections/col/info")
    assert resp.status_code == 500


