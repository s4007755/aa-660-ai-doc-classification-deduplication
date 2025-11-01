"""
Comprehensive Test Suite for API Server (api_server.py)

This test suite provides comprehensive coverage of the API functionality including:
- All API endpoints
- Error handling
- Data validation
- Edge cases
- Performance optimization
"""

import pytest
import sys
import os
from fastapi.testclient import TestClient
from unittest.mock import Mock, MagicMock, patch

# Add src to path for imports (from tests/api/ up to project root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import API server
from api_server import app, get_cached_collections, infer_embedding_model


# Create test client
client = TestClient(app)


class TestRootEndpoint:
    """Test root endpoint functionality."""
    
    def test_root_endpoint_returns_200(self):
        """Test that root endpoint returns 200 status code."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_endpoint_returns_version(self):
        """Test that root endpoint returns version information."""
        response = client.get("/")
        data = response.json()
        assert "version" in data
        assert data["version"] == "1.0.1"
    
    def test_root_endpoint_returns_endpoints(self):
        """Test that root endpoint returns available endpoints."""
        response = client.get("/")
        data = response.json()
        assert "endpoints" in data
        assert "collections" in data["endpoints"]
        assert "collection_info" in data["endpoints"]
    
    def test_root_endpoint_content_type(self):
        """Test that root endpoint returns JSON content type."""
        response = client.get("/")
        assert "application/json" in response.headers["content-type"]


class TestCollectionsEndpoint:
    """Test /collections endpoint."""
    
    @patch('api_server.qdrant_service')
    def test_list_collections_success(self, mock_qdrant):
        """Test successful collection listing."""
        # Mock qdrant service response
        mock_qdrant.list_collections.return_value = {
            "test_collection": {
                "size": 1536,
                "vectors": 100,
                "distance": "Cosine"
            }
        }
        
        response = client.get("/collections")
        assert response.status_code == 200
        data = response.json()
        assert "collections" in data
    
    def test_list_collections_empty(self):
        """Test listing collections when none exist."""
        with patch('api_server.qdrant_service') as mock_qdrant:
            mock_qdrant.list_collections.return_value = {}
            
            response = client.get("/collections")
            assert response.status_code == 200
            data = response.json()
            assert data["collections"] == {}
    
    @patch('api_server.qdrant_service')
    def test_list_collections_error(self, mock_qdrant):
        """Test error handling when listing collections fails."""
        mock_qdrant.list_collections.side_effect = Exception("Connection failed")
        
        response = client.get("/collections")
        assert response.status_code == 500


class TestCollectionInfoEndpoint:
    """Test /collections/{collection_name}/info endpoint."""
    
    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_get_collection_info_success(self, mock_cache, mock_qdrant):
        """Test successful retrieval of collection info."""
        # Mock cache
        mock_cache.return_value = {"test_collection": {"size": 1536}}
        
        # Mock qdrant service
        mock_qdrant.get_collection_info.return_value = {
            "name": "test_collection",
            "dimension": 1536,
            "distance_metric": "Cosine",
            "vector_count": 100
        }
        
        # Mock get_collection_metadata (new requirement)
        mock_qdrant.get_collection_metadata.return_value = None
        
        response = client.get("/collections/test_collection/info")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_collection"
        assert data["dimension"] == 1536
    
    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_get_collection_info_not_found(self, mock_cache, mock_qdrant):
        """Test getting info for non-existent collection."""
        mock_cache.return_value = {}
        
        response = client.get("/collections/nonexistent/info")
        assert response.status_code == 404
    
    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_get_collection_info_error(self, mock_cache, mock_qdrant):
        """Test error handling when getting collection info fails."""
        mock_cache.return_value = {"test_collection": {"size": 1536}}
        mock_qdrant.get_collection_info.side_effect = Exception("Database error")
        
        response = client.get("/collections/test_collection/info")
        assert response.status_code == 500


class TestPointsEndpoint:
    """Test /collections/{collection_name}/points endpoint."""
    
    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_get_all_points_success(self, mock_cache, mock_qdrant):
        """Test successful retrieval of points."""
        mock_cache.return_value = {"test_collection": {"size": 1536}}
        
        mock_qdrant.scroll_vectors.return_value = (
            [
                {
                    "id": 1,
                    "payload": {
                        "source": "test.txt",
                        "cluster_id": 0,
                        "cluster_name": "TestCluster",
                        "predicted_label": "Label1",
                        "confidence": 0.95,
                        "text_content": "Sample text"
                    }
                }
            ],
            None
        )
        
        response = client.get("/collections/test_collection/points")
        assert response.status_code == 200
        data = response.json()
        assert "points" in data
        assert len(data["points"]) == 1
    
    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_get_all_points_with_limit(self, mock_cache, mock_qdrant):
        """Test getting points with custom limit."""
        mock_cache.return_value = {"test_collection": {"size": 1536}}
        mock_qdrant.scroll_vectors.return_value = ([], None)
        
        response = client.get("/collections/test_collection/points?limit=50")
        assert response.status_code == 200

    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_get_all_points_all_true_pages(self, mock_cache, mock_qdrant):
        """Fetch all points across multiple pages when no limit is provided."""
        mock_cache.return_value = {"test_collection": {"size": 1536}}

        # Simulate two pages of 1000 and 500
        first_page = ([{"id": i, "payload": {"source": f"t{i}.txt"}} for i in range(1000)], "offset_1")
        second_page = ([{"id": 1000 + i, "payload": {"source": f"t{1000+i}.txt"}} for i in range(500)], None)

        def scroll_side_effect(collection_name, limit, with_payload, with_vectors, page_offset=None, **kwargs):
            if page_offset is None:
                return first_page
            else:
                return second_page

        mock_qdrant.scroll_vectors.side_effect = scroll_side_effect

        response = client.get("/collections/test_collection/points")
        assert response.status_code == 200
        data = response.json()
        assert data["total_returned"] == 1500
        assert len(data["points"]) == 1500

    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_text_content_truncation(self, mock_cache, mock_qdrant):
        """Ensure text_content is truncated to 500 chars in responses."""
        mock_cache.return_value = {"test_collection": {"size": 1536}}

        long_text = "x" * 1200
        mock_qdrant.scroll_vectors.return_value = ([{"id": 1, "payload": {"source": "s.txt", "text_content": long_text}}], None)

        response = client.get("/collections/test_collection/points?limit=1")
        assert response.status_code == 200
        data = response.json()
        assert len(data["points"]) == 1
        assert len(data["points"][0]["text_content"]) == 500

    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_all_points_empty_page_midstream(self, mock_cache, mock_qdrant):
        """Ensure aggregation stops correctly if an empty page is encountered in the middle."""
        mock_cache.return_value = {"test_collection": {"size": 1536}}

        first_page = ([{"id": i, "payload": {"source": f"t{i}.txt"}} for i in range(1000)], "offset_1")
        empty_page = ([], None)

        def scroll_side_effect(collection_name, limit, with_payload, with_vectors, page_offset=None, **kwargs):
            if page_offset is None:
                return first_page
            else:
                return empty_page

        mock_qdrant.scroll_vectors.side_effect = scroll_side_effect

        response = client.get("/collections/test_collection/points")
        assert response.status_code == 200
        data = response.json()
        assert data["total_returned"] == 1000
        assert len(data["points"]) == 1000

    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_points_with_missing_optional_fields(self, mock_cache, mock_qdrant):
        """Points missing optional fields should serialize with None defaults."""
        mock_cache.return_value = {"test_collection": {"size": 1536}}

        mock_qdrant.scroll_vectors.return_value = (
            [
                {"id": 1, "payload": {"source": "a.txt"}},
                {"id": 2, "payload": {}},
            ],
            None
        )

        response = client.get("/collections/test_collection/points?limit=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data["points"]) == 2
        for p in data["points"]:
            assert "cluster_id" in p
            assert "cluster_name" in p
            assert "predicted_label" in p
            assert "confidence" in p

    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_large_limit_capped_by_page_size(self, mock_cache, mock_qdrant):
        """When limit is large, only one page up to 1000 is fetched and sliced."""
        mock_cache.return_value = {"test_collection": {"size": 1536}}

        # Backend would return many points, but API fetches only one page of 1000 max
        points = [{"id": i, "payload": {"source": f"t{i}.txt"}} for i in range(5000)]
        mock_qdrant.scroll_vectors.return_value = (points, "next")

        response = client.get("/collections/test_collection/points?limit=10000")
        assert response.status_code == 200
        data = response.json()
        assert data["total_returned"] <= 1000
        assert len(data["points"]) <= 1000
    
    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_get_all_points_limit_validation(self, mock_cache, mock_qdrant):
        """Test that limit parameter is validated."""
        mock_cache.return_value = {"test_collection": {"size": 1536}}
        
        # Test limit too high
        response = client.get("/collections/test_collection/points?limit=20000")
        assert response.status_code == 422  # Validation error
        
        # Test negative limit
        response = client.get("/collections/test_collection/points?limit=-1")
        assert response.status_code == 422
    
    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_get_all_points_collection_not_found(self, mock_cache, mock_qdrant):
        """Test getting points from non-existent collection."""
        mock_cache.return_value = {}
        
        response = client.get("/collections/nonexistent/points")
        assert response.status_code == 404


class TestPointByIdEndpoint:
    """Test /collections/{collection_name}/points/{point_id} endpoint."""
    
    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_get_point_by_id_success(self, mock_cache, mock_qdrant):
        """Test successful retrieval of specific point."""
        mock_cache.return_value = {"test_collection": {"size": 1536}}
        
        # Mock qdrant client retrieve method
        mock_client = MagicMock()
        mock_point = MagicMock()
        mock_point.id = 1
        mock_point.payload = {"source": "test.txt"}
        mock_client.retrieve.return_value = [mock_point]
        mock_qdrant.client = mock_client
        
        response = client.get("/collections/test_collection/points/1")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 1
    
    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_get_point_by_id_not_found(self, mock_cache, mock_qdrant):
        """Test getting non-existent point."""
        mock_cache.return_value = {"test_collection": {"size": 1536}}
        
        # Mock qdrant client to return empty
        mock_client = MagicMock()
        mock_client.retrieve.return_value = []
        mock_qdrant.client = mock_client
        mock_qdrant.scroll_vectors.return_value = ([], None)
        
        response = client.get("/collections/test_collection/points/9999")
        assert response.status_code == 404


class TestLabelsEndpoint:
    """Test /collections/{collection_name}/labels endpoints."""
    
    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_get_all_labels_success(self, mock_cache, mock_qdrant):
        """Test successful retrieval of all labels."""
        mock_cache.return_value = {"test_collection": {"size": 1536}}
        
        mock_qdrant.scroll_vectors.return_value = (
            [
                {"id": 1, "payload": {"predicted_label": "Label1"}},
                {"id": 2, "payload": {"predicted_label": "Label2"}},
                {"id": 3, "payload": {"predicted_label": "Label1"}},
            ],
            None
        )
        
        response = client.get("/collections/test_collection/labels")
        assert response.status_code == 200
        data = response.json()
        assert "labels" in data
        assert len(data["labels"]) == 2  # Two unique labels
    
    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_get_points_by_label_success(self, mock_cache, mock_qdrant):
        """Test successful retrieval of points by label."""
        mock_cache.return_value = {"test_collection": {"size": 1536}}
        
        mock_qdrant.scroll_vectors.return_value = (
            [
                {"id": 1, "payload": {"predicted_label": "Label1", "source": "test.txt"}},
            ],
            None
        )
        
        response = client.get("/collections/test_collection/labels/Label1")
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "Label1"
        assert len(data["points"]) == 1
    
    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_get_points_by_label_not_found(self, mock_cache, mock_qdrant):
        """Test getting points with non-existent label."""
        mock_cache.return_value = {"test_collection": {"size": 1536}}
        mock_qdrant.scroll_vectors.return_value = ([], None)
        
        response = client.get("/collections/test_collection/labels/NonexistentLabel")
        assert response.status_code == 404


class TestClustersEndpoint:
    """Test /collections/{collection_name}/clusters endpoints."""
    
    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_get_all_clusters_success(self, mock_cache, mock_qdrant):
        """Test successful retrieval of all clusters."""
        mock_cache.return_value = {"test_collection": {"size": 1536}}
        
        mock_qdrant.scroll_vectors.return_value = (
            [
                {"id": 1, "payload": {"cluster_id": 0, "cluster_name": "Cluster_0"}},
                {"id": 2, "payload": {"cluster_id": 1, "cluster_name": "Cluster_1"}},
                {"id": 3, "payload": {"cluster_id": 0, "cluster_name": "Cluster_0"}},
            ],
            None
        )
        
        response = client.get("/collections/test_collection/clusters")
        assert response.status_code == 200
        data = response.json()
        assert "clusters" in data
        assert len(data["clusters"]) == 2  # Two unique clusters
    
    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_get_points_by_cluster_success(self, mock_cache, mock_qdrant):
        """Test successful retrieval of points by cluster."""
        mock_cache.return_value = {"test_collection": {"size": 1536}}
        
        mock_qdrant.scroll_vectors.return_value = (
            [
                {
                    "id": 1,
                    "payload": {
                        "cluster_id": 0,
                        "cluster_name": "Cluster_0",
                        "source": "test.txt"
                    }
                },
            ],
            None
        )
        
        response = client.get("/collections/test_collection/clusters/0")
        assert response.status_code == 200
        data = response.json()
        assert data["cluster_id"] == 0
        assert len(data["points"]) == 1
    
    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_get_points_by_cluster_not_found(self, mock_cache, mock_qdrant):
        """Test getting points from non-existent cluster."""
        mock_cache.return_value = {"test_collection": {"size": 1536}}
        mock_qdrant.scroll_vectors.return_value = ([], None)
        
        response = client.get("/collections/test_collection/clusters/999")
        assert response.status_code == 404


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_infer_embedding_model_small(self):
        """Test model inference for small embedding."""
        model = infer_embedding_model(1536)
        assert model == "text-embedding-3-small"
    
    def test_infer_embedding_model_large(self):
        """Test model inference for large embedding."""
        model = infer_embedding_model(3072)
        assert model == "text-embedding-3-large"
    
    def test_infer_embedding_model_unknown(self):
        """Test model inference for unknown dimension."""
        model = infer_embedding_model(999)
        assert "unknown-model-999d" in model
    
    @patch('api_server.qdrant_service')
    def test_get_cached_collections(self, mock_qdrant):
        """Test collection caching functionality."""
        mock_qdrant.list_collections.return_value = {
            "test_collection": {"size": 1536}
        }
        
        # First call should fetch from service
        collections = get_cached_collections()
        assert "test_collection" in collections
        
        # Mock should be called once
        assert mock_qdrant.list_collections.call_count == 1


class TestErrorHandling:
    """Test error handling across all endpoints."""
    
    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_collection_not_found_error(self, mock_cache, mock_qdrant):
        """Test 404 error for non-existent collection."""
        mock_cache.return_value = {}
        
        endpoints_to_test = [
            "/collections/nonexistent/info",
            "/collections/nonexistent/points",
            "/collections/nonexistent/labels",
            "/collections/nonexistent/clusters",
        ]
        
        for endpoint in endpoints_to_test:
            response = client.get(endpoint)
            assert response.status_code == 404, f"Failed for endpoint: {endpoint}"
    
    @patch('api_server.qdrant_service')
    def test_internal_server_error(self, mock_qdrant):
        """Test 500 error handling."""
        mock_qdrant.list_collections.side_effect = Exception("Database error")
        
        response = client.get("/collections")
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data


class TestInputValidation:
    """Test input validation and security."""
    
    def test_invalid_limit_parameter(self):
        """Test validation of limit parameter."""
        # Test negative limit
        response = client.get("/collections/test/points?limit=-1")
        assert response.status_code == 422
        
        # Test limit too high
        response = client.get("/collections/test/points?limit=20000")
        assert response.status_code == 422
    
    def test_invalid_offset_parameter(self):
        """Test validation of offset parameter."""
        # Test negative offset
        response = client.get("/collections/test/points?offset=-1")
        assert response.status_code == 422
    
    @patch('api_server.get_cached_collections')
    def test_special_characters_in_collection_name(self, mock_cache):
        """Test handling of special characters in collection name."""
        mock_cache.return_value = {}
        
        # Test with SQL injection attempt
        response = client.get("/collections/'; DROP TABLE users; --/info")
        # Should return 404, not crash
        assert response.status_code in [404, 422]
        
        # Test with path traversal attempt
        response = client.get("/collections/../../../etc/passwd/info")
        assert response.status_code in [404, 422]


class TestPerformance:
    """Test performance-related functionality."""
    
    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_limit_parameter_caps_results(self, mock_cache, mock_qdrant):
        """Test that limit parameter properly caps results."""
        mock_cache.return_value = {"test_collection": {"size": 1536}}
        
        # Generate 1000 mock points
        mock_points = [
            {"id": i, "payload": {"source": f"test{i}.txt"}}
            for i in range(1000)
        ]
        mock_qdrant.scroll_vectors.return_value = (mock_points, None)
        
        response = client.get("/collections/test_collection/points?limit=10")
        assert response.status_code == 200
        data = response.json()
        # The API should respect the limit
        assert len(data["points"]) <= 10
    
    @patch('api_server.qdrant_service')
    def test_collections_cache_reduces_calls(self, mock_qdrant):
        """Test that caching reduces calls to Qdrant service."""
        # Reset global cache state
        import api_server
        api_server.COLLECTIONS_CACHE = {}
        api_server.COLLECTIONS_CACHE_TIME = 0
        
        mock_qdrant.list_collections.return_value = {"test": {"size": 1536}}
        
        # First call
        get_cached_collections()
        call_count_1 = mock_qdrant.list_collections.call_count
        
        # Second call immediately after (should use cache)
        get_cached_collections()
        call_count_2 = mock_qdrant.list_collections.call_count
        
        # Cache should prevent second call
        assert call_count_1 == call_count_2


class TestResponseFormat:
    """Test response format and data structure."""
    
    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_point_response_format(self, mock_cache, mock_qdrant):
        """Test that point responses have correct format."""
        mock_cache.return_value = {"test_collection": {"size": 1536}}
        
        mock_qdrant.scroll_vectors.return_value = (
            [
                {
                    "id": 1,
                    "payload": {
                        "source": "test.txt",
                        "cluster_id": 0,
                        "cluster_name": "TestCluster",
                        "predicted_label": "Label1",
                        "confidence": 0.95,
                        "text_content": "Sample text content for testing purposes"
                    }
                }
            ],
            None
        )
        
        response = client.get("/collections/test_collection/points")
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "collection" in data
        assert "points" in data
        assert "total_returned" in data
        
        # Check point structure
        point = data["points"][0]
        assert "id" in point
        assert "source" in point
        assert "cluster_id" in point
        assert "cluster_name" in point
        assert "predicted_label" in point
        assert "confidence" in point
        assert "text_content" in point
        
        # Check text content is truncated to 500 chars
        assert len(point["text_content"]) <= 500
    
    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_label_info_response_format(self, mock_cache, mock_qdrant):
        """Test that label info responses have correct format."""
        mock_cache.return_value = {"test_collection": {"size": 1536}}
        
        mock_qdrant.scroll_vectors.return_value = (
            [
                {"id": 1, "payload": {"predicted_label": "Label1"}},
                {"id": 2, "payload": {"predicted_label": "Label1"}},
            ],
            None
        )
        
        response = client.get("/collections/test_collection/labels")
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "collection" in data
        assert "labels" in data
        assert "total_labels" in data
        
        # Check label structure
        if data["labels"]:
            label = data["labels"][0]
            assert "label" in label
            assert "count" in label
    
    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_cluster_info_response_format(self, mock_cache, mock_qdrant):
        """Test that cluster info responses have correct format."""
        mock_cache.return_value = {"test_collection": {"size": 1536}}
        
        mock_qdrant.scroll_vectors.return_value = (
            [
                {"id": 1, "payload": {"cluster_id": 0, "cluster_name": "Cluster_0"}},
            ],
            None
        )
        
        response = client.get("/collections/test_collection/clusters")
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "collection" in data
        assert "clusters" in data
        assert "total_clusters" in data
        
        # Check cluster structure
        if data["clusters"]:
            cluster = data["clusters"][0]
            assert "cluster_id" in cluster
            assert "cluster_name" in cluster
            assert "count" in cluster


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_empty_collection(self, mock_cache, mock_qdrant):
        """Test handling of empty collections."""
        mock_cache.return_value = {"empty_collection": {"size": 1536}}
        mock_qdrant.scroll_vectors.return_value = ([], None)
        
        response = client.get("/collections/empty_collection/points")
        assert response.status_code == 200
        data = response.json()
        assert data["points"] == []
        assert data["total_returned"] == 0
    
    @patch('api_server.qdrant_service')
    @patch('api_server.get_cached_collections')
    def test_points_without_optional_fields(self, mock_cache, mock_qdrant):
        """Test handling of points missing optional fields."""
        mock_cache.return_value = {"test_collection": {"size": 1536}}
        
        # Point with minimal payload
        mock_qdrant.scroll_vectors.return_value = (
            [
                {
                    "id": 1,
                    "payload": {
                        "source": "test.txt"
                        # No cluster_id, cluster_name, predicted_label, etc.
                    }
                }
            ],
            None
        )
        
        response = client.get("/collections/test_collection/points")
        assert response.status_code == 200
        data = response.json()
        
        point = data["points"][0]
        assert point["id"] == 1
        assert point["source"] == "test.txt"
        assert point["cluster_id"] is None
        assert point["predicted_label"] is None


class TestCollectionMetadata:
    """Test collection metadata features (created_at, description, clustering info)."""
    
    @patch('api_server.qdrant_service')
    def test_collection_info_with_metadata(self, mock_qdrant):
        """Test collection info endpoint returns metadata fields."""
        from datetime import datetime
        
        # Mock get_cached_collections
        mock_qdrant.list_collections.return_value = {"test_coll": {}}
        
        # Mock get_collection_info
        mock_qdrant.get_collection_info.return_value = {
            "name": "test_coll",
            "vector_count": 100,
            "dimension": 1536,
            "distance_metric": "Cosine",
            "embedding_model": "text-embedding-3-small",
            "created_at": "2025-10-08T14:30:00",
            "description": "Test collection for metadata"
        }
        
        # Mock get_collection_metadata for clustering info
        mock_qdrant.get_collection_metadata.return_value = {
            "_metadata": True,
            "created_at": "2025-10-08T14:30:00",
            "embedding_model": "text-embedding-3-small",
            "description": "Test collection for metadata",
            "clustering_algorithm": "kmeans",
            "num_clusters": 5
        }
        
        with patch('api_server.get_cached_collections', return_value={"test_coll": {}}):
            response = client.get("/collections/test_coll/info")
            assert response.status_code == 200
            data = response.json()
            
            assert data["name"] == "test_coll"
            assert data["created_at"] == "2025-10-08T14:30:00"
            assert data["description"] == "Test collection for metadata"
            assert data["clustering_algorithm"] == "kmeans"
            assert data["num_clusters"] == 5
    
    @patch('api_server.qdrant_service')
    def test_collection_info_without_metadata(self, mock_qdrant):
        """Test collection info for old collections without metadata."""
        # Mock get_cached_collections
        mock_qdrant.list_collections.return_value = {"old_coll": {}}
        
        # Mock get_collection_info (no metadata)
        mock_qdrant.get_collection_info.return_value = {
            "name": "old_coll",
            "vector_count": 50,
            "dimension": 1536,
            "distance_metric": "Cosine",
            "embedding_model": None,
            "created_at": None,
            "description": None
        }
        
        # Mock get_collection_metadata (no metadata point)
        mock_qdrant.get_collection_metadata.return_value = None
        
        with patch('api_server.get_cached_collections', return_value={"old_coll": {}}):
            response = client.get("/collections/old_coll/info")
            assert response.status_code == 200
            data = response.json()
            
            assert data["name"] == "old_coll"
            assert data["created_at"] is None
            assert data["description"] is None
            assert data["clustering_algorithm"] is None
            assert data["num_clusters"] is None
    
    @patch('api_server.qdrant_service')
    def test_list_collections_with_clustering_metadata(self, mock_qdrant):
        """Test /collections endpoint includes clustering metadata."""
        # Mock list_collections
        mock_qdrant.list_collections.return_value = {
            "clustered_coll": {
                "size": 1536,
                "vectors": 200,
                "distance": "Cosine",
                "created_at": "2025-10-08T14:30:00",
                "description": "Clustered collection"
            }
        }
        
        # Mock get_collection_metadata
        def mock_get_metadata(name):
            if name == "clustered_coll":
                return {
                    "clustering_algorithm": "agglomerative",
                    "num_clusters": 10
                }
            return None
        
        mock_qdrant.get_collection_metadata.side_effect = mock_get_metadata
        
        response = client.get("/collections")
        assert response.status_code == 200
        data = response.json()
        
        assert "collections" in data
        assert "clustered_coll" in data["collections"]
        coll_info = data["collections"]["clustered_coll"]
        assert coll_info["clustering_algorithm"] == "agglomerative"
        assert coll_info["num_clusters"] == 10
        assert coll_info["created_at"] == "2025-10-08T14:30:00"
        assert coll_info["description"] == "Clustered collection"
    
    @patch('api_server.qdrant_service')
    def test_description_truncation_not_in_api(self, mock_qdrant):
        """Test that API returns full description (truncation is CLI-only)."""
        long_description = "A" * 200  # 200 character description
        
        mock_qdrant.list_collections.return_value = {"test_coll": {}}
        mock_qdrant.get_collection_info.return_value = {
            "name": "test_coll",
            "vector_count": 100,
            "dimension": 1536,
            "distance_metric": "Cosine",
            "embedding_model": "text-embedding-3-small",
            "created_at": "2025-10-08T14:30:00",
            "description": long_description
        }
        mock_qdrant.get_collection_metadata.return_value = {
            "description": long_description
        }
        
        with patch('api_server.get_cached_collections', return_value={"test_coll": {}}):
            response = client.get("/collections/test_coll/info")
            assert response.status_code == 200
            data = response.json()
            
            # API should return full description, no truncation
            assert data["description"] == long_description
            assert len(data["description"]) == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

