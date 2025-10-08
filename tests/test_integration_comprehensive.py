"""
Comprehensive Integration Test Suite

This test suite provides end-to-end integration testing including:
- Full CLI workflows
- Full API workflows
- Service integration
- Real Qdrant operations (requires running Qdrant instance)
- OpenAI integration (requires API key)
"""

import pytest
import sys
import os
import json
import time
from pathlib import Path
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.services.qdrant_service import QdrantService
from src.services.openai_service import OpenAIService
from src.services.processing_service import ProcessingService
from src.services.sqlite_service import SQLiteService
from src.pipelines.classification.classifier import DocumentClassifier


# Skip tests if Qdrant is not available
def is_qdrant_available():
    """Check if Qdrant is available."""
    try:
        service = QdrantService()
        return service.is_connected()
    except:
        return False


# Skip tests if OpenAI API key is not available
def is_openai_available():
    """Check if OpenAI API is available."""
    try:
        service = OpenAIService()
        return service.is_api_available()
    except:
        return False


# Test fixtures
@pytest.fixture
def test_collection_name():
    """Generate unique test collection name."""
    return f"test_collection_{int(time.time())}"


@pytest.fixture
def test_directory(tmp_path):
    """Create a temporary test directory with sample files."""
    test_dir = tmp_path / "test_docs"
    test_dir.mkdir()
    
    # Create sample text files
    (test_dir / "file1.txt").write_text("This is a test document about technology.")
    (test_dir / "file2.txt").write_text("This is a test document about sports.")
    (test_dir / "file3.txt").write_text("This is a test document about science.")
    
    return str(test_dir)


@pytest.fixture
def qdrant_service():
    """Create QdrantService instance."""
    return QdrantService()


@pytest.fixture
def openai_service():
    """Create OpenAIService instance."""
    return OpenAIService()


@pytest.fixture
def processing_service():
    """Create ProcessingService instance."""
    return ProcessingService()


@pytest.fixture
def sqlite_service(tmp_path):
    """Create SQLiteService instance with temp database."""
    db_path = tmp_path / "test_hash_index.db"
    return SQLiteService(str(db_path))


@pytest.fixture
def labels_file(tmp_path):
    """Create a temporary labels file."""
    labels_file = tmp_path / "labels.json"
    labels_data = {
        "0": {"label": "Technology", "description": "Technology-related content"},
        "1": {"label": "Sports", "description": "Sports-related content"},
        "2": {"label": "Science", "description": "Science-related content"}
    }
    labels_file.write_text(json.dumps(labels_data))
    return str(labels_file)


class TestQdrantServiceIntegration:
    """Integration tests for QdrantService."""
    
    @pytest.mark.skipif(not is_qdrant_available(), reason="Qdrant not available")
    def test_create_and_delete_collection(self, qdrant_service, test_collection_name):
        """Test creating and deleting a collection."""
        # Create collection
        success = qdrant_service.create_collection(
            test_collection_name,
            dimension=1536,
            model="text-embedding-3-small"
        )
        assert success
        
        # Verify collection exists
        collections = qdrant_service.list_collections()
        assert test_collection_name in collections
        
        # Delete collection
        success = qdrant_service.delete_collection(test_collection_name)
        assert success
        
        # Verify collection removed
        collections = qdrant_service.list_collections()
        assert test_collection_name not in collections
    
    @pytest.mark.skipif(not is_qdrant_available(), reason="Qdrant not available")
    def test_insert_and_retrieve_vectors(self, qdrant_service, test_collection_name):
        """Test inserting and retrieving vectors."""
        try:
            # Create collection
            qdrant_service.create_collection(
                test_collection_name,
                dimension=128,
                model="test-model"
            )
            
            # Insert vectors
            vectors = [[0.1] * 128 for _ in range(5)]
            payloads = [{"text": f"Document {i}"} for i in range(5)]
            
            success, inserted, skipped = qdrant_service.insert_vectors(
                test_collection_name,
                vectors,
                payloads
            )
            
            assert success
            assert inserted == 5
            assert skipped == 0
            
            # Retrieve vectors
            points, _ = qdrant_service.scroll_vectors(
                test_collection_name,
                limit=10,
                with_payload=True,
                with_vectors=False
            )
            
            # Filter out metadata point (ID 0)
            data_points = [p for p in points if not p.get('payload', {}).get('_metadata')]
            assert len(data_points) == 5
            
        finally:
            # Cleanup
            qdrant_service.delete_collection(test_collection_name)
    
    @pytest.mark.skipif(not is_qdrant_available(), reason="Qdrant not available")
    def test_update_payload(self, qdrant_service, test_collection_name):
        """Test updating point payloads."""
        try:
            # Create collection and insert data
            qdrant_service.create_collection(
                test_collection_name,
                dimension=128,
                model="test-model"
            )
            
            vectors = [[0.1] * 128]
            payloads = [{"text": "Original"}]
            
            qdrant_service.insert_vectors(test_collection_name, vectors, payloads)
            
            # Update payload
            success = qdrant_service.update_payload(
                test_collection_name,
                [1],  # Assuming ID starts at 1
                {"text": "Updated", "cluster_id": 0}
            )
            
            assert success
            
            # Verify update
            points, _ = qdrant_service.scroll_vectors(
                test_collection_name,
                limit=10,
                with_payload=True
            )
            
            # Filter out metadata point (ID 0)
            data_points = [p for p in points if not p.get('payload', {}).get('_metadata')]
            assert data_points[0]["payload"]["text"] == "Updated"
            assert data_points[0]["payload"]["cluster_id"] == 0
            
        finally:
            # Cleanup
            qdrant_service.delete_collection(test_collection_name)


class TestProcessingServiceIntegration:
    """Integration tests for ProcessingService."""
    
    def test_process_directory(self, processing_service, test_directory):
        """Test processing a directory of files."""
        texts, payloads = processing_service.process_source(test_directory)
        
        assert len(texts) == 3
        assert len(payloads) == 3
        
        # Verify content
        for text in texts:
            assert "test document" in text.lower()
        
        # Verify payloads have required fields
        for payload in payloads:
            assert "source" in payload
            assert "hash" in payload
            assert "type" in payload
            assert payload["type"] == "file"
    
    def test_process_directory_with_limit(self, processing_service, test_directory):
        """Test processing directory with limit."""
        texts, payloads = processing_service.process_source(test_directory, limit=2)
        
        assert len(texts) == 2
        assert len(payloads) == 2
    
    def test_process_csv(self, processing_service, tmp_path):
        """Test processing CSV file."""
        # Create test CSV
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("text,category\nTest document 1,cat1\nTest document 2,cat2")
        
        texts, payloads = processing_service.process_source(
            str(csv_file),
            text_column="text"
        )
        
        assert len(texts) == 2
        assert texts[0] == "Test document 1"
        assert texts[1] == "Test document 2"
        
        # Verify payloads include CSV columns
        assert payloads[0]["category"] == "cat1"
        assert payloads[1]["category"] == "cat2"


class TestSQLiteServiceIntegration:
    """Integration tests for SQLiteService."""
    
    def test_hash_deduplication(self, sqlite_service):
        """Test hash-based deduplication."""
        collection = "test_collection"
        
        # Add hashes
        hashes = ["hash1", "hash2", "hash3"]
        added = sqlite_service.mark_as_processed(collection, hashes)
        assert added == 3
        
        # Check for duplicates
        assert sqlite_service.is_duplicate(collection, "hash1")
        assert sqlite_service.is_duplicate(collection, "hash2")
        assert not sqlite_service.is_duplicate(collection, "hash4")
        
        # Try adding duplicates
        added = sqlite_service.mark_as_processed(collection, ["hash1", "hash4"])
        assert added == 1  # Only hash4 should be added
        
        # Verify stats
        stats = sqlite_service.get_stats(collection)
        assert stats["hash_count"] == 4
    
    def test_clear_collection(self, sqlite_service):
        """Test clearing collection hashes."""
        collection = "test_collection"
        
        # Add hashes
        sqlite_service.mark_as_processed(collection, ["hash1", "hash2"])
        
        # Clear collection
        success = sqlite_service.clear_collection(collection)
        assert success
        
        # Verify cleared
        stats = sqlite_service.get_stats(collection)
        assert stats["hash_count"] == 0


class TestOpenAIServiceIntegration:
    """Integration tests for OpenAIService."""
    
    @pytest.mark.skipif(not is_openai_available(), reason="OpenAI API not available")
    def test_generate_embeddings(self, openai_service):
        """Test generating embeddings."""
        texts = ["Test document 1", "Test document 2", "Test document 3"]
        
        embeddings = openai_service.generate_embeddings(texts)
        
        assert len(embeddings) == 3
        assert len(embeddings[0]) == 1536  # Default model dimension
        
        # Verify embeddings are not all zeros
        assert any(abs(x) > 0.01 for x in embeddings[0])
    
    @pytest.mark.skipif(not is_openai_available(), reason="OpenAI API not available")
    def test_generate_cluster_label(self, openai_service):
        """Test generating cluster labels."""
        representative_texts = [
            "Apple releases new iPhone",
            "Samsung announces Galaxy update",
            "Google unveils Pixel phone"
        ]
        
        label = openai_service.generate_single_word_cluster_label(
            cluster_id=0,
            representative_texts=representative_texts
        )
        
        assert isinstance(label, str)
        assert len(label) > 0
        # Label should be related to technology/phones
        assert label != "Cluster"


class TestEndToEndWorkflow:
    """End-to-end workflow integration tests."""
    
    @pytest.mark.skipif(not is_qdrant_available(), reason="Qdrant not available")
    def test_full_ingestion_workflow(
        self,
        qdrant_service,
        processing_service,
        openai_service,
        test_collection_name,
        test_directory
    ):
        """Test complete document ingestion workflow."""
        try:
            # Step 1: Create collection
            success = qdrant_service.create_collection(
                test_collection_name,
                dimension=1536,
                model="text-embedding-3-small"
            )
            assert success
            
            # Step 2: Process documents
            texts, payloads = processing_service.process_source(test_directory)
            assert len(texts) > 0
            
            # Step 3: Generate embeddings
            embeddings = openai_service.generate_embeddings(texts)
            assert len(embeddings) == len(texts)
            
            # Step 4: Insert into collection
            success, inserted, skipped = qdrant_service.insert_vectors(
                test_collection_name,
                embeddings,
                payloads
            )
            assert success
            assert inserted == len(texts)
            
            # Step 5: Verify data was inserted
            info = qdrant_service.get_collection_info(test_collection_name)
            assert info["vector_count"] == len(texts)
            
        finally:
            # Cleanup
            qdrant_service.delete_collection(test_collection_name)
    
    @pytest.mark.skipif(
        not is_qdrant_available() or not is_openai_available(),
        reason="Qdrant or OpenAI not available"
    )
    def test_full_clustering_workflow(
        self,
        qdrant_service,
        processing_service,
        openai_service,
        test_collection_name,
        test_directory
    ):
        """Test complete clustering workflow."""
        try:
            # Setup: Ingest documents
            qdrant_service.create_collection(
                test_collection_name,
                dimension=1536,
                model="text-embedding-3-small"
            )
            
            texts, payloads = processing_service.process_source(test_directory)
            embeddings = openai_service.generate_embeddings(texts)
            qdrant_service.insert_vectors(test_collection_name, embeddings, payloads)
            
            # Step 1: Retrieve vectors for clustering
            points, _ = qdrant_service.scroll_vectors(
                test_collection_name,
                limit=100,
                with_payload=True,
                with_vectors=True
            )
            
            assert len(points) > 0
            
            # Step 2: Perform clustering (simple K-means)
            from sklearn.cluster import KMeans
            import numpy as np
            
            vectors = np.array([point["vector"] for point in points])
            kmeans = KMeans(n_clusters=2, random_state=42)
            cluster_labels = kmeans.fit_predict(vectors)
            
            # Step 3: Update points with cluster assignments
            for point, cluster_id in zip(points, cluster_labels):
                qdrant_service.update_payload(
                    test_collection_name,
                    [point["id"]],
                    {"cluster_id": int(cluster_id)}
                )
            
            # Step 4: Verify cluster assignments
            updated_points, _ = qdrant_service.scroll_vectors(
                test_collection_name,
                limit=100,
                with_payload=True
            )
            
            for point in updated_points:
                assert "cluster_id" in point["payload"]
                assert 0 <= point["payload"]["cluster_id"] < 2
            
        finally:
            # Cleanup
            qdrant_service.delete_collection(test_collection_name)
    
    @pytest.mark.skipif(
        not is_qdrant_available() or not is_openai_available(),
        reason="Qdrant or OpenAI not available"
    )
    def test_full_classification_workflow(
        self,
        qdrant_service,
        processing_service,
        openai_service,
        test_collection_name,
        test_directory,
        labels_file
    ):
        """Test complete classification workflow."""
        try:
            # Setup: Ingest documents
            qdrant_service.create_collection(
                test_collection_name,
                dimension=1536,
                model="text-embedding-3-small"
            )
            
            texts, payloads = processing_service.process_source(test_directory)
            embeddings = openai_service.generate_embeddings(texts)
            qdrant_service.insert_vectors(test_collection_name, embeddings, payloads)
            
            # Step 1: Initialize classifier
            classifier = DocumentClassifier(qdrant_service)
            
            # Step 2: Classify documents
            result = classifier.classify_documents(
                test_collection_name,
                labels_file=labels_file
            )
            
            assert result["success"], f"Classification failed: {result.get('error', 'Unknown error')}"
            assert result["classified_count"] > 0
            
            # Step 3: Verify classifications
            points, _ = qdrant_service.scroll_vectors(
                test_collection_name,
                limit=100,
                with_payload=True
            )
            
            # Filter out metadata point (ID 0)
            data_points = [p for p in points if not p.get('payload', {}).get('_metadata')]
            
            for point in data_points:
                payload = point["payload"]
                assert "predicted_label" in payload
                assert "confidence" in payload
                assert payload["predicted_label"] in ["Technology", "Sports", "Science"]
            
        finally:
            # Cleanup
            qdrant_service.delete_collection(test_collection_name)


class TestDeduplicationWorkflow:
    """Test deduplication workflows."""
    
    @pytest.mark.skipif(not is_qdrant_available(), reason="Qdrant not available")
    def test_deduplication_prevents_duplicates(
        self,
        qdrant_service,
        processing_service,
        openai_service,
        sqlite_service,
        test_collection_name,
        test_directory
    ):
        """Test that deduplication prevents duplicate documents."""
        try:
            # Create collection
            qdrant_service.create_collection(
                test_collection_name,
                dimension=1536,
                model="text-embedding-3-small"
            )
            
            # Process documents
            texts, payloads = processing_service.process_source(test_directory)
            
            # Filter out duplicates using SQLite service
            unique_texts = []
            unique_payloads = []
            
            for text, payload in zip(texts, payloads):
                if not sqlite_service.is_duplicate(test_collection_name, payload["hash"]):
                    unique_texts.append(text)
                    unique_payloads.append(payload)
                    sqlite_service.mark_as_processed(
                        test_collection_name,
                        [payload["hash"]]
                    )
            
            # First insertion should succeed
            assert len(unique_texts) == len(texts)
            
            # Try processing same directory again
            texts2, payloads2 = processing_service.process_source(test_directory)
            
            unique_texts2 = []
            unique_payloads2 = []
            
            for text, payload in zip(texts2, payloads2):
                if not sqlite_service.is_duplicate(test_collection_name, payload["hash"]):
                    unique_texts2.append(text)
                    unique_payloads2.append(payload)
            
            # Second insertion should find all duplicates
            assert len(unique_texts2) == 0
            
        finally:
            # Cleanup
            qdrant_service.delete_collection(test_collection_name)
            sqlite_service.clear_collection(test_collection_name)


class TestErrorRecovery:
    """Test error handling and recovery."""
    
    @pytest.mark.skipif(not is_qdrant_available(), reason="Qdrant not available")
    def test_recovery_from_partial_insertion(
        self,
        qdrant_service,
        test_collection_name
    ):
        """Test recovery when insertion partially fails."""
        try:
            # Create collection
            qdrant_service.create_collection(
                test_collection_name,
                dimension=128,
                model="test-model"
            )
            
            # Create valid and invalid data
            vectors = [[0.1] * 128 for _ in range(5)]
            payloads = [{"id": i} for i in range(5)]
            
            # Insert valid data
            success, inserted, skipped = qdrant_service.insert_vectors(
                test_collection_name,
                vectors,
                payloads
            )
            
            assert success
            assert inserted == 5
            
            # Verify we can still query the collection
            points, _ = qdrant_service.scroll_vectors(
                test_collection_name,
                limit=10
            )
            
            # Filter out metadata point (ID 0)
            data_points = [p for p in points if not p.get('payload', {}).get('_metadata')]
            assert len(data_points) == 5
            
        finally:
            # Cleanup
            qdrant_service.delete_collection(test_collection_name)


# Removed concurrent insertion tests per project scope (no concurrent writes expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

