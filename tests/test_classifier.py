"""
Unit tests for DocumentClassifier

Tests document classification, label management, and enrichment functionality.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from src.pipelines.classification.classifier import DocumentClassifier


class TestClassifierInitialization:
    """Test document classifier initialization."""
    
    def test_init_default(self):
        """Test default initialization."""
        mock_qdrant = Mock()
        classifier = DocumentClassifier(mock_qdrant)
        
        assert classifier.qdrant_service == mock_qdrant
        assert classifier.log is not None
    
    def test_init_custom_log(self):
        """Test initialization with custom log function."""
        mock_qdrant = Mock()
        mock_log = Mock()
        classifier = DocumentClassifier(mock_qdrant, log_function=mock_log)
        
        assert classifier.log == mock_log


class TestTaxonomyLoading:
    """Test taxonomy loading functionality."""
    
    def test_load_taxonomy_success(self, tmp_path):
        """Test successful taxonomy loading."""
        # Create test taxonomy file
        taxonomy_data = {
            "1": {"label": "Sports", "description": "Sports content"},
            "2": {"label": "Technology", "description": "Tech content"}
        }
        taxonomy_path = tmp_path / "taxonomy.json"
        with open(taxonomy_path, "w", encoding="utf-8") as f:
            json.dump(taxonomy_data, f)
        
        mock_qdrant = Mock()
        classifier = DocumentClassifier(mock_qdrant)
        
        result = classifier.load_taxonomy(str(taxonomy_path))
        
        assert result == taxonomy_data
        assert "1" in result
        assert result["1"]["label"] == "Sports"
    
    def test_load_taxonomy_nonexistent(self):
        """Test loading nonexistent taxonomy file."""
        mock_qdrant = Mock()
        classifier = DocumentClassifier(mock_qdrant)
        
        result = classifier.load_taxonomy("/nonexistent/taxonomy.json")
        
        assert result is None


class TestDocumentClassification:
    """Test document classification functionality."""
    
    @pytest.fixture
    def mock_classifier(self):
        """Create classifier with mocked Qdrant service."""
        mock_qdrant = Mock()
        classifier = DocumentClassifier(mock_qdrant, log_function=Mock())
        return classifier, mock_qdrant
    
    def test_classify_documents_success(self, mock_classifier, tmp_path):
        """Test successful document classification."""
        classifier, mock_qdrant = mock_classifier
        
        # Create labels file
        labels_data = {
            "1": {"label": "Sports", "description": "Sports content"},
            "2": {"label": "Technology", "description": "Tech content"}
        }
        labels_path = tmp_path / "labels.json"
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels_data, f)
        
        # Mock collection info
        mock_qdrant.get_collection_info.return_value = {"vector_count": 2}
        
        # Mock scroll_vectors to return test points
        mock_points = [
            {"id": 1, "vector": [0.1] * 1536, "payload": {"source": "doc1"}},
            {"id": 2, "vector": [0.2] * 1536, "payload": {"source": "doc2"}}
        ]
        mock_qdrant.scroll_vectors.return_value = (mock_points, None)
        
        # Mock embed function
        with patch('src.pipelines.classification.classifier.embed') as mock_embed:
            mock_embed.return_value = [[0.15] * 1536, [0.25] * 1536]
            
            result = classifier.classify_documents("test_collection", labels_file=str(labels_path))
        
        assert result["success"] is True
        assert result["classified_count"] == 2
        assert result["total_documents"] == 2
        assert result["labels_used"] == 2
    
    def test_classify_documents_no_labels_file(self, mock_classifier):
        """Test classification without labels file."""
        classifier, _ = mock_classifier
        
        result = classifier.classify_documents("test_collection")
        
        assert result["success"] is False
        assert "error" in result
    
    def test_classify_documents_with_collection_labels(self, mock_classifier):
        """Test classification using collection-stored labels."""
        classifier, mock_qdrant = mock_classifier
        
        # Mock loading labels from collection
        with patch.object(classifier, '_load_labels_from_collection') as mock_load:
            mock_load.return_value = {
                "1": {"label": "Sports", "description": "Sports content"}
            }
            
            # Mock collection info
            mock_qdrant.get_collection_info.return_value = {"vector_count": 1}
            
            # Mock scroll_vectors
            mock_points = [
                {"id": 1, "vector": [0.1] * 1536, "payload": {"source": "doc1"}}
            ]
            mock_qdrant.scroll_vectors.return_value = (mock_points, None)
            
            # Mock embed
            with patch('src.pipelines.classification.classifier.embed') as mock_embed:
                mock_embed.return_value = [[0.15] * 1536]
                
                result = classifier.classify_documents("test_collection", use_collection_labels=True)
        
        assert result["success"] is True
    
    def test_classify_documents_with_enrichment(self, mock_classifier, tmp_path):
        """Test classification with label enrichment."""
        classifier, mock_qdrant = mock_classifier
        
        # Create labels file
        labels_data = {
            "1": {"label": "Sports"}
        }
        labels_path = tmp_path / "labels.json"
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels_data, f)
        
        # Mock enrichment
        with patch.object(classifier, '_enrich_labels_data') as mock_enrich:
            mock_enrich.return_value = {
                "1": {"label": "Sports", "description": "Sports content", "enriched": True}
            }
            
            # Mock collection operations
            mock_qdrant.get_collection_info.return_value = {"vector_count": 1}
            mock_points = [
                {"id": 1, "vector": [0.1] * 1536, "payload": {"source": "doc1"}}
            ]
            mock_qdrant.scroll_vectors.return_value = (mock_points, None)
            
            with patch('src.pipelines.classification.classifier.embed') as mock_embed:
                mock_embed.return_value = [[0.15] * 1536]
                
                result = classifier.classify_documents(
                    "test_collection",
                    labels_file=str(labels_path),
                    enrich_labels=True
                )
        
        assert result["success"] is True
        mock_enrich.assert_called_once()


class TestLabelEnrichment:
    """Test label enrichment functionality."""
    
    @pytest.fixture
    def mock_classifier(self):
        """Create classifier with mocked Qdrant service."""
        mock_qdrant = Mock()
        classifier = DocumentClassifier(mock_qdrant, log_function=Mock())
        return classifier, mock_qdrant
    
    def test_enrich_labels_success(self, mock_classifier, tmp_path):
        """Test successful label enrichment."""
        classifier, mock_qdrant = mock_classifier
        
        # Create labels file
        labels_data = {
            "1": {"label": "Sports"},
            "2": "Technology"
        }
        labels_path = tmp_path / "labels.json"
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels_data, f)
        
        # Mock enrichment
        with patch.object(classifier, '_enrich_labels_data') as mock_enrich:
            mock_enrich.return_value = {
                "1": {"label": "Sports", "description": "Sports content", "enriched": True},
                "2": {"label": "Technology", "description": "Tech content", "enriched": True}
            }
            
            result = classifier.enrich_labels(str(labels_path))
        
        assert result["success"] is True
        assert "enriched_labels" in result
    
    def test_enrich_labels_with_storage(self, mock_classifier, tmp_path):
        """Test label enrichment with collection storage."""
        classifier, mock_qdrant = mock_classifier
        
        # Create labels file
        labels_data = {"1": {"label": "Sports"}}
        labels_path = tmp_path / "labels.json"
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels_data, f)
        
        # Mock enrichment and storage
        with patch.object(classifier, '_enrich_labels_data') as mock_enrich:
            with patch.object(classifier, '_store_labels_in_collection') as mock_store:
                mock_enrich.return_value = {
                    "1": {"label": "Sports", "description": "Sports content", "enriched": True}
                }
                
                result = classifier.enrich_labels(
                    str(labels_path),
                    store_in_collection=True,
                    collection_name="test_collection"
                )
        
        assert result["success"] is True
        mock_store.assert_called_once()


class TestLabelManagement:
    """Test label management functionality."""
    
    @pytest.fixture
    def mock_classifier(self):
        """Create classifier with mocked Qdrant service."""
        mock_qdrant = Mock()
        classifier = DocumentClassifier(mock_qdrant, log_function=Mock())
        return classifier, mock_qdrant
    
    def test_add_label_to_collection_success(self, mock_classifier):
        """Test successfully adding label to collection."""
        classifier, mock_qdrant = mock_classifier
        
        # Mock next ID
        mock_qdrant._get_next_id.return_value = 100
        
        # Mock embedding
        with patch('src.pipelines.classification.classifier.embed') as mock_embed:
            mock_embed.return_value = [[0.1] * 1536]
            
            result = classifier.add_label_to_collection(
                "test_collection",
                "NewLabel",
                "Description of new label"
            )
        
        assert result["success"] is True
        assert "label_id" in result
        assert "point_id" in result
        mock_qdrant.client.upsert.assert_called_once()
    
    def test_add_label_without_description(self, mock_classifier):
        """Test adding label without explicit description."""
        classifier, mock_qdrant = mock_classifier
        
        # Mock description generation
        with patch.object(classifier, '_generate_label_description') as mock_gen_desc:
            mock_gen_desc.return_value = "Auto-generated description"
            
            # Mock next ID
            mock_qdrant._get_next_id.return_value = 100
            
            # Mock embedding
            with patch('src.pipelines.classification.classifier.embed') as mock_embed:
                mock_embed.return_value = [[0.1] * 1536]
                
                result = classifier.add_label_to_collection(
                    "test_collection",
                    "NewLabel",
                    enrich=True
                )
        
        assert result["success"] is True
        mock_gen_desc.assert_called_once()
    
    def test_get_collection_labels_success(self, mock_classifier):
        """Test getting labels from collection."""
        classifier, _ = mock_classifier
        
        # Mock loading labels
        with patch.object(classifier, '_load_labels_from_collection') as mock_load:
            mock_load.return_value = {
                "1": {"label": "Sports", "description": "Sports content"}
            }
            
            result = classifier.get_collection_labels("test_collection")
        
        assert result["success"] is True
        assert "labels" in result
        assert "1" in result["labels"]
    
    def test_load_labels_from_collection(self, mock_classifier):
        """Test loading labels from collection."""
        classifier, mock_qdrant = mock_classifier
        
        # Mock scroll_vectors to return label points
        mock_label_points = [
            {
                "id": 1,
                "payload": {
                    "label_id": "1",
                    "label_name": "Sports",
                    "description": "Sports content",
                    "enriched": True,
                    "type": "label"
                }
            }
        ]
        mock_qdrant.scroll_vectors.return_value = (mock_label_points, None)
        
        result = classifier._load_labels_from_collection("test_collection")
        
        assert "1" in result
        assert result["1"]["label"] == "Sports"
        assert result["1"]["enriched"] is True


class TestLabelDescriptionGeneration:
    """Test AI description generation for labels."""
    
    @pytest.fixture
    def mock_classifier(self):
        """Create classifier with mocked Qdrant service."""
        mock_qdrant = Mock()
        classifier = DocumentClassifier(mock_qdrant, log_function=Mock())
        return classifier
    
    def test_generate_label_description_with_api(self, mock_classifier):
        """Test description generation with API."""
        classifier = mock_classifier
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            # Patch the OpenAI import inside the function
            with patch('openai.OpenAI') as mock_openai:
                # Mock API response
                mock_client = Mock()
                mock_response = Mock()
                mock_response.choices = [Mock(message=Mock(content="AI-generated description"))]
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client
                
                result = classifier._generate_label_description("Sports")
        
        assert result == "AI-generated description"
    
    def test_generate_label_description_without_api(self, mock_classifier):
        """Test description generation without API."""
        classifier = mock_classifier
        
        with patch.dict('os.environ', {}, clear=True):
            result = classifier._generate_label_description("Technology")
        
        assert "technology" in result.lower()
    
    def test_enrich_labels_data_with_api(self, mock_classifier):
        """Test enriching labels data with API."""
        classifier = mock_classifier
        
        labels_data = {
            "1": {"label": "Sports"},
            "2": "Technology"
        }
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            with patch.object(classifier, '_generate_label_description') as mock_gen:
                mock_gen.return_value = "AI description"
                
                result = classifier._enrich_labels_data(labels_data)
        
        assert result["1"]["enriched"] is True
        assert result["2"]["enriched"] is True


class TestLabelStorage:
    """Test label storage in collection."""
    
    @pytest.fixture
    def mock_classifier(self):
        """Create classifier with mocked Qdrant service."""
        mock_qdrant = Mock()
        classifier = DocumentClassifier(mock_qdrant, log_function=Mock())
        return classifier, mock_qdrant
    
    def test_store_labels_in_collection(self, mock_classifier):
        """Test storing labels in collection."""
        classifier, mock_qdrant = mock_classifier
        
        labels_data = {
            "1": {"label": "Sports", "description": "Sports content", "enriched": True},
            "2": {"label": "Technology", "description": "Tech content", "enriched": True}
        }
        
        # Mock next ID
        mock_qdrant._get_next_id.return_value = 100
        
        # Mock embedding
        with patch('src.pipelines.classification.classifier.embed') as mock_embed:
            mock_embed.return_value = [[0.1] * 1536, [0.2] * 1536]
            
            classifier._store_labels_in_collection("test_collection", labels_data)
        
        # Verify upsert was called
        mock_qdrant.client.upsert.assert_called_once()
        call_args = mock_qdrant.client.upsert.call_args
        assert call_args[1]["collection_name"] == "test_collection"
        assert len(call_args[1]["points"]) == 2


class TestConvenienceFunctions:
    """Test convenience functions for backward compatibility."""
    
    def test_classify_documents_convenience(self, tmp_path):
        """Test classify_documents convenience function."""
        from src.pipelines.classification.classifier import classify_documents
        
        mock_qdrant = Mock()
        
        # Create labels file
        labels_data = {"1": {"label": "Sports"}}
        labels_path = tmp_path / "labels.json"
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels_data, f)
        
        # Mock collection operations
        mock_qdrant.get_collection_info.return_value = {"vector_count": 1}
        mock_points = [{"id": 1, "vector": [0.1] * 1536, "payload": {}}]
        mock_qdrant.scroll_vectors.return_value = (mock_points, None)
        
        with patch('src.pipelines.classification.classifier.embed') as mock_embed:
            mock_embed.return_value = [[0.15] * 1536]
            
            result = classify_documents(
                mock_qdrant,
                "test_collection",
                labels_file=str(labels_path)
            )
        
        assert result["success"] is True
    
    def test_enrich_labels_convenience(self, tmp_path):
        """Test enrich_labels convenience function."""
        from src.pipelines.classification.classifier import enrich_labels
        
        mock_qdrant = Mock()
        
        # Create labels file
        labels_data = {"1": {"label": "Sports"}}
        labels_path = tmp_path / "labels.json"
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels_data, f)
        
        with patch.dict('os.environ', {}, clear=True):
            result = enrich_labels(mock_qdrant, str(labels_path))
        
        assert result["success"] is True
    
    def test_add_label_convenience(self):
        """Test add_label_to_collection convenience function."""
        from src.pipelines.classification.classifier import add_label_to_collection
        
        mock_qdrant = Mock()
        mock_qdrant._get_next_id.return_value = 100
        
        with patch('src.pipelines.classification.classifier.embed') as mock_embed:
            mock_embed.return_value = [[0.1] * 1536]
            
            with patch.dict('os.environ', {}, clear=True):
                result = add_label_to_collection(
                    mock_qdrant,
                    "test_collection",
                    "NewLabel",
                    "Description"
                )
        
        assert result["success"] is True

