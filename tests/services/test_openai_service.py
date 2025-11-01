"""
Unit tests for OpenAIService

Tests all OpenAI service functionality with mocked client.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.services.openai_service import OpenAIService


class TestOpenAIServiceInitialization:
    """Test OpenAI service initialization and configuration."""
    
    def test_init_with_valid_api_key(self):
        """Test initialization with valid API key."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            with patch('openai.OpenAI') as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                
                service = OpenAIService()
                
                assert service.api_key == 'sk-test123'
                assert service.client == mock_client
                assert service.is_api_available() is True
    
    def test_init_without_api_key(self):
        """Test initialization without API key."""
        with patch.dict('os.environ', {}, clear=True):
            service = OpenAIService()
            
            assert service.client is None
            assert service.is_api_available() is False
    
    def test_init_with_invalid_api_key(self):
        """Test initialization with invalid placeholder API key."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'your-api-key-here'}):
            service = OpenAIService()
            
            assert service.client is None
            assert service.is_api_available() is False
    
    def test_custom_log_function(self):
        """Test initialization with custom log function."""
        mock_log = Mock()
        with patch.dict('os.environ', {}, clear=True):
            service = OpenAIService(log_function=mock_log)
            
            # Verify log was called during initialization
            assert mock_log.called


class TestEmbeddingGeneration:
    """Test embedding generation functionality."""
    
    @pytest.fixture
    def mock_service(self):
        """Create a mocked OpenAI service."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            with patch('openai.OpenAI') as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                
                service = OpenAIService()
                return service, mock_client
    
    def test_generate_embeddings_success(self, mock_service):
        """Test successful embedding generation."""
        service, mock_client = mock_service
        
        # Mock API response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        mock_client.embeddings.create.return_value = mock_response
        
        texts = ["Hello world", "Test text"]
        embeddings = service.generate_embeddings(texts)
        
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
    
    def test_generate_embeddings_empty_list(self, mock_service):
        """Test embedding generation with empty list."""
        service, _ = mock_service
        
        embeddings = service.generate_embeddings([])
        
        assert embeddings == []
    
    def test_generate_embeddings_api_failure_fallback(self, mock_service):
        """Test fallback to random embeddings on API failure."""
        service, mock_client = mock_service
        
        # Make API call fail
        mock_client.embeddings.create.side_effect = Exception("API Error")
        
        texts = ["Test text"]
        with patch('src.utils.embedding._generate_random_embeddings') as mock_random:
            mock_random.return_value = [[0.1] * 1536]
            embeddings = service.generate_embeddings(texts)
            
            # Should call fallback
            mock_random.assert_called_once()
    
    def test_generate_embeddings_without_api(self):
        """Test embedding generation when API is not available."""
        with patch.dict('os.environ', {}, clear=True):
            service = OpenAIService()
            
            texts = ["Test text"]
            with patch('src.utils.embedding._generate_random_embeddings') as mock_random:
                mock_random.return_value = [[0.1] * 1536]
                embeddings = service.generate_embeddings(texts)
                
                # Should use random embeddings
                mock_random.assert_called_once()
    
    def test_create_batches_by_tokens_small_texts(self, mock_service):
        """Test batching with small texts that fit in one batch."""
        service, _ = mock_service
        
        # 10 short texts (~50 tokens each = 500 tokens total)
        texts = ["Short text " * 10] * 10
        
        with patch('tiktoken.encoding_for_model') as mock_enc:
            mock_encoder = Mock()
            mock_encoder.encode.return_value = [1] * 50  # 50 tokens per text
            mock_enc.return_value = mock_encoder
            
            batches = service._create_batches_by_tokens(texts, "text-embedding-3-small")
        
        # All should fit in one batch (500 tokens << 180k limit)
        assert len(batches) == 1
        assert len(batches[0]) == 10
    
    def test_create_batches_by_tokens_large_texts(self, mock_service):
        """Test batching with large texts requiring multiple batches."""
        service, _ = mock_service
        
        # 100 texts with 2000 tokens each = 200k tokens total
        texts = ["Large text " * 400] * 100
        
        with patch('tiktoken.encoding_for_model') as mock_enc:
            mock_encoder = Mock()
            mock_encoder.encode.return_value = [1] * 2000  # 2000 tokens per text
            mock_enc.return_value = mock_encoder
            
            batches = service._create_batches_by_tokens(texts, "text-embedding-3-small")
        
        # Should split into multiple batches (180k token limit / 2000 = 90 per batch)
        assert len(batches) >= 2
        # Each batch should have ~90 texts (180k / 2000)
        assert all(len(batch) <= 90 for batch in batches)
    
    def test_create_batches_by_tokens_mixed_sizes(self, mock_service):
        """Test batching with mixed text sizes."""
        service, _ = mock_service
        
        # Mix of small (100 tokens) and large (5000 tokens) texts
        texts = ["Small"] * 50 + ["Large text " * 1000] * 50
        
        with patch('tiktoken.encoding_for_model') as mock_enc:
            mock_encoder = Mock()
            def encode_side_effect(text):
                if "Large" in text:
                    return [1] * 5000  # 5000 tokens
                return [1] * 100  # 100 tokens
            mock_encoder.encode.side_effect = encode_side_effect
            mock_enc.return_value = mock_encoder
            
            batches = service._create_batches_by_tokens(texts, "text-embedding-3-small")
        
        # Should create batches efficiently
        assert len(batches) >= 2
        total_texts = sum(len(batch) for batch in batches)
        assert total_texts == 100  # All texts should be batched
    
    def test_create_batches_respects_max_texts_limit(self, mock_service):
        """Test that batching respects the 2000 document limit."""
        service, _ = mock_service
        
        # 3000 very small texts (10 tokens each)
        texts = ["Tiny"] * 3000
        
        with patch('tiktoken.encoding_for_model') as mock_enc:
            mock_encoder = Mock()
            mock_encoder.encode.return_value = [1] * 10  # 10 tokens per text
            mock_enc.return_value = mock_encoder
            
            batches = service._create_batches_by_tokens(texts, "text-embedding-3-small")
        
        # Should split into at least 2 batches (3000 / 2000 = 1.5)
        assert len(batches) >= 2
        # No batch should exceed 2000 texts
        assert all(len(batch) <= 2000 for batch in batches)
    
    def test_create_batches_without_tiktoken(self, mock_service):
        """Test fallback batching when tiktoken is not available."""
        service, _ = mock_service
        
        texts = ["Medium text " * 50] * 100
        
        # Mock ImportError to trigger fallback
        with patch('tiktoken.encoding_for_model', side_effect=ImportError("tiktoken not found")):
            batches = service._create_batches_by_tokens(texts, "text-embedding-3-small")
        
        # Should still create valid batches using character-based estimation
        assert len(batches) >= 1
        total_texts = sum(len(batch) for batch in batches)
        assert total_texts == 100
    
    def test_create_batches_empty_list(self, mock_service):
        """Test batching with empty text list."""
        service, _ = mock_service
        
        batches = service._create_batches_by_tokens([], "text-embedding-3-small")
        
        assert batches == []
    
    def test_create_batches_single_text(self, mock_service):
        """Test batching with single text."""
        service, _ = mock_service
        
        texts = ["Single text"]
        
        with patch('tiktoken.encoding_for_model') as mock_enc:
            mock_encoder = Mock()
            mock_encoder.encode.return_value = [1] * 100
            mock_enc.return_value = mock_encoder
            
            batches = service._create_batches_by_tokens(texts, "text-embedding-3-small")
        
        assert len(batches) == 1
        assert len(batches[0]) == 1


class TestClusterNaming:
    """Test cluster naming functionality."""
    
    @pytest.fixture
    def mock_service(self):
        """Create a mocked OpenAI service."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            with patch('openai.OpenAI') as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                
                service = OpenAIService()
                return service, mock_client
    
    def test_generate_cluster_name_success(self, mock_service):
        """Test successful cluster name generation."""
        service, mock_client = mock_service
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Technology"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        texts = ["AI technology", "Machine learning", "Deep learning"]
        name = service.generate_cluster_name(texts)
        
        assert name == "Technology"
    
    def test_generate_cluster_name_without_api(self):
        """Test cluster name generation without API."""
        with patch.dict('os.environ', {}, clear=True):
            service = OpenAIService()
            
            texts = ["Test text"]
            name = service.generate_cluster_name(texts)
            
            assert name == "Cluster_Auto"
    
    def test_generate_cluster_name_with_debug(self, mock_service, tmp_path):
        """Test cluster name generation with debug mode."""
        service, mock_client = mock_service
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Science"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        texts = ["Physics", "Chemistry"]
        
        # Change to temp directory
        import os
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            name = service.generate_cluster_name(texts, debug=True)
            
            # Should create debug file
            debug_files = list(tmp_path.glob("cluster_naming_prompt_*.txt"))
            assert len(debug_files) > 0
        finally:
            os.chdir(old_cwd)
    
    def test_generate_single_word_label_success(self, mock_service):
        """Test single-word label generation."""
        service, mock_client = mock_service
        
        # Mock JSON response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='{"label": "Sports"}'))]
        mock_client.chat.completions.create.return_value = mock_response
        
        texts = ["Basketball game", "Football match"]
        label = service.generate_single_word_cluster_label(1, texts)
        
        assert label == "Sports"


class TestLabelEnrichment:
    """Test label enrichment functionality."""
    
    @pytest.fixture
    def mock_service(self):
        """Create a mocked OpenAI service."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            with patch('openai.OpenAI') as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                
                service = OpenAIService()
                return service, mock_client
    
    def test_enrich_label_description_success(self, mock_service):
        """Test successful label description enrichment."""
        service, mock_client = mock_service
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Articles about sports and athletic events"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        description = service.enrich_label_description("Sports")
        
        assert "sports" in description.lower()
    
    def test_enrich_label_description_without_api(self):
        """Test label enrichment without API."""
        with patch.dict('os.environ', {}, clear=True):
            service = OpenAIService()
            
            description = service.enrich_label_description("Technology")
            
            # Should return fallback description
            assert "technology" in description.lower()


class TestTextOperations:
    """Test text summarization and classification."""
    
    @pytest.fixture
    def mock_service(self):
        """Create a mocked OpenAI service."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            with patch('openai.OpenAI') as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                
                service = OpenAIService()
                return service, mock_client
    
    def test_generate_text_summary_success(self, mock_service):
        """Test successful text summarization."""
        service, mock_client = mock_service
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="This is a summary"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        text = "Long text that needs to be summarized" * 10
        summary = service.generate_text_summary(text, max_length=100)
        
        assert len(summary) > 0
        assert "summary" in summary.lower()
    
    def test_generate_text_summary_without_api(self):
        """Test text summarization without API raises error."""
        with patch.dict('os.environ', {}, clear=True):
            service = OpenAIService()
            
            with pytest.raises(RuntimeError, match="OpenAI API not available"):
                service.generate_text_summary("Test text")
    
    def test_classify_text_success(self, mock_service):
        """Test successful text classification."""
        service, mock_client = mock_service
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Sports"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        categories = ["Sports", "Technology", "Politics"]
        result = service.classify_text("Basketball game tonight", categories)
        
        assert result["category"] == "Sports"
        assert result["method"] == "llm"
        assert "confidence" in result
    
    def test_classify_text_without_api(self):
        """Test text classification without API raises error."""
        with patch.dict('os.environ', {}, clear=True):
            service = OpenAIService()
            
            with pytest.raises(RuntimeError, match="OpenAI API not available"):
                service.classify_text("Test", ["Cat1", "Cat2"])
    
    def test_find_closest_category_exact_match(self, mock_service):
        """Test finding closest category with exact match."""
        service, _ = mock_service
        
        categories = ["Sports", "Technology", "Politics"]
        result = service._find_closest_category("sports", categories)
        
        assert result == "Sports"
    
    def test_find_closest_category_partial_match(self, mock_service):
        """Test finding closest category with partial match."""
        service, _ = mock_service
        
        categories = ["Sports News", "Technology", "Politics"]
        result = service._find_closest_category("Sports", categories)
        
        assert result == "Sports News"
    
    def test_find_closest_category_no_match(self, mock_service):
        """Test finding closest category with no match returns first."""
        service, _ = mock_service
        
        categories = ["Sports", "Technology"]
        result = service._find_closest_category("Unknown", categories)
        
        assert result == "Sports"


class TestServiceInfo:
    """Test service information and utilities."""
    
    def test_get_service_info_with_api(self):
        """Test getting service info when API is available."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            with patch('openai.OpenAI'):
                service = OpenAIService()
                
                info = service.get_service_info()
                
                assert info["api_available"] is True
                assert info["has_api_key"] is True
                assert info["client_initialized"] is True
    
    def test_get_service_info_without_api(self):
        """Test getting service info when API is not available."""
        with patch.dict('os.environ', {}, clear=True):
            service = OpenAIService()
            
            info = service.get_service_info()
            
            assert info["api_available"] is False
    
    def test_test_connection_success(self):
        """Test successful API connection test."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            with patch('openai.OpenAI') as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                
                # Mock successful response
                mock_response = Mock()
                mock_response.choices = [Mock(message=Mock(content="Hi"))]
                mock_client.chat.completions.create.return_value = mock_response
                
                service = OpenAIService()
                result = service.test_connection()
                
                assert result is True
    
    def test_test_connection_failure(self):
        """Test failed API connection test."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            with patch('openai.OpenAI') as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                
                # Mock failed response
                mock_client.chat.completions.create.side_effect = Exception("Connection error")
                
                service = OpenAIService()
                result = service.test_connection()
                
                assert result is False
    
    def test_close_service(self):
        """Test closing the service."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            with patch('openai.OpenAI'):
                service = OpenAIService()
                
                service.close()
                
                assert service.client is None

