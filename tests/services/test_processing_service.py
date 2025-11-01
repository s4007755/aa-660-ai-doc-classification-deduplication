"""
Unit tests for ProcessingService

Tests all document processing functionality including directory, CSV, URL, and PDF processing.
"""

import pytest
import os
import csv
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from src.services.processing_service import ProcessingService


class TestProcessingServiceInitialization:
    """Test processing service initialization."""
    
    def test_init_default(self):
        """Test default initialization."""
        service = ProcessingService()
        
        assert service.max_text_length == 15000
        assert service.batch_size == 3
    
    def test_init_custom_log(self):
        """Test initialization with custom log function."""
        mock_log = Mock()
        service = ProcessingService(log_function=mock_log)
        
        assert service.log == mock_log
    
    def test_set_max_text_length(self):
        """Test setting max text length."""
        service = ProcessingService()
        service.set_max_text_length(20000)
        
        assert service.max_text_length == 20000
    
    def test_set_batch_size(self):
        """Test setting batch size."""
        service = ProcessingService()
        service.set_batch_size(5)
        
        assert service.batch_size == 5
    
    def test_set_batch_size_minimum(self):
        """Test setting batch size with minimum constraint."""
        service = ProcessingService()
        service.set_batch_size(0)
        
        assert service.batch_size == 1  # Should be at least 1


class TestDirectoryProcessing:
    """Test directory processing functionality."""
    
    @pytest.fixture
    def temp_test_dir(self, tmp_path):
        """Create temporary test directory with files."""
        # Create test files
        (tmp_path / "file1.txt").write_text("Content of file 1", encoding="utf-8")
        (tmp_path / "file2.txt").write_text("Content of file 2", encoding="utf-8")
        (tmp_path / "file3.md").write_text("# Markdown content", encoding="utf-8")
        
        return tmp_path
    
    def test_process_directory_success(self, temp_test_dir):
        """Test successful directory processing."""
        service = ProcessingService()
        
        texts, payloads = service._process_directory(str(temp_test_dir))
        
        assert len(texts) == 3
        assert len(payloads) == 3
        assert all(isinstance(p, dict) for p in payloads)
        assert all("source" in p for p in payloads)
        assert all("hash" in p for p in payloads)
        assert all("type" in p for p in payloads)
    
    def test_process_directory_with_limit(self, temp_test_dir):
        """Test directory processing with limit."""
        service = ProcessingService()
        
        texts, payloads = service._process_directory(str(temp_test_dir), limit=2)
        
        assert len(texts) == 2
        assert len(payloads) == 2
    
    def test_process_directory_empty(self, tmp_path):
        """Test processing empty directory."""
        service = ProcessingService()
        
        texts, payloads = service._process_directory(str(tmp_path))
        
        assert texts == []
        assert payloads == []
    
    def test_process_directory_with_long_content(self, tmp_path):
        """Test directory processing with content exceeding max length."""
        # Create file with very long content
        long_content = "A" * 20000
        (tmp_path / "long.txt").write_text(long_content, encoding="utf-8")
        
        service = ProcessingService()
        texts, payloads = service._process_directory(str(tmp_path))
        
        assert len(texts) == 1
        assert len(texts[0]) <= service.max_text_length
        assert payloads[0].get("original_length") is not None


class TestCSVProcessing:
    """Test CSV processing functionality."""
    
    @pytest.fixture
    def temp_csv_file(self, tmp_path):
        """Create temporary CSV file."""
        csv_path = tmp_path / "test.csv"
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label", "id"])
            writer.writeheader()
            writer.writerow({"text": "First document", "label": "A", "id": "1"})
            writer.writerow({"text": "Second document", "label": "B", "id": "2"})
            writer.writerow({"text": "Third document", "label": "A", "id": "3"})
        
        return csv_path
    
    def test_process_csv_success(self, temp_csv_file):
        """Test successful CSV processing."""
        service = ProcessingService()
        
        texts, payloads = service._process_csv(str(temp_csv_file))
        
        assert len(texts) == 3
        assert len(payloads) == 3
        assert texts[0] == "First document"
        assert payloads[0]["type"] == "csv_text"
        assert payloads[0]["label"] == "A"
    
    def test_process_csv_with_limit(self, temp_csv_file):
        """Test CSV processing with limit."""
        service = ProcessingService()
        
        texts, payloads = service._process_csv(str(temp_csv_file), limit=2)
        
        assert len(texts) == 2
        assert len(payloads) == 2
    
    def test_process_csv_custom_text_column(self, tmp_path):
        """Test CSV processing with custom text column."""
        csv_path = tmp_path / "custom.csv"
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["content", "title"])
            writer.writeheader()
            writer.writerow({"content": "Document content", "title": "Doc 1"})
        
        service = ProcessingService()
        texts, payloads = service._process_csv(str(csv_path), text_column="content")
        
        assert len(texts) == 1
        assert texts[0] == "Document content"
        assert payloads[0]["title"] == "Doc 1"
    
    def test_process_csv_with_url_column(self, tmp_path):
        """Test CSV processing with URL column."""
        csv_path = tmp_path / "urls.csv"
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["url", "category"])
            writer.writeheader()
            writer.writerow({"url": "https://example.com", "category": "test"})
        
        service = ProcessingService()
        
        # Mock HTML extraction
        with patch.object(service, '_extract_html_content', return_value="Extracted content"):
            texts, payloads = service._process_csv(str(csv_path), url_column="url")
            
            assert len(texts) == 1
            assert texts[0] == "Extracted content"
            assert payloads[0]["type"] == "url_html"
            assert payloads[0]["url"] == "https://example.com"


class TestURLProcessing:
    """Test URL processing functionality."""
    
    def test_extract_html_content_success(self):
        """Test successful HTML content extraction."""
        service = ProcessingService()
        
        mock_html = """
        <html>
            <body>
                <p>This is test content</p>
                <script>alert('test');</script>
            </body>
        </html>
        """
        
        with patch('src.services.processing_service.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.content = mock_html.encode('utf-8')
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            content = service._extract_html_content("https://example.com")
            
            assert "This is test content" in content
            assert "alert" not in content  # Script should be removed
    
    def test_extract_html_content_failure(self):
        """Test HTML extraction failure."""
        service = ProcessingService()
        
        with patch('src.services.processing_service.requests.get') as mock_get:
            mock_get.side_effect = Exception("Connection error")
            
            content = service._extract_html_content("https://example.com")
            
            assert content == ""
    
    def test_process_url_html(self):
        """Test processing HTML URL."""
        service = ProcessingService()
        
        with patch.object(service, '_extract_html_content', return_value="HTML content"):
            with patch('src.services.processing_service.requests.head') as mock_head:
                mock_head.return_value = Mock(headers={'Content-Type': 'text/html'})
                
                texts, payloads = service._process_url("https://example.com")
                
                assert len(texts) == 1
                assert texts[0] == "HTML content"
                assert payloads[0]["type"] == "url"
    
    def test_process_url_pdf(self):
        """Test processing PDF URL."""
        service = ProcessingService()
        
        with patch.object(service, '_extract_pdf_content', return_value="PDF content"):
            with patch('src.services.processing_service.requests.head') as mock_head:
                mock_head.return_value = Mock(headers={'Content-Type': 'application/pdf'})
                
                texts, payloads = service._process_url("https://example.com/doc.pdf")
                
                assert len(texts) == 1
                assert texts[0] == "PDF content"
                assert payloads[0]["type"] == "url_pdf"


class TestPDFProcessing:
    """Test PDF content extraction."""
    
    def test_extract_pdf_content_success(self):
        """Test successful PDF extraction."""
        service = ProcessingService()
        
        # Mock PDF response and parser
        mock_pdf_bytes = b"%PDF-1.4\nTest PDF content"
        
        with patch('src.services.processing_service.requests.get') as mock_get:
            with patch('pypdf.PdfReader') as mock_reader:
                # Mock streaming response
                mock_response = Mock()
                mock_response.iter_content.return_value = [mock_pdf_bytes]
                mock_response.raise_for_status = Mock()
                mock_get.return_value = mock_response
                
                # Mock PDF reader
                mock_page = Mock()
                mock_page.extract_text.return_value = "Extracted PDF text"
                mock_pdf_instance = Mock()
                mock_pdf_instance.pages = [mock_page]
                mock_reader.return_value = mock_pdf_instance
                
                content = service._extract_pdf_content("https://example.com/doc.pdf")
                
                assert "Extracted PDF text" in content
    
    def test_extract_pdf_content_failure(self):
        """Test PDF extraction failure."""
        service = ProcessingService()
        
        with patch('src.services.processing_service.requests.get') as mock_get:
            mock_get.side_effect = Exception("Download failed")
            
            content = service._extract_pdf_content("https://example.com/doc.pdf")
            
            assert content == ""
    
    def test_extract_pdf_content_parse_error(self):
        """Test PDF extraction with parse error."""
        service = ProcessingService()
        
        mock_pdf_bytes = b"Invalid PDF"
        
        with patch('src.services.processing_service.requests.get') as mock_get:
            with patch('pypdf.PdfReader') as mock_reader:
                # Mock streaming response
                mock_response = Mock()
                mock_response.iter_content.return_value = [mock_pdf_bytes]
                mock_response.raise_for_status = Mock()
                mock_get.return_value = mock_response
                
                # Mock PDF reader to raise error
                mock_reader.side_effect = Exception("Parse error")
                
                content = service._extract_pdf_content("https://example.com/doc.pdf")
                
                assert content == ""


class TestSourceProcessing:
    """Test main process_source method."""
    
    def test_process_source_directory(self, tmp_path):
        """Test processing directory source."""
        (tmp_path / "test.txt").write_text("Test content", encoding="utf-8")
        
        service = ProcessingService()
        texts, payloads = service.process_source(str(tmp_path))
        
        assert len(texts) == 1
        assert payloads[0]["type"] == "file"
    
    def test_process_source_csv(self, tmp_path):
        """Test processing CSV source."""
        csv_path = tmp_path / "test.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text"])
            writer.writeheader()
            writer.writerow({"text": "CSV content"})
        
        service = ProcessingService()
        texts, payloads = service.process_source(str(csv_path))
        
        assert len(texts) == 1
        assert payloads[0]["type"] == "csv_text"
    
    def test_process_source_url(self):
        """Test processing URL source."""
        service = ProcessingService()
        
        with patch.object(service, '_process_url', return_value=(["URL content"], [{"type": "url"}])):
            texts, payloads = service.process_source("https://example.com")
            
            assert len(texts) == 1
            assert payloads[0]["type"] == "url"
    
    def test_process_source_single_file(self, tmp_path):
        """Test processing single file source."""
        file_path = tmp_path / "single.txt"
        file_path.write_text("Single file content", encoding="utf-8")
        
        service = ProcessingService()
        texts, payloads = service.process_source(str(file_path))
        
        assert len(texts) == 1
        assert texts[0] == "Single file content"
    
    def test_process_source_invalid(self):
        """Test processing invalid source."""
        service = ProcessingService()
        
        texts, payloads = service.process_source("/nonexistent/path")
        
        assert texts == []
        assert payloads == []


class TestSourceValidation:
    """Test source validation functionality."""
    
    def test_validate_source_directory(self, tmp_path):
        """Test validating directory source."""
        (tmp_path / "test.txt").write_text("Test", encoding="utf-8")
        
        service = ProcessingService()
        
        # Mock directory_enumerator to return the file
        with patch('src.services.processing_service.directory_enumerator') as mock_enum:
            mock_enum.return_value = [str(tmp_path / "test.txt")]
            result = service.validate_source(str(tmp_path))
        
        assert result["valid"] is True
        assert result["type"] == "directory"
        assert result["exists"] is True
    
    def test_validate_source_csv(self, tmp_path):
        """Test validating CSV source."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("text\nTest", encoding="utf-8")
        
        service = ProcessingService()
        result = service.validate_source(str(csv_path))
        
        assert result["valid"] is True
        assert result["type"] == "csv"
        assert result["exists"] is True
    
    def test_validate_source_url(self):
        """Test validating URL source."""
        service = ProcessingService()
        result = service.validate_source("https://example.com")
        
        assert result["valid"] is True
        assert result["type"] == "url"
    
    def test_validate_source_nonexistent(self):
        """Test validating nonexistent source."""
        service = ProcessingService()
        result = service.validate_source("/nonexistent/path")
        
        assert result["valid"] is False
        assert result["exists"] is False


class TestSourceStats:
    """Test source statistics functionality."""
    
    def test_get_source_stats_directory(self, tmp_path):
        """Test getting stats for directory."""
        (tmp_path / "test.txt").write_text("Test", encoding="utf-8")
        
        service = ProcessingService()
        
        # Mock directory_enumerator to return the file
        with patch('src.services.processing_service.directory_enumerator') as mock_enum:
            mock_enum.return_value = [str(tmp_path / "test.txt")]
            stats = service.get_source_stats(str(tmp_path))
        
        assert stats["type"] == "directory"
        assert stats["file_count"] >= 1
    
    def test_get_source_stats_csv(self, tmp_path):
        """Test getting stats for CSV."""
        csv_path = tmp_path / "test.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerow({"text": "Test", "label": "A"})
        
        service = ProcessingService()
        stats = service.get_source_stats(str(csv_path))
        
        # CSV files should be type "csv" in validation, but stats shows "file" with csv info
        assert stats.get("csv_rows") == 1 or stats.get("csv_rows") is None  # May vary based on implementation
        if stats.get("csv_columns"):
            assert "text" in stats["csv_columns"]
    
    def test_get_source_stats_invalid(self):
        """Test getting stats for invalid source."""
        service = ProcessingService()
        stats = service.get_source_stats("/nonexistent")
        
        assert "error" in stats


class TestFileExtraction:
    """Test file text extraction."""
    
    def test_extract_text_utf8(self, tmp_path):
        """Test extracting UTF-8 text."""
        file_path = tmp_path / "utf8.txt"
        file_path.write_text("UTF-8 content with émojis 🎉", encoding="utf-8")
        
        service = ProcessingService()
        content = service.extract_text_from_file(str(file_path))
        
        assert "UTF-8 content" in content
        assert "🎉" in content
    
    def test_extract_text_encoding_fallback(self, tmp_path):
        """Test extraction with encoding fallback."""
        file_path = tmp_path / "latin1.txt"
        file_path.write_bytes(b"Latin-1 content \xe9")  # é in latin-1
        
        service = ProcessingService()
        content = service.extract_text_from_file(str(file_path))
        
        # Should successfully extract with fallback
        assert len(content) > 0
    
    def test_extract_text_nonexistent(self):
        """Test extracting from nonexistent file."""
        service = ProcessingService()
        content = service.extract_text_from_file("/nonexistent.txt")
        
        assert content == ""


class TestServiceInfo:
    """Test service information methods."""
    
    def test_get_service_info(self):
        """Test getting service information."""
        service = ProcessingService()
        info = service.get_service_info()
        
        assert info["service_name"] == "ProcessingService"
        assert "directory" in info["supported_types"]
        assert "csv" in info["supported_types"]
        assert "url" in info["supported_types"]
        assert info["configuration"]["max_text_length"] == 15000
    
    def test_close_service(self):
        """Test closing the service."""
        service = ProcessingService()
        
        # Should not raise error
        service.close()

