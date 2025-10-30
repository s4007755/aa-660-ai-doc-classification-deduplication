import pytest
from unittest.mock import Mock, patch
from src.services.processing_service import ProcessingService


class TestProcessingEdges:
    def test_csv_text_column_mismatch(self, tmp_path):
        svc = ProcessingService(log_function=Mock())
        p = tmp_path / 'bad.csv'
        p.write_text('wrong_col\nvalue', encoding='utf-8')
        texts, payloads = svc.process_source(str(p), text_column='text')
        # Should not crash; may return empty
        assert isinstance(texts, list)

    def test_url_fetch_failure(self):
        svc = ProcessingService(log_function=Mock())
        with patch('src.services.processing_service.requests.get') as rg:
            rg.side_effect = Exception('net')
            texts, payloads = svc.process_source('http://example.com')
            assert texts == [] or isinstance(texts, list)


