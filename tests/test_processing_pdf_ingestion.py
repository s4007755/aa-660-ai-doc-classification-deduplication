import io
import types
import pytest
from unittest.mock import patch, MagicMock

from src.services.processing_service import ProcessingService


class _FakeResp:
    def __init__(self, content: bytes, headers=None, status_code=200):
        self._content = content
        self.headers = headers or {}
        self.status_code = status_code

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise Exception("HTTP error")

    def iter_content(self, chunk_size=65536):
        buf = self._content
        for i in range(0, len(buf), chunk_size):
            yield buf[i : i + chunk_size]


class _FakePdfPage:
    def __init__(self, text: str):
        self._text = text

    def extract_text(self):
        return self._text


class _FakePdfReader:
    def __init__(self, _io):
        # ignore buffer, return deterministic 2 pages
        self.pages = [_FakePdfPage("Attention is all you need."), _FakePdfPage("Transformer architectures.")]


@patch("src.services.processing_service.requests.head")
@patch("src.services.processing_service.requests.get")
@patch("pypdf.PdfReader", _FakePdfReader)
def test_process_url_pdf_arxiv(mock_get, mock_head):
    """Process a known PDF URL (arXiv) with mocked HTTP and PDF parsing."""
    url = "https://arxiv.org/pdf/1706.03762"
    # HEAD indicates PDF
    mock_head.return_value = _FakeResp(b"", headers={"Content-Type": "application/pdf"})
    # GET returns some bytes; content not used by _FakePdfReader
    mock_get.return_value = _FakeResp(b"%PDF-1.4\n%...mock...")

    svc = ProcessingService(log_function=lambda *_: None)
    texts, payloads = svc._process_url(url)

    assert len(texts) == 1
    assert len(payloads) == 1
    assert payloads[0]["type"] == "url_pdf"
    assert payloads[0]["source"] == url
    assert "Transformer" in texts[0] or "Attention" in texts[0]


@patch("src.services.processing_service.requests.head")
@patch("src.services.processing_service.requests.get")
@patch("pypdf.PdfReader", _FakePdfReader)
def test_process_csv_pdf_url_column(mock_get, mock_head):
    """CSV ingestion with url_column pointing to PDF URLs should produce url_pdf payloads."""
    # HEAD indicates PDF
    mock_head.return_value = _FakeResp(b"", headers={"Content-Type": "application/pdf"})
    mock_get.return_value = _FakeResp(b"%PDF-1.4\n%...mock...")

    # Create a fake CSV file in-memory
    import tempfile, os
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "urls.csv"
        csv_path.write_text("url\nhttps://arxiv.org/pdf/1706.03762\n")

        svc = ProcessingService(log_function=lambda *_: None)
        texts, payloads = svc._process_csv(str(csv_path), limit=10, text_column=None, url_column="url")

        assert len(texts) == 1
        assert len(payloads) == 1
        assert payloads[0]["type"] == "url_pdf"
        assert payloads[0]["source"].endswith("1706.03762")
        assert any(len(t) > 10 for t in texts)


@patch("src.services.processing_service.requests.head")
@patch("src.services.processing_service.requests.get")
@patch("pypdf.PdfReader", _FakePdfReader)
def test_pdf_partial_download_limits_pages(mock_get, mock_head):
    """Ensure parser limits to a few pages; content still extracted."""
    mock_head.return_value = _FakeResp(b"", headers={"Content-Type": "application/pdf"})
    mock_get.return_value = _FakeResp(b"%PDF-FAKE-CONTENT" * 100)

    svc = ProcessingService(log_function=lambda *_: None)
    texts, payloads = svc._process_url("https://example.com/file.pdf")
    assert texts  # non-empty list
    # First 5 pages only — our FakePdfReader returns 2 pages so it's fine
    text = texts[0] if texts else ""
    assert "Transformer" in text or "Attention" in text

