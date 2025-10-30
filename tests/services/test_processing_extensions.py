import tempfile
from pathlib import Path
from src.services.processing_service import ProcessingService


def test_directory_ingestion_skips_pdf_docx():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "a.txt").write_text("hello", encoding="utf-8")
        (p / "b.pdf").write_text("%PDF", encoding="utf-8")
        (p / "c.docx").write_text("docx", encoding="utf-8")
        (p / "d.html").write_text("<html>ok</html>", encoding="utf-8")
        svc = ProcessingService(log_function=lambda *_: None)
        texts, payloads = svc._process_directory(str(p))
        sources = [pl.get("source", "") for pl in payloads]
        assert any("a.txt" in s for s in sources)
        assert any("d.html" in s for s in sources)
        assert not any("b.pdf" in s for s in sources)
        assert not any("c.docx" in s for s in sources)


