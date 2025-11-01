from unittest.mock import MagicMock
from src.pipelines.classification.cli import Cli


def test_query_docid_retrieve_fallbacks_to_scroll(capsys):
    cli = Cli()
    cli.collection = "col"
    # simulate retrieve returns empty
    cli.qdrant_service.client.retrieve = MagicMock(return_value=[])
    # simulate scroll returns one matching id on first page
    cli.qdrant_service.scroll_vectors = MagicMock(return_value=([{"id": 42, "payload": {"source": "s"}}], None))

    cli.handle_command("query 42")
    # ensure the fallback scroll path was called
    assert cli.qdrant_service.scroll_vectors.call_count >= 1
    captured = capsys.readouterr().out
    assert "Document ID:" in captured


