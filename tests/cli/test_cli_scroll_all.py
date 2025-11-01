from unittest.mock import MagicMock
from src.pipelines.classification.cli import Cli


def test_scroll_all_paginates():
    cli = Cli()
    cli.collection = "col"
    # two pages
    page1 = ([{"id": 1, "payload": {}}, {"id": 2, "payload": {}}], "cursor1")
    page2 = ([{"id": 3, "payload": {}}], None)
    cli.qdrant_service.scroll_vectors = MagicMock(side_effect=[page1, page2])

    points = cli._scroll_all(with_payload=True, with_vectors=False)
    assert len(points) == 3
    assert cli.qdrant_service.scroll_vectors.call_count == 2


