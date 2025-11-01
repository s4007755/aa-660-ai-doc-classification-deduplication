import numpy as np
from unittest.mock import MagicMock, patch
from src.pipelines.classification.cli import Cli


@patch("src.pipelines.classification.cli.KMeans")
def test_agglomerative_fallback_to_kmeans_for_large_n(mock_kmeans):
    cli = Cli()
    cli.collection = "col"
    # mock qdrant embeddings vectors > 5000
    n = 6000
    dim = 8
    vecs = np.random.rand(n, dim).tolist()
    # build points_list with vectors
    points_page = ([{"id": i + 1, "vector": vecs[i], "payload": {}} for i in range(n)], None)
    cli.qdrant_service.get_collection_info = MagicMock(return_value={"vector_count": n})
    cli.qdrant_service.scroll_vectors = MagicMock(return_value=points_page)
    # mock client payload update methods
    cli.qdrant_service.client = MagicMock()
    # run clustering without num_clusters to trigger unsupervised path
    cli.handle_command("cluster")
    assert mock_kmeans.called


