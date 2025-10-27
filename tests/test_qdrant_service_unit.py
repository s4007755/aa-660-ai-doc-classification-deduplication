from unittest.mock import MagicMock, patch

from src.services.qdrant_service import QdrantService


@patch("src.services.qdrant_service.QdrantClient")
def test_create_delete_list_info(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    # get_collections shape
    mock_client.get_collections.return_value.collections = []

    svc = QdrantService(log_function=lambda *_: None)
    # create
    assert svc.create_collection("c1", 128, "test") is True
    # list
    col = MagicMock()
    col.name = "c1"
    mock_client.get_collections.return_value.collections = [col]
    info_mock = MagicMock()
    info_mock.config.params.vectors.size = 128
    info_mock.config.params.vectors.distance.value = "Cosine"
    info_mock.points_count = 0
    info_mock.status.value = "green"
    mock_client.get_collection.return_value = info_mock
    lst = svc.list_collections()
    assert "c1" in lst.keys()
    # info
    info = svc.get_collection_info("c1")
    assert info["dimension"] == 128
    # delete
    assert svc.delete_collection("c1") is True


@patch("src.services.qdrant_service.QdrantClient")
def test_insert_scroll_update_delete(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    info_mock = MagicMock()
    info_mock.points_count = 0
    info_mock.config.params.vectors.size = 128
    info_mock.config.params.vectors.distance.value = "Cosine"
    info_mock.status.value = "green"
    mock_client.get_collection.return_value = info_mock

    # scroll returns no points initially
    mock_client.scroll.return_value = ([], None)

    svc = QdrantService(log_function=lambda *_: None)
    # insert
    vectors = [[0.1] * 128 for _ in range(3)]
    payloads = [{"k": i} for i in range(3)]
    ok, inserted, skipped = svc.insert_vectors("c1", vectors, payloads)
    assert ok and inserted == 3 and skipped == 0

    # update payload
    mock_client.set_payload.return_value = None
    assert svc.update_payload("c1", [1, 2], {"a": 1}) is True

    # delete points
    mock_client.delete.return_value = None
    assert svc.delete_points("c1", [1]) is True

    # scroll formatting
    # simulate returned points
    class _P:
        def __init__(self, id):
            self.id = id
            self.payload = {"x": 1}
            self.vector = [0.1]

    mock_client.scroll.return_value = ([ _P(1), _P(2) ], None)
    pts, nxt = svc.scroll_vectors("c1", limit=2, with_payload=True, with_vectors=True)
    assert len(pts) == 2 and pts[0]["payload"]["x"] == 1 and "vector" in pts[0]


@patch("src.services.qdrant_service.QdrantClient")
def test_reserve_id_block_empty_collection(mock_client_cls):
    """Test ID reservation for empty collection."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    
    # Empty collection
    info_mock = MagicMock()
    info_mock.points_count = 0
    mock_client.get_collection.return_value = info_mock
    
    svc = QdrantService(log_function=lambda *_: None)
    
    # Should start at 1 (ID 0 is reserved for metadata)
    start_id = svc._reserve_id_block("test_collection", 5)
    assert start_id == 1
    
    # Next reservation should be 6
    start_id = svc._reserve_id_block("test_collection", 3)
    assert start_id == 6


@patch("src.services.qdrant_service.QdrantClient")
def test_reserve_id_block_existing_points(mock_client_cls):
    """Test ID reservation for collection with existing points."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    
    # Collection with 10 points (IDs 0-9, where 0 is metadata)
    info_mock = MagicMock()
    info_mock.points_count = 10
    mock_client.get_collection.return_value = info_mock
    
    # Mock scroll to return max ID
    class _Point:
        def __init__(self, id):
            self.id = id
    
    mock_client.scroll.return_value = ([_Point(9), _Point(8), _Point(7)], None)
    
    svc = QdrantService(log_function=lambda *_: None)
    
    # Should start after max ID (9 + 1 = 10)
    start_id = svc._reserve_id_block("test_collection", 5)
    assert start_id == 10
    
    # Next reservation should be 15
    start_id = svc._reserve_id_block("test_collection", 3)
    assert start_id == 15


@patch("src.services.qdrant_service.QdrantClient")
def test_reserve_id_block_thread_safety(mock_client_cls):
    """Test that ID reservation is thread-safe."""
    import threading
    
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    
    info_mock = MagicMock()
    info_mock.points_count = 0
    mock_client.get_collection.return_value = info_mock
    
    svc = QdrantService(log_function=lambda *_: None)
    
    results = []
    
    def reserve_ids():
        for _ in range(10):
            start_id = svc._reserve_id_block("test_collection", 5)
            results.append(start_id)
    
    # Run 5 threads concurrently
    threads = [threading.Thread(target=reserve_ids) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # All IDs should be unique (no collisions)
    assert len(results) == 50
    assert len(set(results)) == 50  # All unique
    
    # Results should be sequential (no gaps when using blocks)
    sorted_results = sorted(results)
    expected = list(range(1, 251, 5))  # 1, 6, 11, 16, ..., 246
    assert sorted_results == expected


@patch("src.services.qdrant_service.QdrantClient")
def test_insert_vectors_assigns_sequential_ids(mock_client_cls):
    """Test that insert_vectors assigns sequential IDs."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    
    info_mock = MagicMock()
    info_mock.points_count = 0
    info_mock.config.params.vectors.size = 128
    mock_client.get_collection.return_value = info_mock
    
    svc = QdrantService(log_function=lambda *_: None)
    
    vectors = [[0.1] * 128 for _ in range(5)]
    payloads = [{"index": i} for i in range(5)]
    
    # Capture the upsert call
    upsert_calls = []
    def capture_upsert(**kwargs):
        upsert_calls.append(kwargs)
        return None
    
    mock_client.upsert.side_effect = capture_upsert
    
    success, inserted, skipped = svc.insert_vectors("test_collection", vectors, payloads)
    
    assert success
    assert inserted == 5
    
    # Check that IDs are sequential starting from 1
    points = upsert_calls[0]["points"]
    ids = [p.id for p in points]
    assert ids == [1, 2, 3, 4, 5]


@patch("src.services.qdrant_service.QdrantClient")
def test_multiple_insert_batches_sequential_ids(mock_client_cls):
    """Test that multiple insert operations maintain ID sequence."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    
    info_mock = MagicMock()
    info_mock.points_count = 0
    info_mock.config.params.vectors.size = 128
    mock_client.get_collection.return_value = info_mock
    
    svc = QdrantService(log_function=lambda *_: None)
    
    all_ids = []
    
    # Capture IDs from upsert calls
    def capture_upsert(**kwargs):
        points = kwargs["points"]
        all_ids.extend([p.id for p in points])
        return None
    
    mock_client.upsert.side_effect = capture_upsert
    
    # First batch
    vectors1 = [[0.1] * 128 for _ in range(3)]
    payloads1 = [{"batch": 1, "i": i} for i in range(3)]
    svc.insert_vectors("test_collection", vectors1, payloads1)
    
    # Second batch
    vectors2 = [[0.2] * 128 for _ in range(2)]
    payloads2 = [{"batch": 2, "i": i} for i in range(2)]
    svc.insert_vectors("test_collection", vectors2, payloads2)
    
    # Third batch
    vectors3 = [[0.3] * 128 for _ in range(4)]
    payloads3 = [{"batch": 3, "i": i} for i in range(4)]
    svc.insert_vectors("test_collection", vectors3, payloads3)
    
    # IDs should be sequential across all batches
    assert all_ids == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # All unique
    assert len(all_ids) == len(set(all_ids))

