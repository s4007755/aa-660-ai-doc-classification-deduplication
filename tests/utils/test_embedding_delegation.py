from unittest.mock import patch
from src.utils.embedding import embed


@patch("src.services.openai_service.OpenAIService.generate_embeddings")
def test_embed_delegates_to_service(mock_gen):
    mock_gen.return_value = [[0.1, 0.2]]
    out = embed(["hi"], model="text-embedding-3-small")
    assert out == [[0.1, 0.2]]


@patch("src.services.openai_service.OpenAIService.generate_embeddings")
def test_embed_falls_back_on_error(mock_gen):
    mock_gen.side_effect = Exception("boom")
    out = embed(["text"], model="text-embedding-3-small")
    assert len(out) == 1
    assert isinstance(out[0], list)
    assert len(out[0]) > 10


