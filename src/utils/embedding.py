import numpy as np
import random
from src.utils.hash_utils import generate_deterministic_seed
from typing import List

def embed(texts, model: str = "text-embedding-3-small") -> List[List[float]]:
    """Generate embeddings for texts.
    Delegates to OpenAIService when available; falls back to deterministic random.
    """
    if isinstance(texts, str):
        texts = [texts]

    try:
        # Prefer centralized service implementation (handles batching, timeouts, fallbacks)
        from src.services.openai_service import OpenAIService
        service = OpenAIService(log_function=lambda *_: None)
        embeddings = service.generate_embeddings(texts, model=model)
        if embeddings:
            return embeddings
        # Fallback if service returns empty
        return _generate_random_embeddings(texts, model)
    except Exception as e:
        print(f"Embedding service failed: {e}")
        return _generate_random_embeddings(texts, model)

def _generate_random_embeddings(texts, model="text-embedding-3-small"):
    """Generate random embeddings for testing purposes."""
    
    # Model dimensions
    model_dims = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536
    }
    
    dim = model_dims.get(model, 1536)
    
    # Generate deterministic random vectors based on text content
    embeddings = []
    for text in texts:
        # Use text hash as seed for deterministic randomness
        seed = generate_deterministic_seed(text)
        random.seed(seed)
        
        # Generate random vector and normalize it
        vector = [random.gauss(0, 1) for _ in range(dim)]
        norm = sum(x**2 for x in vector) ** 0.5
        
        # Avoid division by zero
        if norm == 0:
            # If norm is zero, generate a new vector
            vector = [random.gauss(0, 1) for _ in range(dim)]
            norm = sum(x**2 for x in vector) ** 0.5
        
        normalized_vector = [x / norm for x in vector]
        
        embeddings.append(normalized_vector)
    
    return embeddings

from tiktoken import encoding_for_model

# cost per 1 mil tokens
MODEL_PRICES = {
    "text-embedding-3-small": 0.02 / 1_000_000,  # $0.02 per 1M tokens
    "text-embedding-3-large": 0.13 / 1_000_000,
    "text-embedding-ada-002": 0.10 / 1_000_000,
}

MAX_TOKENS = {
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
    "text-embedding-ada-002": 8191,
}

def estimate_embedding_cost(texts, model="text-embedding-3-small"):
    enc = encoding_for_model(model)
    total_tokens = 0
    for t in texts:
        tokens = len(enc.encode(t))
        total_tokens += min(tokens, MAX_TOKENS[model])
    return {
        "tokens": total_tokens,
        "cost_usd": round(total_tokens * MODEL_PRICES[model], 6)
    }