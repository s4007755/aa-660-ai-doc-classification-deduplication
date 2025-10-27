import numpy as np
import random
from src.utils.hash_utils import generate_deterministic_seed

def embed(texts, model="text-embedding-3-small"):
    """Generate embeddings for texts. Falls back to random vectors if OpenAI API is not available."""
    
    if isinstance(texts, str):
        texts = [texts]

    try:
        # Try to use OpenAI API first
        from openai import OpenAI
        
        # Try to import API key with proper error handling
        try:
            from src.pipelines.classification.credentials import OPENAI_API_KEY
        except ImportError:
            # Fallback to environment variable if credentials file doesn't exist
            import os
            OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        except Exception as e:
            # If there's any other error with the credentials file, fallback to env
            print(f"Warning: Could not load credentials file: {e}")
            import os
            OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your-api-key-here":
            raise ValueError("No valid OpenAI API key found")
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]

    except Exception as e:
        print(f"OpenAI API failed: {e}")
        print("Falling back to random vectors for testing...")
        
        # Fallback to random vectors for testing
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
    "text-embedding-3-small": 8192,
    "text-embedding-3-large": 8192,
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