"""
OpenAI Service

Wraps OpenAI API calls with better interface and error handling.
Provides embedding generation, LLM operations, and AI-powered features.
"""

from typing import List, Dict, Any, Optional
import os
from src.utils.hash_utils import HashUtils
import json


class OpenAIService:
    """
    Service for managing OpenAI API operations.
    
    This service wraps OpenAI API calls and provides
    a cleaner interface with better error handling and logging.
    """
    
    def __init__(self, log_function=None):
        """
        Initialize OpenAI service.
        
        Args:
            log_function: Optional logging function
        """
        self.log = log_function or print
        self.api_key = None
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client with API key."""
        # Load API key from environment variable only
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        # Validate API key
        if not self.api_key or self.api_key == "your-api-key-here" or self.api_key.strip() == "":
            self.log("No valid OpenAI API key found. Please set OPENAI_API_KEY environment variable.", True)
            self.client = None
            return
            
        # Initialize OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                timeout=15.0,  # Reduced default timeout for all requests
                max_retries=1   # Reduced retries to fail faster
            )
            self.log("OpenAI service initialized successfully")
        except Exception as e:
            self.log(f"Failed to initialize OpenAI service: {e}", True)
            self.client = None
    
    def is_api_available(self) -> bool:
        """Check if OpenAI API is available."""
        return self.client is not None
    
    def _create_batches_by_tokens(self, texts: List[str], model: str = "text-embedding-3-small") -> List[List[str]]:
        """
        Create batches based on actual token counts to maximize efficiency.

        Args:
            texts: List of texts to embed
            model: Embedding model to use

        Returns:
            List of batches, where each batch is a list of texts
        """
        # OpenAI API limits per request (embeddings endpoint)
        # Leave some safety margin to avoid hitting exact limits
        MAX_TOTAL_TOKENS_PER_REQUEST = 180000  # ~90% of 200k limit - safe margin
        MAX_TEXTS_PER_REQUEST = 2000  # ~97% of 2048 limit - safe margin

        if not texts:
            return []

        try:
            # Use tiktoken for accurate token counting
            from tiktoken import encoding_for_model

            # Model-specific token limits per individual text
            model_token_limits = {
                "text-embedding-3-small": 8191,
                "text-embedding-3-large": 8191,
                "text-embedding-ada-002": 8191
            }

            max_tokens_per_text = model_token_limits.get(model, 8191)
            enc = encoding_for_model(model)

            # Calculate actual tokens for all texts
            token_counts = []
            for text in texts:
                try:
                    tokens = len(enc.encode(text))
                    # Cap individual text tokens to model limit
                    capped_tokens = min(tokens, max_tokens_per_text)
                    token_counts.append(capped_tokens)
                except Exception:
                    # If tokenization fails, estimate conservatively
                    estimated_tokens = min(len(text) // 3, max_tokens_per_text)
                    token_counts.append(estimated_tokens)

            # Create batches based on actual token counts
            batches = []
            current_batch = []
            current_batch_tokens = 0

            for i, (text, token_count) in enumerate(zip(texts, token_counts)):
                # Check if adding this text would exceed limits
                if (current_batch_tokens + token_count > MAX_TOTAL_TOKENS_PER_REQUEST or
                    len(current_batch) >= MAX_TEXTS_PER_REQUEST):
                    # Start new batch
                    if current_batch:
                        batches.append(current_batch)
                    current_batch = [text]
                    current_batch_tokens = token_count
                else:
                    # Add to current batch
                    current_batch.append(text)
                    current_batch_tokens += token_count

            # Add final batch
            if current_batch:
                batches.append(current_batch)

            return batches

        except ImportError:
            # Fallback to conservative character-based estimation if tiktoken not available
            self.log("Warning: tiktoken not available, using character-based estimation", True)

            # Conservative character-based estimation
            batches = []
            current_batch = []
            current_batch_chars = 0
            CHARS_PER_TOKEN = 4  # Typical average
            MAX_CHARS_PER_REQUEST = MAX_TOTAL_TOKENS_PER_REQUEST * CHARS_PER_TOKEN

            for text in texts:
                text_chars = len(text)
                
                if (current_batch_chars + text_chars > MAX_CHARS_PER_REQUEST or
                    len(current_batch) >= MAX_TEXTS_PER_REQUEST):
                    if current_batch:
                        batches.append(current_batch)
                    current_batch = [text]
                    current_batch_chars = text_chars
                else:
                    current_batch.append(text)
                    current_batch_chars += text_chars

            if current_batch:
                batches.append(current_batch)

            return batches

    def generate_embeddings(self, texts: List[str], model: str = "text-embedding-3-small", batch_size: int = None) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI API.
        
        Automatically batches based on actual token counts for optimal performance.
        """
        if not texts:
            return []
        
        embeddings = []
        
        # Create optimal batches based on actual token counts (unless batch_size is explicitly provided)
        if batch_size is None:
            batches = self._create_batches_by_tokens(texts, model)
            self.log(f"Created {len(batches)} batches for {len(texts)} texts based on token counts")
        else:
            # Manual batching if batch_size is explicitly provided
            batches = []
            for i in range(0, len(texts), batch_size):
                batches.append(texts[i:i + batch_size])
            self.log(f"Using manual batching: {len(batches)} batches of size {batch_size}")
        
        # Process each batch
        for batch_idx, batch_texts in enumerate(batches):
            if self.is_api_available():
                try:
                    response = self.client.embeddings.create(
                        input=batch_texts,
                        model=model,
                        timeout=30  # Reasonable timeout for larger batches
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    self.log(f"Batch {batch_idx + 1}/{len(batches)}: Generated {len(batch_embeddings)} embeddings")
                except Exception as e:
                    self.log(f"OpenAI API call failed for batch {batch_idx + 1}: {e}", True)
                    from src.utils.embedding import _generate_random_embeddings
                    batch_embeddings = _generate_random_embeddings(batch_texts, model)
                    self.log(f"Falling back to random embeddings for batch of {len(batch_texts)}")
            else:
                from src.utils.embedding import _generate_random_embeddings
                batch_embeddings = _generate_random_embeddings(batch_texts, model)
                self.log(f"OpenAI API not available, using random embeddings for batch {batch_idx + 1}")

            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    
    def generate_cluster_name(self, representative_texts: List[str], debug: bool = False) -> str:
        """
        Generate a cluster name using LLM based on representative texts.

        Args:
            representative_texts: List of representative text excerpts
            debug: Whether to save the prompt for debugging

        Returns:
            Generated cluster name
        """
        try:
            if not self.is_api_available():
                return "Cluster_Auto"

            # Limit to first 3 texts to avoid token limits
            texts = representative_texts[:3]

            prompt = f"""
Based on these representative document excerpts, generate a single, descriptive word or short phrase (2-3 words max) that best represents this cluster:

{chr(10).join(texts)}

Return only the cluster name, no additional text.
"""

            # Save prompt to file if debug is enabled
            if debug:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"cluster_naming_prompt_{timestamp}.txt"

                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write("=" * 60 + "\n")
                        f.write("CLUSTER NAMING PROMPT\n")
                        f.write("=" * 60 + "\n")
                        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Number of representative texts: {len(texts)}\n")
                        f.write("\nPROMPT:\n")
                        f.write("-" * 40 + "\n")
                        f.write(prompt)
                        f.write("\n" + "-" * 40 + "\n")

                        # Add the representative texts
                        f.write("\nREPRESENTATIVE TEXTS:\n")
                        f.write("-" * 40 + "\n")
                        for i, text in enumerate(texts):
                            f.write(f"Text {i+1}:\n{text[:300]}...\n\n")

                    self.log(f"Debug: Saved cluster naming prompt to {filename}")

                except Exception as e:
                    self.log(f"Debug: Failed to save naming prompt file: {e}", True)
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20
            )
            
            cluster_name = response.choices[0].message.content.strip()
            # Clean up the response
            cluster_name = cluster_name.replace('"', '').replace('.', '').strip()
            
            if cluster_name and len(cluster_name) > 0:
                self.log(f"Generated cluster name: '{cluster_name}'")
                return cluster_name
            else:
                return "Cluster_Auto"
                
        except Exception as e:
            self.log(f"LLM cluster naming failed: {e}", True)
            return "Cluster_Auto"

    def generate_single_word_cluster_label(self, cluster_id: int, representative_texts: List[str], debug: bool = False) -> str:
        """
        Generate a strict single-word cluster label via JSON response.
        Falls back to the simpler name generator on failure.
        """
        try:
            if not self.is_api_available():
                return "Cluster"

            # Limit representative docs to avoid token overflow; more than earlier for context
            docs = representative_texts[:10]
            prompt = f"""
You are given representative documents from clusters.
Assign a single-word label to each cluster.
The label must be a simple, broad, everyday word that most people would recognize as a category (like a section of a newspaper). Avoid jargon, abstract words, or specific names of events, people, or products.

Cluster {cluster_id} docs:
{json.dumps(docs, indent=2)}

Return strictly as JSON in this format:
{{
  "label": "<single_word>"
}}
"""

            # Save prompt to file if debug is enabled
            if debug:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"cluster_labelling_prompt_{timestamp}.txt"

                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write("=" * 60 + "\n")
                        f.write(f"CLUSTER LABELING PROMPT - Cluster {cluster_id}\n")
                        f.write("=" * 60 + "\n")
                        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Cluster ID: {cluster_id}\n")
                        f.write(f"Number of representative texts: {len(docs)}\n")
                        f.write("\nPROMPT:\n")
                        f.write("-" * 40 + "\n")
                        f.write(prompt)
                        f.write("\n" + "-" * 40 + "\n")

                        # Add the representative texts
                        f.write("\nREPRESENTATIVE TEXTS:\n")
                        f.write("-" * 40 + "\n")
                        for i, text in enumerate(docs):
                            f.write(f"Text {i+1}:\n{text[:500]}...\n\n")

                    self.log(f"Debug: Saved cluster labeling prompt to {filename}")

                except Exception as e:
                    self.log(f"Debug: Failed to save prompt file: {e}", True)
            
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            label = data.get("label", "Cluster")
            # Normalize to single token-ish label
            label = label.strip().split()[0].strip('".,').title() or "Cluster"
            self.log(f"Generated strict cluster label: '{label}'")
            return label
        except Exception as e:
            self.log(f"Strict cluster labeling failed, falling back: {e}", True)
            return self.generate_cluster_name(representative_texts, debug=debug)

    def enrich_label_description(self, label_name: str, existing_desc: str = "") -> str:
        """
        Generate AI-enhanced description for a label.
        
        Args:
            label_name: Name of the label
            existing_desc: Existing description (optional)
            
        Returns:
            Enhanced description
        """
        try:
            if not self.is_api_available():
                return f"Content related to {label_name.lower()} topics and themes."
            
            prompt = f"""
Generate a comprehensive, mutually exclusive description for the category "{label_name}".
The description should be 1-2 sentences that clearly define what content belongs to this category.
Make it specific enough to distinguish from other categories but broad enough to cover the topic area.

{f"Existing description: {existing_desc}" if existing_desc else ""}

Return only the description text, no additional formatting.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            
            description = response.choices[0].message.content.strip()
            self.log(f"Generated description for '{label_name}': {description[:50]}...")
            return description
            
        except Exception as e:
            self.log(f"AI description generation failed: {e}", True)
            # Fallback to basic description
            return f"Content related to {label_name.lower()} topics and themes."

    def generate_text_summary(self, text: str, max_length: int = 200) -> str:
        """
        Generate a summary of the given text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Generated summary
        """
        if not self.is_api_available():
            raise RuntimeError("OpenAI API not available. Please set OPENAI_API_KEY environment variable.")
        
        try:
            
            prompt = f"""
Summarize the following text in {max_length} characters or less:

{text}

Return only the summary, no additional text.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_length // 4  # Rough token estimation
            )
            
            summary = response.choices[0].message.content.strip()
            self.log(f"Generated summary: {summary[:50]}...")
            return summary
            
        except Exception as e:
            self.log(f"Text summarization failed: {e}", True)
            # Fallback to truncated text
            return text[:max_length] + "..." if len(text) > max_length else text
    
    def classify_text(self, text: str, categories: List[str]) -> Dict[str, Any]:
        """
        Classify text into one of the given categories.
        
        Args:
            text: Text to classify
            categories: List of possible categories
            
        Returns:
            Classification result with category and confidence
        """
        if not self.is_api_available():
            raise RuntimeError("OpenAI API not available. Please set OPENAI_API_KEY environment variable.")
        
        try:
            
            categories_str = ", ".join(categories)
            prompt = f"""
Classify the following text into one of these categories: {categories_str}

Text: {text}

Respond with only the category name that best fits the text.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50
            )
            
            predicted_category = response.choices[0].message.content.strip()
            
            # Validate that the predicted category is in our list
            if predicted_category not in categories:
                # Find closest match
                predicted_category = self._find_closest_category(predicted_category, categories)
            
            return {
                "category": predicted_category,
                "confidence": 0.8,  # LLM doesn't provide confidence scores
                "method": "llm"
            }
            
        except Exception as e:
            self.log(f"Text classification failed: {e}", True)
            raise RuntimeError(f"Failed to classify text: {e}")
    
    def _find_closest_category(self, predicted: str, categories: List[str]) -> str:
        """Find the closest matching category."""
        predicted_lower = predicted.lower()
        
        # Try exact match first
        for category in categories:
            if predicted_lower == category.lower():
                return category
        
        # Try partial match
        for category in categories:
            if predicted_lower in category.lower() or category.lower() in predicted_lower:
                return category
        
        # Return first category if no match found
        return categories[0] if categories else "unknown"
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the OpenAI service.
        
        Returns:
            Dictionary with service information
        """
        return {
            "api_available": self.is_api_available(),
            "has_api_key": bool(self.api_key and self.api_key != "your-api-key-here"),
            "client_initialized": self.client is not None,
            "api_key_length": len(self.api_key) if self.api_key else 0
        }
    
    def test_connection(self) -> bool:
        """
        Test the OpenAI API connection.
        
        Returns:
            True if connection is working, False otherwise
        """
        try:
            if not self.is_api_available():
                return False
            
            # Simple test call
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.1,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            
            self.log("OpenAI API connection test successful")
            return True
            
        except Exception as e:
            self.log(f"OpenAI API connection test failed: {e}", True)
            return False
    
    def close(self):
        """Close the OpenAI service."""
        try:
            if self.client:
                # OpenAI client doesn't have an explicit close method
                self.client = None
                self.log("OpenAI service closed")
        except Exception as e:
            self.log(f"Error closing OpenAI service: {e}", True)
