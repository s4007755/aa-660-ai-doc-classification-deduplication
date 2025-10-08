"""
Standardized hash utilities for the document classification system.

This module provides consistent hash functions across the entire codebase,
ensuring deterministic, secure, and standardized hashing for:
- Text content deduplication
- File content verification  
- Label identification
- Random seed generation
"""

import hashlib
import os
from typing import Union, BinaryIO


class HashUtils:
    """
    Standardized hash utilities with consistent algorithms and encoding.
    
    Uses SHA256 as the primary algorithm for:
    - Security (collision resistance)
    - Consistency (deterministic across runs)
    - Performance (reasonable speed)
    """
    
    # Standard algorithm for all text/content hashing
    DEFAULT_ALGORITHM = "sha256"
    
    # Alternative algorithms for specific use cases
    ALGORITHMS = {
        "sha256": hashlib.sha256,
        "sha1": hashlib.sha1,
        "md5": hashlib.md5,
        "blake2b": hashlib.blake2b,
    }
    
    @staticmethod
    def hash_text(text: str, algorithm: str = DEFAULT_ALGORITHM) -> str:
        """
        Hash text content using standardized algorithm.
        
        Args:
            text: Text content to hash
            algorithm: Hash algorithm to use (default: sha256)
            
        Returns:
            Hexadecimal hash string
            
        Example:
            >>> HashUtils.hash_text("Hello World")
            'a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e'
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Normalize text: strip whitespace and convert to lowercase for consistency
        normalized_text = text.strip().lower()
        
        # Use standardized encoding
        text_bytes = normalized_text.encode("utf-8")
        
        # Generate hash
        hash_func = HashUtils.ALGORITHMS.get(algorithm, HashUtils.ALGORITHMS[HashUtils.DEFAULT_ALGORITHM])
        return hash_func(text_bytes).hexdigest()
    
    @staticmethod
    def hash_file(file_obj: BinaryIO, algorithm: str = DEFAULT_ALGORITHM) -> str:
        """
        Hash file content using standardized algorithm.
        
        Args:
            file_obj: Binary file object to hash
            algorithm: Hash algorithm to use (default: sha256)
            
        Returns:
            Hexadecimal hash string
            
        Example:
            >>> with open("document.pdf", "rb") as f:
            ...     file_hash = HashUtils.hash_file(f)
        """
        hash_func = HashUtils.ALGORITHMS.get(algorithm, HashUtils.ALGORITHMS[HashUtils.DEFAULT_ALGORITHM])
        hasher = hash_func()
        
        # Reset file pointer to beginning
        file_obj.seek(0)
        
        # Read file in chunks for memory efficiency
        chunk_size = 8192  # 8KB chunks
        while chunk := file_obj.read(chunk_size):
            hasher.update(chunk)
        
        return hasher.hexdigest()
    
    @staticmethod
    def hash_file_path(file_path: str, algorithm: str = DEFAULT_ALGORITHM) -> str:
        """
        Hash file content from file path.
        
        Args:
            file_path: Path to file to hash
            algorithm: Hash algorithm to use (default: sha256)
            
        Returns:
            Hexadecimal hash string
        """
        try:
            with open(file_path, "rb") as f:
                return HashUtils.hash_file(f, algorithm)
        except (OSError, IOError) as e:
            raise ValueError(f"Cannot hash file '{file_path}': {e}")
    
    @staticmethod
    def hash_bytes(data: bytes, algorithm: str = DEFAULT_ALGORITHM) -> str:
        """
        Hash binary data using standardized algorithm.
        
        Args:
            data: Binary data to hash
            algorithm: Hash algorithm to use (default: sha256)
            
        Returns:
            Hexadecimal hash string
            
        Example:
            >>> HashUtils.hash_bytes(b"hello world")
            'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
        """
        """
        Hash binary data using standardized algorithm.
        
        Args:
            data: Binary data to hash
            algorithm: Hash algorithm to use (default: sha256)
            
        Returns:
            Hexadecimal hash string
        """
        hash_func = HashUtils.ALGORITHMS.get(algorithm, HashUtils.ALGORITHMS[HashUtils.DEFAULT_ALGORITHM])
        return hash_func(data).hexdigest()
    
    @staticmethod
    def hash_path(path: str, algorithm: str = DEFAULT_ALGORITHM) -> str:
        """
        Hash file path string (not content) for path-based identification.
        
        Args:
            path: File path to hash
            algorithm: Hash algorithm to use (default: sha256)
            
        Returns:
            Hexadecimal hash string
        """
        # Normalize path for consistency across platforms
        normalized_path = os.path.normpath(path).lower()
        return HashUtils.hash_text(normalized_path, algorithm)
    
    @staticmethod
    def generate_deterministic_seed(text: str, max_value: int = 2**32) -> int:
        """
        Generate deterministic seed from text for random number generation.
        
        Args:
            text: Text to generate seed from
            max_value: Maximum value for seed (default: 2^32)
            
        Returns:
            Integer seed value
            
        Example:
            >>> HashUtils.generate_deterministic_seed("Hello World")
            1234567890  # Deterministic across runs
        """
        # Use SHA256 hash for deterministic seed generation
        hash_value = HashUtils.hash_text(text)
        
        # Convert first 8 characters of hash to integer
        # This ensures deterministic behavior across runs
        seed_str = hash_value[:8]
        seed = int(seed_str, 16)
        
        # Ensure seed is within bounds
        return seed % max_value
    
    @staticmethod
    def create_label_hash(label_id: str, label_text: str, algorithm: str = DEFAULT_ALGORITHM) -> str:
        """
        Create standardized hash for labels combining ID and text.
        
        Args:
            label_id: Label identifier
            label_text: Label text content
            algorithm: Hash algorithm to use (default: sha256)
            
        Returns:
            Hexadecimal hash string
            
        Example:
            >>> HashUtils.create_label_hash("0", "Politics")
            'label_0_politics_hash_value'
        """
        # Combine ID and text for unique label identification
        combined = f"{label_id}:{label_text.strip().lower()}"
        return f"label_{HashUtils.hash_text(combined, algorithm)}"
    
    @staticmethod
    def verify_hash_consistency(text: str, expected_hash: str, algorithm: str = DEFAULT_ALGORITHM) -> bool:
        """
        Verify that text produces expected hash.
        
        Args:
            text: Text to verify
            expected_hash: Expected hash value
            algorithm: Hash algorithm to use (default: sha256)
            
        Returns:
            True if hash matches, False otherwise
        """
        actual_hash = HashUtils.hash_text(text, algorithm)
        return actual_hash == expected_hash


# Convenience functions for backward compatibility and ease of use
def hash_text(text: str) -> str:
    """Convenience function for text hashing."""
    return HashUtils.hash_text(text)


def hash_file(file_obj: BinaryIO) -> str:
    """Convenience function for file hashing."""
    return HashUtils.hash_file(file_obj)


def hash_file_path(file_path: str) -> str:
    """Convenience function for file path hashing."""
    return HashUtils.hash_file_path(file_path)


def generate_deterministic_seed(text: str, max_value: int = 2**32) -> int:
    """Convenience function for deterministic seed generation."""
    return HashUtils.generate_deterministic_seed(text, max_value)


def create_label_hash(label_id: str, label_text: str) -> str:
    """Convenience function for label hashing."""
    return HashUtils.create_label_hash(label_id, label_text)
