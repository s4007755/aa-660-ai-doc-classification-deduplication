"""
Core Domain Models

This module contains the core domain entities for the document classification system.
These are pure domain models without external dependencies.
"""

from .document import Document
from .collection import Collection, DistanceMetric
from .label import Label

__all__ = [
    'Document',
    'Collection', 
    'DistanceMetric',
    'Label'
]
