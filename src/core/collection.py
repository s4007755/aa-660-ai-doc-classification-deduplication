"""
Collection Domain Model

Represents a vector collection with metadata and operations.
This is a pure domain entity without external dependencies.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class DistanceMetric(Enum):
    """Distance metrics for vector similarity."""
    COSINE = "Cosine"
    EUCLIDEAN = "Euclidean"
    DOT = "Dot"


@dataclass
class Collection:
    """
    Collection entity representing a vector collection.
    
    This is a rich domain model that encapsulates collection-related
    business logic and maintains data integrity.
    """
    
    # Core properties
    name: str
    dimension: int
    model: str
    
    # Configuration
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    
    # Metadata
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Statistics
    vector_count: int = 0
    label_count: int = 0
    cluster_count: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Status
    is_active: bool = True
    
    def set_description(self, description: str):
        """Set collection description."""
        self.description = description
        self.updated_at = datetime.now()
    
    def add_tag(self, tag: str):
        """Add a tag to the collection."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now()
    
    def remove_tag(self, tag: str):
        """Remove a tag from the collection."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.now()
    
    def update_metadata(self, key: str, value: Any):
        """Update collection metadata."""
        self.metadata[key] = value
        self.updated_at = datetime.now()
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value with default."""
        return self.metadata.get(key, default)
    
    def has_tag(self, tag: str) -> bool:
        """Check if collection has a specific tag."""
        return tag in self.tags
    
    def update_stats(self, vector_count: int = None, label_count: int = None, cluster_count: int = None):
        """Update collection statistics."""
        if vector_count is not None:
            self.vector_count = vector_count
        if label_count is not None:
            self.label_count = label_count
        if cluster_count is not None:
            self.cluster_count = cluster_count
        
        self.updated_at = datetime.now()
    
    def increment_vector_count(self, count: int = 1):
        """Increment vector count."""
        self.vector_count += count
        self.updated_at = datetime.now()
    
    def increment_label_count(self, count: int = 1):
        """Increment label count."""
        self.label_count += count
        self.updated_at = datetime.now()
    
    def increment_cluster_count(self, count: int = 1):
        """Increment cluster count."""
        self.cluster_count += count
        self.updated_at = datetime.now()
    
    def is_empty(self) -> bool:
        """Check if collection is empty."""
        return self.vector_count == 0
    
    def has_labels(self) -> bool:
        """Check if collection has labels."""
        return self.label_count > 0
    
    def has_clusters(self) -> bool:
        """Check if collection has clusters."""
        return self.cluster_count > 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            "vector_count": self.vector_count,
            "label_count": self.label_count,
            "cluster_count": self.cluster_count,
            "is_empty": self.is_empty(),
            "has_labels": self.has_labels(),
            "has_clusters": self.has_clusters()
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get collection configuration."""
        return {
            "name": self.name,
            "dimension": self.dimension,
            "model": self.model,
            "distance_metric": self.distance_metric.value,
            "description": self.description,
            "is_active": self.is_active
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert collection to dictionary representation."""
        return {
            "name": self.name,
            "dimension": self.dimension,
            "model": self.model,
            "distance_metric": self.distance_metric.value,
            "description": self.description,
            "metadata": self.metadata,
            "tags": self.tags,
            "vector_count": self.vector_count,
            "label_count": self.label_count,
            "cluster_count": self.cluster_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_active": self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Collection':
        """Create Collection from dictionary representation."""
        return cls(
            name=data["name"],
            dimension=data["dimension"],
            model=data["model"],
            distance_metric=DistanceMetric(data.get("distance_metric", "Cosine")),
            description=data.get("description"),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            vector_count=data.get("vector_count", 0),
            label_count=data.get("label_count", 0),
            cluster_count=data.get("cluster_count", 0),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            is_active=data.get("is_active", True)
        )
    
    def __str__(self) -> str:
        """String representation of the collection."""
        return f"Collection(name='{self.name}', vectors={self.vector_count}, model='{self.model}')"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Collection(name='{self.name}', dimension={self.dimension}, "
                f"model='{self.model}', vectors={self.vector_count}, "
                f"labels={self.label_count}, clusters={self.cluster_count})")
