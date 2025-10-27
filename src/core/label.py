"""
Label Domain Model

Represents a classification label with metadata and operations.
This is a pure domain entity without external dependencies.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
from src.utils.hash_utils import HashUtils


@dataclass
class Label:
    """
    Label entity representing a classification label.
    
    This is a rich domain model that encapsulates label-related
    business logic and maintains data integrity.
    """
    
    # Core properties
    label_id: str
    label_name: str
    description: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    
    # Status flags
    is_enriched: bool = False
    is_custom: bool = False
    is_active: bool = True
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Computed properties
    _hash: Optional[str] = None
    
    def __post_init__(self):
        """Compute hash after initialization."""
        self._compute_hash()
    
    def _compute_hash(self):
        """Compute label hash."""
        self._hash = HashUtils.create_label_hash(self.label_id, self.label_name)
    
    @property
    def hash(self) -> str:
        """Get label hash."""
        return self._hash or ""
    
    def set_description(self, description: str):
        """Set label description."""
        self.description = description
        self.updated_at = datetime.now()
    
    def enrich_description(self, enriched_description: str):
        """Enrich label with AI-generated description."""
        self.description = enriched_description
        self.is_enriched = True
        self.updated_at = datetime.now()
    
    def add_tag(self, tag: str):
        """Add a tag to the label."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now()
    
    def remove_tag(self, tag: str):
        """Remove a tag from the label."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.now()
    
    def update_metadata(self, key: str, value: Any):
        """Update label metadata."""
        self.metadata[key] = value
        self.updated_at = datetime.now()
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value with default."""
        return self.metadata.get(key, default)
    
    def has_tag(self, tag: str) -> bool:
        """Check if label has a specific tag."""
        return tag in self.tags
    
    def mark_as_custom(self):
        """Mark label as custom (user-created)."""
        self.is_custom = True
        self.updated_at = datetime.now()
    
    def mark_as_enriched(self):
        """Mark label as enriched with AI description."""
        self.is_enriched = True
        self.updated_at = datetime.now()
    
    def deactivate(self):
        """Deactivate the label."""
        self.is_active = False
        self.updated_at = datetime.now()
    
    def activate(self):
        """Activate the label."""
        self.is_active = True
        self.updated_at = datetime.now()
    
    def get_display_text(self) -> str:
        """Get display text for the label."""
        if self.description:
            return f"{self.label_name}: {self.description}"
        return self.label_name
    
    def get_full_text(self) -> str:
        """Get full text including name and description."""
        if self.description:
            return f"{self.label_name}: {self.description}"
        return self.label_name
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get status information about the label."""
        return {
            "is_enriched": self.is_enriched,
            "is_custom": self.is_custom,
            "is_active": self.is_active,
            "has_description": bool(self.description),
            "tag_count": len(self.tags),
            "metadata_count": len(self.metadata)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert label to dictionary representation."""
        return {
            "label_id": self.label_id,
            "label_name": self.label_name,
            "description": self.description,
            "metadata": self.metadata,
            "tags": self.tags,
            "is_enriched": self.is_enriched,
            "is_custom": self.is_custom,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "hash": self.hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Label':
        """Create Label from dictionary representation."""
        label = cls(
            label_id=data["label_id"],
            label_name=data["label_name"],
            description=data.get("description"),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            is_enriched=data.get("is_enriched", False),
            is_custom=data.get("is_custom", False),
            is_active=data.get("is_active", True),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )
        
        # Set hash if provided
        if "hash" in data:
            label._hash = data["hash"]
        
        return label
    
    @classmethod
    def create_custom(cls, label_name: str, description: str = None) -> 'Label':
        """Create a custom label."""
        # Generate deterministic label ID
        label_id = f"custom_{len(label_name)}_{HashUtils.generate_deterministic_seed(label_name, 1000)}"
        
        label = cls(
            label_id=label_id,
            label_name=label_name,
            description=description,
            is_custom=True
        )
        
        return label
    
    def __str__(self) -> str:
        """String representation of the label."""
        return f"Label(id='{self.label_id}', name='{self.label_name}')"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Label(id='{self.label_id}', name='{self.label_name}', "
                f"enriched={self.is_enriched}, custom={self.is_custom}, "
                f"active={self.is_active})")
