"""
Document Domain Model

Represents a document with content, metadata, and operations.
This is a pure domain entity without external dependencies.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
from src.utils.hash_utils import HashUtils


@dataclass
class Document:
    """
    Document entity representing a file or text content.
    
    This is a rich domain model that encapsulates document-related
    business logic and maintains data integrity.
    """
    
    # Core properties
    path: str
    content: str
    name: str
    extension: str
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    summary: Optional[str] = None
    category: Optional[str] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Hashes (computed properties)
    _hash_binary: Optional[str] = None
    _hash_path: Optional[str] = None
    _hash_content: Optional[str] = None
    _hash_title: Optional[str] = None
    
    def __post_init__(self):
        """Compute hashes after initialization."""
        self._compute_hashes()
    
    def _compute_hashes(self):
        """Compute all document hashes."""
        self._hash_path = HashUtils.hash_path(self.path)
        self._hash_content = HashUtils.hash_text(self.content or "")
        self._hash_title = HashUtils.hash_text(self.name)
    
    @property
    def hash_binary(self) -> str:
        """Get binary hash of the file content."""
        return self._hash_binary or ""
    
    @property
    def hash_path(self) -> str:
        """Get hash of the file path."""
        return self._hash_path or ""
    
    @property
    def hash_content(self) -> str:
        """Get hash of the document content."""
        return self._hash_content or ""
    
    @property
    def hash_title(self) -> str:
        """Get hash of the document title/name."""
        return self._hash_title or ""
    
    def set_binary_hash(self, binary_hash: str):
        """Set the binary hash (computed from file content)."""
        self._hash_binary = binary_hash
    
    def set_category(self, category: str):
        """Set the document category."""
        self.category = category
        self.updated_at = datetime.now()
    
    def add_tag(self, tag: str):
        """Add a tag to the document."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now()
    
    def remove_tag(self, tag: str):
        """Remove a tag from the document."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.now()
    
    def update_metadata(self, key: str, value: Any):
        """Update document metadata."""
        self.metadata[key] = value
        self.updated_at = datetime.now()
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value with default."""
        return self.metadata.get(key, default)
    
    def has_tag(self, tag: str) -> bool:
        """Check if document has a specific tag."""
        return tag in self.tags
    
    def is_empty(self) -> bool:
        """Check if document content is empty."""
        return not self.content or self.content.strip() == ""
    
    def get_content_preview(self, max_length: int = 200) -> str:
        """Get a preview of the document content."""
        if not self.content:
            return ""
        
        if len(self.content) <= max_length:
            return self.content
        
        return self.content[:max_length] + "..."
    
    def get_size_info(self) -> Dict[str, int]:
        """Get size information about the document."""
        return {
            "content_length": len(self.content or ""),
            "metadata_count": len(self.metadata),
            "tags_count": len(self.tags),
            "path_length": len(self.path)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary representation."""
        return {
            "path": self.path,
            "name": self.name,
            "extension": self.extension,
            "content": self.content,
            "metadata": self.metadata,
            "tags": self.tags,
            "summary": self.summary,
            "category": self.category,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "hashes": {
                "binary": self.hash_binary,
                "path": self.hash_path,
                "content": self.hash_content,
                "title": self.hash_title
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create Document from dictionary representation."""
        doc = cls(
            path=data["path"],
            content=data["content"],
            name=data["name"],
            extension=data["extension"],
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            summary=data.get("summary"),
            category=data.get("category"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )
        
        # Set hashes if provided
        if "hashes" in data:
            hashes = data["hashes"]
            doc._hash_binary = hashes.get("binary")
            doc._hash_path = hashes.get("path")
            doc._hash_content = hashes.get("content")
            doc._hash_title = hashes.get("title")
        
        return doc
    
    def __hash__(self) -> int:
        """Hash function for use in sets and as dictionary keys."""
        return int(self.hash_path[:8], 16) if self.hash_path else 0
    
    def __str__(self) -> str:
        """String representation of the document."""
        return f"Document(name='{self.name}', path='{self.path}', category='{self.category}')"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Document(name='{self.name}', extension='{self.extension}', "
                f"path='{self.path}', content_length={len(self.content or '')}, "
                f"category='{self.category}', tags={self.tags})")
