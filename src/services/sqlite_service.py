"""
SQLite Service

Wraps the existing HashIndex.py functionality with better interface and error handling.
Provides SQLite3 hash index operations for deduplication.
"""

from typing import List, Set, Dict, Any, Optional
from pathlib import Path
import sqlite3
from src.utils.HashIndex import HashIndex


class SQLiteService:
    """
    Service for managing SQLite3 hash index operations.
    
    This service wraps the existing HashIndex functionality and provides
    a cleaner interface with better error handling and logging.
    """
    
    def __init__(self, db_path: str = "config/hash_index.db", log_function=None):
        """
        Initialize SQLite service.
        
        Args:
            db_path: Path to SQLite database file
            log_function: Optional logging function
        """
        self.db_path = Path(db_path)
        self.log = log_function or print
        self._hash_index = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Ensure the hash index is initialized."""
        if not self._initialized:
            try:
                self._hash_index = HashIndex(str(self.db_path))
                self._initialized = True
                self.log(f"SQLite service initialized with database: {self.db_path}")
                
                # Create collection metadata table if it doesn't exist
                self._create_metadata_table()
            except Exception as e:
                self.log(f"Failed to initialize SQLite service: {e}", True)
                raise
    
    def _create_metadata_table(self):
        """Create collection metadata table for storing creation timestamps and other metadata."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS collection_metadata (
                    collection_name TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT,
                    clustering_method TEXT,
                    num_clusters INTEGER
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            self.log(f"Failed to create metadata table: {e}", True)
    
    def is_duplicate(self, collection: str, hash_value: str) -> bool:
        """
        Check if a hash already exists in the collection.
        
        Args:
            collection: Collection name
            hash_value: Hash value to check
            
        Returns:
            True if hash exists (duplicate), False otherwise
        """
        try:
            self._ensure_initialized()
            return self._hash_index.exists(collection, hash_value)
        except Exception as e:
            self.log(f"Failed to check duplicate: {e}", True)
            return False
    
    def mark_as_processed(self, collection: str, hash_values: List[str]) -> int:
        """
        Mark hash values as processed in the collection.
        
        Args:
            collection: Collection name
            hash_values: List of hash values to mark as processed
            
        Returns:
            Number of hashes successfully added
        """
        try:
            self._ensure_initialized()
            
            # Filter out already existing hashes
            new_hashes = []
            for hash_value in hash_values:
                if not self._hash_index.exists(collection, hash_value):
                    new_hashes.append(hash_value)
            
            if new_hashes:
                self._hash_index.bulk_add(collection, new_hashes)
                self.log(f"Marked {len(new_hashes)} new hashes as processed in '{collection}'")
            
            return len(new_hashes)
            
        except Exception as e:
            self.log(f"Failed to mark hashes as processed: {e}", True)
            return 0
    
    def get_processed_hashes(self, collection: str) -> Set[str]:
        """
        Get all processed hashes for a collection.
        
        Args:
            collection: Collection name
            
        Returns:
            Set of processed hash values
        """
        try:
            self._ensure_initialized()
            return self._hash_index.load_all(collection)
        except Exception as e:
            self.log(f"Failed to get processed hashes: {e}", True)
            return set()
    
    def clear_collection(self, collection: str) -> bool:
        """
        Clear all hashes for a collection.
        
        Args:
            collection: Collection name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._ensure_initialized()
            
            # Connect directly to database to drop table
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f"DROP TABLE IF EXISTS {collection}")
            conn.commit()
            conn.close()
            
            # Remove from initialized set if it exists
            if hasattr(self._hash_index, 'initialized'):
                self._hash_index.initialized.discard(collection)
            
            self.log(f"Dropped SQLite hash table for collection '{collection}'")
            return True
            
        except Exception as e:
            self.log(f"Failed to clear collection: {e}", True)
            return False
    
    def get_stats(self, collection: str) -> Dict[str, Any]:
        """
        Get statistics for a collection.
        
        Args:
            collection: Collection name
            
        Returns:
            Dictionary with collection statistics
        """
        try:
            self._ensure_initialized()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (collection,))
            
            if not cursor.fetchone():
                conn.close()
                return {
                    "collection": collection,
                    "hash_count": 0,
                    "table_exists": False
                }
            
            # Get hash count
            cursor.execute(f"SELECT COUNT(*) FROM {collection}")
            hash_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "collection": collection,
                "hash_count": hash_count,
                "table_exists": True
            }
            
        except Exception as e:
            self.log(f"Failed to get stats: {e}", True)
            return {
                "collection": collection,
                "hash_count": 0,
                "table_exists": False,
                "error": str(e)
            }
    
    def get_all_collections(self) -> List[str]:
        """
        Get all collection names that have hash tables.
        
        Returns:
            List of collection names
        """
        try:
            self._ensure_initialized()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table'
            """)
            
            collections = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return collections
            
        except Exception as e:
            self.log(f"Failed to get collections: {e}", True)
            return []
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get general database information.
        
        Returns:
            Dictionary with database information
        """
        try:
            self._ensure_initialized()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get database file size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            # Get table count
            cursor.execute("""
                SELECT COUNT(*) FROM sqlite_master 
                WHERE type='table'
            """)
            table_count = cursor.fetchone()[0]
            
            # Get total hash count across all tables
            total_hashes = 0
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table'
            """)
            
            for table_name in cursor.fetchall():
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name[0]}")
                    total_hashes += cursor.fetchone()[0]
                except:
                    pass  # Skip tables that can't be queried
            
            conn.close()
            
            return {
                "database_path": str(self.db_path),
                "database_size_bytes": db_size,
                "table_count": table_count,
                "total_hash_count": total_hashes,
                "is_initialized": self._initialized
            }
            
        except Exception as e:
            self.log(f"Failed to get database info: {e}", True)
            return {
                "database_path": str(self.db_path),
                "error": str(e),
                "is_initialized": self._initialized
            }
    
    def close(self):
        """Close the database connection."""
        try:
            if self._hash_index:
                self._hash_index.close()
                self._initialized = False
                self.log("SQLite service closed")
        except Exception as e:
            self.log(f"Error closing SQLite service: {e}", True)
    
    def __enter__(self):
        """Context manager entry."""
        self._ensure_initialized()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
