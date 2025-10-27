"""
Qdrant Service

Wraps the existing VecDB.py functionality with better interface and error handling.
Provides Qdrant vector database operations and management.
"""

from typing import List, Dict, Any, Optional, Tuple
import threading
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue, MatchAny
from src.core import Collection, DistanceMetric


class QdrantService:
    """
    Service for managing Qdrant vector database operations.
    
    This service wraps the existing VecDB functionality and provides
    a cleaner interface with better error handling and logging.
    """
    
    def __init__(self, host: str = "localhost", port: int = 6333, log_function=None):
        """
        Initialize Qdrant service.
        
        Args:
            host: Qdrant host
            port: Qdrant port
            log_function: Optional logging function
        """
        self.host = host
        self.port = port
        self.log = log_function or print
        self.client = None
        self.connected = False
        self._id_counters = {}  # Store counters per collection
        self._id_lock = threading.Lock()  # Thread safety
        
        self._connect()
    
    def _connect(self) -> None:
        """Connect to Qdrant server."""
        try:
            self.log("Initializing Qdrant client...")
            self.client = QdrantClient(
                host=self.host, 
                port=self.port, 
                timeout=5.0,  # Reduced timeout for faster failures
                prefer_grpc=False  # Use HTTP for compatibility
            )
            self.client.get_collections()
            self.log(f"Connected to Qdrant on {self.host}:{self.port}")
            self.connected = True
        except Exception as e:
            self.log(f"Failed to connect to Qdrant: {e}", True)
            self.connected = False
    
    def is_connected(self) -> bool:
        """Check if connected to Qdrant."""
        return self.connected
    
    def reconnect(self, host: str = None, port: int = None):
        """Reconnect to Qdrant server."""
        if host:
            self.host = host
        if port:
            self.port = port
        self._connect()
    
    def create_collection(self, name: str, dimension: int, model: str, description: str = None) -> bool:
        """
        Create a new collection.
        
        Args:
            name: Collection name
            dimension: Vector dimension
            model: Model name
            description: Optional description of the collection
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.connected:
                self.log("Cannot create collection: not connected to Qdrant.", True)
                return False
            
            # Check if collection already exists
            collections = self.client.get_collections()
            existing_names = [col.name for col in collections.collections]
            
            if name in existing_names:
                self.log(f"Collection '{name}' already exists.", True)
                return False
            
            # Create collection
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                )
            )
            
            # Store collection metadata as a special point with ID 0
            try:
                from datetime import datetime
                metadata_payload = {
                    "_metadata": True,
                    "created_at": datetime.now().isoformat(),
                    "embedding_model": model,
                    "dimension": dimension
                }
                
                # Add description if provided
                if description:
                    metadata_payload["description"] = description
                
                metadata_point = PointStruct(
                    id=0,
                    vector=[0.0] * dimension,  # Zero vector for metadata point
                    payload=metadata_payload
                )
                self.client.upsert(
                    collection_name=name,
                    points=[metadata_point]
                )
                self.log(f"Collection '{name}' created with model '{model}' (vector size {dimension})")
            except Exception as meta_error:
                self.log(f"Collection '{name}' created with model '{model}' (vector size {dimension})")
                self.log(f"Warning: Could not store collection metadata: {meta_error}", True)
            
            return True
            
        except Exception as e:
            self.log(f"Failed to create collection '{name}': {e}", True)
            return False
    
    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            name: Collection name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.connected:
                self.log("Cannot delete collection: not connected to Qdrant.", True)
                return False
            
            self.client.delete_collection(name)
            self.log(f"Collection '{name}' deleted")
            return True
            
        except Exception as e:
            self.log(f"Failed to delete collection '{name}': {e}", True)
            return False
    
    def list_collections(self) -> Dict[str, Dict[str, Any]]:
        """
        List all collections with their information.
        
        Returns:
            Dictionary mapping collection names to their info
        """
        try:
            if not self.connected:
                self.log("Cannot list collections: not connected to Qdrant.", True)
                return {}
            
            collections = self.client.get_collections()
            result = {}
            
            for collection in collections.collections:
                try:
                    info = self.client.get_collection(collection.name)
                    metadata = self.get_collection_metadata(collection.name)
                    
                    # Adjust count to exclude metadata point if it exists
                    actual_count = info.points_count
                    if metadata:
                        actual_count = max(0, info.points_count - 1)
                    
                    result[collection.name] = {
                        "size": info.config.params.vectors.size,
                        "vectors": actual_count,
                        "distance": info.config.params.vectors.distance.value,
                        "created_at": metadata.get('created_at') if metadata else None,
                        "description": metadata.get('description') if metadata else None
                    }
                except Exception as e:
                    result[collection.name] = {
                        "size": "unknown",
                        "vectors": "unknown", 
                        "distance": "unknown",
                        "created_at": None,
                        "error": str(e)
                    }
            
            return result
            
        except Exception as e:
            self.log(f"Failed to list collections: {e}", True)
            return {}
    
    def get_collection_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata from the special metadata point (ID 0) in the collection.
        
        Args:
            name: Collection name
            
        Returns:
            Metadata dict or None if not found
        """
        try:
            if not self.connected:
                return None
            
            # Try to retrieve the metadata point (ID 0)
            points = self.client.retrieve(
                collection_name=name,
                ids=[0],
                with_payload=True
            )
            
            if points and len(points) > 0:
                payload = points[0].payload
                if payload.get("_metadata"):
                    return payload
            
            return None
            
        except Exception as e:
            # Metadata point doesn't exist (old collection)
            return None
    
    def update_collection_metadata(self, name: str, metadata_updates: Dict[str, Any]) -> bool:
        """
        Update the metadata point (ID 0) with new information.
        
        Args:
            name: Collection name
            metadata_updates: Dictionary of metadata fields to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.connected:
                return False
            
            # Get current metadata
            current_metadata = self.get_collection_metadata(name)
            
            if current_metadata:
                # Update existing metadata
                self.client.set_payload(
                    collection_name=name,
                    payload=metadata_updates,
                    points=[0]
                )
                return True
            else:
                # Create metadata point if it doesn't exist
                from datetime import datetime
                info = self.client.get_collection(name)
                dimension = info.config.params.vectors.size
                
                metadata_point = PointStruct(
                    id=0,
                    vector=[0.0] * dimension,
                    payload={
                        "_metadata": True,
                        "created_at": datetime.now().isoformat(),
                        **metadata_updates
                    }
                )
                self.client.upsert(
                    collection_name=name,
                    points=[metadata_point]
                )
                return True
                
        except Exception as e:
            self.log(f"Failed to update collection metadata: {e}", True)
            return False
    
    def get_collection_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a collection.
        
        Args:
            name: Collection name
            
        Returns:
            Collection information or None if not found
        """
        try:
            if not self.connected:
                self.log("Cannot get collection info: not connected to Qdrant.", True)
                return None
            
            info = self.client.get_collection(name)
            
            # Get metadata from special metadata point
            metadata = self.get_collection_metadata(name)
            embedding_model = metadata.get('embedding_model') if metadata else None
            created_at = metadata.get('created_at') if metadata else None
            description = metadata.get('description') if metadata else None
            
            # Adjust vector count to exclude metadata point (ID 0) if it exists
            actual_vector_count = info.points_count
            if metadata:
                actual_vector_count = max(0, info.points_count - 1)
            
            return {
                "name": name,
                "dimension": info.config.params.vectors.size,
                "distance_metric": info.config.params.vectors.distance.value,
                "vector_count": actual_vector_count,
                "status": info.status.value,
                "embedding_model": embedding_model,
                "created_at": created_at,
                "description": description
            }
            
        except Exception as e:
            self.log(f"Failed to get collection info for '{name}': {e}", True)
            return None
    
    def get_collection_model(self, name: str) -> Optional[str]:
        """
        Get the embedding model used for a collection.
        
        Args:
            name: Collection name
            
        Returns:
            Model name or None if not found/not stored
        """
        try:
            info = self.get_collection_info(name)
            if info:
                return info.get('embedding_model')
            return None
        except Exception as e:
            self.log(f"Failed to get collection model for '{name}': {e}", True)
            return None
    
    def _reserve_id_block(self, collection_name: str, count: int) -> int:
        """Reserve a contiguous block of IDs atomically and return the starting ID.

        This eliminates race conditions across concurrent insertions by ensuring
        each batch receives a unique, non-overlapping ID range.
        """
        with self._id_lock:
            try:
                if collection_name not in self._id_counters:
                    collection_info = self.client.get_collection(collection_name)
                    if collection_info.points_count == 0:
                        self._id_counters[collection_name] = 1
                    else:
                        max_id = 0
                        try:
                            scroll_result = self.client.scroll(
                                collection_name=collection_name,
                                limit=collection_info.points_count,
                                with_payload=False,
                                with_vectors=False,
                            )
                            points, _ = scroll_result
                            if points:
                                max_id = max(int(point.id) for point in points)
                                self.log(f"Found max ID {max_id} in collection '{collection_name}'")
                        except Exception as e:
                            self.log(f"Could not scan for max ID: {e}", True)
                            max_id = collection_info.points_count
                        self._id_counters[collection_name] = max_id + 1

                start_id = self._id_counters[collection_name]
                # Reserve the requested block in one step
                self._id_counters[collection_name] += max(1, int(count))
                return start_id

            except Exception as e:
                self.log(f"Failed to reserve ID block for collection '{collection_name}': {e}", True)
                try:
                    collection_info = self.client.get_collection(collection_name)
                    # Best-effort fallback: return next available based on current count
                    return collection_info.points_count + 1
                except:
                    return 1
    
    def insert_vectors(self, collection_name: str, vectors: List[List[float]], payloads: List[Dict[str, Any]], batch_size: int = 1000, sqlite_service=None) -> Tuple[bool, int, int]:
        """
        Insert vectors into a collection with batch processing and duplicate checking.
        
        Args:
            collection_name: Collection name
            vectors: List of vector embeddings
            payloads: List of payload dictionaries
            batch_size: Number of vectors to insert per batch
            sqlite_service: Optional SQLite service for duplicate checking
            
        Returns:
            Tuple of (success, inserted_count, skipped_count)
        """
        try:
            if not self.connected:
                self.log("Cannot insert vectors: not connected to Qdrant.", True)
                return False, 0, 0
            
            if not vectors or not payloads or len(vectors) != len(payloads):
                self.log("Vectors and payloads must have equal, non-empty lengths.", True)
                return False, 0, 0
            
            # Filter out duplicates if SQLite service is provided
            filtered_vectors = []
            filtered_payloads = []
            duplicate_hashes = []
            
            if sqlite_service:
                self.log("Checking for duplicates using SQLite hash index...")
                for vec, payload in zip(vectors, payloads):
                    if 'hash' in payload:
                        hash_value = payload['hash']
                        if sqlite_service.is_duplicate(collection_name, hash_value):
                            duplicate_hashes.append(hash_value)
                        else:
                            filtered_vectors.append(vec)
                            filtered_payloads.append(payload)
                    else:
                        # If no hash in payload, include it (shouldn't happen with proper processing)
                        filtered_vectors.append(vec)
                        filtered_payloads.append(payload)
                
                if duplicate_hashes:
                    self.log(f"Skipped {len(duplicate_hashes)} duplicate documents")
            else:
                # No duplicate checking, use all vectors
                filtered_vectors = vectors
                filtered_payloads = payloads
            
            if not filtered_vectors:
                self.log("No new vectors to insert after duplicate filtering.")
                return True, 0, len(duplicate_hashes)
            
            # Reserve a unique ID range atomically for the filtered insertion set
            start_id = self._reserve_id_block(collection_name, len(filtered_vectors))
            
            total_inserted = 0
            total_skipped = len(duplicate_hashes)
            
            # Process filtered vectors in batches to avoid payload size limits
            for i in range(0, len(filtered_vectors), batch_size):
                batch_vectors = filtered_vectors[i:i + batch_size]
                batch_payloads = filtered_payloads[i:i + batch_size]
                
                points = []
                batch_inserted = 0
                
                for j, (vec, payload) in enumerate(zip(batch_vectors, batch_payloads)):
                    points.append(PointStruct(
                        id=start_id + i + j,
                        vector=vec,
                        payload=payload
                    ))
                    batch_inserted += 1
                
                # Insert this batch with retry logic
                max_retries = 3
                batch_success = False
                
                for retry in range(max_retries):
                    try:
                        self.client.upsert(collection_name=collection_name, points=points)
                        total_inserted += batch_inserted
                        self.log(f"Inserted batch {i//batch_size + 1}/{(len(filtered_vectors) + batch_size - 1)//batch_size} ({batch_inserted} vectors)")
                        # After a successful upsert, record hashes for this batch only
                        if sqlite_service:
                            batch_hashes = [p.get('hash') for p in batch_payloads if isinstance(p, dict) and 'hash' in p]
                            if batch_hashes:
                                sqlite_service.mark_as_processed(collection_name, batch_hashes)
                        batch_success = True
                        break
                    except Exception as e:
                        if "timed out" in str(e).lower() and retry < max_retries - 1:
                            self.log(f"Batch {i//batch_size + 1} timed out, retrying ({retry + 1}/{max_retries})...")
                            continue
                        else:
                            self.log(f"Failed to insert batch {i//batch_size + 1}: {e}", True)
                            total_skipped += batch_inserted
                            break
            
            self.log(f"Inserted {total_inserted} vectors into collection '{collection_name}'")
            return True, total_inserted, total_skipped
            
        except Exception as e:
            self.log(f"Failed to insert vectors: {e}", True)
            return False, 0, 0
    
    def search_vectors(self, collection_name: str, query_vector: List[float], limit: int = 10, filter_conditions: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            collection_name: Collection name
            query_vector: Query vector
            limit: Maximum number of results
            filter_conditions: Optional filter conditions
            
        Returns:
            List of search results
        """
        try:
            if not self.connected:
                self.log("Cannot search vectors: not connected to Qdrant.", True)
                return []
            
            # Build filter if conditions provided
            search_filter = None
            if filter_conditions:
                must_conditions = []
                for key, value in filter_conditions.items():
                    if isinstance(value, list):
                        must_conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
                    else:
                        must_conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
                
                if must_conditions:
                    search_filter = Filter(must=must_conditions)
            
            # Perform search
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                })
            
            return formatted_results
            
        except Exception as e:
            self.log(f"Failed to search vectors: {e}", True)
            return []
    
    def scroll_vectors(self, collection_name: str, limit: int = 100, filter_conditions: Dict[str, Any] = None, with_payload: bool = True, with_vectors: bool = False, page_offset: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Scroll through vectors in a collection.
        
        Args:
            collection_name: Collection name
            limit: Maximum number of vectors to return
            filter_conditions: Optional filter conditions
            with_payload: Include payload in results
            with_vectors: Include vectors in results
            
        Returns:
            Tuple of (points_list, next_page_offset)
        """
        try:
            if not self.connected:
                self.log("Cannot scroll vectors: not connected to Qdrant.", True)
                return [], None
            
            # Build filter if conditions provided
            scroll_filter = None
            if filter_conditions:
                must_conditions = []
                for key, value in filter_conditions.items():
                    if isinstance(value, list):
                        must_conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
                    else:
                        must_conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
                
                if must_conditions:
                    scroll_filter = Filter(must=must_conditions)
            
            # Perform scroll with optimized settings
            scroll_result = self.client.scroll(
                collection_name=collection_name,
                limit=min(limit, 10000),  # Cap limit for performance
                scroll_filter=scroll_filter,
                with_payload=with_payload,
                with_vectors=with_vectors,
                offset=page_offset
            )
            
            points, next_page_offset = scroll_result
            
            # Format points
            formatted_points = []
            for point in points:
                point_data = {
                    "id": point.id,
                    "payload": point.payload if with_payload else None,
                    "vector": point.vector if with_vectors else None
                }
                formatted_points.append(point_data)
            
            return formatted_points, next_page_offset
            
        except Exception as e:
            self.log(f"Failed to scroll vectors: {e}", True)
            return [], None
    
    def update_payload(self, collection_name: str, point_ids: List[int], payload: Dict[str, Any]) -> bool:
        """
        Update payload for specific points.
        
        Args:
            collection_name: Collection name
            point_ids: List of point IDs to update
            payload: New payload data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.connected:
                self.log("Cannot update payload: not connected to Qdrant.", True)
                return False
            
            self.client.set_payload(
                collection_name=collection_name,
                payload=payload,
                points=point_ids
            )
            
            self.log(f"Updated payload for {len(point_ids)} points in collection '{collection_name}'")
            return True
            
        except Exception as e:
            self.log(f"Failed to update payload: {e}", True)
            return False
    
    def delete_points(self, collection_name: str, point_ids: List[int]) -> bool:
        """
        Delete specific points from a collection.
        
        Args:
            collection_name: Collection name
            point_ids: List of point IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.connected:
                self.log("Cannot delete points: not connected to Qdrant.", True)
                return False
            
            self.client.delete(
                collection_name=collection_name,
                points_selector=point_ids
            )
            
            self.log(f"Deleted {len(point_ids)} points from collection '{collection_name}'")
            return True
            
        except Exception as e:
            self.log(f"Failed to delete points: {e}", True)
            return False
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a collection.
        
        Args:
            collection_name: Collection name
            
        Returns:
            Dictionary with collection statistics
        """
        try:
            if not self.connected:
                return {"error": "Not connected to Qdrant"}
            
            info = self.client.get_collection(collection_name)
            
            # Get sample of vectors for detailed analysis
            sample_points, _ = self.scroll_vectors(collection_name, limit=1000, with_payload=True, with_vectors=False)
            
            # Analyze payloads
            doc_types = {}
            cluster_counts = {}
            classification_counts = {}
            
            for point in sample_points:
                payload = point.get("payload", {})
                
                # Count by type
                doc_type = payload.get('type', 'unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                
                # Count clusters
                if 'cluster_id' in payload:
                    cluster_id = payload['cluster_id']
                    cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
                
                # Count classifications
                if 'predicted_label' in payload:
                    label = payload['predicted_label']
                    classification_counts[label] = classification_counts.get(label, 0) + 1
            
            return {
                "collection_name": collection_name,
                "dimension": info.config.params.vectors.size,
                "distance_metric": info.config.params.vectors.distance.value,
                "total_vectors": info.points_count,
                "status": info.status.value,
                "document_types": doc_types,
                "cluster_counts": cluster_counts,
                "classification_counts": classification_counts,
                "sample_size": len(sample_points)
            }
            
        except Exception as e:
            self.log(f"Failed to get collection stats: {e}", True)
            return {"error": str(e)}
    
    def close(self):
        """Close the Qdrant connection."""
        try:
            if self.client:
                # QdrantClient doesn't have an explicit close method
                self.client = None
                self.connected = False
                self.log("Qdrant service closed")
        except Exception as e:
            self.log(f"Error closing Qdrant service: {e}", True)
