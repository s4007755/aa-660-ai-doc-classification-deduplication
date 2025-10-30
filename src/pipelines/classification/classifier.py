"""
Document Classification Module

This module provides comprehensive document classification functionality including:
- Text classification using cosine similarity
- Label management and enrichment
- Collection-based label storage
- AI-powered label description generation
"""

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct
from src.utils.embedding import embed
from src.utils.hash_utils import HashUtils


class DocumentClassifier:
    """
    Document classification system using vector similarity and AI-powered label management.
    """
    
    def __init__(self, qdrant_service, log_function=None):
        """
        Initialize the document classifier.
        
        Args:
            qdrant_service: QdrantService instance for vector operations
            log_function: Optional logging function (default: print)
        """
        self.qdrant_service = qdrant_service
        self.log = log_function or print
        
    def _allocate_point_ids(self, collection_name: str, count: int) -> int:
        """Allocate point ID(s) using the available service method for compatibility.
        Returns the starting ID for the allocated range.
        """
        # Prefer legacy single-ID API when tests/mock expect it
        if count == 1 and hasattr(self.qdrant_service, "_get_next_id"):
            try:
                return int(self.qdrant_service._get_next_id(collection_name))
            except Exception:
                pass
        # Use modern block reservation if available
        if hasattr(self.qdrant_service, "_reserve_id_block"):
            try:
                return int(self.qdrant_service._reserve_id_block(collection_name, count))
            except Exception:
                pass
        # Fallback: compute next id from collection stats (best-effort)
        try:
            info = self.qdrant_service.client.get_collection(collection_name)
            # points_count includes metadata id 0; next id is points_count + 1
            return int(info.points_count) + 1
        except Exception:
            return 1

    def load_taxonomy(self, taxonomy_path):
        """
        Load taxonomy from a JSON file.
        
        Args:
            taxonomy_path: Path to JSON file containing taxonomy
            
        Returns:
            dict: Loaded taxonomy data or None if failed
        """
        try:
            with open(taxonomy_path, 'r', encoding='utf-8') as f:
                taxonomy_data = json.load(f)
            self.log(f"Loaded taxonomy from {taxonomy_path}")
            return taxonomy_data
        except Exception as e:
            self.log(f"Failed to load taxonomy from {taxonomy_path}: {e}", True)
            return None
        
    def classify_documents(self, collection_name, labels_file=None, use_collection_labels=False, enrich_labels=False):
        """
        Classify documents in a collection using provided labels or collection-stored labels.
        
        Args:
            collection_name: Name of the collection to classify
            labels_file: Path to JSON file containing labels (optional if use_collection_labels=True)
            use_collection_labels: Whether to use labels stored in the collection
            enrich_labels: Whether to enrich labels with AI-generated descriptions
            
        Returns:
            dict: Classification results including success status and statistics
        """
        try:
            # Load labels and generate embeddings
            if use_collection_labels:
                labels_data = self._load_labels_from_collection(collection_name)
                if not labels_data:
                    return {"success": False, "error": "No labels found in collection"}
            else:
                if not labels_file:
                    return {"success": False, "error": "Labels file required"}
                
                try:
                    with open(labels_file, 'r') as f:
                        labels_data = json.load(f)
                except Exception as e:
                    return {"success": False, "error": f"Failed to load labels file: {e}"}
            
            # Enrich labels if requested
            if enrich_labels:
                self.log("Enriching labels with AI-generated descriptions...")
                labels_data = self._enrich_labels_data(labels_data)
            
            self.log("Generating embeddings for labels...")
            
            # Extract label names and IDs
            label_texts = []
            label_names = []
            label_ids = []
            for label_id, label_data in labels_data.items():
                if isinstance(label_data, dict):
                    label_text = label_data.get('label', '')
                    if label_data.get('description'):
                        label_text += f": {label_data['description']}"
                else:
                    label_text = str(label_data)
                
                label_texts.append(label_text)
                label_names.append(label_data.get('label', str(label_data)) if isinstance(label_data, dict) else str(label_data))
                label_ids.append(label_id)
            
            # Generate embeddings for labels
            label_embeddings = embed(label_texts)
            
            if not label_embeddings:
                return {"success": False, "error": "Failed to generate label embeddings"}
            
            # Get collection info to check size
            collection_info = self.qdrant_service.get_collection_info(collection_name)
            total_points = collection_info['vector_count']
            
            # Process all vectors - use a large limit to ensure we get all points
            self.log(f"Retrieving all {total_points} vectors for classification...")
            points_list, _ = self.qdrant_service.scroll_vectors(
                collection_name, max(total_points * 2, 1000), with_payload=True, with_vectors=True
            )
            
            if not points_list:
                return {"success": False, "error": "No vectors found in collection"}
            
            # Filter out metadata point (ID 0)
            data_points = [p for p in points_list if not p.get('payload', {}).get('_metadata')]
            
            if not data_points:
                return {"success": False, "error": "No data vectors found in collection"}
            
            vectors = np.array([point['vector'] for point in data_points])
            point_ids = [point['id'] for point in data_points]
            
            # Calculate cosine similarity and classify
            self.log("Classifying documents...")
            similarities = cosine_similarity(vectors, label_embeddings)
            predicted_indices = np.argmax(similarities, axis=1)
            
            # Update vectors with classifications
            classified_count = 0
            for i, (point_id, pred_idx) in enumerate(zip(point_ids, predicted_indices)):
                predicted_label = label_names[pred_idx]
                predicted_label_id = label_ids[pred_idx]
                confidence = similarities[i][pred_idx]
                
                success = self.qdrant_service.update_payload(
                    collection_name=collection_name,
                    point_ids=[point_id],
                    payload={
                        "predicted_label": predicted_label,
                        "predicted_label_id": predicted_label_id,
                        "confidence": float(confidence),
                        "classification_method": "cosine_similarity"
                    }
                )
                
                if not success:
                    self.log(f"Failed to update payload for point {point_id}", True)
                    continue
                classified_count += 1
            
            self.log(f"Classification completed! Classified {classified_count} documents.")
            
            return {
                "success": True,
                "classified_count": classified_count,
                "total_documents": len(data_points),
                "labels_used": len(label_names)
            }
            
        except Exception as e:
            self.log(f"Classification failed: {e}", True)
            return {"success": False, "error": str(e)}
    
    def enrich_labels(self, labels_file, store_in_collection=False, collection_name=None):
        """
        Enrich labels with AI-generated descriptions.
        
        Args:
            labels_file: Path to JSON file containing labels
            store_in_collection: Whether to store enriched labels in collection
            collection_name: Collection name (required if store_in_collection=True)
            
        Returns:
            dict: Enrichment results
        """
        try:
            with open(labels_file, 'r') as f:
                labels_data = json.load(f)
            
            self.log("Enriching labels with AI-generated descriptions...")
            enriched_data = self._enrich_labels_data(labels_data)
            
            if store_in_collection and collection_name:
                self._store_labels_in_collection(collection_name, enriched_data)
                self.log("Enriched labels stored in collection.")
            
            return {"success": True, "enriched_labels": enriched_data}
            
        except Exception as e:
            self.log(f"Label enrichment failed: {e}", True)
            return {"success": False, "error": str(e)}
    
    def add_label_to_collection(self, collection_name, label_name, description=None, enrich=False):
        """
        Add a new label to the collection.
        
        Args:
            collection_name: Name of the collection
            label_name: Name of the label to add
            description: Optional description for the label
            enrich: If True and no description is provided, generate an AI description
            
        Returns:
            dict: Addition results
        """
        try:
            # Determine description and enrichment flag (opt-in enrichment)
            final_description = description
            enriched_flag = False
            if not description and enrich:
                final_description = self._generate_label_description(label_name)
                enriched_flag = True
            
            # Generate embedding for the label
            label_text = label_name
            if final_description:
                label_text += f": {final_description}"
            
            embeddings = embed([label_text])
            if not embeddings:
                return {"success": False, "error": "Failed to generate label embedding"}
            
            # Reserve one ID for this label
            label_id = f"custom_{len(label_name)}_{HashUtils.generate_deterministic_seed(label_name, 1000)}"
            point_id = self._allocate_point_ids(collection_name, 1)
            
            # Create label metadata
            metadata = {
                "label_id": label_id,
                "label_name": label_name,
                "description": final_description,
                "type": "label",
                "enriched": enriched_flag,
                "custom": True,
                "hash": HashUtils.create_label_hash(label_id, label_text)
            }
            
            # Insert label into collection
            point = PointStruct(
                id=point_id,
                vector=embeddings[0],
                payload=metadata
            )
            
            self.qdrant_service.client.upsert(collection_name=collection_name, points=[point])
            self.log(f"Added label '{label_name}' to collection '{collection_name}'")
            
            return {"success": True, "label_id": label_id, "point_id": point_id}
            
        except Exception as e:
            self.log(f"Failed to add label: {e}", True)
            return {"success": False, "error": str(e)}
    
    def get_collection_labels(self, collection_name):
        """
        Get all labels stored in a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            dict: Labels data
        """
        try:
            labels_data = self._load_labels_from_collection(collection_name)
            return {"success": True, "labels": labels_data}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _enrich_labels_data(self, labels_data):
        """Enrich labels with AI-generated descriptions."""
        try:
            from openai import OpenAI
            
            # Try to import API key with proper error handling
            try:
                from src.pipelines.classification.credentials import OPENAI_API_KEY
            except ImportError:
                import os
                OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            except Exception as e:
                self.log(f"Warning: Could not load credentials file: {e}", True)
                import os
                OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            
            if not OPENAI_API_KEY or OPENAI_API_KEY == "your-api-key-here" or OPENAI_API_KEY.strip() == "":
                self.log("No valid OpenAI API key found. Skipping enrichment.", True)
                return labels_data
            
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            enriched_data = {}
            for label_id, label_data in labels_data.items():
                if isinstance(label_data, dict):
                    label_name = label_data.get('label', '')
                    existing_desc = label_data.get('description', '')
                    
                    if not existing_desc:
                        description = self._generate_label_description(label_name, existing_desc)
                        enriched_data[label_id] = {
                            'label': label_name,
                            'description': description,
                            'enriched': True
                        }
                    else:
                        enriched_data[label_id] = label_data.copy()
                        enriched_data[label_id]['enriched'] = False
                else:
                    # Handle simple string labels
                    label_name = str(label_data)
                    description = self._generate_label_description(label_name)
                    enriched_data[label_id] = {
                        'label': label_name,
                        'description': description,
                        'enriched': True
                    }
            
            return enriched_data
            
        except Exception as e:
            self.log(f"Label enrichment failed: {e}", True)
            return labels_data  # Return original data if enrichment fails
    
    def _generate_label_description(self, label_name, existing_desc=""):
        """Generate AI description for a label."""
        try:
            from openai import OpenAI
            
            # Try to import API key with proper error handling
            try:
                from src.pipelines.classification.credentials import OPENAI_API_KEY
            except ImportError:
                import os
                OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            except Exception as e:
                self.log(f"Warning: Could not load credentials file: {e}", True)
                import os
                OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            
            if not OPENAI_API_KEY or OPENAI_API_KEY == "your-api-key-here" or OPENAI_API_KEY.strip() == "":
                return f"Content related to {label_name.lower()} topics and themes."
            
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            prompt = f"""
Generate a concise, professional description for the label "{label_name}".
The description should be 1-2 sentences that clearly explain what this category represents.
{f"Current description: {existing_desc}" if existing_desc else ""}

Return only the description, no additional text.
"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.log(f"AI description generation failed: {e}", True)
            # Fallback to basic description
            return f"Content related to {label_name.lower()} topics and themes."
    
    def _store_labels_in_collection(self, collection_name, labels_data):
        """Store labels as special points in the collection."""
        try:
            self.log("Storing labels in collection...")
            
            # Generate embeddings for labels
            label_texts = []
            label_metadata = []
            
            for label_id, label_data in labels_data.items():
                label_text = label_data['label']
                if label_data.get('description'):
                    label_text += f": {label_data['description']}"
                
                label_texts.append(label_text)
                label_metadata.append({
                    "label_id": label_id,
                    "label_name": label_data['label'],
                    "description": label_data.get('description', ''),
                    "type": "label",
                    "enriched": label_data.get('enriched', False),
                    "hash": HashUtils.create_label_hash(label_id, label_text)
                })
            
            # Generate embeddings
            label_embeddings = embed(label_texts)
            
            if not label_embeddings:
                self.log("Failed to generate label embeddings.", True)
                return
            
            # Reserve a contiguous ID block for all labels
            start_id = self._allocate_point_ids(collection_name, len(label_embeddings))
            
            # Create label points
            points = []
            for i, (embedding, metadata) in enumerate(zip(label_embeddings, label_metadata)):
                points.append(PointStruct(
                    id=start_id + i,
                    vector=embedding,
                    payload=metadata
                ))
            
            # Insert labels into collection
            self.qdrant_service.client.upsert(collection_name=collection_name, points=points)
            self.log(f"Stored {len(points)} labels in collection '{collection_name}'")
            
        except Exception as e:
            self.log(f"Failed to store labels: {e}", True)
    
    def _load_labels_from_collection(self, collection_name):
        """Load labels from collection with pagination to ensure completeness."""
        try:
            labels_data = {}
            next_offset = None
            while True:
                points_list, next_offset = self.qdrant_service.scroll_vectors(
                    collection_name,
                    1000,
                    with_payload=True,
                    with_vectors=False,
                    filter_conditions={"type": "label"},
                    page_offset=next_offset
                )
                if not points_list:
                    break
                for point in points_list:
                    payload = point.get('payload', {})
                    label_id = payload.get('label_id')
                    if label_id:
                        labels_data[label_id] = {
                            'label': payload.get('label_name', ''),
                            'description': payload.get('description', ''),
                            'enriched': payload.get('enriched', False)
                        }
                if not next_offset:
                    break
            return labels_data
        except Exception as e:
            self.log(f"Failed to load labels from collection: {e}", True)
            return {}


# Convenience functions for backward compatibility
def classify_documents(vecdb_client, collection_name, labels_file=None, use_collection_labels=False, enrich_labels=False, log_function=None):
    """Convenience function for document classification."""
    classifier = DocumentClassifier(vecdb_client, log_function)
    return classifier.classify_documents(collection_name, labels_file, use_collection_labels, enrich_labels)


def enrich_labels(vecdb_client, labels_file, store_in_collection=False, collection_name=None, log_function=None):
    """Convenience function for label enrichment."""
    classifier = DocumentClassifier(vecdb_client, log_function)
    return classifier.enrich_labels(labels_file, store_in_collection, collection_name)


def add_label_to_collection(vecdb_client, collection_name, label_name, description=None, enrich=False, log_function=None):
    """Convenience function for adding labels to collection."""
    classifier = DocumentClassifier(vecdb_client, log_function)
    return classifier.add_label_to_collection(collection_name, label_name, description, enrich=enrich)
