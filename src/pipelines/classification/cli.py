from src.services.qdrant_service import QdrantService
from src.services.openai_service import OpenAIService
from src.services.processing_service import ProcessingService
from src.services.sqlite_service import SQLiteService
from src.pipelines.classification.classifier import DocumentClassifier
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markup import escape
from datetime import datetime
import argparse
import json
import numpy as np
import shutil
import traceback
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

# Constants
CLUSTER_REPRESENTATIVE_LIMIT = 25  # Number of docs per cluster
TEXT_PREVIEW_LIMIT = 800  # Character limit for text previews (conservative for token limits)
REQUEST_TIMEOUT = 10  # Timeout for HTTP requests
MAX_QUERY_RESULTS = 50  # Default limit for query results
LARGE_RESULT_THRESHOLD = 50  # Threshold for showing "too many results" message

class NoExitArgParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)

class Cli:
    
    def __init__(self, host="localhost", port=6333):
        self.console = Console()
        
        # Initialize services
        self.qdrant_service = QdrantService(host, port, self.log)
        self.openai_service = OpenAIService(log_function=self.log)
        self.processing_service = ProcessingService(log_function=self.log)
        self.sqlite_service = SQLiteService(log_function=self.log)
        
        # Initialize classifier with services
        self.classifier = DocumentClassifier(self.qdrant_service, self.log)
        
        self.collection = None
        self.host = host
        self.port = port

    def _hr(self, pad: int = 0, char: str = "─") -> str:
        """Return a horizontal rule scaled to the current terminal width.
        pad reduces the number of characters to account for indentation.
        """
        try:
            width = self.console.width or shutil.get_terminal_size((100, 24)).columns
        except Exception:
            width = 100
        line_len = max(10, int(width) - max(0, int(pad)))
        return char * line_len

    def run(self):
        print("Welcome to Document Classifier CLI. Type 'help' for commands.\n")
        while True:
            try:
                command = input(self._prompt_text()).strip()
                if command:
                    self.handle_command(command)
            except EOFError:
                # Handle EOF gracefully - this happens when stdin is closed
                print("\nReceived EOF signal. Exiting...")
                break
            except KeyboardInterrupt:
                print("\nReceived interrupt signal. Exiting...")
                break
            except Exception as e:
                print(f"\nUnexpected error: {e}")
                print("Exiting...")
                break

    def log(self, msg, error=False):
        now = datetime.now().strftime("%H:%M:%S")
        color = "red" if error else "green"
        self.console.print(f"[[{color}]{now}[/{color}]] {msg}")

    def _prompt_text(self):
        if self.qdrant_service and self.qdrant_service.is_connected():
            prompt = f"\033[1;32m{self.host}\033[0m:\033[1;34m{self.port}\033[0m"
        else:
            prompt = "\033[1;31moffline\033[0m"
        if self.collection:
            prompt += f" (\033[1;36m{self.collection}\033[0m)"
        return prompt + " $ "
    
    
    def _retry(self, host, port):
        """
        Retry connection to Qdrant server.
        
        Args:
            host: Qdrant host
            port: Qdrant port
        """
        self.host = host
        self.port = port
        self.qdrant_service = QdrantService(host, port, self.log)
        self.classifier = DocumentClassifier(self.qdrant_service, self.log)

    def _cluster_command(self, num_clusters=None, debug=False):
        """Perform clustering on the current collection."""
        try:
            # Get collection info using QdrantService
            collection_info = self.qdrant_service.get_collection_info(self.collection)
            total_points = collection_info['vector_count']
            
            if total_points == 0:
                self.log("No vectors found in collection.", True)
                return
            
            self.log(f"Retrieving vectors from collection...")
            self.log(f"Processing all {total_points} vectors for clustering...")
            
            # Retrieve vectors using QdrantService - no limit, get all vectors
            points_list, _ = self.qdrant_service.scroll_vectors(
                self.collection, total_points, with_payload=True, with_vectors=True
            )
            
            if not points_list:
                self.log("No vectors found in collection.", True)
                return
            
            
            vectors = np.array([point['vector'] for point in points_list])
            point_ids = [int(point['id']) for point in points_list]
            
            self.log(f"Retrieved {len(vectors)} vectors for clustering...")
            
            # Filter out zero vectors (metadata vectors) which cause issues with cosine similarity
            non_zero_mask = np.any(vectors != 0, axis=1)
            zero_vector_count = len(vectors) - np.sum(non_zero_mask)
            
            if zero_vector_count > 0:
                self.log(f"Found {zero_vector_count} metadata vectors (zero vectors), skipping them for clustering...")
                vectors = vectors[non_zero_mask]
                point_ids = [point_ids[i] for i in range(len(point_ids)) if non_zero_mask[i]]
            
            if len(vectors) == 0:
                self.log("No valid document vectors found for clustering.", True)
                return
            
            self.log(f"Using {len(vectors)} document vectors for clustering...")
            
            # Perform clustering
            if num_clusters is not None:
                # K-means clustering
                algorithm = "kmeans"
                # Ensure requested clusters do not exceed number of distinct vectors
                try:
                    unique_vectors = np.unique(vectors, axis=0)
                    effective_clusters = max(1, min(num_clusters, len(unique_vectors)))
                    if effective_clusters != num_clusters:
                        self.log(f"Adjusted number of clusters from {num_clusters} to {effective_clusters} due to duplicate/insufficient distinct points")
                except Exception:
                    effective_clusters = num_clusters
                self.log(f"Performing K-means clustering with {effective_clusters} clusters...")
                kmeans = KMeans(n_clusters=effective_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(vectors)
            else:
                # Unsupervised clustering (Agglomerative with distance threshold)
                algorithm = "agglomerative"
                self.log("Performing unsupervised clustering using Agglomerative Clustering...")
                
                # Calculate optimal distance threshold based on data size
                # For embeddings, cosine distance works better than euclidean
                sample_size = min(1000, len(vectors))
                sample_vectors = vectors[:sample_size]
                distances = cosine_distances(sample_vectors)
                # Use 75th percentile of distances as threshold
                distance_threshold = np.percentile(distances[distances > 0], 75)
                
                self.log(f"Using distance threshold: {distance_threshold:.3f}")
                
                # Use precomputed distance matrix to avoid cosine metric issues with zero vectors
                # Calculate full distance matrix for all vectors
                self.log("Computing distance matrix...")
                full_distances = cosine_distances(vectors)
                
                agglo = AgglomerativeClustering(
                    n_clusters=None,  # Let distance threshold determine clusters
                    distance_threshold=distance_threshold,
                    linkage='average',  # Average linkage works well for text
                    metric='precomputed'  # Use precomputed distance matrix
                )
                cluster_labels = agglo.fit_predict(full_distances)
            
            # Update vectors with cluster assignments using QdrantService
            self.log("Updating vectors with cluster assignments...")
            
            # Group point ids by cluster to minimize set_payload calls
            cluster_to_point_ids = {}
            for pid, cid in zip(point_ids, cluster_labels):
                cluster_to_point_ids.setdefault(int(cid), []).append(int(pid))
            
            # Update payloads per-cluster in large batches
            batch_size = 1000  # Update 1000 points at a time
            for cid, ids in cluster_to_point_ids.items():
                payload_common = {"cluster_id": int(cid), "algorithm": algorithm}
                for i in range(0, len(ids), batch_size):
                    batch_ids = ids[i:i + batch_size]
                    try:
                        self.qdrant_service.client.set_payload(
                            collection_name=self.collection,
                            payload=payload_common,
                            points=batch_ids
                        )
                    except Exception as e:
                        self.log(f"Failed to update cluster payloads for cluster {cid} batch {i//batch_size + 1}: {e}", True)
                self.log(f"Updated cluster assignments for cluster {cid} ({len(ids)} points)")
            
            # Generate cluster names using LLM
            self.log("Generating cluster names...")
            cluster_names = self._generate_cluster_names(vectors, cluster_labels, points_list, debug=debug)
            
            # After naming, write cluster_name into payloads grouped by cluster
            try:
                self.log("Writing cluster names to payloads...")
                # Build mapping from cluster_id to list of point_ids
                cluster_to_point_ids = {}
                for pid, cid in zip(point_ids, cluster_labels):
                    cluster_to_point_ids.setdefault(int(cid), []).append(int(pid))
                
                # For each cluster, set the cluster_name for all its points in batches
                name_batch_size = 1000
                for cid, ids in cluster_to_point_ids.items():
                    if cid == -1:
                        # Skip noise
                        continue
                    name = cluster_names.get(cid, f"Cluster_{cid}")
                    for i in range(0, len(ids), name_batch_size):
                        batch_ids = ids[i:i + name_batch_size]
                        try:
                            self.qdrant_service.client.set_payload(
                                collection_name=self.collection,
                                payload={
                                    "cluster_name": str(name)
                                },
                                points=batch_ids
                            )
                        except Exception as e:
                            self.log(f"Failed to write cluster_name for cluster {cid} batch {i//name_batch_size + 1}: {e}", True)
                self.log("Cluster names written to payloads")
            except Exception as e:
                self.log(f"Failed writing cluster names to payloads: {e}", True)
            
            # Update collection metadata with clustering information
            num_clusters = len(set(cluster_labels))
            # Prepare clusters summary for fast API usage
            try:
                clusters_summary = {}
                for cid, ids in cluster_to_point_ids.items():
                    if cid == -1:
                        continue
                    name = cluster_names.get(cid, f"Cluster_{cid}")
                    clusters_summary[str(int(cid))] = {
                        "cluster_id": int(cid),
                        "cluster_name": str(name),
                        "count": int(len(ids))
                    }
                self.qdrant_service.update_collection_metadata(
                    self.collection,
                    {
                        "clustering_algorithm": algorithm,
                        "num_clusters": int(num_clusters),
                        "clustered_at": datetime.now().isoformat(),
                        "clusters_summary": clusters_summary
                    }
                )
            except Exception as meta_err:
                # Fallback to minimal metadata if summary fails
                self.qdrant_service.update_collection_metadata(
                    self.collection,
                    {
                        "clustering_algorithm": algorithm,
                        "num_clusters": int(num_clusters),
                        "clustered_at": datetime.now().isoformat()
                    }
                )
                self.log(f"Warning: failed to write clusters summary: {meta_err}", True)
            
            self.log(f"Clustering completed! Found {num_clusters} clusters.")
            
        except Exception as e:
            self.log(f"Clustering failed: {e}", True)

    def _classify_command(self, labels_file=None, use_collection_labels=False, enrich_labels=False):
        """Classify documents using the DocumentClassifier."""
        result = self.classifier.classify_documents(
            self.collection, 
            labels_file, 
            use_collection_labels, 
            enrich_labels
        )
        
        if not result["success"]:
            self.log(result["error"], True)

    def _enrich_labels_command(self, labels_file, store_in_collection=False):
        """Enrich labels using the DocumentClassifier."""
        result = self.classifier.enrich_labels(
            labels_file, 
            store_in_collection, 
            self.collection if store_in_collection else None
        )
        
        if not result["success"]:
            self.log(result["error"], True)

    def _add_label_command(self, label_name, description=None, enrich=False):
        """Add a new label to the collection."""
        result = self.classifier.add_label_to_collection(self.collection, label_name, description, enrich=enrich)
        
        if not result["success"]:
            self.log(result["error"], True)

    def _scan_inferred_labels(self, total_points):
        """Helper: Scan documents for inferred labels (predicted_label)."""
        inferred = {}
        page_offset = None
        while True:
            points_list, page_offset = self.qdrant_service.scroll_vectors(
                self.collection, 10000, with_payload=True, with_vectors=False, page_offset=page_offset
            )
            if not points_list:
                break
            for p in points_list:
                payload = p.get('payload', {}) or {}
                if payload.get('_metadata'):
                    continue
                label = payload.get('predicted_label')
                if label:
                    inferred[label] = inferred.get(label, 0) + 1
            if not page_offset:
                break
        return inferred
    
    def _scan_clusters(self):
        """Helper: Scan documents for cluster assignments."""
        cluster_counts = {}
        page_offset = None
        while True:
            points_list, page_offset = self.qdrant_service.scroll_vectors(
                self.collection, 10000, with_payload=True, with_vectors=False, page_offset=page_offset
            )
            if not points_list:
                break
            for point in points_list:
                payload = point.get('payload', {}) or {}
                if payload.get('_metadata'):
                    continue
                cluster_id = payload.get('cluster_id')
                if cluster_id is not None:
                    cluster_name = payload.get('cluster_name', f'Cluster_{cluster_id}')
                    key = (cluster_id, cluster_name)
                    cluster_counts[key] = cluster_counts.get(key, 0) + 1
            if not page_offset:
                break
        return cluster_counts

    def _list_labels_command(self):
        """List both stored and inferred labels in the collection."""
        try:
            # Get collection info
            collection_info = self.qdrant_service.get_collection_info(self.collection)
            total_points = collection_info['vector_count']
            
            if total_points == 0:
                self.console.print(f"\n[yellow]No vectors in collection '{self.collection}'[/yellow]\n")
                return
            
            # Get stored labels
            result = self.classifier.get_collection_labels(self.collection)
            if not result["success"]:
                self.log(result["error"], True)
                return
            
            stored_labels = result["labels"]
            
            # Try labels summary from collection metadata first
            inferred = {}
            try:
                coll_metadata = self.qdrant_service.get_collection_metadata(self.collection)
                labels_summary = coll_metadata.get('labels_summary') if coll_metadata else None
                total_docs_meta = coll_metadata.get('total_documents') if coll_metadata else None
                if labels_summary and isinstance(labels_summary, dict) and len(labels_summary) > 0:
                    # Use cached summary
                    self.console.print(f"\n[bold cyan]Inferred Labels (cached summary):[/bold cyan] [yellow]{self.collection}[/yellow]")
                    self.console.print(f"[dim]{self._hr()}[/dim]")
                    label_table = Table(show_header=True, box=None, padding=(0, 2))
                    label_table.add_column("Label Name", style="blue")
                    label_table.add_column("Documents", style="green", justify="right")
                    label_table.add_column("Percentage", style="cyan", justify="right")
                    total_for_pct = total_docs_meta or total_points or 1
                    for name, count in sorted(labels_summary.items(), key=lambda x: (-x[1], x[0])):
                        percentage = (count / total_for_pct) * 100
                        label_table.add_row(name, f"{int(count):,}", f"{percentage:.1f}%")
                    self.console.print(label_table)
                    self.console.print(f"\n[dim]Total: {len(labels_summary)} inferred labels[/dim]")
                    if not stored_labels:
                        self.console.print("[dim]Store labels explicitly via: enrich-labels --store-in-collection or add-label[/dim]")
                    self.console.print()
                    return
            except Exception:
                # Ignore summary errors and fall back to scan
                pass

            # Fallback: scan to infer labels
            self.console.print(f"\n[dim]Scanning {total_points} documents for labels...[/dim]")
            inferred = self._scan_inferred_labels(total_points)
            
            # Show stored labels if they exist
            if stored_labels:
                self.console.print(f"\n[bold cyan]Stored Labels in Collection:[/bold cyan] [yellow]{self.collection}[/yellow]")
                self.console.print(f"[dim]{self._hr()}[/dim]")
                
                label_table = Table(show_header=True, box=None, padding=(0, 2))
                label_table.add_column("Label ID", style="yellow")
                label_table.add_column("Label Name", style="blue")
                label_table.add_column("Description", style="dim")
                label_table.add_column("Enriched", style="cyan", justify="center")
                
                for label_id, label_data in sorted(stored_labels.items(), key=lambda x: x[1].get('label', '')):
                    description = label_data.get('description') or ''
                    truncated_desc = description[:50] + ('...' if len(description) > 50 else '') if description else ''
                    label_table.add_row(
                        label_id,
                        label_data.get('label', ''),
                        truncated_desc,
                        "✓" if label_data.get('enriched', False) else "-"
                    )
                
                self.console.print(label_table)
                self.console.print(f"\n[dim]Total: {len(stored_labels)} stored labels[/dim]")
            
            # Show inferred labels if they exist
            if inferred:
                if stored_labels:
                    self.console.print()  # Extra spacing between sections
                
                self.console.print(f"\n[bold cyan]Inferred Labels (from classified documents):[/bold cyan] [yellow]{self.collection}[/yellow]")
                self.console.print(f"[dim]{self._hr()}[/dim]")
                
                label_table = Table(show_header=True, box=None, padding=(0, 2))
                label_table.add_column("Label Name", style="blue")
                label_table.add_column("Documents", style="green", justify="right")
                label_table.add_column("Percentage", style="cyan", justify="right")
                
                for name, count in sorted(inferred.items(), key=lambda x: (-x[1], x[0])):
                    percentage = (count / total_points) * 100
                    label_table.add_row(
                        name,
                        f"{count:,}",
                        f"{percentage:.1f}%"
                    )
                
                self.console.print(label_table)
                self.console.print(f"\n[dim]Total: {len(inferred)} inferred labels[/dim]")
                
                if not stored_labels:
                    self.console.print("[dim]Store labels explicitly via: enrich-labels --store-in-collection or add-label[/dim]")
            
            # If neither stored nor inferred labels exist
            if not stored_labels and not inferred:
                self.console.print(f"\n[yellow]No labels found in collection '{self.collection}'[/yellow]")
                self.console.print("[dim]Run 'classify' command to assign labels to documents[/dim]")
            
            self.console.print()
                
        except Exception as e:
            self.log(f"List labels failed: {e}", True)

    def _list_clusters_command(self):
        """List all clusters in the collection."""
        try:
            # Get collection info
            collection_info = self.qdrant_service.get_collection_info(self.collection)
            total_points = collection_info['vector_count']
            
            if total_points == 0:
                self.console.print(f"\n[yellow]No vectors in collection '{self.collection}'[/yellow]")
                return
            
            # Prefer clusters summary from collection metadata
            try:
                coll_metadata = self.qdrant_service.get_collection_metadata(self.collection)
                clusters_summary = coll_metadata.get('clusters_summary') if coll_metadata else None
                if clusters_summary and isinstance(clusters_summary, dict) and len(clusters_summary) > 0:
                    self.console.print(f"\n[bold cyan]Clusters in Collection (cached summary):[/bold cyan] [yellow]{self.collection}[/yellow]")
                    self.console.print(f"[dim]{self._hr()}[/dim]")
                    cluster_table = Table(show_header=True, box=None, padding=(0, 2))
                    cluster_table.add_column("Cluster ID", style="yellow", justify="center")
                    cluster_table.add_column("Cluster Name", style="magenta")
                    cluster_table.add_column("Documents", style="green", justify="right")
                    cluster_table.add_column("Percentage", style="cyan", justify="right")
                    # Compute total from summary
                    total_from_summary = sum(int(v.get('count', 0)) for v in clusters_summary.values()) or max(1, total_points)
                    for key, v in sorted(clusters_summary.items(), key=lambda x: int(x[0])):
                        cid = int(v.get('cluster_id', int(key)))
                        cname = str(v.get('cluster_name', f'Cluster_{cid}'))
                        cnt = int(v.get('count', 0))
                        pct = (cnt / total_from_summary) * 100
                        cluster_table.add_row(str(cid), cname, f"{cnt:,}", f"{pct:.1f}%")
                    self.console.print(cluster_table)
                    self.console.print(f"\n[dim]Total: {len(clusters_summary)} clusters[/dim]")
                    self.console.print(f"[dim]Query a cluster with: query cluster:0 or query cluster:ClusterName[/dim]\n")
                    return
            except Exception:
                # Ignore and fall back
                pass

            # Fallback: scan all points to collect cluster information
            self.console.print(f"\n[dim]Scanning {total_points} documents for clusters...[/dim]")
            cluster_counts = self._scan_clusters()
            
            if not cluster_counts:
                self.console.print(f"\n[yellow]No clusters found in collection '{self.collection}'[/yellow]")
                self.console.print("[dim]Run 'cluster' command to create clusters[/dim]\n")
                return
            
            # Display clusters
            self.console.print(f"\n[bold cyan]Clusters in Collection:[/bold cyan] [yellow]{self.collection}[/yellow]")
            self.console.print(f"[dim]{self._hr()}[/dim]")
            
            cluster_table = Table(show_header=True, box=None, padding=(0, 2))
            cluster_table.add_column("Cluster ID", style="yellow", justify="center")
            cluster_table.add_column("Cluster Name", style="magenta")
            cluster_table.add_column("Documents", style="green", justify="right")
            cluster_table.add_column("Percentage", style="cyan", justify="right")
            
            # Sort by cluster ID
            for (cluster_id, cluster_name), count in sorted(cluster_counts.items()):
                percentage = (count / total_points) * 100
                cluster_table.add_row(
                    str(cluster_id),
                    cluster_name,
                    f"{count:,}",
                    f"{percentage:.1f}%"
                )
            
            self.console.print(cluster_table)
            self.console.print(f"\n[dim]Total: {len(cluster_counts)} clusters[/dim]")
            self.console.print(f"[dim]Query a cluster with: query cluster:0 or query cluster:ClusterName[/dim]\n")
            
        except Exception as e:
            self.log(f"List clusters failed: {e}", True)

    def _query_command(self, query, limit=MAX_QUERY_RESULTS):
        """Query documents by cluster, URL, directory, or docID."""
        try:
            # Parse query for special filters
            filter_conditions = None
            
            # Check for cluster queries
            if query.startswith("cluster:"):
                # Format: cluster:0 or cluster:Economy
                cluster_value = query.split(":", 1)[1]
                if cluster_value.isdigit():
                    # Query by cluster ID
                    filter_conditions = {"cluster_id": int(cluster_value)}
                    print(f"\n=== Documents in Cluster ID {cluster_value} ===")
                else:
                    # Query by cluster name
                    filter_conditions = {"cluster_name": cluster_value}
                    print(f"\n=== Documents in Cluster '{cluster_value}' ===")
                
                points_list, _ = self.qdrant_service.scroll_vectors(
                    self.collection, limit, with_payload=True, with_vectors=False,
                    filter_conditions=filter_conditions
                )
            
            elif query.startswith("label:"):
                # Query by classification label
                label_value = query.split(":", 1)[1]
                filter_conditions = {"predicted_label": label_value}
                print(f"\n=== Documents with Label '{label_value}' ===")
                
                points_list, _ = self.qdrant_service.scroll_vectors(
                    self.collection, limit, with_payload=True, with_vectors=False,
                    filter_conditions=filter_conditions
                )
            
            elif query.startswith("http"):
                # URL query
                points_list, _ = self.qdrant_service.scroll_vectors(
                    self.collection, limit, with_payload=True, with_vectors=False,
                    filter_conditions={"source": query}
                )
            
            elif query.isdigit():
                # Document ID query
                points_list, _ = self.qdrant_service.scroll_vectors(
                    self.collection, 1, with_payload=True, with_vectors=False,
                    filter_conditions={"id": int(query)}
                )
            
            else:
                # Free-text/source path queries are not supported
                points_list = []
        
            if not points_list:
                self.console.print(f"\n[yellow]No documents found for query:[/yellow] [white]{query}[/white]")
                self.console.print("\n[cyan]Query Tips:[/cyan]")
                self.console.print("  • Use [green]cluster:0[/green] or [green]cluster:Economy[/green] to query by cluster")
                self.console.print("  • Use [green]label:Sports[/green] to query by classification label")
                return
            
            self.console.print(f"\n[green]Found {len(points_list)} document(s)[/green]")
            self.console.print(f"[dim]{self._hr()}[/dim]")
            
            for i, point in enumerate(points_list):
                payload = point.get('payload', {})
                
                # Print document header
                self.console.print(f"\n[bold cyan][{i+1}][/bold cyan] [bold]Document ID:[/bold] [yellow]{point['id']}[/yellow]")
                
                # Source
                source = payload.get('source', 'N/A')
                self.console.print(f"    [dim]Source:[/dim] {source}")
                
                # Cluster info
                if 'cluster_id' in payload:
                    cluster_id = payload['cluster_id']
                    cluster_name = payload.get('cluster_name', f'Cluster_{cluster_id}')
                    self.console.print(f"    [dim]Cluster:[/dim] [magenta]{cluster_name}[/magenta] [dim](ID: {cluster_id})[/dim]")
                
                # Classification info
                if 'predicted_label' in payload:
                    label = payload['predicted_label']
                    confidence = payload.get('confidence', 0)
                    self.console.print(f"    [dim]Label:[/dim] [blue]{label}[/blue] [dim](confidence: {confidence:.1%})[/dim]")
                
                # Text preview
                text_preview = payload.get('text_content', '')[:150]
                if len(payload.get('text_content', '')) > 150:
                    text_preview += "..."
                
                if text_preview:
                    self.console.print(f"    [dim]Preview:[/dim] [white]{text_preview}[/white]")
                
                self.console.print(f"    [dim]{self._hr(pad=4)}[/dim]")
                
        except Exception as e:
            self.log(f"Query failed: {e}", True)

    def _stats_command(self):
        """Show collection statistics."""
        try:
            # Get collection info using QdrantService
            collection_info = self.qdrant_service.get_collection_info(self.collection)
            
            # Collection overview
            self.console.print(f"\n[bold cyan]Collection Statistics:[/bold cyan] [yellow]{self.collection}[/yellow]")
            self.console.print(f"[dim]{self._hr()}[/dim]")
            
            total_vectors = collection_info['vector_count']
            
            # Get clustering metadata from collection metadata point
            coll_metadata = self.qdrant_service.get_collection_metadata(self.collection)
            clustering_algorithm = coll_metadata.get('clustering_algorithm') if coll_metadata else None
            
            # Get ALL vectors for detailed stats and detect classifications
            doc_types = {}
            has_classifications = False
            if total_vectors > 0:
                points_list, _ = self.qdrant_service.scroll_vectors(
                    self.collection, total_vectors, with_payload=True, with_vectors=False
                )
                
                # Document types and check for classifications
                for point in points_list:
                    payload = point.get('payload', {})
                    # Check if it's metadata (has _metadata field)
                    if payload.get('_metadata'):
                        doc_type = 'metadata'
                    else:
                        doc_type = payload.get('type', 'unknown')
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                    
                    # Check for classifications
                    if not has_classifications and 'predicted_label' in payload:
                        has_classifications = True
            
            # Basic info table
            info_table = Table(show_header=False, box=None, padding=(0, 2))
            info_table.add_column("Property", style="dim")
            info_table.add_column("Value", style="white")
            
            info_table.add_row("Total Vectors", f"{total_vectors:,}")
            info_table.add_row("Vector Dimension", str(collection_info['dimension']))
            info_table.add_row("Distance Metric", collection_info['distance_metric'])
            
            # Show creation date if available
            if collection_info.get('created_at'):
                try:
                    dt = datetime.fromisoformat(collection_info['created_at'])
                    created_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                    info_table.add_row("Created", created_str)
                except:
                    pass
            
            # Show full description if available
            if collection_info.get('description'):
                info_table.add_row("Description", collection_info['description'])
            
            if clustering_algorithm:
                info_table.add_row("Clustering Method", clustering_algorithm.capitalize())
            if has_classifications:
                info_table.add_row("Classification", "Enabled")
            
            self.console.print(info_table)
            
            if doc_types:
                self.console.print(f"\n[bold cyan]Document Types[/bold cyan]")
                doc_table = Table(show_header=False, box=None, padding=(0, 2))
                doc_table.add_column("Type", style="white")
                doc_table.add_column("Count", style="green", justify="right")
                
                for doc_type, count in sorted(doc_types.items()):
                    doc_table.add_row(doc_type, f"{count:,}")
                
                self.console.print(doc_table)
            
            # Clusters - use helper method
            if total_vectors > 0:
                cluster_counts = self._scan_clusters()
                
                if cluster_counts:
                    self.console.print(f"\n[bold cyan]Clusters[/bold cyan] [dim]({len(cluster_counts)} total)[/dim]")
                    
                    cluster_table = Table(show_header=True, box=None, padding=(0, 2))
                    cluster_table.add_column("Cluster Name", style="magenta")
                    cluster_table.add_column("ID", style="yellow", justify="center")
                    cluster_table.add_column("Documents", style="green", justify="right")
                    
                    for (cluster_id, cluster_name), count in sorted(cluster_counts.items()):
                        cluster_table.add_row(cluster_name, str(cluster_id), f"{count:,}")
                    
                    self.console.print(cluster_table)
            
            # Classifications - use helper method
            if has_classifications:
                classification_counts = self._scan_inferred_labels(total_vectors)
                
                if classification_counts:
                    self.console.print(f"\n[bold cyan]Classifications[/bold cyan] [dim]({len(classification_counts)} total)[/dim]")
                    class_table = Table(show_header=True, box=None, padding=(0, 2))
                    class_table.add_column("Label", style="blue")
                    class_table.add_column("Documents", style="green", justify="right")
                    
                    for label, count in sorted(classification_counts.items(), key=lambda x: x[1], reverse=True):
                        class_table.add_row(label, f"{count:,}")
                    
                    self.console.print(class_table)
            
            self.console.print()
            
        except Exception as e:
            self.log(f"Stats command failed: {e}", True)


    def _generate_cluster_names(self, vectors, cluster_labels, points, debug=False):
        """Generate cluster names using OpenAIService."""
        try:
            
            # Validate inputs
            if cluster_labels is None or len(cluster_labels) == 0:
                self.log("No cluster labels provided for naming", True)
                return {}
            
            if vectors is None or len(vectors) == 0:
                self.log("No vectors provided for naming", True)
                return {}
            
            if points is None or len(points) == 0:
                self.log("No points provided for naming", True)
                return {}
            
            cluster_names = {}
            unique_clusters = set(cluster_labels)
            
            # For each cluster, get representative documents
            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise points in DBSCAN
                    cluster_names[cluster_id] = "Noise"
                    continue
                
                cluster_mask = (cluster_labels == cluster_id)
                cluster_vectors = vectors[cluster_mask]
                cluster_points = [points[i] for i in np.where(cluster_mask)[0]]
                
                # Get representative documents (closest to centroid)
                if len(cluster_vectors) > 0:
                    centroid = np.mean(cluster_vectors, axis=0)
                    distances = euclidean_distances(cluster_vectors, centroid.reshape(1, -1)).flatten()
                    top_indices = np.argsort(distances)[:CLUSTER_REPRESENTATIVE_LIMIT]  # Top closest to centroid
                    
                    representative_texts = []
                    for idx in top_indices:
                        payload = cluster_points[idx].get('payload', {})
                        
                        # Use actual text content if available
                        if 'text_content' in payload:
                            text_content = payload['text_content']
                            if text_content and len(text_content.strip()) > 10:  # Only use meaningful text
                                representative_texts.append(text_content[:TEXT_PREVIEW_LIMIT])
                                continue
                        
                        # Fallback to metadata if no text content
                        text_content = ""
                        if 'source' in payload:
                            source = payload['source']
                            if '-csv-' in source or source.endswith('.csv'):
                                text_content = f"Document from {source.split('/')[-1] if '/' in source else source}"
                            else:
                                text_content = f"Document from {source}"
                        
                        if 'type' in payload:
                            text_content += f" (Type: {payload['type']})"
                        if 'Label' in payload:
                            text_content += f" (Label: {payload['Label']})"
                        
                        if text_content:
                            representative_texts.append(text_content[:TEXT_PREVIEW_LIMIT])
                    
                    if representative_texts:
                        # Generate strict single-word label using OpenAIService
                        cluster_names[cluster_id] = self.openai_service.generate_single_word_cluster_label(cluster_id, representative_texts, debug=debug)
                    else:
                        cluster_names[cluster_id] = f"Cluster_{cluster_id}"
                else:
                    cluster_names[cluster_id] = f"Cluster_{cluster_id}"
            
            return cluster_names
            
        except Exception as e:
            self.log(f"Failed to generate cluster names: {e}", True)
            # Return default names
            return {i: f"Cluster_{i}" for i in set(cluster_labels)}
    

    def _use(self, collection):
        collections = self.qdrant_service.list_collections()
        if collection in collections:
            self.collection = collection
            print(f"Using collection: {self.collection}")
        else:
            print("Collection not found. Please first create collection or use ls to view available collections.")

    def handle_command(self, command):
        if not command:
            return

        # Parse command with proper quote handling
        import shlex
        try:
            parts = shlex.split(command)
        except ValueError:
            # Fallback to simple split if shlex fails
            parts = command.split()
        
        cmd = parts[0]
        args = parts[1:]

        if cmd == "use":
            if not args:
                self.log("Usage: use <collection>", True)
                return
            self._use(args[0])

        elif cmd == "show":
            self.console.print()
            status_table = Table(show_header=False, box=None, padding=(0, 2))
            status_table.add_column("Property", style="dim")
            status_table.add_column("Value")
            
            if self.qdrant_service.is_connected():
                status_table.add_row("Connection", f"[green]Connected[/green] to {self.host}:{self.port}")
                if self.collection:
                    status_table.add_row("Active Collection", f"[yellow]{self.collection}[/yellow]")
                else:
                    status_table.add_row("Active Collection", "[dim]None selected[/dim]")
            else:
                status_table.add_row("Connection", "[red]Not connected[/red] to Qdrant")
            
            self.console.print(status_table)
            self.console.print()

        elif cmd == "ls":
            collections = self.qdrant_service.list_collections()
            
            if collections:
                self.console.print(f"\n[bold cyan]Available Collections[/bold cyan] [dim]({len(collections)} total)[/dim]")
                self.console.print(f"[dim]{self._hr()}[/dim]")
                
                coll_table = Table(show_header=True, box=None, padding=(0, 2))
                coll_table.add_column("Collection Name", style="yellow")
                coll_table.add_column("Vectors", style="green", justify="right")
                coll_table.add_column("Dimension", style="cyan", justify="center")
                coll_table.add_column("Distance", style="white")
                coll_table.add_column("Created", style="dim")
                coll_table.add_column("Description", style="dim", no_wrap=False)
                
                for col_name, col_info in sorted(collections.items()):
                    # Handle different key names from list_collections
                    vector_count = col_info.get('vectors', col_info.get('vector_count', 0))
                    dimension = col_info.get('size', col_info.get('dimension', 'N/A'))
                    distance = col_info.get('distance', 'N/A')
                    created_at = col_info.get('created_at')
                    description = col_info.get('description')
                    
                    # Format creation date
                    if created_at:
                        try:
                            dt = datetime.fromisoformat(created_at)
                            created_str = dt.strftime("%Y-%m-%d %H:%M")
                        except:
                            created_str = created_at[:16] if len(created_at) > 16 else created_at
                    else:
                        created_str = "N/A"
                    
                    # Truncate description to 40 characters
                    if description:
                        desc_str = description[:40] + "..." if len(description) > 40 else description
                    else:
                        desc_str = ""
                    
                    coll_table.add_row(
                        col_name,
                        f"{vector_count:,}" if isinstance(vector_count, int) else str(vector_count),
                        str(dimension),
                        distance,
                        created_str,
                        desc_str
                    )
                
                self.console.print(coll_table)
                self.console.print()
            else:
                self.console.print("\n[yellow]No collections found[/yellow]")
                self.console.print("[dim]Create a collection with:[/dim] [green]create <name>[/green]\n")

        elif cmd == "ls-models":
            # Show available embedding models and dimensions
            model_dims = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536
            }
            t = Table(show_header=True, box=None, padding=(0, 2))
            t.add_column("Model", style="yellow")
            t.add_column("Dimension", style="cyan", justify="right")
            for m, d in model_dims.items():
                t.add_row(m, str(d))
            self.console.print("\n[bold cyan]Embedding Models[/bold cyan]")
            self.console.print(f"[dim]{self._hr()}[/dim]")
            self.console.print(t)
            self.console.print()

        elif cmd == "create":
            if not args:
                self.log("Usage: create <collection_name> [model] [--description \"...\"]", True)
                return
            
            # Parse arguments for create command
            parser = NoExitArgParser(prog="create", add_help=False)
            parser.add_argument("name", help="Collection name")
            parser.add_argument("model", nargs="?", default="text-embedding-3-small", help="Embedding model")
            parser.add_argument("--description", "-d", type=str, help="Collection description")
            
            try:
                parsed = parser.parse_args(args)
            except ValueError as e:
                self.log(f"Invalid arguments: {e}", True)
                self.log("Usage: create <collection_name> [model] [--description \"...\"]", True)
                return
            
            name = parsed.name
            model = parsed.model
            description = parsed.description
            
            # Validate description if provided
            if description and not description.strip():
                self.log("Description cannot be empty", True)
                return
            
            # Check for invalid flag usage (like -dsadasdsa)
            if description and description.startswith('-') and len(description) > 2:
                self.log(f"Invalid flag: '{description}'. Use --description \"...\" for descriptions", True)
                return
            
            # Get vector dimension based on model
            model_dims = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536
            }
            dim = model_dims.get(model, 1536)
            
            success = self.qdrant_service.create_collection(name, dim, model, description)
            if success:
                if description:
                    print(f"Collection '{name}' created successfully with model '{model}'")
                    print(f"Description: {description}")
                else:
                    print(f"Collection '{name}' created successfully with model '{model}'")
            else:
                self.log(f"Failed to create collection '{name}'", True)
            return
        
        elif cmd == "rm":
            # Remove a collection from Qdrant and clear its SQLite hash table
            parser = NoExitArgParser(prog="rm", add_help=False)
            parser.add_argument("name", nargs="?", help="Collection name (defaults to current)")
            parser.add_argument("--yes", "-y", action="store_true", help="Confirm deletion without prompt")
            try:
                opts = parser.parse_args(args)
                target = opts.name or self.collection
                if not target:
                    self.log("No collection specified or selected.", True)
                    return
                if not opts.yes:
                    confirm = input(f"This will delete collection '{target}' and its dedup index. Type 'yes' to confirm: ").strip().lower()
                    if confirm != "yes":
                        self.log("Deletion cancelled.")
                        return
                # Delete Qdrant collection (best-effort)
                q_ok = self.qdrant_service.delete_collection(target)
                # Clear SQLite table regardless of Qdrant outcome
                s_ok = self.sqlite_service.clear_collection(target)
                if q_ok:
                    self.log(f"Deleted collection '{target}' from Qdrant")
                else:
                    self.log(f"Qdrant deletion reported failure for '{target}'", True)
                # SQLiteService logs its own result; avoid duplicate CLI message
                if self.collection == target:
                    self.collection = None
            except ValueError as e:
                self.log(f"Invalid rm arguments: {e}. Usage: rm [name] [--yes]", True)


        elif cmd == "source":
            if self.collection is None:
                self.log("No collection selected.", True)
                return
            
            parser = NoExitArgParser(prog="source", add_help=False)
            parser.add_argument("path", help="Path to directory, CSV file, or URL")
            parser.add_argument("--limit", type=int, help="Limit number of documents to process")
            parser.add_argument("--text-column", help="Column name for text content in CSV")
            parser.add_argument("--url-column", help="Column name for URLs in CSV")
            
            try:
                opts = parser.parse_args(args)
                self._source_command(opts.path, opts.limit, opts.text_column, opts.url_column)
            except ValueError as e:
                self.log(f"Invalid source arguments: {e}. Usage: source <path> [--limit N] [--text-column COL] [--url-column COL]", True)

        elif cmd == "cluster":
            if self.collection is None:
                self.log("No collection selected.", True)
                return
            
            parser = NoExitArgParser(prog="cluster", add_help=False)
            parser.add_argument("--num-clusters", type=int, help="Number of clusters for K-means (if not specified, uses unsupervised clustering)")
            parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed logging and prompt saving")

            try:
                opts = parser.parse_args(args)
                self._cluster_command(opts.num_clusters, debug=opts.debug)
            except ValueError as e:
                self.log(f"Invalid cluster arguments: {e}. Usage: cluster [--num-clusters N] [--debug]", True)

        elif cmd == "classify":
            if self.collection is None:
                self.log("No collection selected.", True)
                return
            
            parser = NoExitArgParser(prog="classify", add_help=False)
            parser.add_argument("labels_file", nargs="?", help="Path to labels.json file (optional if labels stored in collection)")
            parser.add_argument("--use-collection-labels", action="store_true", help="Use labels stored in collection")
            parser.add_argument("--enrich", action="store_true", help="Enrich labels with AI-generated descriptions")
            
            try:
                opts = parser.parse_args(args)
                if opts.use_collection_labels:
                    self._classify_command(None, use_collection_labels=True, enrich_labels=opts.enrich)
                else:
                    if not opts.labels_file:
                        self.log("Must provide labels file or use --use-collection-labels", True)
                        return
                    self._classify_command(opts.labels_file, use_collection_labels=False, enrich_labels=opts.enrich)
            except ValueError as e:
                self.log(f"Invalid classify arguments: {e}. Usage: classify <labels.json> [--use-collection-labels] [--enrich]", True)

        elif cmd == "query":
            if self.collection is None:
                self.log("No collection selected.", True)
                return
            
            if not args:
                self.log("Usage: query <url|directory|docID|cluster:ID|cluster:Name|label:Name>", True)
                return
            
            # Join all args to support paths with spaces
            query_string = " ".join(args)
            self._query_command(query_string)

        elif cmd == "stats":
            if self.collection is None:
                self.log("No collection selected.", True)
                return
            
            self._stats_command()

        elif cmd == "retry":
            parser = NoExitArgParser(prog="retry", add_help=False)
            parser.add_argument("--host", default="localhost", help="Qdrant host")
            parser.add_argument("--port", type=int, default=6333, help="Qdrant port")
            
            try:
                opts = parser.parse_args(args)
                self._retry(opts.host, opts.port)
                print(f"Retrying connection to {opts.host}:{opts.port}")
            except ValueError as e:
                self.log(f"Invalid retry arguments: {e}. Usage: retry [--host HOST] [--port PORT]", True)

        elif cmd == "add-label":
            if self.collection is None:
                self.log("No collection selected.", True)
                return
            
            parser = NoExitArgParser(prog="add-label", add_help=False)
            parser.add_argument("label_name", help="Name of the label to add")
            parser.add_argument("--description", help="Description for the label")
            parser.add_argument("--enrich", action="store_true", help="Generate AI description when none provided")
            
            try:
                opts = parser.parse_args(args)
                self._add_label_command(
                    opts.label_name,
                    opts.description if opts.description else None,
                    enrich=opts.enrich
                )
            except ValueError as e:
                self.log(f"Invalid add-label arguments: {e}. Usage: add-label <label> [--description]", True)

        elif cmd == "rm-label":
            if self.collection is None:
                self.log("No collection selected.", True)
                return
            
            parser = NoExitArgParser(prog="rm-label", add_help=False)
            parser.add_argument("target", help="Label ID or label name to remove")
            parser.add_argument("--by", choices=["id", "name"], help="Match by 'id' (label_id) or 'name' (label_name)")
            parser.add_argument("--yes", "-y", action="store_true", help="Confirm deletion without prompt")
            
            try:
                opts = parser.parse_args(args)
                target = opts.target
                match_by = opts.by
                
                # Auto-detect if not provided: treat strings starting with 'custom_' as IDs
                if not match_by:
                    match_by = "id" if target.startswith("custom_") else "name"
                
                if not opts.yes:
                    confirm = input(f"This will delete label(s) by {match_by}='{target}' from collection '{self.collection}'. Type 'yes' to confirm: ").strip().lower()
                    if confirm != "yes":
                        self.log("Deletion cancelled.")
                        return
                
                # Build filter conditions
                filter_conditions = {"type": "label"}
                if match_by == "id":
                    filter_conditions["label_id"] = target
                else:
                    filter_conditions["label_name"] = target
                
                # Find label points
                points_list, _ = self.qdrant_service.scroll_vectors(
                    self.collection, 1000, with_payload=True, with_vectors=False,
                    filter_conditions=filter_conditions
                )
                
                if not points_list:
                    self.log("No matching labels found.")
                    return
                
                point_ids = [int(p["id"]) for p in points_list]
                ok = self.qdrant_service.delete_points(self.collection, point_ids)
                if ok:
                    self.log(f"Deleted {len(point_ids)} label point(s) from collection '{self.collection}'")
                else:
                    self.log("Failed to delete label point(s).", True)
            except ValueError as e:
                self.log(f"Invalid rm-label arguments: {e}. Usage: rm-label <label_id|label_name> [--by id|name] [--yes]", True)

        elif cmd == "list-labels":
            if self.collection is None:
                self.log("No collection selected.", True)
                return
            
            self._list_labels_command()

        elif cmd == "list-clusters":
            if self.collection is None:
                self.log("No collection selected.", True)
                return
            
            self._list_clusters_command()

        elif cmd == "help":
            self.console.print("\n[bold cyan]Available Commands[/bold cyan]")
            self.console.print(f"[dim]{self._hr()}[/dim]")
            
            help_table = Table(show_header=True, box=None, padding=(0, 2), show_edge=False)
            help_table.add_column("Command", style="green", no_wrap=True)
            help_table.add_column("Description", style="white")
            
            # Collection
            help_table.add_row("[bold magenta]Collection[/bold magenta]", "")
            help_table.add_row("  [green]use[/green] [cyan]<collection>[/cyan]", "Select collection")
            help_table.add_row("  [green]show[/green]", "Show status")
            help_table.add_row("  [green]ls[/green]", "List collections")
            help_table.add_row("  [green]ls-models[/green]", "List embedding models")
            help_table.add_row(f"  [green]create[/green] [cyan]<name>[/cyan] [cyan]{escape('[model]')}[/cyan] [cyan]-d[/cyan]", "Create collection")
            help_table.add_row(f"  [green]rm[/green] [cyan]{escape('[name]')}[/cyan] [cyan]{escape('[--yes]')}[/cyan]", "Remove collection + dedup index")
            
            # Data operations
            help_table.add_row("", "")
            help_table.add_row("[bold magenta]Data[/bold magenta]", "")
            help_table.add_row(f"  [green]source[/green] [cyan]<path>[/cyan] [cyan]{escape('[--limit]')}[/cyan] [cyan]{escape('[--text-column]')}[/cyan] [cyan]{escape('[--url-column]')}[/cyan]", "Ingest data")
            help_table.add_row("  [green]query[/green] [cyan]<query>[/cyan]", "Search (cluster/label/url/id)")
            help_table.add_row("  [green]stats[/green]", "Collection stats")
            
            # Analysis
            help_table.add_row("", "")
            help_table.add_row("[bold magenta]Analysis[/bold magenta]", "")
            help_table.add_row(f"  [green]cluster[/green] [cyan]{escape('[--num-clusters]')}[/cyan] [cyan]{escape('[--debug]')}[/cyan]", "Cluster + name")
            help_table.add_row(f"  [green]classify[/green] [cyan]<labels.json>[/cyan] [cyan]{escape('[--use-collection-labels]')}[/cyan] [cyan]{escape('[--enrich]')}[/cyan]", "Assign labels")
            help_table.add_row(f"  [green]add-label[/green] [cyan]<label>[/cyan] [cyan]{escape('[--description]')}[/cyan] [cyan]{escape('[--enrich]')}[/cyan]", "Store label point")
            help_table.add_row(f"  [green]rm-label[/green] [cyan]<label|label_id>[/cyan] [cyan]{escape('[--by]')}[/cyan] [cyan]{escape('[--yes]')}[/cyan]", "Remove label points")
            help_table.add_row("  [green]list-labels[/green]", "Show labels (stored/inferred)")
            help_table.add_row("  [green]list-clusters[/green]", "Show all clusters")
            
            # System
            help_table.add_row("[bold magenta]System[/bold magenta]", "")
            help_table.add_row(f"  [green]retry[/green] [cyan]{escape('[--host]')}[/cyan] [cyan]{escape('[--port]')}[/cyan]", "Reconnect")
            help_table.add_row("  [green]help[/green]", "Show help")
            help_table.add_row("  [green]exit[/green] / [green]quit[/green]", "Exit")
            
            self.console.print(help_table)
            
            self.console.print("\n[dim]Query Examples:[/dim]")
            examples = Table(show_header=False, box=None, padding=(0, 2))
            examples.add_column("Command", style="green", no_wrap=True)
            examples.add_column("Comment", style="dim")
            examples.add_row("query cluster:0", "# Get all docs in cluster 0")
            examples.add_row("query cluster:Technology", "# Get all docs in Technology cluster")
            examples.add_row("query label:Sports", "# Get all docs with Sports label")
            self.console.print(examples)

        elif cmd in ("exit", "quit"):
            exit(0)

        else:
            print(f"Unknown command: {cmd}")

    def _source_command(self, path, limit=None, text_column=None, url_column=None):
        """Load data from directory, CSV, or URL."""
        try:
            self.log(f"Loading data from: {path}")

            # Use ProcessingService to load data
            texts, payloads = self.processing_service.process_source(path, limit, text_column=text_column, url_column=url_column)

            if not texts:
                self.log("No data loaded from source.", True)
                return

            self.log(f"Loaded {len(texts)} items. Checking for duplicates...")

            # Check for duplicates before generating embeddings
            filtered_texts = []
            filtered_payloads = []
            duplicate_count = 0
            
            for text, payload in zip(texts, payloads):
                if 'hash' in payload:
                    hash_value = payload['hash']
                    # Only treat as duplicate if the service explicitly returns True.
                    # This avoids MagicMock truthiness marking everything as duplicate in tests.
                    is_dup = self.sqlite_service.is_duplicate(self.collection, hash_value)
                    if isinstance(is_dup, bool) and is_dup:
                        duplicate_count += 1
                    else:
                        filtered_texts.append(text)
                        filtered_payloads.append(payload)
                else:
                    # If no hash in payload, include it (shouldn't happen with proper processing)
                    filtered_texts.append(text)
                    filtered_payloads.append(payload)
            
            if duplicate_count > 0:
                self.log(f"Skipped {duplicate_count} duplicate documents")
            
            if not filtered_texts:
                self.log("No new documents to process after duplicate filtering.")
                return
            
            self.log(f"Processing {len(filtered_texts)} new documents. Generating embeddings...")

            # Get the embedding model used for this collection
            collection_model = self.qdrant_service.get_collection_model(self.collection)
            if collection_model:
                self.log(f"Using collection's embedding model: {collection_model}")
                embeddings = self.openai_service.generate_embeddings(filtered_texts, model=collection_model)
            else:
                # Fallback to default model if collection model not found
                self.log("Embedding model not found, using default: text-embedding-3-small")
                embeddings = self.openai_service.generate_embeddings(filtered_texts)

            if not embeddings:
                self.log("Failed to generate any embeddings.", True)
                return

            if len(embeddings) != len(filtered_texts):
                self.log(f"Embedding count mismatch. Expected {len(filtered_texts)}, got {len(embeddings)}.", True)
                return

            # Check if embeddings are real (non-random) by checking if they look like random data
            # Random embeddings typically have very small values close to 0
            sample_embedding = embeddings[0]
            if sample_embedding and all(abs(x) < 0.01 for x in sample_embedding[:10]):
                self.log("Warning: Generated embeddings appear to be random/fallback embeddings. Check your OpenAI API connection.", True)

            self.log(f"Generated {len(embeddings)} embeddings. Inserting into collection...")

            # Insert into collection using QdrantService (no duplicate checking needed since we already filtered,
            # and hashes are marked post-success inside the service)
            success, inserted_count, skipped_count = self.qdrant_service.insert_vectors(
                self.collection, embeddings, filtered_payloads, sqlite_service=self.sqlite_service
            )

            if success:
                self.log(f"Successfully inserted {inserted_count} items into collection '{self.collection}'")
                self.log("Source command completed successfully.")
            else:
                self.log("Vector insertion failed.", True)

        except Exception as e:
            self.log(f"Source command failed: {e}", True)
            self.log(f"Full traceback: {traceback.format_exc()}", True)

if __name__ == "__main__":
    cli = Cli()
    cli.run()
