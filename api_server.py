#!/usr/bin/env python3
"""
FastAPI client to expose Qdrant endpoints for document clustering queries.

Endpoints:
- GET /collections - List all collections
- GET /collections/{collection_name}/info - Get collection info
- GET /collections/{collection_name}/points - Get all points with labels
- GET /collections/{collection_name}/points/{point_id} - Get specific point
- GET /collections/{collection_name}/labels - Get all unique labels
- GET /collections/{collection_name}/labels/{label} - Get points by label
- GET /collections/{collection_name}/clusters - Get all clusters
- GET /collections/{collection_name}/clusters/{cluster_id} - Get points by cluster
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
from contextlib import asynccontextmanager

# Add src to Python path
sys.path.insert(0, '.')

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import uvicorn
from rich.logging import RichHandler
from rich.console import Console

# Import our services
from src.services.qdrant_service import QdrantService

# Setup Rich logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console, show_path=False)]
)

logger = logging.getLogger("api_server")

# Constants
MAX_SCROLL_LIMIT = 10000  # Maximum limit for scroll operations
DEFAULT_PAGE_SIZE = 1000  # Default page size for pagination
MAX_TEXT_PREVIEW = 500  # Maximum characters for text preview

# Lazy initialization for Qdrant service (avoids connection during import)
_qdrant_service = None

def _get_qdrant_service():
    """Get or create Qdrant service instance (lazy initialization)."""
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
        logger.info("[green]✓[/green] Qdrant service initialized", extra={"markup": True})
    return _qdrant_service

# Lifespan event handler for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup: Initialize Qdrant service eagerly (for production performance)
    _get_qdrant_service()
    yield
    # Shutdown: Add any cleanup code here if needed in the future

app = FastAPI(
    title="Document Clustering API",
    description="REST API for querying clustered documents in Qdrant",
    version="1.0.1",
    lifespan=lifespan,
    # docs_url="/docs",      # Default: Swagger UI at /docs
    # redoc_url="/redoc",    # Default: ReDoc at /redoc
    # openapi_url="/openapi.json"  # Default: OpenAPI schema
    
    # To disable ReDoc (keep only Swagger):
    # redoc_url=None,
    
    # To disable both docs:
    # docs_url=None,
    # redoc_url=None,
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing."""
    start_time = datetime.now()
    
    # Log request
    logger.info(f"[cyan]{request.method}[/cyan] {request.url.path}", extra={"markup": True})
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = (datetime.now() - start_time).total_seconds()
    
    # Log response with color based on status
    if response.status_code < 300:
        status_color = "green"
    elif response.status_code < 400:
        status_color = "yellow"
    else:
        status_color = "red"
    
    logger.info(
        f"[{status_color}]{response.status_code}[/{status_color}] "
        f"{request.url.path} - [{duration:.3f}s]",
        extra={"markup": True}
    )
    
    return response

# Module-level accessor that initializes on first use
class _QdrantServiceProxy:
    """Proxy that delays QdrantService initialization until first access."""
    def __getattr__(self, name):
        return getattr(_get_qdrant_service(), name)

qdrant_service = _QdrantServiceProxy()

# Performance optimizations
COLLECTIONS_CACHE = {}
COLLECTIONS_CACHE_TTL = 30  # Cache for 30 seconds
COLLECTIONS_CACHE_TIME = 0

def infer_embedding_model(dimension: int) -> str:
    """Infer embedding model from vector dimension."""
    model_mapping = {
        1536: "text-embedding-3-small",
        3072: "text-embedding-3-large", 
        512: "text-embedding-ada-002",
        1024: "text-embedding-ada-002",  # Some older versions
        768: "text-embedding-ada-002",   # Some older versions
    }
    return model_mapping.get(dimension, f"unknown-model-{dimension}d")

def get_cached_collections():
    """Get collections with caching to reduce Qdrant calls."""
    import time
    global COLLECTIONS_CACHE, COLLECTIONS_CACHE_TIME
    
    current_time = time.time()
    if (current_time - COLLECTIONS_CACHE_TIME) > COLLECTIONS_CACHE_TTL or not COLLECTIONS_CACHE:
        logger.debug("Refreshing collections cache")
        COLLECTIONS_CACHE = qdrant_service.list_collections()
        COLLECTIONS_CACHE_TIME = current_time
        logger.debug(f"Cached {len(COLLECTIONS_CACHE)} collections")
    
    return COLLECTIONS_CACHE

# Pydantic models
class PointResponse(BaseModel):
    id: int
    source: str
    cluster_id: Optional[int] = None
    cluster_name: Optional[str] = None
    predicted_label: Optional[str] = None
    confidence: Optional[float] = None
    text_content: Optional[str] = None

class CollectionInfo(BaseModel):
    name: str
    vector_count: int
    dimension: int
    distance_metric: str
    embedding_model: str
    created_at: Optional[str] = None
    description: Optional[str] = None
    clustering_algorithm: Optional[str] = None
    num_clusters: Optional[int] = None

class LabelInfo(BaseModel):
    label: str
    count: int

class ClusterInfo(BaseModel):
    cluster_id: int
    cluster_name: str
    count: int

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "Document Clustering API",
        "version": "1.0.1",
        "endpoints": {
            "collections": "/collections",
            "collection_info": "/collections/{name}/info",
            "all_points": "/collections/{name}/points",
            "point_by_id": "/collections/{name}/points/{point_id}",
            "labels": "/collections/{name}/labels",
            "points_by_label": "/collections/{name}/labels/{label}",
            "clusters": "/collections/{name}/clusters",
            "points_by_cluster": "/collections/{name}/clusters/{cluster_id}"
        }
    }

@app.get("/collections")
async def list_collections():
    """List all available collections in Qdrant with metadata.

    Note: This endpoint queries Qdrant directly (no cache) to reflect current state
    and to surface backend errors appropriately for tests and clients.
    
    Returns metadata including:
    - created_at: Collection creation timestamp
    - description: User-provided collection description
    - clustering_algorithm: Method used for clustering (kmeans/agglomerative)
    - num_clusters: Number of clusters in the collection
    """
    try:
        collections_dict = qdrant_service.list_collections()

        # Enhance collections with model and metadata information
        enhanced_collections = {}
        for name, info in collections_dict.items():
            if not isinstance(info, dict):
                continue  # Skip invalid entries
            
            # Safely get dimension/size
            dimension = info.get('size') or info.get('dimension')
            if not dimension:
                continue  # Skip if no dimension info
            
            embedding_model = infer_embedding_model(dimension)
            
            # Get additional metadata
            metadata = qdrant_service.get_collection_metadata(name)
            
            enhanced_collections[name] = {
                **info,
                "embedding_model": embedding_model,
                "clustering_algorithm": metadata.get('clustering_algorithm') if metadata else None,
                "num_clusters": metadata.get('num_clusters') if metadata else None
            }

        return {"collections": enhanced_collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")

@app.get("/collections/{collection_name}/info", response_model=CollectionInfo)
async def get_collection_info(collection_name: str):
    """Get information about a specific collection including metadata."""
    try:
        # Check if collection exists using cache
        collections = get_cached_collections()
        if collection_name not in collections:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
        
        info = qdrant_service.get_collection_info(collection_name)
        if not info:
            raise HTTPException(status_code=500, detail=f"Failed to get collection info for '{collection_name}'")
        embedding_model = info.get('embedding_model') or infer_embedding_model(info['dimension'])
        
        # Get collection metadata for clustering info
        metadata = qdrant_service.get_collection_metadata(collection_name)
        clustering_algorithm = metadata.get('clustering_algorithm') if metadata else None
        num_clusters = metadata.get('num_clusters') if metadata else None
        
        return CollectionInfo(
            name=collection_name,
            vector_count=info['vector_count'],
            dimension=info['dimension'],
            distance_metric=info['distance_metric'],
            embedding_model=embedding_model,
            created_at=info.get('created_at'),
            description=info.get('description'),
            clustering_algorithm=clustering_algorithm,
            num_clusters=num_clusters
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collection info: {str(e)}")

@app.get("/collections/{collection_name}/points")
async def get_all_points(
    collection_name: str,
    limit: Optional[int] = Query(None, ge=1, le=MAX_SCROLL_LIMIT, description="Maximum number of points to return; if omitted, returns all"),
    offset: int = Query(0, ge=0, description="Offset for pagination (used when limit is provided)"),
):
    """Get all points in a collection with their labels."""
    try:
        # Check if collection exists using cache
        collections = get_cached_collections()
        if collection_name not in collections:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
        
        # Get points
        if limit is None:
            # Fetch all pages iteratively (heavy; use with caution)
            aggregated_points = []
            page_offset = None
            page_size = DEFAULT_PAGE_SIZE  # hard paging size for stability
            while True:
                page_points, page_offset = qdrant_service.scroll_vectors(
                    collection_name,
                    limit=page_size,
                    with_payload=True,
                    with_vectors=False,
                    page_offset=page_offset
                )
                if not page_points:
                    break
                aggregated_points.extend(page_points)
                if not page_offset:
                    break
            points = aggregated_points
            next_offset = None
        else:
            # Get a single page respecting limit
            page_size = min(limit, DEFAULT_PAGE_SIZE)
            points, next_offset = qdrant_service.scroll_vectors(
                collection_name,
                limit=page_size,
                with_payload=True,
                with_vectors=False
            )
        
        # Format response (slice to ensure API respects limit even if backend over-returns)
        formatted_points = []
        for point in (points if limit is None else points[:page_size]):
            payload = point.get('payload', {})
            formatted_points.append(PointResponse(
                id=point['id'],
                source=payload.get('source', ''),
                cluster_id=payload.get('cluster_id'),
                cluster_name=payload.get('cluster_name'),
                predicted_label=payload.get('predicted_label'),
                confidence=payload.get('confidence'),
                text_content=payload.get('text_content', '')[:500] if payload.get('text_content') else None
            ))
        
        return {
            "collection": collection_name,
            "points": formatted_points,
            "total_returned": len(formatted_points),
            "next_offset": next_offset
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get points: {str(e)}")

@app.get("/collections/{collection_name}/points/{point_id}", response_model=PointResponse)
async def get_point_by_id(collection_name: str, point_id: int):
    """Get a specific point by ID."""
    try:
        # Check if collection exists using cache
        collections = get_cached_collections()
        if collection_name not in collections:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
        
        # Get specific point using retrieve method
        try:
            point = qdrant_service.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False
            )
            points = [{"id": p.id, "payload": p.payload} for p in point]
        except Exception as e:
            # Fallback to scroll if retrieve fails
            points, _ = qdrant_service.scroll_vectors(
                collection_name,
                limit=1000,  # Get more points to find the specific one
                with_payload=True,
                with_vectors=False
            )
            # Filter by ID in Python
            points = [p for p in points if p['id'] == point_id]
        
        if not points:
            raise HTTPException(status_code=404, detail=f"Point {point_id} not found in collection '{collection_name}'")
        
        point = points[0]
        payload = point.get('payload', {})
        
        return PointResponse(
            id=point['id'],
            source=payload.get('source', ''),
            cluster_id=payload.get('cluster_id'),
            cluster_name=payload.get('cluster_name'),
            predicted_label=payload.get('predicted_label'),
            confidence=payload.get('confidence'),
            text_content=payload.get('text_content', '')[:500] if payload.get('text_content') else None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get point: {str(e)}")

@app.get("/collections/{collection_name}/labels")
async def get_all_labels(
    collection_name: str,
    limit: int = Query(1000, ge=1, le=MAX_SCROLL_LIMIT, description="Page size for label aggregation"),
    page_offset: Optional[str] = Query(None, description="Opaque paging cursor from previous call"),
    use_cache: bool = Query(True, description="Use labels summary from metadata when available")
):
    """Get labels with per-page aggregation.

    Returns label bucket counts for a single page of points, plus next_offset for iteration.
    """
    try:
        # Check if collection exists using cache
        collections = get_cached_collections()
        if collection_name not in collections:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")

        # If cache requested and available, return full summary (no paging)
        if use_cache:
            try:
                metadata = qdrant_service.get_collection_metadata(collection_name)
                summary = metadata.get("labels_summary") if metadata else None
                if isinstance(summary, dict) and len(summary) > 0:
                    labels = [LabelInfo(label=label, count=count) for label, count in sorted(summary.items())]
                    return {
                        "collection": collection_name,
                        "labels": labels,
                        "total_labels": len(labels),
                        "source": "metadata",
                        "next_offset": None
                    }
            except Exception:
                pass

        # Fetch one page of points (per-page aggregation)
        points, next_offset = qdrant_service.scroll_vectors(
            collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            page_offset=page_offset
        )

        # Aggregate labels for this page
        label_counts: Dict[str, int] = {}
        for point in points:
            payload = point.get('payload', {})
            label = payload.get('predicted_label')
            if label:
                label_counts[label] = label_counts.get(label, 0) + 1

        labels = [LabelInfo(label=label, count=count) for label, count in sorted(label_counts.items())]

        return {
            "collection": collection_name,
            "labels": labels,
            "page_size": limit,
            "returned": len(points),
            "next_offset": next_offset,
            "total_labels": len(labels)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get labels: {str(e)}")

@app.get("/collections/{collection_name}/labels/{label}")
async def get_points_by_label(
    collection_name: str, 
    label: str,
    limit: int = Query(100, ge=1, le=MAX_SCROLL_LIMIT, description="Maximum number of points to return")
):
    """Get all points with a specific label."""
    try:
        # Check if collection exists using cache
        collections = get_cached_collections()
        if collection_name not in collections:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
        
        # Get points with specific label
        points, _ = qdrant_service.scroll_vectors(
            collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            filter_conditions={"predicted_label": label}
        )
        
        if not points:
            raise HTTPException(status_code=404, detail=f"No points found with label '{label}' in collection '{collection_name}'")
        
        # Format response
        formatted_points = []
        for point in points:
            payload = point.get('payload', {})
            formatted_points.append(PointResponse(
                id=point['id'],
                source=payload.get('source', ''),
                cluster_id=payload.get('cluster_id'),
                cluster_name=payload.get('cluster_name'),
                predicted_label=payload.get('predicted_label'),
                confidence=payload.get('confidence'),
                text_content=payload.get('text_content', '')[:500] if payload.get('text_content') else None
            ))
        
        return {
            "collection": collection_name,
            "label": label,
            "points": formatted_points,
            "total_found": len(formatted_points)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get points by label: {str(e)}")

@app.get("/collections/{collection_name}/clusters")
async def get_all_clusters(
    collection_name: str,
    limit: int = Query(1000, ge=1, le=MAX_SCROLL_LIMIT, description="Page size for cluster aggregation"),
    page_offset: Optional[str] = Query(None, description="Opaque paging cursor from previous call"),
    use_cache: bool = Query(True, description="Use clusters summary from metadata when available")
):
    """Get clusters with per-page aggregation.

    Returns cluster bucket counts for a single page of points, plus next_offset for iteration.
    """
    try:
        # Check if collection exists using cache
        collections = get_cached_collections()
        if collection_name not in collections:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")

        # If cache requested and available, return full summary (no paging)
        if use_cache:
            try:
                metadata = qdrant_service.get_collection_metadata(collection_name)
                summary = metadata.get("clusters_summary") if metadata else None
                if isinstance(summary, dict) and len(summary) > 0:
                    clusters = [
                        ClusterInfo(
                            cluster_id=int(v.get("cluster_id", int(k))),
                            cluster_name=str(v.get("cluster_name", f"Cluster_{k}")),
                            count=int(v.get("count", 0))
                        )
                        for k, v in sorted(summary.items(), key=lambda x: int(x[0]))
                    ]
                    return {
                        "collection": collection_name,
                        "clusters": clusters,
                        "total_clusters": len(clusters),
                        "source": "metadata",
                        "next_offset": None
                    }
            except Exception:
                pass

        # Fetch one page of points (per-page aggregation)
        points, next_offset = qdrant_service.scroll_vectors(
            collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            page_offset=page_offset
        )

        # Aggregate clusters for this page
        cluster_counts: Dict[tuple[int, str], int] = {}
        for point in points:
            payload = point.get('payload', {})
            cluster_id = payload.get('cluster_id')
            if cluster_id is None:
                continue
            cluster_name = payload.get('cluster_name', f'Cluster_{cluster_id}')
            key = (int(cluster_id), str(cluster_name))
            cluster_counts[key] = cluster_counts.get(key, 0) + 1

        clusters = [
            ClusterInfo(cluster_id=cid, cluster_name=cname, count=count)
            for (cid, cname), count in sorted(cluster_counts.items())
        ]

        return {
            "collection": collection_name,
            "clusters": clusters,
            "page_size": limit,
            "returned": len(points),
            "next_offset": next_offset,
            "total_clusters": len(clusters)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get clusters: {str(e)}")

@app.get("/collections/{collection_name}/clusters/{cluster_id}")
async def get_points_by_cluster(
    collection_name: str, 
    cluster_id: int,
    limit: int = Query(100, ge=1, le=MAX_SCROLL_LIMIT, description="Maximum number of points to return")
):
    """Get all points in a specific cluster."""
    try:
        # Check if collection exists using cache
        collections = get_cached_collections()
        if collection_name not in collections:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
        
        # Get points with specific cluster_id
        points, _ = qdrant_service.scroll_vectors(
            collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            filter_conditions={"cluster_id": cluster_id}
        )
        
        if not points:
            raise HTTPException(status_code=404, detail=f"No points found in cluster {cluster_id} in collection '{collection_name}'")
        
        # Format response
        formatted_points = []
        cluster_name = None
        for point in points:
            payload = point.get('payload', {})
            if cluster_name is None:
                cluster_name = payload.get('cluster_name', f'Cluster_{cluster_id}')
            formatted_points.append(PointResponse(
                id=point['id'],
                source=payload.get('source', ''),
                cluster_id=payload.get('cluster_id'),
                cluster_name=payload.get('cluster_name'),
                predicted_label=payload.get('predicted_label'),
                confidence=payload.get('confidence'),
                text_content=payload.get('text_content', '')[:500] if payload.get('text_content') else None
            ))
        
        return {
            "collection": collection_name,
            "cluster_id": cluster_id,
            "cluster_name": cluster_name,
            "points": formatted_points,
            "total_found": len(formatted_points)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get points by cluster: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"[red]Unhandled exception:[/red] {str(exc)}", extra={"markup": True}, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    # Pretty startup banner
    console.print("\n[bold cyan]╔═══════════════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [bold white]Document Clustering API Server[/bold white]                       [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]╚═══════════════════════════════════════════════════════╝[/bold cyan]\n")
    
    console.print("[bold]Server Information:[/bold]")
    console.print(f"  [dim]•[/dim] [cyan]API Endpoint:[/cyan]     http://localhost:8000")
    console.print(f"  [dim]•[/dim] [cyan]Documentation:[/cyan]    http://localhost:8000/docs")
    console.print(f"  [dim]•[/dim] [cyan]Version:[/cyan]          1.0.1")
    
    # console.print("\n[bold]Performance Optimizations:[/bold]")
    # console.print(f"  [dim]•[/dim] [green]Collections caching (30s TTL)[/green]")
    # console.print(f"  [dim]•[/dim] [green]Request/response logging with timing[/green]")
    # console.print(f"  [dim]•[/dim] [green]Rich tracebacks for debugging[/green]")
    
    # console.print("\n[bold]Features:[/bold]")
    # console.print(f"  [dim]•[/dim] [yellow]Collection metadata (created_at, description)[/yellow]")
    # console.print(f"  [dim]•[/dim] [yellow]Clustering information (algorithm, num_clusters)[/yellow]")
    # console.print(f"  [dim]•[/dim] [yellow]Full-text search and filtering[/yellow]")
    
    console.print(f"\n[bold green]Server starting...[/bold green]\n")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=1,  # Single worker for now
        loop="asyncio",
        access_log=False,  # We handle logging with middleware
        log_config=None  # Disable uvicorn's default logging
    )
