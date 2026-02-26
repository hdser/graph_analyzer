"""
Snapshots Router

API endpoints for historical network snapshot functionality.
"""

import json
import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse

from ..config import HAS_SSE
from ..models.snapshot import (
    SnapshotCreateRequest,
    SnapshotBatchRequest,
    SnapshotSuggestRequest,
    SnapshotRebuildRequest,
    SnapshotInfo,
    SnapshotData,
    SnapshotListResponse,
    SnapshotSuggestResponse,
    SnapshotProgress,
    SnapshotStatus,
    StorageStats,
)
from ..services.snapshot_service import SnapshotService, snapshot_service


router = APIRouter(prefix="/api/snapshots", tags=["snapshots"])


def get_snapshot_service() -> SnapshotService:
    """Get snapshot service instance."""
    return snapshot_service


def parse_snapshot_id(snapshot_id: str) -> tuple:
    """
    Parse snapshot_id into (base_sql_file, block_number).
    
    Format: {base_sql_file}_block_{block_number}
    Example: crc_v2_trusts_block_12345678
    """
    parts = snapshot_id.rsplit("_block_", 1)
    if len(parts) != 2:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid snapshot_id format: {snapshot_id}"
        )
    
    base_sql_file = parts[0]
    try:
        block_number = int(parts[1])
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid block number in snapshot_id: {snapshot_id}"
        )
    
    return base_sql_file, block_number


# =============================================================================
# List & Get Endpoints
# =============================================================================

@router.get("", response_model=SnapshotListResponse)
async def list_snapshots(
    base_sql_file: Optional[str] = Query(None, description="Filter by SQL file"),
    service: SnapshotService = Depends(get_snapshot_service)
):
    """
    List all available snapshots.
    
    Optionally filter by base_sql_file to get snapshots for a specific network.
    """
    snapshots = service.list_snapshots(base_sql_file)
    return SnapshotListResponse(snapshots=snapshots, total_count=len(snapshots))


@router.get("/available-sql-files")
async def get_available_sql_files(
    service: SnapshotService = Depends(get_snapshot_service)
):
    """
    Get list of SQL files that have snapshot templates available.
    """
    sql_files = service.get_available_snapshot_sql_files()
    return {"sql_files": sql_files}


@router.get("/storage-stats", response_model=StorageStats)
async def get_storage_stats(
    service: SnapshotService = Depends(get_snapshot_service)
):
    """
    Get snapshot storage usage statistics.
    """
    return service.storage.get_storage_stats()


@router.get("/{snapshot_id}", response_model=SnapshotInfo)
async def get_snapshot_info(
    snapshot_id: str,
    service: SnapshotService = Depends(get_snapshot_service)
):
    """
    Get snapshot metadata by ID.
    
    Snapshot ID format: {base_sql_file}_block_{block_number}
    """
    base_sql_file, block_number = parse_snapshot_id(snapshot_id)
    
    info = service.get_snapshot_info(base_sql_file, block_number)
    if not info:
        raise HTTPException(status_code=404, detail=f"Snapshot not found: {snapshot_id}")
    
    return info


@router.get("/{snapshot_id}/data", response_model=SnapshotData)
async def get_snapshot_data(
    snapshot_id: str,
    service: SnapshotService = Depends(get_snapshot_service)
):
    """
    Get full snapshot data including edges, layout, and metrics.
    
    This endpoint returns all data needed to render the snapshot in the frontend.
    """
    base_sql_file, block_number = parse_snapshot_id(snapshot_id)
    
    try:
        data = service.get_snapshot(base_sql_file, block_number)
        return data
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{snapshot_id}/nodes")
async def get_snapshot_nodes(
    snapshot_id: str,
    service: SnapshotService = Depends(get_snapshot_service)
):
    """
    Get snapshot nodes with positions and metrics (no edges).
    
    Fast endpoint for initial render - loads nodes first, edges separately.
    Returns Cytoscape-compatible node elements.
    """
    from fastapi.responses import JSONResponse
    
    base_sql_file, block_number = parse_snapshot_id(snapshot_id)
    
    try:
        # Load nodes data only
        nodes_data = await asyncio.to_thread(
            service.storage.load_snapshot_nodes, base_sql_file, block_number
        )
        
        # Get metadata for counts
        metadata = await asyncio.to_thread(
            service.storage.load_snapshot_metadata, base_sql_file, block_number
        )
        
        return JSONResponse(content={
            "elements": nodes_data["elements"],
            "metadata": {
                "snapshot_id": metadata.snapshot_id if metadata else snapshot_id,
                "base_sql_file": base_sql_file,
                "block_number": metadata.block_number if metadata else block_number,
                "block_timestamp": metadata.block_timestamp.isoformat() if metadata and metadata.block_timestamp else None,
                "label": metadata.label if metadata else None,
                "node_count": len(nodes_data["elements"]),
                "edge_count": metadata.edge_count if metadata else 0,
                "metrics_computed": metadata.metrics_computed if metadata else []
            }
        })
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{snapshot_id}/layout")
async def get_snapshot_layout(
    snapshot_id: str,
    service: SnapshotService = Depends(get_snapshot_service)
):
    """
    Get snapshot layout (positions only) - ultra lightweight for animation.
    
    Returns just {node_id: {x, y}} for fast position-only updates.
    """
    from fastapi.responses import JSONResponse
    
    base_sql_file, block_number = parse_snapshot_id(snapshot_id)
    
    try:
        # Load just the layout dict
        positions = await asyncio.to_thread(
            service.storage.load_snapshot_layout_dict, base_sql_file, block_number
        )
        
        return JSONResponse(content={
            "positions": positions,
            "count": len(positions)
        })
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{snapshot_id}/edges")
async def get_snapshot_edges(
    snapshot_id: str,
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    limit: int = Query(50000, ge=1, le=100000, description="Max edges to return"),
    service: SnapshotService = Depends(get_snapshot_service)
):
    """
    Get snapshot edges with pagination.
    
    Returns edges in batches for incremental loading.
    """
    from fastapi.responses import JSONResponse
    
    base_sql_file, block_number = parse_snapshot_id(snapshot_id)
    
    try:
        edges_data = await asyncio.to_thread(
            service.storage.load_snapshot_edges, 
            base_sql_file, block_number, offset, limit
        )
        
        return JSONResponse(content=edges_data)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{snapshot_id}/edges/lightweight")
async def get_snapshot_edges_lightweight(
    snapshot_id: str,
    service: SnapshotService = Depends(get_snapshot_service)
):
    """
    Get ALL snapshot edges in lightweight format for animation.
    
    Returns just [source, target] pairs - no Cytoscape wrapping, no metadata.
    Much faster and smaller than the full edges endpoint.
    """
    from fastapi.responses import JSONResponse
    
    base_sql_file, block_number = parse_snapshot_id(snapshot_id)
    
    try:
        # Load edges directly from parquet - just source/target columns
        edges = await asyncio.to_thread(
            service.storage.load_snapshot_edges_lightweight,
            base_sql_file, block_number
        )
        
        return JSONResponse(content={
            "edges": edges,  # List of [source, target] pairs
            "count": len(edges)
        })
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# =============================================================================
# Create Endpoints
# =============================================================================

@router.post("/create", response_model=SnapshotInfo)
async def create_snapshot(
    request: SnapshotCreateRequest,
    service: SnapshotService = Depends(get_snapshot_service)
):
    """
    Create a single snapshot.
    
    If snapshot already exists, returns existing snapshot info.
    """
    try:
        # Run in thread pool to not block event loop
        snapshot_info = await asyncio.to_thread(
            service.create_snapshot, request
        )
        return snapshot_info
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Snapshot creation failed: {str(e)}")


@router.post("/create-batch")
async def create_batch(
    request: SnapshotBatchRequest,
    service: SnapshotService = Depends(get_snapshot_service)
):
    """
    Create multiple snapshots with SSE progress reporting.
    
    Returns Server-Sent Events stream with progress updates.
    Event types:
    - progress: Current snapshot being created
    - complete: Individual snapshot complete
    - error: Error for individual snapshot
    - done: All snapshots complete
    """
    if not HAS_SSE:
        # Fallback: create all at once and return
        results = await asyncio.to_thread(
            service.create_batch, request
        )
        return {"snapshots": [s.model_dump() for s in results], "total": len(results)}
    
    from sse_starlette.sse import EventSourceResponse
    
    async def generate():
        total = len(request.block_numbers)
        created = []
        errors = []
        
        # Sort block numbers ascending
        block_numbers = sorted(request.block_numbers)
        
        for idx, block_number in enumerate(block_numbers, 1):
            try:
                # Progress event
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "current": idx,
                        "total": total,
                        "block_number": block_number,
                        "status": "computing"
                    })
                }
                
                # Create snapshot
                create_request = SnapshotCreateRequest(
                    base_sql_file=request.base_sql_file,
                    block_number=block_number,
                    metrics_mode=request.metrics_mode
                )
                
                snapshot_info = await asyncio.to_thread(
                    service.create_snapshot, create_request
                )
                created.append(snapshot_info)
                
                # Complete event
                yield {
                    "event": "complete",
                    "data": json.dumps({
                        "current": idx,
                        "total": total,
                        "snapshot_id": snapshot_info.snapshot_id,
                        "node_count": snapshot_info.node_count,
                        "edge_count": snapshot_info.edge_count
                    })
                }
                
            except Exception as e:
                errors.append({"block_number": block_number, "error": str(e)})
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "block_number": block_number,
                        "error": str(e)
                    })
                }
        
        # Done event
        yield {
            "event": "done",
            "data": json.dumps({
                "total_created": len(created),
                "total_errors": len(errors),
                "snapshots": [s.snapshot_id for s in created]
            })
        }
    
    return EventSourceResponse(generate())


@router.post("/rebuild")
async def rebuild_snapshots(
    request: SnapshotRebuildRequest,
    service: SnapshotService = Depends(get_snapshot_service)
):
    """
    Rebuild snapshots from scratch.

    If block_numbers is None, rebuilds all existing snapshots for the given
    base_sql_file. Optionally deletes existing snapshots first.

    Returns SSE progress stream (same event format as create-batch).
    """
    # Determine which blocks to rebuild
    if request.block_numbers:
        block_numbers = sorted(request.block_numbers)
    else:
        # Rebuild all existing snapshots
        existing = await asyncio.to_thread(
            service.list_snapshots, request.base_sql_file
        )
        block_numbers = sorted([s.block_number for s in existing])

    if not block_numbers:
        return {"error": "No snapshots found to rebuild", "total": 0}

    # Delete existing if requested
    if request.delete_existing:
        for block in block_numbers:
            try:
                await asyncio.to_thread(
                    service.delete_snapshot, request.base_sql_file, block
                )
            except Exception:
                pass  # Continue even if delete fails

    if not HAS_SSE:
        # Fallback: rebuild all at once
        batch_req = SnapshotBatchRequest(
            base_sql_file=request.base_sql_file,
            block_numbers=block_numbers,
            metrics_mode=request.metrics_mode,
        )
        results = await asyncio.to_thread(service.create_batch, batch_req)
        return {
            "snapshots": [s.model_dump() for s in results],
            "total": len(results),
        }

    from sse_starlette.sse import EventSourceResponse

    async def generate():
        total = len(block_numbers)
        created = []
        errors = []

        for idx, block_number in enumerate(block_numbers, 1):
            try:
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "current": idx,
                        "total": total,
                        "block_number": block_number,
                        "status": "rebuilding",
                    }),
                }

                create_request = SnapshotCreateRequest(
                    base_sql_file=request.base_sql_file,
                    block_number=block_number,
                    metrics_mode=request.metrics_mode,
                )

                snapshot_info = await asyncio.to_thread(
                    service.create_snapshot, create_request
                )
                created.append(snapshot_info)

                yield {
                    "event": "complete",
                    "data": json.dumps({
                        "current": idx,
                        "total": total,
                        "snapshot_id": snapshot_info.snapshot_id,
                        "node_count": snapshot_info.node_count,
                        "edge_count": snapshot_info.edge_count,
                    }),
                }

            except Exception as e:
                errors.append({"block_number": block_number, "error": str(e)})
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "block_number": block_number,
                        "error": str(e),
                    }),
                }

        yield {
            "event": "done",
            "data": json.dumps({
                "total_created": len(created),
                "total_errors": len(errors),
                "snapshots": [s.snapshot_id for s in created],
            }),
        }

    return EventSourceResponse(generate())


@router.post("/suggest", response_model=SnapshotSuggestResponse)
async def suggest_block_numbers(
    request: SnapshotSuggestRequest,
    service: SnapshotService = Depends(get_snapshot_service)
):
    """
    Get suggested block numbers for snapshot creation.
    
    Returns block numbers at the start of each time period (daily/weekly/monthly)
    within the specified date range.
    """
    try:
        suggestions = await asyncio.to_thread(
            service.suggest_block_numbers, request
        )
        return suggestions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")


# =============================================================================
# Delete Endpoints
# =============================================================================

@router.delete("/{snapshot_id}")
async def delete_snapshot(
    snapshot_id: str,
    service: SnapshotService = Depends(get_snapshot_service)
):
    """
    Delete a snapshot.
    
    This removes the snapshot files and updates the index.
    """
    base_sql_file, block_number = parse_snapshot_id(snapshot_id)
    
    success = await asyncio.to_thread(
        service.delete_snapshot, base_sql_file, block_number
    )
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Snapshot not found: {snapshot_id}")
    
    return {"success": True, "snapshot_id": snapshot_id}


# =============================================================================
# Comparison Endpoints
# =============================================================================

@router.get("/compare/{from_snapshot_id}/{to_snapshot_id}")
async def compare_snapshots(
    from_snapshot_id: str,
    to_snapshot_id: str,
    service: SnapshotService = Depends(get_snapshot_service)
):
    """
    Compare two snapshots and return the differences.
    
    Returns:
    - added_nodes: Nodes in 'to' but not in 'from'
    - removed_nodes: Nodes in 'from' but not in 'to'
    - retained_nodes: Nodes in both snapshots
    - added_edges: Edges in 'to' but not in 'from'
    - removed_edges: Edges in 'from' but not in 'to'
    """
    from fastapi.responses import JSONResponse
    
    try:
        from_base, from_block = parse_snapshot_id(from_snapshot_id)
        to_base, to_block = parse_snapshot_id(to_snapshot_id)
        
        # Verify same base SQL file
        if from_base != to_base:
            raise HTTPException(
                status_code=400, 
                detail="Cannot compare snapshots from different networks"
            )
        
        comparison = await asyncio.to_thread(
            service.compare_snapshots,
            from_base, from_block, to_block
        )
        
        return JSONResponse(content=comparison)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/animation/{base_sql_file}")
async def get_animation_data(
    base_sql_file: str,
    service: SnapshotService = Depends(get_snapshot_service)
):
    """
    Get all snapshots for animation with compact layout data.

    Returns minimal data needed for animation:
    - snapshots: List of snapshot metadata with layouts
    """
    from fastapi.responses import JSONResponse

    try:
        animation_data = await asyncio.to_thread(
            service.get_animation_data, base_sql_file
        )

        return JSONResponse(content=animation_data)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# =============================================================================
# Preload Endpoints (for TimeZoomBar)
# =============================================================================

@router.get("/preload/{base_sql_file}")
async def preload_snapshots(
    base_sql_file: str,
    snapshot_ids: str = Query(..., description="Comma-separated snapshot IDs to preload"),
    service: SnapshotService = Depends(get_snapshot_service)
):
    """
    Preload multiple snapshots for timeline navigation.

    Used by TimeZoomBar to preload adjacent snapshots for smooth scrubbing.
    Returns lightweight data (layout positions + metadata) for each snapshot.

    Args:
        base_sql_file: The base SQL file name
        snapshot_ids: Comma-separated list of snapshot IDs to preload

    Returns:
        preloaded: Dict mapping snapshot_id to preloaded data
        failed: List of snapshot_ids that failed to preload
    """
    from fastapi.responses import JSONResponse

    ids = [s.strip() for s in snapshot_ids.split(",") if s.strip()]

    if len(ids) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 snapshots can be preloaded at once"
        )

    preloaded = {}
    failed = []

    for snapshot_id in ids:
        try:
            _, block_number = parse_snapshot_id(snapshot_id)

            # Load layout (positions) - lightweight
            positions = await asyncio.to_thread(
                service.storage.load_snapshot_layout_dict,
                base_sql_file,
                block_number
            )

            # Load metadata
            metadata = await asyncio.to_thread(
                service.storage.load_snapshot_metadata,
                base_sql_file,
                block_number
            )

            preloaded[snapshot_id] = {
                "positions": positions,
                "node_count": len(positions),
                "edge_count": metadata.edge_count if metadata else 0,
                "block_number": block_number,
                "block_timestamp": metadata.block_timestamp.isoformat() if metadata and metadata.block_timestamp else None,
                "label": metadata.label if metadata else None
            }

        except Exception as e:
            failed.append({"snapshot_id": snapshot_id, "error": str(e)})

    return JSONResponse(content={
        "preloaded": preloaded,
        "failed": failed,
        "preload_count": len(preloaded)
    })