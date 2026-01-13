"""
Capacity Flow API Router

API endpoints for capacity graph and max flow operations.

Location: web_viewer/backend/routers/capacity_flow.py
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from ..services.capacity_flow_service import capacity_flow_service

router = APIRouter(prefix="/api/capacity-flow", tags=["capacity-flow"])


# Request/Response Models

class BuildRequest(BaseModel):
    """Request to build capacity graph."""
    graph_id: Optional[str] = None
    include_groups: bool = True
    force_rebuild: bool = False


class MaxFlowRequest(BaseModel):
    """Request to compute max flow."""
    source: str
    sink: str
    graph_id: Optional[str] = None
    backend: Optional[str] = None
    algorithm: Optional[str] = None
    cutoff: Optional[int] = None
    decompose_paths: bool = True
    simplify_paths: bool = True
    max_paths: Optional[int] = Field(default=100, ge=1, le=10000)


# Endpoints

@router.get("/status")
async def get_status():
    """Check if capacity flow is available."""
    return {
        "available": capacity_flow_service.is_available(),
        "backends": capacity_flow_service.get_available_backends()
    }


@router.get("/backends")
async def get_backends():
    """Get available backends with algorithm info."""
    return {
        "backends": capacity_flow_service.get_available_backends(),
        "algorithms": capacity_flow_service.get_algorithms()
    }


@router.get("/algorithms")
async def get_algorithms():
    """Get available algorithms (combined backend + algorithm)."""
    return {"algorithms": capacity_flow_service.get_algorithms()}


@router.get("/stats")
async def get_stats(graph_id: Optional[str] = None):
    """Get capacity graph statistics."""
    return capacity_flow_service.get_stats(graph_id)


@router.post("/build")
async def build_graph(request: BuildRequest):
    """Build capacity graph from trust network data."""
    if not capacity_flow_service.is_available():
        raise HTTPException(status_code=503, detail="Capacity flow engine not available")
    
    result = await capacity_flow_service.build_capacity_graph(
        graph_id=request.graph_id,
        include_groups=request.include_groups,
        force_rebuild=request.force_rebuild
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Build failed"))
    
    return result


@router.get("/graph/nodes")
async def get_graph_nodes(
    graph_id: Optional[str] = None,
    use_trust_layout: bool = True
):
    """
    Get capacity graph nodes with positions.
    Loads nodes first for fast initial display.
    """
    if not capacity_flow_service.is_available():
        raise HTTPException(status_code=503, detail="Capacity flow engine not available")
    
    result = capacity_flow_service.get_capacity_graph_nodes(graph_id, use_trust_layout)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.get("/graph/edges")
async def get_graph_edges(
    graph_id: Optional[str] = None,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=10000, ge=100, le=50000),
    edge_type: Optional[str] = None
):
    """
    Get capacity graph edges in batches.
    
    Args:
        offset: Starting edge index
        limit: Maximum edges to return
        edge_type: Filter by type (balance, trust, mint)
    """
    if not capacity_flow_service.is_available():
        raise HTTPException(status_code=503, detail="Capacity flow engine not available")
    
    result = capacity_flow_service.get_capacity_graph_edges(
        graph_id=graph_id,
        offset=offset,
        limit=limit,
        edge_type=edge_type
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.get("/graph")
async def get_graph_data(graph_id: Optional[str] = None):
    """
    Get full capacity graph data for visualization.
    For large graphs, use /graph/nodes and /graph/edges instead.
    """
    if not capacity_flow_service.is_available():
        raise HTTPException(status_code=503, detail="Capacity flow engine not available")
    
    result = capacity_flow_service.get_capacity_graph_data(graph_id)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.post("/max-flow")
async def compute_max_flow(request: MaxFlowRequest):
    """Compute maximum flow between source and sink."""
    if not capacity_flow_service.is_available():
        raise HTTPException(status_code=503, detail="Capacity flow engine not available")
    
    result = await capacity_flow_service.compute_max_flow(
        source=request.source,
        sink=request.sink,
        graph_id=request.graph_id,
        backend=request.backend,
        algorithm=request.algorithm,
        cutoff=request.cutoff,
        decompose_paths=request.decompose_paths,
        simplify_paths=request.simplify_paths,
        max_paths=request.max_paths
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Computation failed"))
    
    return result


@router.post("/find-paths")
async def find_paths(request: MaxFlowRequest):
    """Alias for max-flow with path decomposition."""
    request.decompose_paths = True
    return await compute_max_flow(request)


@router.get("/node/{address}/capacity")
async def get_node_capacity(
    address: str,
    direction: str = Query(default="both", pattern="^(in|out|both)$"),
    graph_id: Optional[str] = None
):
    """Get capacity information for a node."""
    if not capacity_flow_service.is_available():
        raise HTTPException(status_code=503, detail="Capacity flow not available")
    
    result = capacity_flow_service.get_node_capacity(address, direction, graph_id)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.delete("/clear")
async def clear_cache(graph_id: Optional[str] = None):
    """Clear capacity graph cache."""
    return capacity_flow_service.clear(graph_id)