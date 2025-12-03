"""
Network Router

API endpoints for network/graph management:
- Loading networks from SQL
- Getting graph elements
- State queries
- Cache management
- Neighbor queries
- Node updates for auto-reload
"""

from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..models.requests import LoadConfig
from ..services.network_service import network_service
from ..config import settings, HAS_ANOMALY, HAS_SSE

from engines.graph_metrics import METRIC_CATEGORIES, METRIC_PRESETS

if HAS_ANOMALY:
    from engines.anomaly_engine import AnomalyEngine
    from engines.composite_engine import CompositeMetricEngine


router = APIRouter(prefix="/api", tags=["network"])


class NeighborRequest(BaseModel):
    """Request model for neighbor queries."""
    node_ids: List[str]
    direction: str = "both"  # "in", "out", or "both"


@router.get("/config")
async def get_config():
    """Get application configuration including available SQL files and features."""
    config = {
        "sql_files": network_service.available_sql_files,
        "node_properties_files": network_service.available_node_properties_files,
        "metric_modes": {
            "presets": {k: list(v) for k, v in METRIC_PRESETS.items()},
            "categories": {k: v for k, v in METRIC_CATEGORIES.items()}
        },
        "cytoscape_desktop_available": network_service.cytoscape_available,
        "cached_layouts": network_service.list_cached_layouts(),
        "anomaly_available": HAS_ANOMALY,
        "auto_reload_available": HAS_SSE,
        # UI mode
        "hide_data_source_ui": settings.HIDE_DATA_SOURCE_UI,
        "default_sql_files": settings.DEFAULT_SQL_FILES,
        "default_properties_files": settings.DEFAULT_PROPERTIES_FILES,
        "default_metrics_mode": settings.DEFAULT_METRICS_MODE,
        "auto_reload_interval": settings.AUTO_RELOAD_DEFAULT_INTERVAL,
    }
    
    # Add anomaly algorithms if available
    if HAS_ANOMALY and network_service.anomaly_engine:
        config["anomaly_algorithms"] = AnomalyEngine.get_available_algorithms()
        config["composite_operations"] = CompositeMetricEngine.get_available_operations()
    
    return config


@router.post("/load")
def load_network(config: LoadConfig):
    """Load network from SQL files."""
    try:
        state = network_service.load_network(config)
        return state
    except Exception as e:
        print(f"Error loading network: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graphs/{graph_id}/elements")
def get_graph_elements(
    graph_id: str,
    mode: str = Query("full", pattern="^(full|nodes_only)$"),
):
    """
    Return graph elements for Cytoscape.js.

    The `mode` parameter allows loading only nodes for large graphs to
    keep the initial payload light.
    """
    try:
        elements = network_service.get_graph_elements(graph_id, mode=mode)
        return {"elements": elements, "count": len(elements)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graphs/{graph_id}/edges")
def get_graph_edges(
    graph_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(50000, ge=1, le=200000),
):
    """
    Return a chunk of edges for the given graph.

    This is used to incrementally stream edges to the frontend so that
    the initial graph preview can display nodes quickly while edges
    are loaded in batches.
    """
    try:
        return network_service.get_graph_edges_chunk(graph_id, offset, limit)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graphs/{graph_id}/node-updates")
def get_node_updates(
    graph_id: str,
    node_ids: Optional[str] = Query(None, description="Comma-separated node IDs")
):
    """
    Get updated node data for incremental frontend refresh.
    
    Used by auto-reload to update the frontend display without full reload.
    Returns current node attributes including metrics and properties.
    """
    try:
        parsed_node_ids = None
        if node_ids:
            parsed_node_ids = [n.strip() for n in node_ids.split(",") if n.strip()]
        
        updates = network_service.get_node_updates(graph_id, parsed_node_ids)
        return {"updates": updates, "count": len(updates)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/network/graphs/{graph_id}/neighbors")
def get_neighbors(graph_id: str, request: NeighborRequest):
    """
    Get neighbors of specified nodes.
    
    This allows finding neighbors even when edges aren't loaded on the frontend.
    Uses the NetworkX graph stored in memory.
    """
    try:
        return network_service.get_neighbors(
            graph_id, 
            request.node_ids, 
            request.direction
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cached-layouts")
async def list_cached_layouts():
    """List all cached layouts."""
    return network_service.list_cached_layouts()


@router.delete("/cached-layouts")
async def clear_cached_layouts(graph_id: Optional[str] = None):
    """Clear cached layouts, optionally for a specific graph."""
    network_service.clear_layout_cache(graph_id)
    return {"status": "cleared", "graph_id": graph_id}


@router.get("/state")
async def get_state():
    """
    Get current application state.
    
    Returns information about loaded graphs, node/edge counts,
    and computed metrics.
    """
    if not network_service.graphs:
        return {
            "loaded": False,
            "loaded_graphs": [],
            "node_count": 0,
            "edge_count": 0,
            "metrics_computed": []
        }
    
    total_nodes = sum(G.number_of_nodes() for G in network_service.graphs.values())
    total_edges = sum(G.number_of_edges() for G in network_service.graphs.values())
    
    metrics_computed = []
    if network_service.metrics_dfs:
        for df in network_service.metrics_dfs.values():
            metrics_computed = [c for c in df.columns if c != 'avatar']
            break
    
    return {
        "loaded": True,
        "loaded_graphs": list(network_service.graphs.keys()),
        "node_count": total_nodes,
        "edge_count": total_edges,
        "metrics_computed": metrics_computed,
        "cytoscape_available": network_service.cytoscape_available,
        "anomaly_available": HAS_ANOMALY,
        "auto_reload_available": HAS_SSE
    }