"""
Network Router

API endpoints for network/graph management:
- Loading networks from SQL
- Getting graph elements
- State queries
- Cache management
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..models.requests import LoadConfig
from ..services.network_service import network_service
from ..config import HAS_ANOMALY, HAS_SSE

from ...engines.graph_metrics import METRIC_CATEGORIES, METRIC_PRESETS

if HAS_ANOMALY:
    from ...engines.anomaly_engine import AnomalyEngine
    from ...engines.composite_engine import CompositeMetricEngine


router = APIRouter(prefix="/api", tags=["network"])


@router.get("/config")
async def get_config():
    """Get application configuration including available SQL files and features."""
    config = {
        "sql_files": network_service.available_sql_files,
        "metric_modes": {
            "presets": {k: list(v) for k, v in METRIC_PRESETS.items()},
            "categories": {k: v for k, v in METRIC_CATEGORIES.items()}
        },
        "cytoscape_desktop_available": network_service.cytoscape_available,
        "cached_layouts": network_service.list_cached_layouts(),
        "anomaly_available": HAS_ANOMALY,
        "auto_reload_available": HAS_SSE
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
async def get_current_state():
    """Get current application state."""
    if not network_service.graphs:
        return {"loaded": False}
    
    total_nodes = sum(len(df) for df in network_service.metrics_dfs.values())
    return {
        "loaded": True,
        "graphs": list(network_service.graphs.keys()),
        "cytoscape_available": network_service.cytoscape_available,
        "node_count": total_nodes,
        "anomaly_available": HAS_ANOMALY,
        "auto_reload_available": HAS_SSE
    }