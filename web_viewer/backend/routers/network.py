"""
Network Router

API endpoints for network/graph management:
- Loading networks from SQL
- Getting graph elements
- State queries
- Cache management
- Neighbor queries
- Node updates for auto-reload
- API properties management
"""

from typing import Optional, List

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..models.requests import LoadConfig
from ..services.network_service import network_service
from ..services.api_properties_service import api_properties_service
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
        # API properties configuration
        "api_properties": {
            "enabled": bool(settings.EXTERNAL_API_PROVIDERS),
            "providers": api_properties_service.available_providers,
            "base_url": settings.EXTERNAL_API_BASE_URL,
            "cache_ttl_seconds": settings.EXTERNAL_API_CACHE_TTL,
        },
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
            "metrics_computed": [],
            "api_properties_loaded": {}
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
        "auto_reload_available": HAS_SSE,
        "api_properties_loaded": network_service._api_properties_loaded,
        "api_properties_source": network_service._api_properties_source
    }


@router.get("/api-properties/providers")
async def list_api_properties_providers():
    """List available API properties providers and their status."""
    return {
        "providers": api_properties_service.available_providers,
        "all_columns": api_properties_service.all_columns_provided,
        "cache_ttl_seconds": settings.EXTERNAL_API_CACHE_TTL,
        "base_url": settings.EXTERNAL_API_BASE_URL
    }


@router.get("/api-properties/cache")
async def list_api_properties_caches():
    """List cached API properties with metadata."""
    return {
        "caches": network_service.cache_service.list_api_properties_caches()
    }


@router.delete("/api-properties/cache")
async def clear_api_properties_cache(
    provider: Optional[str] = Query(None, description="Provider name to clear"),
    version: Optional[str] = Query(None, description="Version to clear")
):
    """Clear API properties cache, optionally for specific provider/version."""
    network_service.cache_service.clear_api_properties_cache(provider, version)
    return {
        "status": "cleared",
        "provider": provider,
        "version": version
    }


@router.post("/api-properties/refresh")
async def refresh_api_properties(
    version: str = Query("v2", description="Version to refresh"),
    providers: Optional[str] = Query(None, description="Comma-separated provider names")
):
    """
    Refresh API properties by fetching fresh data from APIs.
    
    This fetches new data regardless of cache state and updates the cache.
    """
    try:
        provider_list = None
        if providers:
            provider_list = [p.strip() for p in providers.split(",") if p.strip()]
        
        df, provider_cols, source = network_service.load_api_properties(
            version=version,
            providers=provider_list,
            skip_cache=True  # Always fetch fresh
        )
        
        if df.empty:
            return {
                "status": "no_data",
                "message": "No data fetched from API providers",
                "providers_queried": provider_list or settings.EXTERNAL_API_PROVIDERS
            }
        
        return {
            "status": "success",
            "rows_fetched": len(df),
            "columns": list(df.columns),
            "provider_columns": provider_cols,
            "source": source
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nodes/data")
async def get_all_node_data(
    limit: int = Query(10000, ge=1, le=100000),
    offset: int = Query(0, ge=0)
):
    """
    Get all node data (metrics + properties) for data exploration.
    
    Returns paginated list of all nodes with their attributes.
    """
    import pandas as pd
    
    df = network_service.get_all_node_data_df()
    if df is None or df.empty:
        return {
            "nodes": [],
            "columns": [],
            "total": 0,
            "offset": offset,
            "limit": limit
        }
    
    total = len(df)
    
    # Get column info (types)
    columns = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        col_type = "string"
        if "int" in dtype or "float" in dtype:
            col_type = "number"
        elif "bool" in dtype:
            col_type = "boolean"
        columns.append({"name": col, "type": col_type})
    
    # Paginate
    df_page = df.iloc[offset:offset + limit]
    
    # Convert to records, handling special types
    nodes = []
    for _, row in df_page.iterrows():
        node = {}
        for col in df.columns:
            val = row[col]
            # Check for list/array BEFORE checking isna (isna fails on arrays)
            if isinstance(val, (list, np.ndarray)):
                # Keep arrays as-is for frontend display
                node[col] = val if isinstance(val, list) else val.tolist()
            elif isinstance(val, dict):
                node[col] = val
            elif val is None:
                node[col] = None
            elif isinstance(val, float) and np.isnan(val):
                node[col] = None
            else:
                try:
                    if pd.isna(val):
                        node[col] = None
                    else:
                        node[col] = val
                except (ValueError, TypeError):
                    node[col] = val
        nodes.append(node)
    
    return {
        "nodes": nodes,
        "columns": columns,
        "total": total,
        "offset": offset,
        "limit": limit
    }