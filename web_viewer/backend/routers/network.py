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
- Layout management
"""

from typing import Optional, List

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..models.requests import LoadConfig
from ..services.network_service import network_service
from ..services.api_properties_service import api_properties_service
from ..config import settings, HAS_ANOMALY, HAS_SSE

from engines.metrics import METRIC_CATEGORIES, METRIC_PRESETS

if HAS_ANOMALY:
    from engines.anomaly_engine import AnomalyEngine
    from engines.composite_engine import CompositeMetricEngine


router = APIRouter(prefix="/api", tags=["network"])


class NeighborRequest(BaseModel):
    """Request model for neighbor queries."""
    node_ids: List[str]
    direction: str = "both"


class LayoutRequest(BaseModel):
    """Request model for layout recomputation."""
    backend: Optional[str] = None
    algorithm: Optional[str] = None
    save_as_base: bool = False
    from_scratch: bool = True  # If False, use existing layout as starting point (warm start)
    # Backend-specific parameters
    iterations: Optional[int] = None
    scale: Optional[float] = None


class SetDefaultLayoutRequest(BaseModel):
    """Request to set a layout as the default."""
    filename: str


@router.get("/config")
async def get_config():
    """Get application configuration including available SQL files and features."""
    formatted_presets = {}
    for preset_name, preset_data in METRIC_PRESETS.items():
        if isinstance(preset_data, dict):
            metrics_list = preset_data.get('metrics', [])
            for cat in preset_data.get('categories', []):
                if cat in METRIC_CATEGORIES:
                    cat_data = METRIC_CATEGORIES[cat]
                    if isinstance(cat_data, dict):
                        metrics_list.extend(cat_data.get('metrics', []))
            formatted_presets[preset_name] = list(set(metrics_list))
        else:
            formatted_presets[preset_name] = list(preset_data)
    
    formatted_categories = {}
    for cat_name, cat_data in METRIC_CATEGORIES.items():
        if isinstance(cat_data, dict):
            formatted_categories[cat_name] = cat_data.get('description', cat_name)
        else:
            formatted_categories[cat_name] = str(cat_data)
    
    config = {
        "sql_files": network_service.available_sql_files,
        "node_properties_files": network_service.available_node_properties_files,
        "metric_modes": {
            "presets": formatted_presets,
            "categories": formatted_categories
        },
        "cytoscape_desktop_available": network_service.cytoscape_available,
        "cached_layouts": network_service.list_cached_layouts(),
        "anomaly_available": HAS_ANOMALY,
        "auto_reload_available": HAS_SSE,
        "hide_data_source_ui": settings.HIDE_DATA_SOURCE_UI,
        "default_sql_files": settings.DEFAULT_SQL_FILES,
        "default_properties_files": settings.DEFAULT_PROPERTIES_FILES,
        "default_metrics_mode": settings.DEFAULT_METRICS_MODE,
        "auto_reload_interval": settings.AUTO_RELOAD_DEFAULT_INTERVAL,
        "api_properties": {
            "enabled": bool(settings.EXTERNAL_API_PROVIDERS),
            "providers": api_properties_service.available_providers,
            "base_url": settings.EXTERNAL_API_BASE_URL,
            "cache_ttl_seconds": settings.EXTERNAL_API_CACHE_TTL,
        },
        # Layout backends info
        "layout_backends": network_service.layout_service.get_available_backends(),
        "layout_backend_priority": settings.LAYOUT_BACKEND_PRIORITY,
    }
    
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
    """Return graph elements for Cytoscape.js."""
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
    """Return a chunk of edges for the given graph."""
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
    """Get updated node data for incremental frontend refresh."""
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
    """Get neighbors of specified nodes."""
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


# =============================================================================
# LAYOUT ENDPOINTS
# =============================================================================

@router.get("/layout/backends")
async def get_layout_backends():
    """Get list of available layout backends and their capabilities."""
    return {
        "backends": network_service.layout_service.get_available_backends(),
        "priority": settings.LAYOUT_BACKEND_PRIORITY
    }


@router.get("/layout/saved/{graph_id}")
async def get_saved_layouts(graph_id: str):
    """
    Get list of all saved layouts for a graph.
    
    Returns layouts with metadata including backend, algorithm, and timestamps.
    """
    print(f"[LAYOUT API] Getting saved layouts for graph_id: {graph_id}")
    print(f"[LAYOUT API] Layouts dir: {network_service.cache_service.layouts_dir}")
    
    # List all parquet files in layouts dir
    import os
    layouts_dir = network_service.cache_service.layouts_dir
    if layouts_dir.exists():
        all_files = list(layouts_dir.glob("*.parquet"))
        print(f"[LAYOUT API] All parquet files in layouts dir: {[f.name for f in all_files]}")
        matching = list(layouts_dir.glob(f"{graph_id}*.parquet"))
        print(f"[LAYOUT API] Files matching '{graph_id}*.parquet': {[f.name for f in matching]}")
    else:
        print(f"[LAYOUT API] Layouts dir does not exist!")
    
    layouts = network_service.cache_service.list_layouts_for_graph(graph_id)
    print(f"[LAYOUT API] Found {len(layouts)} layouts: {layouts}")
    return {
        "graph_id": graph_id,
        "layouts": layouts,
        "count": len(layouts)
    }


@router.post("/layout/set-default/{graph_id}")
async def set_default_layout(graph_id: str, request: SetDefaultLayoutRequest):
    """
    Set an existing layout as the default (base) layout.
    
    Args:
        graph_id: Graph identifier
        request: Contains filename of the layout to set as default
    """
    try:
        base_path = network_service.cache_service.set_layout_as_default(
            graph_id, 
            request.filename
        )
        
        return {
            "status": "success",
            "graph_id": graph_id,
            "source": request.filename,
            "base_path": base_path
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/layout/load/{graph_id}/{filename}")
async def load_specific_layout(graph_id: str, filename: str):
    """
    Load a specific saved layout and apply it to the current graph.
    
    Args:
        graph_id: Graph identifier
        filename: Layout filename (with or without .parquet extension)
    """
    if graph_id not in network_service.graphs:
        raise HTTPException(status_code=404, detail=f"Graph not found: {graph_id}")
    
    positions = network_service.cache_service.get_layout_by_filename(filename)
    
    if not positions:
        raise HTTPException(status_code=404, detail=f"Layout not found: {filename}")
    
    # Update stored layout
    network_service.layouts[graph_id] = positions
    
    return {
        "graph_id": graph_id,
        "filename": filename,
        "node_count": len(positions),
        "status": "loaded"
    }


@router.delete("/layout/{graph_id}/{filename}")
async def delete_layout(graph_id: str, filename: str):
    """Delete a specific saved layout."""
    deleted = network_service.cache_service.delete_layout(filename)
    
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Layout not found: {filename}")
    
    return {
        "status": "deleted",
        "filename": filename
    }


@router.post("/layout/recompute/{graph_id}")
async def recompute_layout(graph_id: str, request: LayoutRequest):
    """
    Recompute layout for a loaded graph.
    
    Args:
        graph_id: Graph identifier
        request: Layout configuration including:
            - backend: Specific backend to use (igraph, fa2, cytoscape_desktop)
            - algorithm: Algorithm for backends with multiple options
            - from_scratch: If False, use existing layout as starting point (warm start)
            - save_as_base: Save result as the default layout
            - iterations: Number of iterations (for iterative algorithms)
            - scale: Output scale factor
        
    Returns:
        Layout computation result with backend/algorithm info
    """
    if graph_id not in network_service.graphs:
        raise HTTPException(status_code=404, detail=f"Graph not found: {graph_id}")
    
    G = network_service.graphs[graph_id]
    
    try:
        # Build kwargs from request
        kwargs = {}
        if request.iterations:
            kwargs['iterations'] = request.iterations
        if request.scale:
            kwargs['scale'] = request.scale
        
        positions, backend_used, algo_used, comp_time = network_service.layout_service.recompute_layout(
            G,
            graph_id,
            backend=request.backend,
            algorithm=request.algorithm,
            from_scratch=request.from_scratch,
            **kwargs
        )
        
        if not positions:
            raise HTTPException(status_code=500, detail="Layout computation failed")
        
        # Update stored layout
        network_service.layouts[graph_id] = positions
        
        # Save named layout with backend/algorithm in filename
        layout_path = network_service.cache_service.save_named_layout(
            graph_id, 
            positions, 
            backend_used, 
            algo_used
        )
        
        # Also save as base if requested
        if request.save_as_base:
            network_service.cache_service.save_base_layout(graph_id, positions)
        
        return {
            "graph_id": graph_id,
            "backend": backend_used,
            "algorithm": algo_used,
            "node_count": len(positions),
            "computation_time": comp_time,
            "layout_file": layout_path,
            "saved_as_base": request.save_as_base,
            "warm_start": not request.from_scratch
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[LAYOUT] Recompute error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/layout/test/{backend}")
async def test_layout_backend(
    backend: str,
    nodes: int = Query(100, ge=10, le=10000),
    algorithm: Optional[str] = None
):
    """
    Test a layout backend with a synthetic graph.
    
    Useful for validating backend availability and performance.
    """
    import networkx as nx
    import time
    
    # Generate test graph
    G = nx.erdos_renyi_graph(nodes, 0.02, directed=True)
    
    try:
        start = time.time()
        positions, algo, comp_time = network_service.layout_service.compute_layout(
            G,
            "test_graph",
            preferred_backend=backend,
            algorithm=algorithm
        )
        
        return {
            "backend": backend,
            "algorithm": algo,
            "success": positions is not None,
            "node_count": len(positions) if positions else 0,
            "edge_count": G.number_of_edges(),
            "computation_time": time.time() - start
        }
    except Exception as e:
        return {
            "backend": backend,
            "success": False,
            "error": str(e)
        }


# =============================================================================
# CACHE ENDPOINTS
# =============================================================================

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
    """Get current application state."""
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
        "api_properties_source": network_service._api_properties_source,
        "layout_backends_available": [
            b["id"] for b in network_service.layout_service.get_available_backends()
            if b.get("available", False)
        ]
    }


# =============================================================================
# API PROPERTIES ENDPOINTS
# =============================================================================

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
    """Refresh API properties by fetching fresh data from APIs."""
    try:
        provider_list = None
        if providers:
            provider_list = [p.strip() for p in providers.split(",") if p.strip()]
        
        df, provider_cols, source = network_service.load_api_properties(
            version=version,
            providers=provider_list,
            skip_cache=True
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


# =============================================================================
# NODE DATA ENDPOINTS
# =============================================================================

@router.get("/nodes/data")
async def get_all_node_data(
    limit: int = Query(10000, ge=1, le=100000),
    offset: int = Query(0, ge=0)
):
    """Get all node data (metrics + properties) for data exploration."""
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
    
    columns = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        col_type = "string"
        if "int" in dtype or "float" in dtype:
            col_type = "number"
        elif "bool" in dtype:
            col_type = "boolean"
        columns.append({"name": col, "type": col_type})
    
    df_page = df.iloc[offset:offset + limit]
    
    nodes = []
    for _, row in df_page.iterrows():
        node = {}
        for col in df.columns:
            val = row[col]
            if isinstance(val, (list, np.ndarray)):
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