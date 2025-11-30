"""
Composite Metrics Router

API endpoints for composite metric creation and management.
"""

from typing import Optional, List, Dict, Any

import numpy as np
from fastapi import APIRouter, HTTPException

from ..models.requests import CompositeMetricConfig, CompositePreviewRequest
from ..models.responses import CompositeMetricResult, CompositePreviewResponse
from ..services.network_service import network_service
from ..config import HAS_ANOMALY

if HAS_ANOMALY:
    from engines.composite_engine import CompositeMetricEngine


router = APIRouter(prefix="/api/metrics/composite", tags=["composite"])


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to a JSON-compatible float."""
    if value is None:
        return default
    try:
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


@router.get("/operations")
async def get_composite_operations():
    """Get available composite metric operations."""
    if not HAS_ANOMALY:
        raise HTTPException(
            status_code=503, 
            detail="Composite metrics not available"
        )
    return CompositeMetricEngine.get_available_operations()


@router.get("/saved")
async def get_saved_composites(version: Optional[str] = None):
    """Get list of saved composite metrics."""
    if not HAS_ANOMALY or not network_service.composite_engine:
        raise HTTPException(
            status_code=503, 
            detail="Composite metrics not available"
        )
    
    composites = network_service.composite_engine.get_saved_composites(version)
    return {"composites": composites}


@router.post("/preview", response_model=CompositePreviewResponse)
async def preview_composite_metric(request: CompositePreviewRequest):
    """
    Preview a composite metric without saving.
    
    Returns the computed values and statistics for visualization,
    allowing users to see the result before creating the metric.
    """
    if not HAS_ANOMALY or not network_service.composite_engine:
        raise HTTPException(
            status_code=503, 
            detail="Composite metrics not available"
        )
    
    if not network_service.graphs:
        raise HTTPException(
            status_code=400, 
            detail="No graphs loaded. Please load networks first."
        )
    
    try:
        # Get metrics DataFrame
        version = list(network_service.metrics_dfs.keys())[0] if network_service.metrics_dfs else None
        df = network_service.get_metrics_dataframe(version)
        
        if df is None or df.empty:
            raise HTTPException(
                status_code=400, 
                detail="No metrics data available. Run metrics first."
            )
        
        # Filter to specific nodes if requested
        original_df = df.copy()
        if request.node_ids:
            id_col = 'avatar' if 'avatar' in df.columns else 'id' if 'id' in df.columns else None
            if id_col:
                df = df[df[id_col].astype(str).isin(request.node_ids)]
            else:
                df = df[df.index.astype(str).isin(request.node_ids)]
            
            if df.empty:
                raise HTTPException(
                    status_code=400,
                    detail="No matching nodes found for the specified node_ids"
                )
        
        # Validate metrics exist
        missing = [m for m in request.metrics if m not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Metrics not found: {missing}"
            )
        
        # Create composite without saving
        result_series, metadata = network_service.composite_engine.create_composite(
            df=df,
            name="preview",
            metrics=request.metrics,
            operation=request.operation,
            weights=request.weights,
            normalize=request.normalize,
            save=False
        )
        
        # Determine ID column
        id_col = 'avatar' if 'avatar' in df.columns else 'id' if 'id' in df.columns else None
        
        # Build values for visualization
        values = []
        for idx, row in df.iterrows():
            if id_col:
                node_id = str(row[id_col])
            else:
                node_id = str(idx)
            
            # Get composite value
            if node_id in result_series.index:
                composite_val = float(result_series[node_id])
            elif idx in result_series.index:
                composite_val = float(result_series[idx])
            else:
                # Try positional
                try:
                    pos = df.index.get_loc(idx)
                    composite_val = float(result_series.iloc[pos])
                except:
                    composite_val = 0.0
            
            values.append({
                'id': node_id,
                'metric1': _safe_float(row[request.metrics[0]]),
                'metric2': _safe_float(row[request.metrics[1]]),
                'composite': _safe_float(composite_val)
            })
        
        # Calculate correlations
        m1 = df[request.metrics[0]].values.astype(float)
        m2 = df[request.metrics[1]].values.astype(float)
        composite_values = result_series.values.astype(float)
        
        # Handle NaN values for correlation calculation
        m1 = np.nan_to_num(m1, nan=0.0)
        m2 = np.nan_to_num(m2, nan=0.0)
        composite_values = np.nan_to_num(composite_values, nan=0.0)
        
        # Calculate correlations safely
        def safe_corr(a, b):
            if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
                return 0.0
            try:
                corr = np.corrcoef(a, b)[0, 1]
                return _safe_float(corr)
            except:
                return 0.0
        
        correlations = {
            'input_correlation': safe_corr(m1, m2),
            'm1_composite': safe_corr(m1, composite_values),
            'm2_composite': safe_corr(m2, composite_values)
        }
        
        # Build histogram data
        hist_counts, bin_edges = np.histogram(composite_values, bins=30)
        histogram = {
            'bins': [_safe_float(b) for b in bin_edges.tolist()],
            'counts': [int(c) for c in hist_counts.tolist()]
        }
        
        return CompositePreviewResponse(
            formula=metadata['formula'],
            statistics={
                'min': _safe_float(metadata['statistics']['min']),
                'max': _safe_float(metadata['statistics']['max']),
                'mean': _safe_float(metadata['statistics']['mean']),
                'std': _safe_float(metadata['statistics']['std']),
                'median': _safe_float(metadata['statistics']['median']),
            },
            values=values,
            correlations=correlations,
            histogram=histogram,
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Composite preview error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create")
def create_composite_metric(config: CompositeMetricConfig):
    """
    Create a composite metric from existing metrics.
    
    Optionally saves to cache for reuse across sessions.
    """
    if not HAS_ANOMALY or not network_service.composite_engine:
        raise HTTPException(
            status_code=503, 
            detail="Composite metrics not available"
        )
    
    if not network_service.graphs:
        raise HTTPException(
            status_code=400, 
            detail="No graphs loaded. Please load networks first."
        )
    
    try:
        # Get metrics DataFrame
        version = config.version
        if not version:
            version = list(network_service.metrics_dfs.keys())[0] if network_service.metrics_dfs else None
        
        df = network_service.get_metrics_dataframe(version)
        if df is None or df.empty:
            raise HTTPException(
                status_code=400, 
                detail="No metrics data available. Run metrics first."
            )
        
        # Create composite metric
        result_series, metadata = network_service.composite_engine.create_composite(
            df=df,
            name=config.name,
            metrics=config.metrics,
            operation=config.operation,
            weights=config.weights,
            normalize=config.normalize,
            save=config.save,
            version=version or "default"
        )
        
        # Build node_updates from the result series directly
        # This ensures all nodes get the update, not just those in graphs
        node_updates = []
        for node_id, value in result_series.items():
            node_updates.append({
                'id': str(node_id),
                config.name: _safe_float(float(value))
            })
        
        # Also apply to graphs if loaded
        for gid, G in network_service.graphs.items():
            graph_version = network_service._extract_version(gid)
            if version and graph_version != version:
                continue
            
            for node_id, value in result_series.items():
                if G.has_node(node_id):
                    G.nodes[node_id][config.name] = float(value)
        
        return CompositeMetricResult(
            metric_name=config.name,
            formula=metadata['formula'],
            node_updates=node_updates,
            statistics=metadata['statistics'],
            saved=metadata.get('saved', False),
            composite_id=metadata.get('id')
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Composite metric error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{composite_id}")
async def delete_composite_metric(composite_id: str):
    """Delete a saved composite metric by ID or name."""
    if not HAS_ANOMALY or not network_service.composite_engine:
        raise HTTPException(
            status_code=503, 
            detail="Composite metrics not available"
        )
    
    # Try deleting by ID first
    success = network_service.composite_engine.delete_composite(composite_id)
    
    # If not found by ID, try by name
    if not success:
        success = network_service.composite_engine.delete_composite_by_name(composite_id)
    
    if not success:
        raise HTTPException(
            status_code=404, 
            detail=f"Composite {composite_id} not found"
        )
    
    return {"status": "deleted", "composite_id": composite_id}


@router.post("/{composite_id}/apply")
def apply_composite_metric(composite_id: str, version: Optional[str] = None):
    """Apply a saved composite metric to current graph data."""
    if not HAS_ANOMALY or not network_service.composite_engine:
        raise HTTPException(
            status_code=503, 
            detail="Composite metrics not available"
        )
    
    if not network_service.graphs:
        raise HTTPException(
            status_code=400, 
            detail="No graphs loaded"
        )
    
    try:
        # Get the composite config
        composite = network_service.composite_engine.get_composite_by_id(composite_id)
        if not composite:
            raise HTTPException(
                status_code=404, 
                detail=f"Composite {composite_id} not found"
            )
        
        # Get metrics DataFrame
        if not version:
            version = composite.get('version', 'default')
        
        df = network_service.get_metrics_dataframe(version)
        if df is None or df.empty:
            raise HTTPException(
                status_code=400, 
                detail="No metrics data available"
            )
        
        # Apply composite
        result_series, metadata = network_service.composite_engine.apply_saved_composite(
            composite_id, df
        )
        
        if result_series is None:
            raise HTTPException(
                status_code=404, 
                detail=f"Composite {composite_id} not found"
            )
        
        # Apply to graphs
        node_updates = []
        metric_name = composite['name']
        
        for gid, G in network_service.graphs.items():
            graph_version = network_service._extract_version(gid)
            if version != 'default' and graph_version != version:
                continue
            
            for node_id, value in result_series.items():
                if G.has_node(node_id):
                    G.nodes[node_id][metric_name] = float(value)
                    node_updates.append({
                        'id': node_id,
                        metric_name: float(value)
                    })
        
        return {
            "metric_name": metric_name,
            "node_updates": node_updates,
            "count": len(node_updates)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))