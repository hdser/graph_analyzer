"""
Composite Metrics Router

API endpoints for composite metric creation and management.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException

from backend.models.requests import CompositeMetricConfig
from backend.models.responses import CompositeMetricResult
from backend.services.network_service import network_service
from backend.config import HAS_ANOMALY

if HAS_ANOMALY:
    from engines.composite_engine import CompositeMetricEngine


router = APIRouter(prefix="/api/metrics/composite", tags=["composite"])


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


@router.post("")
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
        
        # Apply to graphs
        node_updates = []
        
        for gid, G in network_service.graphs.items():
            graph_version = network_service._extract_version(gid)
            if version and graph_version != version:
                continue
            
            for node_id, value in result_series.items():
                if G.has_node(node_id):
                    G.nodes[node_id][config.name] = float(value)
                    node_updates.append({
                        'id': node_id,
                        config.name: float(value)
                    })
        
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