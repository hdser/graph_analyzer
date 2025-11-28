"""
Anomaly Detection Router

API endpoints for anomaly detection functionality.
"""

from fastapi import APIRouter, HTTPException

from ..models.requests import AnomalyDetectionConfig
from ..models.responses import AnomalyDetectionResult
from ..services.network_service import network_service
from ..config import HAS_ANOMALY

if HAS_ANOMALY:
    from engines.anomaly_engine import AnomalyEngine


router = APIRouter(prefix="/api/anomaly", tags=["anomaly"])


@router.get("/algorithms")
async def get_anomaly_algorithms():
    """Get available anomaly detection algorithms with parameters."""
    if not HAS_ANOMALY:
        raise HTTPException(
            status_code=503, 
            detail="Anomaly detection not available. Install scikit-learn."
        )
    return AnomalyEngine.get_available_algorithms()


@router.post("/detect")
def detect_anomalies(config: AnomalyDetectionConfig):
    """
    Run anomaly detection on graph metrics.
    
    Returns anomaly scores as new metric that can be used for
    coloring/filtering in the visualization.
    """
    if not HAS_ANOMALY or not network_service.anomaly_engine:
        raise HTTPException(
            status_code=503, 
            detail="Anomaly detection not available"
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
            # Use first available version
            version = list(network_service.metrics_dfs.keys())[0] if network_service.metrics_dfs else None
        
        df = network_service.get_metrics_dataframe(version)
        if df is None or df.empty:
            raise HTTPException(
                status_code=400, 
                detail="No metrics data available. Run metrics first."
            )
        
        # Run anomaly detection
        result = network_service.anomaly_engine.detect_anomalies(
            df=df,
            metrics=config.metrics,
            algorithm=config.algorithm,
            parameters=config.parameters
        )
        
        # Prepare response
        response = AnomalyDetectionResult(
            metric_name=config.name,
            algorithm=result.algorithm,
            n_anomalies=result.n_anomalies,
            n_total=result.n_total,
            anomaly_percentage=(result.n_anomalies / result.n_total * 100) if result.n_total > 0 else 0,
            computation_time=result.computation_time,
            top_anomalies=result.top_anomalies,
            score_statistics=result.statistics,
            metrics_used=result.metrics_used,
            parameters_used=result.parameters
        )
        
        # Apply to graph if requested
        if config.apply_to_graph:
            node_updates = []
            
            for gid, G in network_service.graphs.items():
                graph_version = network_service._extract_version(gid)
                if version and graph_version != version:
                    continue
                
                for node_id, score in result.scores.items():
                    if G.has_node(node_id):
                        G.nodes[node_id][config.name] = score
                        G.nodes[node_id][f"{config.name}_is_anomaly"] = result.binary_labels.get(node_id, False)
                        node_updates.append({
                            'id': node_id,
                            config.name: score,
                            f"{config.name}_is_anomaly": result.binary_labels.get(node_id, False)
                        })
            
            response.node_updates = node_updates
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Anomaly detection error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/composites")
async def get_composite_metrics(version: str = None):
    """Get list of saved composite metrics."""
    if not HAS_ANOMALY or not network_service.composite_engine:
        raise HTTPException(
            status_code=503, 
            detail="Composite metrics not available"
        )
    
    return network_service.composite_engine.get_saved_composites(version)