"""
Metrics Router

API endpoints for metrics computation.
"""

from fastapi import APIRouter, HTTPException

from ..models.requests import MetricsConfig
from ..services.network_service import network_service


router = APIRouter(prefix="/api", tags=["metrics"])


@router.post("/metrics")
def update_metrics(config: MetricsConfig):
    """
    Recompute metrics for loaded graphs.
    
    This updates node attributes with new metric values based on
    the specified metrics mode.
    """
    try:
        result = network_service.update_metrics(config)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error updating metrics: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))