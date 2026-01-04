"""
Routers Package

FastAPI routers for the Graph Analyzer API.

Location: web_viewer/backend/routers/__init__.py
"""

from .network import router as network_router
from .metrics import router as metrics_router
from .anomaly import router as anomaly_router
from .composite import router as composite_router
from .auto_reload import router as auto_reload_router
from .snapshots import router as snapshots_router
from .snapshot_analysis import router as snapshot_analysis_router
from .timeseries import router as timeseries_router
from .temporal_composite import router as temporal_composite_router
from .graph_algorithms import router as graph_algorithms_router

__all__ = [
    "network_router",
    "metrics_router",
    "anomaly_router",
    "composite_router",
    "auto_reload_router",
    "snapshots_router",
    "snapshot_analysis_router",
    "timeseries_router",
    "temporal_composite_router",
    "graph_algorithms_router",
]