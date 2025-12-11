"""
Routers Package

API endpoint routers.
"""

from .network import router as network_router
from .metrics import router as metrics_router
from .anomaly import router as anomaly_router
from .composite import router as composite_router
from .auto_reload import router as auto_reload_router
from .snapshots import router as snapshots_router

__all__ = [
    "network_router",
    "metrics_router",
    "anomaly_router",
    "composite_router",
    "auto_reload_router",
    "snapshots_router",
]