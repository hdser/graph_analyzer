"""
Routers Package

API endpoint routers.
"""

from backend.routers.network import router as network_router
from backend.routers.metrics import router as metrics_router
from backend.routers.anomaly import router as anomaly_router
from backend.routers.composite import router as composite_router
from backend.routers.auto_reload import router as auto_reload_router

__all__ = [
    "network_router",
    "metrics_router",
    "anomaly_router",
    "composite_router",
    "auto_reload_router",
]