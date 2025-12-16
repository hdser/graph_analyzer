"""
Services Package

Business logic services for the Graph Analyzer.
"""

# Services with singleton instances
from .network_service import network_service, NetworkService
from .api_properties_service import api_properties_service, APIPropertiesService
from .snapshot_service import snapshot_service, SnapshotService
from .snapshot_analysis_service import snapshot_analysis_service, SnapshotAnalysisService

# Services as classes only (no singleton exported at module level)
from .cache_service import CacheService
from .layout_service import LayoutService
from .auto_reload_service import AutoReloadManager
from .snapshot_storage import SnapshotStorage
from .snapshot_layout import SnapshotLayout

__all__ = [
    # Singletons
    "network_service",
    "api_properties_service",
    "snapshot_service",
    "snapshot_analysis_service",
    # Classes
    "NetworkService",
    "APIPropertiesService",
    "SnapshotService",
    "SnapshotAnalysisService",
    "CacheService",
    "LayoutService",
    "AutoReloadManager",
    "SnapshotStorage",
    "SnapshotLayout",
]