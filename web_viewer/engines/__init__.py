"""
Engines Package

Core computation engines for graph analysis:
- GraphMetrics: Comprehensive graph metrics computation
- AnomalyEngine: Anomaly detection with multiple algorithms
- CompositeMetricEngine: Composite metric creation
"""

from engines.graph_metrics import GraphMetrics, METRIC_CATEGORIES, METRIC_PRESETS

__all__ = [
    "GraphMetrics",
    "METRIC_CATEGORIES",
    "METRIC_PRESETS",
]

# Conditionally export anomaly-related classes
try:
    from engines.anomaly_engine import AnomalyEngine, AnomalyResult
    from engines.composite_engine import CompositeMetricEngine
    __all__.extend(["AnomalyEngine", "AnomalyResult", "CompositeMetricEngine"])
except ImportError:
    pass