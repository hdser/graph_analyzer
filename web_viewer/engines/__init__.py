"""
Engines Package

Core computation engines for graph analysis:
- GraphMetrics: Comprehensive graph metrics computation
- AnomalyEngine: Anomaly detection with multiple algorithms
- CompositeMetricEngine: Composite metric creation
- MetricProfiler: Automatic metric analysis and configuration
"""

from .graph_metrics import GraphMetrics, METRIC_CATEGORIES, METRIC_PRESETS

__all__ = [
    "GraphMetrics",
    "METRIC_CATEGORIES",
    "METRIC_PRESETS",
]

# Conditionally export anomaly-related classes
try:
    from .anomaly_config import (
        MetricConfig,
        MetricTransform,
        AlgorithmConfig,
        ThresholdMethod,
        NaNStrategy,
        ScoreNormalization,
        GlobalScaling,
    )
    from .anomaly_engine import AnomalyEngine
    from .result_builder import AnomalyResult, GroupAnomalyStats, ThresholdInfo
    from .metric_profiler import MetricProfiler, MetricProfile
    from .preprocessing import Preprocessor
    from .composite_engine import CompositeMetricEngine
    
    __all__.extend([
        "MetricConfig",
        "MetricTransform", 
        "AlgorithmConfig",
        "ThresholdMethod",
        "NaNStrategy",
        "ScoreNormalization",
        "GlobalScaling",
        "AnomalyEngine",
        "AnomalyResult",
        "GroupAnomalyStats",
        "ThresholdInfo",
        "MetricProfiler",
        "MetricProfile",
        "Preprocessor",
        "CompositeMetricEngine",
    ])
except ImportError as e:
    print(f"[WARNING] Some anomaly detection features unavailable: {e}")