"""
Engines Package

Computation engines for graph metrics, anomaly detection, and analysis.
"""

# Graph metrics
from .graph_metrics import GraphMetrics, METRIC_CATEGORIES, METRIC_PRESETS

# Anomaly detection engine and config
from .anomaly_engine import AnomalyEngine
from .anomaly_config import (
    MetricConfig,
    MetricTransform,
    AlgorithmConfig,
    NaNStrategy,
    GlobalScaling,
    ScoreNormalization,
    ThresholdMethod,
    AggregationMethod,
    TailSide,
)

# Metric profiler
from .metric_profiler import MetricProfiler, MetricProfile

# Composite metric engine
from .composite_engine import CompositeMetricEngine

# Preprocessing
from .preprocessing import Preprocessor, ChunkedPreprocessor

# Result builder
from .result_builder import ResultBuilder

# Timeseries engine
from .timeseries_engine import TimeseriesEngine, timeseries_engine

# Temporal composite engine
from .temporal_composite_engine import TemporalCompositeEngine, temporal_composite_engine

__all__ = [
    # Graph metrics
    "GraphMetrics",
    "METRIC_CATEGORIES",
    "METRIC_PRESETS",
    # Anomaly
    "AnomalyEngine",
    "MetricConfig",
    "MetricTransform",
    "AlgorithmConfig",
    "NaNStrategy",
    "GlobalScaling",
    "ScoreNormalization",
    "ThresholdMethod",
    "AggregationMethod",
    "TailSide",
    # Profiler
    "MetricProfiler",
    "MetricProfile",
    # Composite
    "CompositeMetricEngine",
    # Preprocessing
    "Preprocessor",
    "ChunkedPreprocessor",
    # Result builder
    "ResultBuilder",
    # Timeseries
    "TimeseriesEngine",
    "timeseries_engine",
    # Temporal composite
    "TemporalCompositeEngine",
    "temporal_composite_engine",
]