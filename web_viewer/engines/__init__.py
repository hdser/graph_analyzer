"""
Engines Package

Main package for graph analysis engines including:
- Anomaly detection
- Composite metrics
- Graph metrics computation
- Timeseries analysis
- Temporal composite analysis
"""

# Metrics engine (new modular system)
from .metrics import (
    MetricEngine,
    MetricComputer,
    MetricResolver,
    METRIC_REGISTRY,
    METRIC_CATEGORIES,
    METRIC_PRESETS,
    list_all_metrics,
    list_categories,
    list_presets,
    get_metric,
    get_category_metrics,
    get_preset_metrics,
)

# Anomaly detection configuration
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
    ANOMALY_PRESETS,
)

# Anomaly detection engine
from .anomaly_engine import AnomalyEngine

# Composite metrics engine
from .composite_engine import CompositeMetricEngine

# Metric profiler
from .metric_profiler import MetricProfiler, MetricProfile

# Preprocessing
from .preprocessing import Preprocessor, ChunkedPreprocessor

# Result building
from .result_builder import AnomalyResult, ResultBuilder, GroupAnomalyStats

# Parallel processing
from .parallel import ParallelExecutor, get_optimal_workers

# Timeseries engine
from .timeseries_engine import TimeseriesEngine

# Temporal composite engine
from .temporal_composite_engine import TemporalCompositeEngine


__all__ = [
    # Metrics
    "MetricEngine",
    "MetricComputer",
    "MetricResolver",
    "METRIC_REGISTRY",
    "METRIC_CATEGORIES",
    "METRIC_PRESETS",
    "list_all_metrics",
    "list_categories",
    "list_presets",
    "get_metric",
    "get_category_metrics",
    "get_preset_metrics",
    # Anomaly config
    "MetricConfig",
    "MetricTransform",
    "AlgorithmConfig",
    "NaNStrategy",
    "GlobalScaling",
    "ScoreNormalization",
    "ThresholdMethod",
    "AggregationMethod",
    "TailSide",
    "ANOMALY_PRESETS",
    # Engines
    "AnomalyEngine",
    "CompositeMetricEngine",
    "MetricProfiler",
    "MetricProfile",
    "TimeseriesEngine",
    "TemporalCompositeEngine",
    # Preprocessing
    "Preprocessor",
    "ChunkedPreprocessor",
    # Results
    "AnomalyResult",
    "ResultBuilder",
    "GroupAnomalyStats",
    # Parallel
    "ParallelExecutor",
    "get_optimal_workers",
]