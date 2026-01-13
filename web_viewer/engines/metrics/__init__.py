"""
Metrics Package

Modular graph metrics computation with granular metric selection.

Usage:
    from engines.metrics import MetricEngine, METRIC_REGISTRY, METRIC_CATEGORIES, METRIC_PRESETS
    
    engine = MetricEngine(graph)
    df = engine.compute(metrics=["pagerank", "eigentrust"])
    # or
    df = engine.compute(categories=["centrality", "trust"])
    # or
    df = engine.compute(preset="essential")
"""

from .registry import (
    MetricDefinition,
    METRIC_REGISTRY,
    METRIC_CATEGORIES,
    METRIC_PRESETS,
    get_metric,
    get_category_metrics,
    get_preset_metrics,
    list_all_metrics,
    list_categories,
    list_presets,
)
from .resolver import MetricResolver
from .computer import MetricComputer, MetricEngine

__all__ = [
    # Core classes
    "MetricDefinition",
    "MetricResolver", 
    "MetricComputer",
    "MetricEngine",
    # Registry data
    "METRIC_REGISTRY",
    "METRIC_CATEGORIES", 
    "METRIC_PRESETS",
    # Helper functions
    "get_metric",
    "get_category_metrics",
    "get_preset_metrics",
    "list_all_metrics",
    "list_categories",
    "list_presets",
]