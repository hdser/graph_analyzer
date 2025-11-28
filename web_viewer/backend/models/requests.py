"""
Request Models

Pydantic models for API request validation.
"""

from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


class LoadConfig(BaseModel):
    """Configuration for loading network data."""
    sql_files: List[str] = Field(
        default_factory=list,
        description="List of SQL file names to execute"
    )
    use_cached_layout: bool = Field(
        default=True,
        description="Whether to use cached layout if available"
    )
    skip_sql: bool = Field(
        default=False,
        description="Skip SQL execution and use cached data"
    )
    metrics_mode: str = Field(
        default="basic",
        description="Metrics computation mode: basic, essential, standard, comprehensive, or full"
    )


class MetricsConfig(BaseModel):
    """Configuration for metrics computation."""
    metrics_mode: str = Field(
        default="basic",
        description="Metrics computation mode"
    )
    metrics_graph_id: Optional[str] = Field(
        default=None,
        description="Specific graph ID to compute metrics for (None = all)"
    )


class AnomalyDetectionConfig(BaseModel):
    """Configuration for anomaly detection."""
    name: str = Field(
        description="Name for the anomaly score metric"
    )
    metrics: List[str] = Field(
        description="List of metric names to analyze"
    )
    algorithm: str = Field(
        default="isolation_forest",
        description="Algorithm: zscore, iqr, isolation_forest, lof, dbscan, mahalanobis"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Algorithm-specific parameters"
    )
    apply_to_graph: bool = Field(
        default=True,
        description="Whether to apply scores as node attributes"
    )
    version: Optional[str] = Field(
        default=None,
        description="Graph version to apply to (v1, v2, etc.)"
    )


class CompositeMetricConfig(BaseModel):
    """Configuration for composite metric creation."""
    name: str = Field(
        description="Name for the composite metric"
    )
    metrics: List[str] = Field(
        min_length=2,
        max_length=2,
        description="Two source metrics to combine"
    )
    operation: str = Field(
        default="multiply",
        description="Operation: multiply, add, subtract, divide, maximum, minimum, average, weighted_sum, norm_multiply"
    )
    weights: Optional[List[float]] = Field(
        default=None,
        description="Weights for weighted_sum operation"
    )
    normalize: bool = Field(
        default=False,
        description="Normalize inputs to [0,1] before operation"
    )
    save: bool = Field(
        default=True,
        description="Save composite to cache for reuse"
    )
    version: Optional[str] = Field(
        default=None,
        description="Graph version identifier"
    )


class AutoReloadConfig(BaseModel):
    """Configuration for automatic background reloading."""
    enabled: bool = Field(
        default=True,
        description="Whether auto-reload is enabled"
    )
    interval_seconds: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Reload interval in seconds (60-3600)"
    )
    sql_files: Optional[List[str]] = Field(
        default=None,
        description="SQL files to reload (None = use current)"
    )
    compute_metrics: bool = Field(
        default=True,
        description="Recompute metrics after reload"
    )
    metrics_mode: str = Field(
        default="basic",
        description="Metrics mode for recomputation"
    )