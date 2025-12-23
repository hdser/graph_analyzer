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
    node_properties_files: List[str] = Field(
        default_factory=list,
        description="List of SQL files for fixed node properties (from properties directory)"
    )
    use_cached_layout: bool = Field(
        default=True,
        description="Whether to use cached layout if available"
    )
    skip_sql: bool = Field(
        default=False,
        description="Skip SQL execution and use cached data"
    )
    preset: Optional[str] = Field(
        default="basic",
        description="Metrics preset: basic, essential, moderate, comprehensive, all, trust_analysis, influence, structure"
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="Metrics categories to compute"
    )
    metrics: Optional[List[str]] = Field(
        default=None,
        description="Individual metrics to compute"
    )


class MetricsConfig(BaseModel):
    """Configuration for metrics computation with granular selection."""
    
    # Selection strategies (mutually exclusive, priority order: metrics > categories > preset)
    preset: Optional[str] = Field(
        default=None,
        description="Preset name: basic, essential, moderate, comprehensive, all, trust_analysis, influence, structure"
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="List of category names: topology, centrality, clustering, community, trust, etc."
    )
    metrics: Optional[List[str]] = Field(
        default=None,
        description="List of individual metric names: pagerank, betweenness_centrality, etc."
    )
    exclude_metrics: Optional[List[str]] = Field(
        default=None,
        description="Metrics to exclude from final selection"
    )
    
    # Parameters
    metric_parameters: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Per-metric parameter overrides: {metric_name: {param: value}}"
    )
    
    # Filters
    skip_expensive: bool = Field(
        default=False,
        description="Skip metrics with cost='very_high'"
    )
    
    # Target
    metrics_graph_id: Optional[str] = Field(
        default=None,
        description="Specific graph ID to compute metrics for (None = all)"
    )


class MetricTransformRequest(BaseModel):
    """Per-metric transform configuration for anomaly detection preprocessing."""
    log: bool = Field(False, description="Apply log1p transform")
    clip_min: Optional[float] = Field(None, description="Lower bound clipping")
    clip_max: Optional[float] = Field(None, description="Upper bound clipping")
    drop: bool = Field(False, description="Exclude from analysis")
    weight: float = Field(1.0, description="Importance weight for weighted aggregation")
    fill_value: Optional[float] = Field(None, description="Custom NaN fill value")


class MetricConfigRequest(BaseModel):
    """Metric preprocessing configuration for anomaly detection."""
    id_column: str = Field("avatar", description="Column containing node IDs")
    group_by: Optional[str] = Field(None, description="Column for group-aware detection")
    nan_strategy: str = Field("zero", description="NaN handling: zero, mean, median, drop")
    per_metric: Dict[str, MetricTransformRequest] = Field(
        default_factory=dict,
        description="Per-metric transform configurations"
    )
    global_scaling: str = Field("none", description="Global scaling: none, standard, robust, minmax")
    min_group_size: int = Field(3, description="Minimum samples for group detection")
    use_float32: bool = Field(False, description="Use float32 for memory efficiency")


class AlgorithmConfigRequest(BaseModel):
    """Full algorithm configuration for anomaly detection."""
    algorithm: str = Field("isolation_forest", description="Algorithm name")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Algorithm parameters")
    top_n: int = Field(20, description="Number of top anomalies to return")
    score_normalization: str = Field("minmax", description="Score normalization: minmax, rank, none")
    threshold_method: str = Field("fixed", description="Threshold method: fixed, percentile, auto")
    threshold_value: float = Field(0.5, description="Threshold value")


class AnomalyDetectionRequest(BaseModel):
    """Request for anomaly detection analysis."""
    name: str = Field(
        default="anomaly_score",
        description="Name for the anomaly score metric"
    )
    metrics: List[str] = Field(
        ...,
        description="List of metric names to analyze"
    )
    algorithm: str = Field(
        default="isolation_forest",
        description="Algorithm: zscore, iqr, isolation_forest, lof, dbscan, mahalanobis"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Algorithm-specific parameters"
    )
    config: Optional[MetricConfigRequest] = Field(
        default=None,
        description="Metric preprocessing configuration"
    )
    algorithm_config: Optional[AlgorithmConfigRequest] = Field(
        default=None,
        description="Full algorithm configuration (overrides algorithm and parameters)"
    )
    sample_size: Optional[int] = Field(
        default=None,
        description="Sample size for large datasets (None = use all data)"
    )
    apply_to_graph: bool = Field(
        default=True,
        description="Whether to apply scores as node attributes"
    )
    node_ids: Optional[List[str]] = Field(
        default=None,
        description="Filter analysis to specific node IDs (None = all nodes)"
    )


class PCARequest(BaseModel):
    """Request for PCA analysis."""
    metrics: List[str] = Field(
        ...,
        description="List of metric names for PCA"
    )
    n_components: str = Field(
        default="auto",
        description="Number of components: 'auto', '2', '3', '5', '10', or variance ratio like '0.95'"
    )
    standardize: bool = Field(
        default=True,
        description="Whether to standardize features before PCA"
    )
    node_ids: Optional[List[str]] = Field(
        default=None,
        description="Filter analysis to specific node IDs (None = all nodes)"
    )


class ProfileMetricsRequest(BaseModel):
    """Request to profile metrics for preprocessing recommendations."""
    metrics: List[str] = Field(..., description="List of metrics to profile")


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


class CompositePreviewRequest(BaseModel):
    """Request for composite metric preview (without saving)."""
    metrics: List[str] = Field(
        min_length=2,
        max_length=2,
        description="Two source metrics to combine"
    )
    operation: str = Field(
        default="multiply",
        description="Operation to apply"
    )
    weights: Optional[List[float]] = Field(
        default=None,
        description="Weights for weighted_sum operation"
    )
    normalize: bool = Field(
        default=False,
        description="Normalize inputs to [0,1] before operation"
    )
    node_ids: Optional[List[str]] = Field(
        default=None,
        description="Filter to specific node IDs (None = all nodes)"
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
    node_properties_files: Optional[List[str]] = Field(
        default=None,
        description="Node properties files to reload (None = use current)"
    )
    compute_metrics: bool = Field(
        default=True,
        description="Recompute metrics after reload"
    )
    preset: Optional[str] = Field(
        default="basic",
        description="Metrics preset for recomputation"
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="Metrics categories for recomputation"
    )
    metrics: Optional[List[str]] = Field(
        default=None,
        description="Individual metrics for recomputation"
    )