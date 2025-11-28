"""
Response Models

Pydantic models for API responses.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from pydantic import BaseModel, Field


class NetworkState(BaseModel):
    """Response for network loading operation."""
    loaded_graphs: List[str] = Field(
        description="List of loaded graph IDs"
    )
    node_count: int = Field(
        description="Total number of nodes"
    )
    edge_count: int = Field(
        description="Total number of edges"
    )
    metrics_computed: List[str] = Field(
        default_factory=list,
        description="List of computed metric names"
    )
    computation_time: float = Field(
        description="Total computation time in seconds"
    )
    layout_computation_time: float = Field(
        default=0.0,
        description="Layout computation time in seconds"
    )
    layout_algorithm: str = Field(
        default="unknown",
        description="Algorithm used for layout"
    )
    layout_cached: bool = Field(
        default=False,
        description="Whether layout was loaded from cache"
    )
    data_source: str = Field(
        default="sql",
        description="Data source: sql or cache"
    )


class AlgorithmParameterResponse(BaseModel):
    """Parameter specification for an anomaly detection algorithm."""
    name: str
    type: str
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[List[Any]] = None
    description: str = ""


class AlgorithmInfoResponse(BaseModel):
    """Information about an anomaly detection algorithm."""
    name: str
    display_name: str
    description: str
    complexity: str
    multivariate: bool
    requires_sklearn: bool
    parameters: Dict[str, AlgorithmParameterResponse]


class MetricProfileResponse(BaseModel):
    """Profile of a single metric from profiling analysis."""
    name: str
    dtype: str
    n_samples: int
    n_unique: int
    n_missing: int
    n_zeros: int
    n_negative: int
    n_inf: int
    min: float
    max: float
    mean: float
    median: float
    std: float
    skewness: float
    kurtosis: float
    p25: float
    p75: float
    p95: float
    p99: float
    iqr: float
    suggested_transform: Dict[str, Any]
    warnings: List[str]


class ProfileMetricsResponse(BaseModel):
    """Response from metric profiling analysis."""
    profiles: Dict[str, MetricProfileResponse]
    suggested_config: Dict[str, Any]
    report: str


class ThresholdInfoResponse(BaseModel):
    """Information about how anomaly threshold was determined."""
    method: str
    value: float
    percentile: Optional[float] = None
    auto_reason: Optional[str] = None


class GroupAnomalyStatsResponse(BaseModel):
    """Statistics for a single group in group-aware anomaly detection."""
    group_value: Any
    n_samples: int
    n_anomalies: int
    anomaly_rate: float
    mean_score: float
    std_score: float
    threshold_used: float
    top_anomalies: List[Dict[str, Any]]


class AnomalyDetectionResponse(BaseModel):
    """Response for anomaly detection operation."""
    metric_name: str = Field(
        description="Name of the anomaly score metric"
    )
    algorithm: str = Field(
        description="Algorithm used for detection"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Algorithm parameters used"
    )
    metrics_used: List[str] = Field(
        default_factory=list,
        description="Input metrics analyzed"
    )
    threshold_info: ThresholdInfoResponse = Field(
        description="Threshold determination info"
    )
    n_anomalies: int = Field(
        description="Number of anomalies detected"
    )
    n_total: int = Field(
        description="Total number of nodes analyzed"
    )
    anomaly_percentage: float = Field(
        description="Percentage of nodes marked as anomalies"
    )
    computation_time: float = Field(
        description="Computation time in seconds"
    )
    statistics: Dict[str, float] = Field(
        default_factory=dict,
        description="Score statistics (min, max, mean, std, percentiles)"
    )
    top_anomalies: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top N most anomalous nodes with details"
    )
    group_results: Optional[Dict[str, GroupAnomalyStatsResponse]] = Field(
        default=None,
        description="Per-group results for group-aware detection"
    )
    preprocessing_stats: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Preprocessing statistics per metric"
    )
    node_updates: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Node attribute updates for frontend application"
    )


class CompositeMetricResult(BaseModel):
    """Response for composite metric creation."""
    metric_name: str = Field(
        description="Name of the composite metric"
    )
    formula: str = Field(
        description="Formula string (e.g., 'metric1 × metric2')"
    )
    node_updates: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Node attribute updates for frontend"
    )
    statistics: Dict[str, float] = Field(
        default_factory=dict,
        description="Statistics of computed metric"
    )
    saved: bool = Field(
        default=False,
        description="Whether composite was saved to cache"
    )
    composite_id: Optional[str] = Field(
        default=None,
        description="ID of saved composite"
    )


class SavedCompositeResponse(BaseModel):
    """Information about a saved composite metric."""
    id: str
    name: str
    formula: str
    operation: str
    source_metrics: List[str]
    normalize: bool
    created_at: str


class SavedCompositesListResponse(BaseModel):
    """List of saved composite metrics."""
    composites: List[SavedCompositeResponse]


class AutoReloadStatus(BaseModel):
    """Status of auto-reload feature."""
    enabled: bool = Field(
        description="Whether auto-reload is active"
    )
    interval_seconds: int = Field(
        description="Current reload interval"
    )
    last_reload_time: Optional[datetime] = Field(
        default=None,
        description="Time of last reload"
    )
    next_reload_time: Optional[datetime] = Field(
        default=None,
        description="Scheduled time of next reload"
    )
    reload_in_progress: bool = Field(
        default=False,
        description="Whether a reload is currently running"
    )
    current_node_count: int = Field(
        default=0,
        description="Current number of nodes"
    )
    last_reload_duration: Optional[float] = Field(
        default=None,
        description="Duration of last reload in seconds"
    )
    last_reload_nodes_added: int = Field(
        default=0,
        description="Nodes added in last reload"
    )
    last_reload_nodes_removed: int = Field(
        default=0,
        description="Nodes removed in last reload"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if last reload failed"
    )