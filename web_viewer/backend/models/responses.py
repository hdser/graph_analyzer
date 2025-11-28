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


class AnomalyDetectionResult(BaseModel):
    """Response for anomaly detection operation."""
    metric_name: str = Field(
        description="Name of the anomaly score metric"
    )
    algorithm: str = Field(
        description="Algorithm used"
    )
    n_anomalies: int = Field(
        description="Number of anomalies detected"
    )
    n_total: int = Field(
        description="Total number of nodes analyzed"
    )
    anomaly_percentage: float = Field(
        description="Percentage of anomalies"
    )
    computation_time: float = Field(
        description="Computation time in seconds"
    )
    top_anomalies: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top N most anomalous nodes"
    )
    score_statistics: Dict[str, float] = Field(
        default_factory=dict,
        description="Score statistics (min, max, mean, std, percentiles)"
    )
    metrics_used: List[str] = Field(
        default_factory=list,
        description="Input metrics analyzed"
    )
    parameters_used: Dict[str, Any] = Field(
        default_factory=dict,
        description="Algorithm parameters used"
    )
    node_updates: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Node attribute updates for frontend"
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