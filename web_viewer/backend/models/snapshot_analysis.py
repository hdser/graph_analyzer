"""
Snapshot Analysis Models

Pydantic models for snapshot analysis functionality including:
- Analysis configuration
- Analysis results
- Metric statistics
- Anomaly summaries
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class AnalysisStatus(str, Enum):
    """Status of an analysis operation."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Statistics Models
# =============================================================================

class MetricStatistics(BaseModel):
    """Statistics for a single metric."""
    metric_name: str
    count: int = Field(description="Number of non-null values")
    min: float
    max: float
    mean: float
    std: float
    median: float
    q25: float = Field(description="25th percentile")
    q75: float = Field(description="75th percentile")
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None


class AnomalyResultSummary(BaseModel):
    """Summary of anomaly detection results."""
    algorithm: str
    metrics_used: List[str]
    parameters: Dict[str, Any]
    
    total_nodes: int
    anomaly_count: int
    anomaly_percentage: float
    
    threshold_method: str
    threshold_value: float
    
    # Score distribution
    score_min: float
    score_max: float
    score_mean: float
    score_std: float
    
    # Top anomalies (node IDs with highest scores)
    top_anomaly_ids: List[str] = Field(default_factory=list, max_length=100)
    
    computation_time_seconds: float


# =============================================================================
# Request Models
# =============================================================================

class SnapshotAnalysisConfig(BaseModel):
    """Configuration for running analysis on a snapshot."""
    
    # Metrics configuration
    metrics_mode: str = Field(
        default="essential",
        description="Metrics computation mode: 'basic', 'essential', 'moderate', 'all'"
    )
    recompute_metrics: bool = Field(
        default=False,
        description="Force recompute metrics even if already computed"
    )
    
    # Anomaly detection configuration
    run_anomaly_detection: bool = Field(
        default=False,
        description="Whether to run anomaly detection"
    )
    anomaly_algorithm: str = Field(
        default="isolation_forest",
        description="Algorithm: 'zscore', 'iqr', 'isolation_forest', 'lof', 'dbscan', 'mahalanobis'"
    )
    anomaly_metrics: List[str] = Field(
        default_factory=list,
        description="Metrics to use for anomaly detection (empty = use defaults)"
    )
    anomaly_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Algorithm-specific parameters"
    )
    anomaly_threshold_method: str = Field(
        default="percentile",
        description="Threshold method: 'percentile', 'std', 'iqr', 'fixed'"
    )
    anomaly_threshold_value: float = Field(
        default=95.0,
        description="Threshold value (interpretation depends on method)"
    )
    
    # Storage options
    save_results: bool = Field(
        default=True,
        description="Whether to save analysis results to disk"
    )
    save_per_node_data: bool = Field(
        default=True,
        description="Whether to save per-node metric values"
    )


class BatchAnalysisConfig(BaseModel):
    """Configuration for batch analysis of multiple snapshots."""
    base_sql_file: str = Field(description="Base SQL file name")
    block_numbers: List[int] = Field(description="List of block numbers to analyze")
    config: SnapshotAnalysisConfig = Field(
        default_factory=SnapshotAnalysisConfig,
        description="Analysis configuration to apply to all snapshots"
    )
    parallel: bool = Field(
        default=False,
        description="Whether to run analyses in parallel (uses more memory)"
    )


# =============================================================================
# Response Models
# =============================================================================

class SnapshotAnalysisResult(BaseModel):
    """Complete results from snapshot analysis."""
    
    # Identification
    snapshot_id: str
    base_sql_file: str
    block_number: int
    block_timestamp: Optional[datetime] = None
    
    # Analysis metadata
    analysis_id: str = Field(description="Unique ID for this analysis run")
    analysis_timestamp: datetime
    analysis_config: SnapshotAnalysisConfig
    status: AnalysisStatus
    
    # Network summary
    node_count: int
    edge_count: int
    
    # Metrics results
    metrics_computed: List[str] = Field(default_factory=list)
    metric_statistics: Dict[str, MetricStatistics] = Field(default_factory=dict)
    
    # Anomaly results (if run)
    anomaly_results: Optional[AnomalyResultSummary] = None
    
    # Performance
    computation_time_seconds: float
    metrics_computation_time: float = 0.0
    anomaly_computation_time: float = 0.0
    
    # Error information (if failed)
    error_message: Optional[str] = None


class AnalysisProgressUpdate(BaseModel):
    """Progress update for long-running analysis operations."""
    snapshot_id: str
    status: AnalysisStatus
    stage: str = Field(description="Current stage: 'loading', 'metrics', 'anomaly', 'saving'")
    progress_percent: int = Field(ge=0, le=100)
    message: str
    current_metric: Optional[str] = None
    elapsed_seconds: float = 0.0


class BatchAnalysisResult(BaseModel):
    """Result of batch analysis operation."""
    base_sql_file: str
    total_requested: int
    total_completed: int
    total_failed: int
    
    results: List[SnapshotAnalysisResult] = Field(default_factory=list)
    failed_snapshots: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of {block_number, error} for failed analyses"
    )
    
    total_computation_time_seconds: float


class AnalyzedSnapshotInfo(BaseModel):
    """Information about a snapshot that has been analyzed."""
    snapshot_id: str
    block_number: int
    block_timestamp: Optional[datetime] = None
    
    # Analysis summary
    has_analysis: bool = True
    analysis_timestamp: datetime
    metrics_computed: List[str]
    has_anomaly_results: bool
    anomaly_count: Optional[int] = None
    
    # Quick stats
    node_count: int
    edge_count: int


class AnalyzedSnapshotsListResponse(BaseModel):
    """Response for listing analyzed snapshots."""
    base_sql_file: str
    snapshots: List[AnalyzedSnapshotInfo]
    total_count: int


class MetricValuesResponse(BaseModel):
    """Response for getting metric values from a snapshot."""
    snapshot_id: str
    metric_name: str
    node_count: int
    
    # Statistics
    statistics: MetricStatistics
    
    # Values (optional, can be large)
    values: Optional[Dict[str, float]] = Field(
        default=None,
        description="Node ID to metric value mapping"
    )
    
    # Histogram data for visualization
    histogram_bins: List[float] = Field(default_factory=list)
    histogram_counts: List[int] = Field(default_factory=list)


# =============================================================================
# Internal Storage Models
# =============================================================================

class AnalysisMetadata(BaseModel):
    """Metadata stored in analysis_meta.json for each snapshot analysis."""
    analysis_id: str
    snapshot_id: str
    base_sql_file: str
    block_number: int
    
    # Configuration used
    config: SnapshotAnalysisConfig
    
    # Results summary
    status: AnalysisStatus
    metrics_computed: List[str]
    has_anomaly_results: bool
    anomaly_count: Optional[int] = None
    
    # Timestamps
    analysis_started: datetime
    analysis_completed: Optional[datetime] = None
    
    # Performance
    computation_time_seconds: float
    
    # File references
    files: Dict[str, str] = Field(
        default_factory=lambda: {
            "full_metrics": "full_metrics.parquet",
            "anomaly_results": "anomaly_results.json",
            "metric_statistics": "metric_statistics.json"
        }
    )