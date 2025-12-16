"""
Timeseries Models

Pydantic models for timeseries analysis including:
- Metric timeseries data
- Node trajectories
- Trend analysis
- Distribution comparisons
- Cohort analysis
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class AggregationType(str, Enum):
    """Aggregation methods for timeseries."""
    MEAN = "mean"
    MEDIAN = "median"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    STD = "std"
    COUNT = "count"
    P10 = "p10"
    P25 = "p25"
    P75 = "p75"
    P90 = "p90"


class TrendDirection(str, Enum):
    """Direction of a trend."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


class CohortType(str, Enum):
    """Types of cohort definitions."""
    FIRST_SEEN_BLOCK = "first_seen_block"
    FIRST_SEEN_DATE = "first_seen_date"
    METRIC_RANGE = "metric_range"
    CUSTOM = "custom"


# =============================================================================
# Data Point Models
# =============================================================================

class TimeseriesPoint(BaseModel):
    """Single point in a timeseries."""
    block_number: int
    timestamp: Optional[datetime] = None
    value: float
    
    # Optional metadata
    node_count: Optional[int] = None  # For aggregated metrics
    sample_size: Optional[int] = None


class TrajectoryPoint(BaseModel):
    """Single point in a node's trajectory."""
    block_number: int
    timestamp: Optional[datetime] = None
    value: Optional[float] = None  # None if node didn't exist at this point
    exists: bool = True  # Whether node existed in this snapshot


class NetworkSummaryPoint(BaseModel):
    """Network-level statistics at a point in time."""
    block_number: int
    timestamp: Optional[datetime] = None
    
    node_count: int
    edge_count: int
    density: float
    
    # Component stats
    num_weakly_connected: Optional[int] = None
    largest_wcc_size: Optional[int] = None
    num_strongly_connected: Optional[int] = None
    largest_scc_size: Optional[int] = None
    
    # Optional derived stats
    avg_degree: Optional[float] = None
    avg_clustering: Optional[float] = None


# =============================================================================
# Statistics Models
# =============================================================================

class TimeseriesStatistics(BaseModel):
    """Statistics for a timeseries."""
    count: int = Field(description="Number of data points")
    min: float
    max: float
    mean: float
    std: float
    
    # Trend info
    first_value: float
    last_value: float
    total_change: float
    percent_change: float
    
    # Volatility
    coefficient_of_variation: Optional[float] = None


class TrendAnalysis(BaseModel):
    """Detailed trend analysis for a metric timeseries."""
    metric: str
    aggregation: AggregationType
    
    # Basic trend
    trend_direction: TrendDirection
    slope: float = Field(description="Linear regression slope")
    intercept: float
    r_squared: float = Field(description="Coefficient of determination")
    
    # Statistical significance
    p_value: float
    is_significant: bool = Field(description="p_value < 0.05")
    
    # Change metrics
    absolute_change: float
    percent_change: float
    
    # Time range
    start_block: int
    end_block: int
    num_points: int
    
    # Volatility assessment
    volatility: float = Field(description="Standard deviation of changes")
    max_drawdown: Optional[float] = None


class HistogramData(BaseModel):
    """Histogram bin data for a distribution."""
    bins: List[float] = Field(description="Bin edges (n+1 values for n bins)")
    counts: List[int] = Field(description="Count in each bin (n values)")
    

class DistributionComparison(BaseModel):
    """Comparison of metric distributions between two snapshots."""
    metric: str
    from_block: int
    to_block: int
    from_timestamp: Optional[datetime] = None
    to_timestamp: Optional[datetime] = None
    
    # Sample sizes
    from_count: int
    to_count: int
    
    # Histogram data for visualization
    from_histogram: Optional[HistogramData] = None
    to_histogram: Optional[HistogramData] = None
    
    # Kolmogorov-Smirnov test
    ks_statistic: float
    ks_pvalue: float
    distributions_differ: bool = Field(description="KS test significant at p<0.05")
    
    # Location shifts
    mean_shift: float
    mean_shift_percent: float
    median_shift: float
    median_shift_percent: float
    
    # Spread changes
    std_change: float
    std_change_percent: float
    
    # Percentile changes
    percentile_changes: Dict[str, float] = Field(
        default_factory=dict,
        description="Changes in p10, p25, p50, p75, p90"
    )
    
    # Shape changes
    skewness_change: Optional[float] = None
    kurtosis_change: Optional[float] = None


# =============================================================================
# Response Models
# =============================================================================

class TimeseriesData(BaseModel):
    """Timeseries data for a metric."""
    base_sql_file: str
    metric: str
    aggregation: AggregationType
    
    # Data points
    data_points: List[TimeseriesPoint]
    
    # Statistics
    statistics: TimeseriesStatistics
    
    # Trend (optional)
    trend: Optional[TrendAnalysis] = None
    
    # Metadata
    snapshots_included: int
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None


class NetworkTimeseriesData(BaseModel):
    """Network-level statistics over time."""
    base_sql_file: str
    
    # Data points
    data_points: List[NetworkSummaryPoint]
    
    # Growth statistics
    node_growth_rate: float = Field(description="Average nodes added per snapshot")
    edge_growth_rate: float = Field(description="Average edges added per snapshot")
    
    # Time range
    snapshots_included: int
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None


class NodeTrajectory(BaseModel):
    """Single node's metric values over time."""
    node_id: str
    metric: str
    
    # Trajectory data
    values: List[TrajectoryPoint]
    
    # Node lifecycle
    first_seen_block: int
    first_seen_timestamp: Optional[datetime] = None
    last_seen_block: int
    last_seen_timestamp: Optional[datetime] = None
    
    # Existence stats
    snapshots_present: int
    snapshots_missing: int
    
    # Value statistics (only for points where node exists)
    statistics: Optional[TimeseriesStatistics] = None
    trend: Optional[TrendAnalysis] = None


class NodeTrajectoriesResponse(BaseModel):
    """Response containing multiple node trajectories."""
    base_sql_file: str
    metric: str
    
    trajectories: Dict[str, NodeTrajectory]
    
    # Summary
    nodes_requested: int
    nodes_found: int
    nodes_not_found: List[str] = Field(default_factory=list)
    
    # Common time range
    block_numbers: List[int]
    timestamps: List[Optional[datetime]]


# =============================================================================
# Cohort Analysis Models
# =============================================================================

class CohortDefinition(BaseModel):
    """Definition of a cohort for analysis."""
    cohort_type: CohortType
    name: str = Field(description="Human-readable cohort name")
    
    # For first_seen cohorts
    first_seen_block_start: Optional[int] = None
    first_seen_block_end: Optional[int] = None
    first_seen_date_start: Optional[datetime] = None
    first_seen_date_end: Optional[datetime] = None
    
    # For metric-based cohorts
    metric_name: Optional[str] = None
    metric_min: Optional[float] = None
    metric_max: Optional[float] = None
    metric_at_block: Optional[int] = None  # When to evaluate the metric
    
    # For custom cohorts
    node_ids: Optional[List[str]] = None


class CohortStatistics(BaseModel):
    """Statistics for a cohort at a point in time."""
    block_number: int
    timestamp: Optional[datetime] = None
    
    # Cohort membership
    cohort_size: int
    still_active: int  # Nodes still present in network
    churned: int  # Nodes no longer present
    
    # Metric statistics
    metric_mean: Optional[float] = None
    metric_median: Optional[float] = None
    metric_std: Optional[float] = None


class CohortTrajectory(BaseModel):
    """Trajectory of a cohort over time."""
    cohort_definition: CohortDefinition
    metric: str
    aggregation: AggregationType
    
    # Trajectory data
    data_points: List[CohortStatistics]
    
    # Cohort info
    initial_size: int
    final_size: int
    retention_rate: float
    
    # Metric trends for the cohort
    trend: Optional[TrendAnalysis] = None


class CohortComparisonResponse(BaseModel):
    """Comparison of multiple cohorts."""
    base_sql_file: str
    metric: str
    
    cohorts: List[CohortTrajectory]
    
    # Comparative statistics
    best_performing_cohort: str
    worst_performing_cohort: str
    highest_retention_cohort: str


# =============================================================================
# Request Models
# =============================================================================

class TimeseriesRequest(BaseModel):
    """Request for timeseries data."""
    metric: str
    aggregation: AggregationType = AggregationType.MEAN
    include_trend: bool = True
    
    # Optional filtering
    start_block: Optional[int] = None
    end_block: Optional[int] = None
    max_points: Optional[int] = None


class NodeTrajectoriesRequest(BaseModel):
    """Request for node trajectories."""
    node_ids: List[str] = Field(max_length=100)
    metric: str
    include_statistics: bool = True
    include_trend: bool = False


class DistributionComparisonRequest(BaseModel):
    """Request for distribution comparison."""
    metric: str
    from_block: int
    to_block: int
    percentiles: List[int] = Field(default=[10, 25, 50, 75, 90])


class CohortAnalysisRequest(BaseModel):
    """Request for cohort analysis."""
    cohort_definitions: List[CohortDefinition]
    metric: str
    aggregation: AggregationType = AggregationType.MEAN
    include_trends: bool = True