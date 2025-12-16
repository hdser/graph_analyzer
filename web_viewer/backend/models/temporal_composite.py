"""
Temporal Composite Models

Pydantic models for temporal composite metrics including:
- Temporal operations (velocity, acceleration, stability, etc.)
- Temporal composite configuration
- Pre-built temporal metric presets
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class TemporalOperation(str, Enum):
    """Available temporal operations."""
    
    # Rate of change operations
    VELOCITY = "velocity"  # First derivative: (current - past) / time
    ACCELERATION = "acceleration"  # Second derivative: change in velocity
    
    # Stability operations
    STABILITY = "stability"  # 1 - normalized_std (0=volatile, 1=stable)
    VOLATILITY = "volatility"  # Coefficient of variation over window
    
    # Trend operations
    MOMENTUM = "momentum"  # Weighted moving average trend indicator
    TREND_STRENGTH = "trend_strength"  # R-squared of linear fit
    
    # Age-based operations
    AGE = "age"  # Blocks since first appearance
    AGE_WEIGHTED = "age_weighted"  # Metric weighted by normalized age
    TENURE_RATIO = "tenure_ratio"  # Presence ratio (snapshots present / total)
    
    # Relative operations
    RELATIVE_TO_COHORT = "relative_to_cohort"  # Metric relative to same-age peers
    PERCENTILE_RANK = "percentile_rank"  # Percentile rank over time
    Z_SCORE_TEMPORAL = "z_score_temporal"  # Z-score relative to historical self
    
    # Composite temporal
    GROWTH_SCORE = "growth_score"  # Combines velocity + stability + age


class CohortReference(str, Enum):
    """Reference group for relative metrics."""
    ALL_NODES = "all"
    SAME_AGE = "same_age"  # Nodes that appeared in same snapshot
    RECENT = "recent"  # Nodes from last N snapshots
    SIMILAR_INITIAL = "similar_initial"  # Nodes with similar initial metric value


class CombineOperation(str, Enum):
    """How to combine temporal metric with another metric."""
    MULTIPLY = "multiply"
    ADD = "add"
    SUBTRACT = "subtract"
    DIVIDE = "divide"
    AVERAGE = "average"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    WEIGHTED_SUM = "weighted_sum"


# =============================================================================
# Configuration Models
# =============================================================================

class TemporalOperationConfig(BaseModel):
    """Configuration for a temporal operation."""
    operation: TemporalOperation
    
    # Window parameters
    window_blocks: int = Field(
        default=5,
        ge=2,
        le=50,
        description="Number of past snapshots to consider"
    )
    
    # Weighting for momentum/weighted operations
    decay_factor: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Exponential decay factor for weighted operations"
    )
    
    # Age weighting
    age_weight: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Weight factor for age-based operations"
    )
    
    # Cohort reference
    cohort_reference: CohortReference = Field(
        default=CohortReference.ALL_NODES,
        description="Reference group for relative metrics"
    )
    
    # Normalization
    normalize_output: bool = Field(
        default=True,
        description="Normalize output to [0, 1] range"
    )


class TemporalCompositeConfig(BaseModel):
    """Full configuration for creating a temporal composite metric."""
    
    # Identity
    name: str = Field(
        description="Name for the new metric",
        min_length=1,
        max_length=64
    )
    
    # Base metric
    base_metric: str = Field(description="Source metric to apply temporal operation to")
    
    # Temporal operation
    temporal_config: TemporalOperationConfig = Field(
        default_factory=TemporalOperationConfig
    )
    
    # Optional combination with another metric
    combine_with: Optional[str] = Field(
        default=None,
        description="Optional second metric to combine with"
    )
    combine_operation: CombineOperation = Field(
        default=CombineOperation.MULTIPLY
    )
    combine_weights: List[float] = Field(
        default=[0.5, 0.5],
        description="Weights for weighted_sum operation"
    )
    
    # Target snapshot
    base_sql_file: str
    target_block: int
    
    # Output options
    save: bool = Field(
        default=True,
        description="Whether to save to composite metrics cache"
    )


# =============================================================================
# Result Models
# =============================================================================

class TemporalMetricStatistics(BaseModel):
    """Statistics for a computed temporal metric."""
    count: int
    min: float
    max: float
    mean: float
    std: float
    median: float
    
    # Temporal-specific stats
    nodes_with_history: int = Field(description="Nodes with enough history for computation")
    nodes_without_history: int = Field(description="Nodes with insufficient history")
    
    # Distribution info
    q25: float
    q75: float
    skewness: Optional[float] = None


class TemporalCompositeResult(BaseModel):
    """Result of temporal composite computation."""
    
    # Identity
    name: str
    temporal_composite_id: str = Field(description="Unique ID for this computation")
    
    # Configuration used
    base_metric: str
    temporal_operation: TemporalOperation
    formula_description: str = Field(
        description="Human-readable formula, e.g., 'd(pagerank)/dt over 5 blocks'"
    )
    
    # Target info
    base_sql_file: str
    target_block: int
    target_timestamp: Optional[datetime] = None
    
    # Time window used
    blocks_used: List[int]
    time_span_seconds: Optional[float] = None
    snapshots_used: int = 0
    
    # Results
    node_count: int
    statistics: TemporalMetricStatistics
    
    # Per-node values (can be large, optional)
    values: Optional[Dict[str, float]] = Field(
        default=None,
        description="Node ID to metric value mapping"
    )
    
    # Histogram for visualization
    histogram_bins: List[float] = Field(default_factory=list)
    histogram_counts: List[int] = Field(default_factory=list)
    
    # Sample values (top and bottom)
    top_nodes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Nodes with highest values"
    )
    bottom_nodes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Nodes with lowest values"
    )
    
    # Performance
    computation_time_seconds: float
    
    # Storage
    saved: bool = False
    cache_key: Optional[str] = None


class TemporalPreviewResult(BaseModel):
    """Preview of temporal metric without saving."""
    
    name: str
    formula_description: str
    
    # Statistics
    statistics: TemporalMetricStatistics
    
    # Histogram for visualization
    histogram_bins: List[float]
    histogram_counts: List[int]
    
    # Sample values (top and bottom)
    top_nodes: List[Dict[str, Any]] = Field(
        description="Nodes with highest values"
    )
    bottom_nodes: List[Dict[str, Any]] = Field(
        description="Nodes with lowest values"
    )
    
    # Correlation with base metric
    correlation_with_base: float


# =============================================================================
# Preset Models
# =============================================================================

class TemporalPresetInfo(BaseModel):
    """Information about a pre-built temporal metric preset."""
    preset_id: str
    name: str
    display_name: str
    description: str
    
    # Configuration
    base_metric: str
    temporal_operation: TemporalOperation
    default_window: int
    
    # Use case
    category: str = Field(description="Category: 'growth', 'stability', 'influence', 'risk'")
    use_case: str = Field(description="When to use this metric")


class TemporalPresetsResponse(BaseModel):
    """Response listing available presets."""
    presets: List[TemporalPresetInfo]
    categories: List[str]


# =============================================================================
# Request Models
# =============================================================================

class TemporalComputeRequest(BaseModel):
    """Request to compute a temporal metric."""
    config: TemporalCompositeConfig


class TemporalPreviewRequest(BaseModel):
    """Request to preview a temporal metric."""
    config: TemporalCompositeConfig
    sample_size: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of top/bottom nodes to include"
    )


class TemporalPresetRequest(BaseModel):
    """Request to apply a preset temporal metric."""
    base_sql_file: str
    target_block: int
    window_blocks: Optional[int] = Field(
        default=None,
        description="Override default window size"
    )
    save: bool = True


# =============================================================================
# Batch Operations
# =============================================================================

class BatchTemporalConfig(BaseModel):
    """Configuration for computing temporal metrics across multiple snapshots."""
    base_sql_file: str
    block_numbers: List[int]
    
    # Which temporal metrics to compute
    configs: List[TemporalCompositeConfig]
    
    # Options
    save_all: bool = True
    compute_parallel: bool = False


class BatchTemporalResult(BaseModel):
    """Result of batch temporal computation."""
    base_sql_file: str
    blocks_processed: int
    metrics_computed: int
    
    # Results per block
    results_by_block: Dict[int, List[TemporalCompositeResult]]
    
    # Aggregate statistics
    total_computation_time_seconds: float
    failed_computations: List[Dict[str, Any]] = Field(default_factory=list)


# =============================================================================
# Available Operations Response
# =============================================================================

class TemporalOperationInfo(BaseModel):
    """Information about a temporal operation."""
    operation: TemporalOperation
    name: str
    description: str
    formula: str
    
    # Requirements
    min_window: int
    default_window: int
    requires_cohort: bool = False
    
    # Output characteristics
    output_range: str = Field(description="e.g., '[-inf, inf]', '[0, 1]'")
    interpretation: str


class AvailableOperationsResponse(BaseModel):
    """Response listing available temporal operations."""
    operations: List[TemporalOperationInfo]
    
    # Grouped by category
    rate_of_change: List[str]
    stability: List[str]
    age_based: List[str]
    relative: List[str]