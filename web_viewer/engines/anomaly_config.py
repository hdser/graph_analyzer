"""
Anomaly Detection Configuration

Dataclasses for configuring anomaly detection behavior.
All configuration is explicit, serializable, and metric-agnostic.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import json


class NaNStrategy(str, Enum):
    """Strategy for handling NaN values."""
    ZERO = "zero"
    MEAN = "mean"
    MEDIAN = "median"
    DROP = "drop"


class GlobalScaling(str, Enum):
    """Global scaling method for all metrics."""
    NONE = "none"
    STANDARD = "standard"      # (x - mean) / std
    ROBUST = "robust"          # (x - median) / IQR
    MINMAX = "minmax"          # (x - min) / (max - min)


class ScoreNormalization(str, Enum):
    """Method for normalizing anomaly scores."""
    MINMAX = "minmax"          # Scale to [0, 1]
    RANK = "rank"              # Percentile rank
    NONE = "none"              # Raw scores


class ThresholdMethod(str, Enum):
    """Method for determining anomaly threshold."""
    FIXED = "fixed"            # Use parameter directly
    PERCENTILE = "percentile"  # Top X% are anomalies
    AUTO = "auto"              # Algorithm-specific automatic


class AggregationMethod(str, Enum):
    """Method for aggregating per-metric scores."""
    MAX = "max"
    MEAN = "mean"
    L2 = "l2"                  # Euclidean norm
    WEIGHTED = "weighted"      # Use metric weights


class TailSide(str, Enum):
    """Which tail(s) to consider as anomalies."""
    BOTH = "both"
    HIGH = "high"              # Only high values
    LOW = "low"                # Only low values


@dataclass
class MetricTransform:
    """
    Per-metric preprocessing configuration.
    
    Attributes:
        log: Apply log1p transform (for right-skewed data)
        clip_min: Lower bound for clipping
        clip_max: Upper bound for clipping
        drop: Exclude this metric from analysis
        weight: Importance weight for weighted aggregation
        fill_value: Custom fill value for NaN (overrides global nan_strategy)
    """
    log: bool = False
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None
    drop: bool = False
    weight: float = 1.0
    fill_value: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'log': self.log,
            'clip_min': self.clip_min,
            'clip_max': self.clip_max,
            'drop': self.drop,
            'weight': self.weight,
            'fill_value': self.fill_value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricTransform':
        """Create from dictionary."""
        return cls(
            log=data.get('log', False),
            clip_min=data.get('clip_min'),
            clip_max=data.get('clip_max'),
            drop=data.get('drop', False),
            weight=data.get('weight', 1.0),
            fill_value=data.get('fill_value'),
        )


@dataclass
class MetricConfig:
    """
    Configuration for metric preprocessing in anomaly detection.
    
    This is completely metric-agnostic - no hardcoded column names.
    
    Attributes:
        id_column: Column name containing node identifiers
        group_by: Optional column for group-aware detection
        nan_strategy: How to handle NaN values globally
        per_metric: Per-metric preprocessing configuration
        global_scaling: Global scaling method applied after per-metric transforms
        min_group_size: Minimum samples required for group-aware detection
        use_float32: Use float32 for memory efficiency (slight precision loss)
    """
    id_column: str = "avatar"
    group_by: Optional[str] = None
    nan_strategy: NaNStrategy = NaNStrategy.ZERO
    per_metric: Dict[str, MetricTransform] = field(default_factory=dict)
    global_scaling: GlobalScaling = GlobalScaling.NONE
    min_group_size: int = 3
    use_float32: bool = False
    
    def get_metric_transform(self, metric: str) -> MetricTransform:
        """Get transform config for a metric, with defaults."""
        return self.per_metric.get(metric, MetricTransform())
    
    def set_metric_transform(self, metric: str, transform: MetricTransform) -> None:
        """Set transform config for a metric."""
        self.per_metric[metric] = transform
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id_column': self.id_column,
            'group_by': self.group_by,
            'nan_strategy': self.nan_strategy.value,
            'per_metric': {k: v.to_dict() for k, v in self.per_metric.items()},
            'global_scaling': self.global_scaling.value,
            'min_group_size': self.min_group_size,
            'use_float32': self.use_float32,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricConfig':
        """Create from dictionary."""
        per_metric = {}
        if 'per_metric' in data:
            for k, v in data['per_metric'].items():
                per_metric[k] = MetricTransform.from_dict(v) if isinstance(v, dict) else v
        
        return cls(
            id_column=data.get('id_column', 'avatar'),
            group_by=data.get('group_by'),
            nan_strategy=NaNStrategy(data.get('nan_strategy', 'zero')),
            per_metric=per_metric,
            global_scaling=GlobalScaling(data.get('global_scaling', 'none')),
            min_group_size=data.get('min_group_size', 3),
            use_float32=data.get('use_float32', False),
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MetricConfig':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class AlgorithmConfig:
    """
    Configuration for anomaly detection algorithm.
    
    Attributes:
        algorithm: Algorithm name (zscore, iqr, isolation_forest, lof, dbscan, mahalanobis)
        parameters: Algorithm-specific parameters
        top_n: Number of top anomalies to return in results
        score_normalization: How to normalize anomaly scores
        threshold_method: How to determine the anomaly threshold
        threshold_value: Value for threshold (interpretation depends on threshold_method)
    """
    algorithm: str = "isolation_forest"
    parameters: Dict[str, Any] = field(default_factory=dict)
    top_n: int = 20
    score_normalization: ScoreNormalization = ScoreNormalization.MINMAX
    threshold_method: ThresholdMethod = ThresholdMethod.FIXED
    threshold_value: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'algorithm': self.algorithm,
            'parameters': self.parameters,
            'top_n': self.top_n,
            'score_normalization': self.score_normalization.value,
            'threshold_method': self.threshold_method.value,
            'threshold_value': self.threshold_value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlgorithmConfig':
        """Create from dictionary."""
        return cls(
            algorithm=data.get('algorithm', 'isolation_forest'),
            parameters=data.get('parameters', {}),
            top_n=data.get('top_n', 20),
            score_normalization=ScoreNormalization(data.get('score_normalization', 'minmax')),
            threshold_method=ThresholdMethod(data.get('threshold_method', 'fixed')),
            threshold_value=data.get('threshold_value', 0.5),
        )


# Preset configurations for common use cases
ANOMALY_PRESETS: Dict[str, Dict[str, Any]] = {
    "quick_zscore": {
        "description": "Fast statistical detection using z-scores",
        "algorithm_config": AlgorithmConfig(
            algorithm="zscore",
            parameters={"threshold": 3.0, "aggregation": "max"},
            score_normalization=ScoreNormalization.MINMAX,
        ),
        "metric_config": MetricConfig(
            global_scaling=GlobalScaling.NONE,
        ),
    },
    "thorough_isolation_forest": {
        "description": "Comprehensive tree-based detection",
        "algorithm_config": AlgorithmConfig(
            algorithm="isolation_forest",
            parameters={"n_estimators": 200, "contamination": 0.1},
            score_normalization=ScoreNormalization.MINMAX,
        ),
        "metric_config": MetricConfig(
            global_scaling=GlobalScaling.STANDARD,
        ),
    },
    "density_based_lof": {
        "description": "Local density-based anomaly detection",
        "algorithm_config": AlgorithmConfig(
            algorithm="lof",
            parameters={"n_neighbors": 20, "contamination": 0.1},
            score_normalization=ScoreNormalization.MINMAX,
        ),
        "metric_config": MetricConfig(
            global_scaling=GlobalScaling.STANDARD,
        ),
    },
    "robust_mahalanobis": {
        "description": "Distance-based detection with correlation awareness",
        "algorithm_config": AlgorithmConfig(
            algorithm="mahalanobis",
            parameters={"alpha": 0.99, "robust": True},
            score_normalization=ScoreNormalization.MINMAX,
        ),
        "metric_config": MetricConfig(
            global_scaling=GlobalScaling.NONE,
        ),
    },
}


def get_preset(name: str) -> tuple:
    """
    Get a preset configuration by name.
    
    Returns:
        Tuple of (AlgorithmConfig, MetricConfig)
    """
    if name not in ANOMALY_PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(ANOMALY_PRESETS.keys())}")
    
    preset = ANOMALY_PRESETS[name]
    return preset["algorithm_config"], preset["metric_config"]


def list_presets() -> Dict[str, str]:
    """List all available presets with descriptions."""
    return {name: preset["description"] for name, preset in ANOMALY_PRESETS.items()}