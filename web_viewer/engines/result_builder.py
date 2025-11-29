"""
Anomaly Detection Result Builder

Constructs AnomalyResult objects from raw algorithm outputs.
Handles score normalization, statistics computation, and result formatting.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .anomaly_config import ScoreNormalization, ThresholdMethod


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to a JSON-compatible float.
    
    Handles NaN, inf, -inf, and numpy types.
    """
    if value is None:
        return default
    try:
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def _sanitize_dict(d: Dict[str, Any], default: float = 0.0) -> Dict[str, Any]:
    """
    Recursively sanitize a dictionary to ensure all float values are JSON-compatible.
    """
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result[key] = _sanitize_dict(value, default)
        elif isinstance(value, (float, np.floating)):
            result[key] = _safe_float(value, default)
        elif isinstance(value, (int, np.integer)):
            result[key] = int(value)
        elif isinstance(value, np.ndarray):
            result[key] = [_safe_float(v, default) for v in value.flatten()]
        elif isinstance(value, list):
            result[key] = [
                _sanitize_dict(v, default) if isinstance(v, dict)
                else _safe_float(v, default) if isinstance(v, (float, np.floating))
                else v
                for v in value
            ]
        else:
            result[key] = value
    return result


@dataclass
class ThresholdInfo:
    """Information about how the anomaly threshold was determined."""
    method: str
    value: float
    percentile: Optional[float] = None
    auto_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return _sanitize_dict({
            'method': self.method,
            'value': self.value,
            'percentile': self.percentile,
            'auto_reason': self.auto_reason,
        })


@dataclass
class PreprocessingStats:
    """Statistics about preprocessing applied to a metric."""
    original_dtype: str
    n_missing: int
    n_inf: int
    n_zeros: int
    transform_applied: List[str]
    original_range: Tuple[float, float]
    final_range: Tuple[float, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return _sanitize_dict({
            'original_dtype': self.original_dtype,
            'n_missing': self.n_missing,
            'n_inf': self.n_inf,
            'n_zeros': self.n_zeros,
            'transform_applied': self.transform_applied,
            'original_range': [_safe_float(v) for v in self.original_range],
            'final_range': [_safe_float(v) for v in self.final_range],
        })


@dataclass
class GroupAnomalyStats:
    """Statistics for a single group in group-aware detection."""
    group_value: Any
    n_samples: int
    n_anomalies: int
    anomaly_rate: float
    mean_score: float
    std_score: float
    threshold_used: float
    top_anomalies: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return _sanitize_dict({
            'group_value': self.group_value,
            'n_samples': self.n_samples,
            'n_anomalies': self.n_anomalies,
            'anomaly_rate': self.anomaly_rate,
            'mean_score': self.mean_score,
            'std_score': self.std_score,
            'threshold_used': self.threshold_used,
            'top_anomalies': self.top_anomalies,
        })


@dataclass
class AnomalyResult:
    """
    Result of anomaly detection.
    
    Enhanced with additional diagnostic information.
    """
    # Core results
    scores: Dict[str, float]
    binary_labels: Dict[str, bool]
    
    # Algorithm info
    algorithm: str
    parameters: Dict[str, Any]
    metrics_used: List[str]
    
    # Threshold info
    threshold_info: ThresholdInfo
    
    # Summary
    n_anomalies: int
    n_total: int
    computation_time: float
    
    # Statistics
    statistics: Dict[str, Any]
    top_anomalies: List[Dict[str, Any]]
    
    # Extended info
    raw_scores: Optional[Dict[str, float]] = None
    per_metric_scores: Optional[Dict[str, Dict[str, float]]] = None
    group_results: Optional[Dict[str, GroupAnomalyStats]] = None
    preprocessing_stats: Optional[Dict[str, PreprocessingStats]] = None
    
    @property
    def anomaly_rate(self) -> float:
        """Percentage of anomalies."""
        if self.n_total == 0:
            return 0.0
        return 100.0 * self.n_anomalies / self.n_total
    
    def get_anomaly_ids(self) -> List[str]:
        """Get list of anomaly node IDs."""
        return [k for k, v in self.binary_labels.items() if v]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Sanitize scores dict
        sanitized_scores = {k: _safe_float(v) for k, v in self.scores.items()}
        
        result = {
            'scores': sanitized_scores,
            'binary_labels': self.binary_labels,
            'algorithm': self.algorithm,
            'parameters': _sanitize_dict(self.parameters),
            'metrics_used': self.metrics_used,
            'threshold_info': self.threshold_info.to_dict(),
            'n_anomalies': self.n_anomalies,
            'n_total': self.n_total,
            'anomaly_rate': _safe_float(self.anomaly_rate),
            'computation_time': _safe_float(self.computation_time),
            'statistics': _sanitize_dict(self.statistics),
            'top_anomalies': [_sanitize_dict(a) for a in self.top_anomalies],
        }
        
        if self.raw_scores is not None:
            result['raw_scores'] = {k: _safe_float(v) for k, v in self.raw_scores.items()}
        
        if self.per_metric_scores is not None:
            result['per_metric_scores'] = {
                metric: {k: _safe_float(v) for k, v in scores.items()}
                for metric, scores in self.per_metric_scores.items()
            }
        
        if self.group_results is not None:
            result['group_results'] = {
                k: v.to_dict() for k, v in self.group_results.items()
            }
        
        if self.preprocessing_stats is not None:
            result['preprocessing_stats'] = {
                k: v.to_dict() for k, v in self.preprocessing_stats.items()
            }
        
        return result


class ResultBuilder:
    """
    Builds AnomalyResult objects from raw detection outputs.
    
    Handles:
    - Score normalization (minmax, rank, none)
    - Statistics computation (vectorized)
    - Top anomalies extraction
    - Threshold determination
    """
    
    def __init__(self):
        pass
    
    def build(
        self,
        df: pd.DataFrame,
        avatars: List[str],
        raw_scores: np.ndarray,
        anomaly_mask: np.ndarray,
        algorithm: str,
        params: Dict[str, Any],
        metrics: List[str],
        score_normalization: ScoreNormalization,
        threshold_method: ThresholdMethod,
        threshold_value: float,
        computation_time: float,
        top_n: int = 20,
        per_metric_scores: Optional[Dict[str, np.ndarray]] = None,
        preprocessing_stats: Optional[Dict[str, PreprocessingStats]] = None,
        group_results: Optional[Dict[str, GroupAnomalyStats]] = None,
    ) -> AnomalyResult:
        """
        Build AnomalyResult from raw algorithm output.
        
        Args:
            df: Original DataFrame (for extracting metric values)
            avatars: List of node IDs
            raw_scores: Raw anomaly scores from algorithm
            anomaly_mask: Boolean array indicating anomalies
            algorithm: Algorithm name
            params: Algorithm parameters used
            metrics: List of metric names used
            score_normalization: How to normalize scores
            threshold_method: How threshold was determined
            threshold_value: Threshold value used
            computation_time: Time taken for computation
            top_n: Number of top anomalies to include
            per_metric_scores: Optional per-metric score breakdown
            preprocessing_stats: Optional preprocessing statistics
            group_results: Optional group-aware results
            
        Returns:
            Complete AnomalyResult object
        """
        # Clean raw scores first - replace NaN/inf with 0
        raw_scores = np.asarray(raw_scores, dtype=np.float64)
        raw_scores = np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize scores
        normalized_scores = self._normalize_scores(raw_scores, score_normalization)
        
        # Build dictionaries with safe float conversion
        score_dict = {
            avatar: _safe_float(score) 
            for avatar, score in zip(avatars, normalized_scores)
        }
        
        label_dict = {
            avatar: bool(label) 
            for avatar, label in zip(avatars, anomaly_mask)
        }
        
        # Raw scores dict (before normalization)
        raw_score_dict = {
            avatar: _safe_float(score) 
            for avatar, score in zip(avatars, raw_scores)
        }
        
        # Compute statistics
        statistics = self._compute_statistics(normalized_scores)
        
        # Build threshold info
        threshold_info = self._build_threshold_info(
            threshold_method, threshold_value, normalized_scores, anomaly_mask
        )
        
        # Extract top anomalies with metric values
        top_anomalies = self._extract_top_anomalies(
            df, avatars, normalized_scores, anomaly_mask, metrics, top_n
        )
        
        # Build per-metric score dicts if provided
        per_metric_score_dicts = None
        if per_metric_scores is not None:
            per_metric_score_dicts = {}
            for metric, scores in per_metric_scores.items():
                # Clean per-metric scores
                clean_scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
                per_metric_score_dicts[metric] = {
                    avatar: _safe_float(score) 
                    for avatar, score in zip(avatars, clean_scores)
                }
        
        return AnomalyResult(
            scores=score_dict,
            binary_labels=label_dict,
            algorithm=algorithm,
            parameters=params,
            metrics_used=metrics,
            threshold_info=threshold_info,
            n_anomalies=int(np.sum(anomaly_mask)),
            n_total=len(avatars),
            computation_time=_safe_float(computation_time),
            statistics=statistics,
            top_anomalies=top_anomalies,
            raw_scores=raw_score_dict,
            per_metric_scores=per_metric_score_dicts,
            group_results=group_results,
            preprocessing_stats=preprocessing_stats,
        )
    
    def _normalize_scores(
        self, 
        scores: np.ndarray, 
        method: ScoreNormalization
    ) -> np.ndarray:
        """
        Normalize scores to [0, 1] range.
        
        Args:
            scores: Raw anomaly scores
            method: Normalization method
            
        Returns:
            Normalized scores
        """
        scores = np.asarray(scores, dtype=np.float64)
        
        # Clean NaN/inf values first
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        
        if method == ScoreNormalization.NONE:
            return scores
        
        if method == ScoreNormalization.RANK:
            # Percentile rank (0 to 1)
            ranked = scipy_stats.rankdata(scores, method='average') / len(scores)
            return np.nan_to_num(ranked, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Default: minmax normalization
        if np.std(scores) < 1e-10:
            return np.zeros_like(scores)
        
        min_val = np.min(scores)
        max_val = np.max(scores)
        
        if max_val - min_val < 1e-10:
            return np.zeros_like(scores)
        
        normalized = (scores - min_val) / (max_val - min_val)
        normalized = np.clip(normalized, 0.0, 1.0)
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    def _compute_statistics(self, scores: np.ndarray) -> Dict[str, Any]:
        """
        Compute summary statistics for anomaly scores.
        
        All operations are vectorized for performance.
        All outputs are sanitized to be JSON-compatible.
        """
        scores = np.asarray(scores, dtype=np.float64)
        
        # Clean NaN/inf values
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Handle empty case
        if len(scores) == 0:
            return {
                'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0, 'median': 0.0,
                'p25': 0.0, 'p75': 0.0, 'p90': 0.0, 'p95': 0.0, 'p99': 0.0,
                'skewness': 0.0, 'kurtosis': 0.0,
            }
        
        # Compute percentiles in single call
        percentiles = np.percentile(scores, [25, 50, 75, 90, 95, 99])
        
        # Compute skewness and kurtosis with NaN handling
        skewness = 0.0
        kurtosis = 0.0
        if len(scores) > 2:
            try:
                skew_val = scipy_stats.skew(scores)
                skewness = _safe_float(skew_val, 0.0)
            except Exception:
                skewness = 0.0
        if len(scores) > 3:
            try:
                kurt_val = scipy_stats.kurtosis(scores)
                kurtosis = _safe_float(kurt_val, 0.0)
            except Exception:
                kurtosis = 0.0
        
        return {
            'min': _safe_float(np.min(scores)),
            'max': _safe_float(np.max(scores)),
            'mean': _safe_float(np.mean(scores)),
            'std': _safe_float(np.std(scores)),
            'median': _safe_float(percentiles[1]),
            'p25': _safe_float(percentiles[0]),
            'p75': _safe_float(percentiles[2]),
            'p90': _safe_float(percentiles[3]),
            'p95': _safe_float(percentiles[4]),
            'p99': _safe_float(percentiles[5]),
            'skewness': skewness,
            'kurtosis': kurtosis,
        }
    
    def _build_threshold_info(
        self,
        method: ThresholdMethod,
        value: float,
        scores: np.ndarray,
        anomaly_mask: np.ndarray,
    ) -> ThresholdInfo:
        """Build threshold information object."""
        value = _safe_float(value)
        
        if method == ThresholdMethod.PERCENTILE:
            actual_percentile = 100 * (1 - np.sum(anomaly_mask) / len(anomaly_mask))
            return ThresholdInfo(
                method=method.value,
                value=value,
                percentile=_safe_float(actual_percentile),
            )
        elif method == ThresholdMethod.AUTO:
            # Compute what percentile the threshold corresponds to
            if np.sum(anomaly_mask) > 0:
                min_anomaly_score = np.min(scores[anomaly_mask])
                percentile = 100 * np.sum(scores < min_anomaly_score) / len(scores)
            else:
                percentile = 100.0
            return ThresholdInfo(
                method=method.value,
                value=value,
                percentile=_safe_float(percentile),
                auto_reason="Algorithm-determined threshold",
            )
        else:
            return ThresholdInfo(
                method=method.value,
                value=value,
            )
    
    def _extract_top_anomalies(
        self,
        df: pd.DataFrame,
        avatars: List[str],
        scores: np.ndarray,
        anomaly_mask: np.ndarray,
        metrics: List[str],
        top_n: int,
    ) -> List[Dict[str, Any]]:
        """
        Extract top N anomalous nodes with their details.
        
        Sorted by score descending.
        Includes original metric values for each node.
        """
        # Get indices sorted by score descending
        sorted_indices = np.argsort(scores)[::-1]
        top_n = min(top_n, len(sorted_indices))
        
        # Build a lookup from avatar to dataframe row
        # Try multiple ID column strategies
        avatar_to_row = self._build_avatar_lookup(df, avatars)
        
        top_anomalies = []
        for idx in sorted_indices[:top_n]:
            avatar = avatars[idx]
            
            anomaly_info = {
                'id': avatar,
                'score': _safe_float(scores[idx]),
                'is_anomaly': bool(anomaly_mask[idx]),
                'rank': len(top_anomalies) + 1,
            }
            
            # Add metric values using the lookup or direct index
            if avatar in avatar_to_row:
                row_idx = avatar_to_row[avatar]
                for metric in metrics:
                    if metric in df.columns:
                        try:
                            val = df.iloc[row_idx][metric]
                            anomaly_info[metric] = _safe_float(val) if pd.notna(val) else 0.0
                        except (IndexError, KeyError):
                            anomaly_info[metric] = 0.0
            else:
                # Fallback: try direct index if avatars align with df index
                try:
                    for metric in metrics:
                        if metric in df.columns:
                            val = df.iloc[idx][metric]
                            anomaly_info[metric] = _safe_float(val) if pd.notna(val) else 0.0
                except (IndexError, KeyError):
                    # Last resort - just add 0.0 for all metrics
                    for metric in metrics:
                        anomaly_info[metric] = 0.0
            
            top_anomalies.append(anomaly_info)
        
        return top_anomalies
    
    def _build_avatar_lookup(
        self,
        df: pd.DataFrame,
        avatars: List[str],
    ) -> Dict[str, int]:
        """
        Build a lookup from avatar string to dataframe row index.
        
        Tries multiple strategies to match avatars to rows:
        1. 'avatar' column
        2. 'id' column
        3. Index name
        4. String-converted index values
        """
        lookup = {}
        
        # Strategy 1: 'avatar' column
        if 'avatar' in df.columns:
            avatar_col = df['avatar'].astype(str)
            for i, val in enumerate(avatar_col):
                if val in avatars:
                    lookup[val] = i
            if lookup:
                return lookup
        
        # Strategy 2: 'id' column
        if 'id' in df.columns:
            id_col = df['id'].astype(str)
            for i, val in enumerate(id_col):
                if val in avatars:
                    lookup[val] = i
            if lookup:
                return lookup
        
        # Strategy 3: Use index
        idx_vals = df.index.astype(str).tolist()
        for i, val in enumerate(idx_vals):
            if val in avatars:
                lookup[val] = i
        if lookup:
            return lookup
        
        # Strategy 4: Direct position mapping (if lengths match)
        if len(df) == len(avatars):
            for i, avatar in enumerate(avatars):
                lookup[avatar] = i
            return lookup
        
        return lookup
    
    def merge_group_results(
        self,
        group_results: Dict[Any, AnomalyResult],
        algorithm: str,
        params: Dict[str, Any],
        metrics: List[str],
        total_computation_time: float,
        top_n: int = 20,
    ) -> AnomalyResult:
        """
        Merge results from multiple groups into a single AnomalyResult.
        
        Used for group-aware detection.
        """
        all_scores = {}
        all_labels = {}
        all_raw_scores = {}
        all_top_anomalies = []
        
        group_stats = {}
        
        total_anomalies = 0
        total_samples = 0
        
        for group_key, result in group_results.items():
            # Merge scores and labels (sanitize scores)
            all_scores.update({k: _safe_float(v) for k, v in result.scores.items()})
            all_labels.update(result.binary_labels)
            if result.raw_scores:
                all_raw_scores.update({k: _safe_float(v) for k, v in result.raw_scores.items()})
            
            # Collect group stats
            group_stats[group_key] = GroupAnomalyStats(
                group_value=group_key,
                n_samples=result.n_total,
                n_anomalies=result.n_anomalies,
                anomaly_rate=_safe_float(result.anomaly_rate),
                mean_score=_safe_float(result.statistics.get('mean', 0.0)),
                std_score=_safe_float(result.statistics.get('std', 0.0)),
                threshold_used=_safe_float(result.threshold_info.value),
                top_anomalies=result.top_anomalies[:5],  # Top 5 per group
            )
            
            # Add top anomalies from this group
            for item in result.top_anomalies:
                item_copy = _sanitize_dict(item.copy())
                item_copy['group'] = group_key
                all_top_anomalies.append(item_copy)
            
            total_anomalies += result.n_anomalies
            total_samples += result.n_total
        
        # Sort all top anomalies by score and take top_n
        all_top_anomalies.sort(key=lambda x: x.get('score', 0), reverse=True)
        top_anomalies = all_top_anomalies[:top_n]
        
        # Recompute ranks
        for i, item in enumerate(top_anomalies):
            item['rank'] = i + 1
        
        # Compute global statistics from merged scores
        all_scores_array = np.array(list(all_scores.values()))
        statistics = self._compute_statistics(all_scores_array)
        
        # Threshold info for merged result
        threshold_info = ThresholdInfo(
            method='group_aware',
            value=0.0,  # Not applicable for merged
            auto_reason=f"Merged from {len(group_results)} groups",
        )
        
        return AnomalyResult(
            scores=all_scores,
            binary_labels=all_labels,
            algorithm=algorithm,
            parameters=params,
            metrics_used=metrics,
            threshold_info=threshold_info,
            n_anomalies=total_anomalies,
            n_total=total_samples,
            computation_time=_safe_float(total_computation_time),
            statistics=statistics,
            top_anomalies=top_anomalies,
            raw_scores=all_raw_scores if all_raw_scores else None,
            group_results=group_stats,
        )