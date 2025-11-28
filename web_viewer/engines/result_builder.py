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


@dataclass
class ThresholdInfo:
    """Information about how the anomaly threshold was determined."""
    method: str
    value: float
    percentile: Optional[float] = None
    auto_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'value': self.value,
            'percentile': self.percentile,
            'auto_reason': self.auto_reason,
        }


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
        return {
            'original_dtype': self.original_dtype,
            'n_missing': self.n_missing,
            'n_inf': self.n_inf,
            'n_zeros': self.n_zeros,
            'transform_applied': self.transform_applied,
            'original_range': list(self.original_range),
            'final_range': list(self.final_range),
        }


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
        return {
            'group_value': self.group_value,
            'n_samples': self.n_samples,
            'n_anomalies': self.n_anomalies,
            'anomaly_rate': self.anomaly_rate,
            'mean_score': self.mean_score,
            'std_score': self.std_score,
            'threshold_used': self.threshold_used,
            'top_anomalies': self.top_anomalies,
        }


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
        result = {
            'scores': self.scores,
            'binary_labels': self.binary_labels,
            'algorithm': self.algorithm,
            'parameters': self.parameters,
            'metrics_used': self.metrics_used,
            'threshold_info': self.threshold_info.to_dict(),
            'n_anomalies': self.n_anomalies,
            'n_total': self.n_total,
            'anomaly_rate': self.anomaly_rate,
            'computation_time': self.computation_time,
            'statistics': self.statistics,
            'top_anomalies': self.top_anomalies,
        }
        
        if self.raw_scores is not None:
            result['raw_scores'] = self.raw_scores
        
        if self.per_metric_scores is not None:
            result['per_metric_scores'] = self.per_metric_scores
        
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
        # Normalize scores
        normalized_scores = self._normalize_scores(raw_scores, score_normalization)
        
        # Build dictionaries
        score_dict = {
            avatar: float(score) 
            for avatar, score in zip(avatars, normalized_scores)
        }
        
        label_dict = {
            avatar: bool(label) 
            for avatar, label in zip(avatars, anomaly_mask)
        }
        
        # Raw scores dict (before normalization)
        raw_score_dict = {
            avatar: float(score) 
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
                per_metric_score_dicts[metric] = {
                    avatar: float(score) 
                    for avatar, score in zip(avatars, scores)
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
            computation_time=computation_time,
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
        
        if method == ScoreNormalization.NONE:
            return scores
        
        if method == ScoreNormalization.RANK:
            # Percentile rank (0 to 1)
            return scipy_stats.rankdata(scores, method='average') / len(scores)
        
        # Default: minmax normalization
        if np.std(scores) < 1e-10:
            return np.zeros_like(scores)
        
        min_val = np.min(scores)
        max_val = np.max(scores)
        
        if max_val - min_val < 1e-10:
            return np.zeros_like(scores)
        
        normalized = (scores - min_val) / (max_val - min_val)
        return np.clip(normalized, 0.0, 1.0)
    
    def _compute_statistics(self, scores: np.ndarray) -> Dict[str, Any]:
        """
        Compute summary statistics for anomaly scores.
        
        All operations are vectorized for performance.
        """
        scores = np.asarray(scores, dtype=np.float64)
        
        # Handle empty or all-nan case
        if len(scores) == 0 or np.all(np.isnan(scores)):
            return {
                'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0, 'median': 0.0,
                'p25': 0.0, 'p75': 0.0, 'p90': 0.0, 'p95': 0.0, 'p99': 0.0,
                'skewness': 0.0, 'kurtosis': 0.0,
            }
        
        # Compute percentiles in single call
        percentiles = np.percentile(scores, [25, 50, 75, 90, 95, 99])
        
        return {
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'median': float(percentiles[1]),
            'p25': float(percentiles[0]),
            'p75': float(percentiles[2]),
            'p90': float(percentiles[3]),
            'p95': float(percentiles[4]),
            'p99': float(percentiles[5]),
            'skewness': float(scipy_stats.skew(scores)) if len(scores) > 2 else 0.0,
            'kurtosis': float(scipy_stats.kurtosis(scores)) if len(scores) > 3 else 0.0,
        }
    
    def _build_threshold_info(
        self,
        method: ThresholdMethod,
        value: float,
        scores: np.ndarray,
        anomaly_mask: np.ndarray,
    ) -> ThresholdInfo:
        """Build threshold information object."""
        if method == ThresholdMethod.PERCENTILE:
            actual_percentile = 100 * (1 - np.sum(anomaly_mask) / len(anomaly_mask))
            return ThresholdInfo(
                method=method.value,
                value=value,
                percentile=actual_percentile,
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
                percentile=percentile,
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
                'score': float(scores[idx]),
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
                            anomaly_info[metric] = float(val) if pd.notna(val) else 0.0
                        except (IndexError, KeyError):
                            anomaly_info[metric] = 0.0
            else:
                # Fallback: try direct index if avatars align with df index
                try:
                    for metric in metrics:
                        if metric in df.columns:
                            val = df.iloc[idx][metric]
                            anomaly_info[metric] = float(val) if pd.notna(val) else 0.0
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
            # Merge scores and labels
            all_scores.update(result.scores)
            all_labels.update(result.binary_labels)
            if result.raw_scores:
                all_raw_scores.update(result.raw_scores)
            
            # Collect group stats
            group_stats[group_key] = GroupAnomalyStats(
                group_value=group_key,
                n_samples=result.n_total,
                n_anomalies=result.n_anomalies,
                anomaly_rate=result.anomaly_rate,
                mean_score=result.statistics.get('mean', 0.0),
                std_score=result.statistics.get('std', 0.0),
                threshold_used=result.threshold_info.value,
                top_anomalies=result.top_anomalies[:5],  # Top 5 per group
            )
            
            # Add top anomalies from this group
            for item in result.top_anomalies:
                item_copy = item.copy()
                item_copy['group'] = group_key
                all_top_anomalies.append(item_copy)
            
            total_anomalies += result.n_anomalies
            total_samples += result.n_total
        
        # Sort all top anomalies by score and take top_n
        all_top_anomalies.sort(key=lambda x: x['score'], reverse=True)
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
            computation_time=total_computation_time,
            statistics=statistics,
            top_anomalies=top_anomalies,
            raw_scores=all_raw_scores if all_raw_scores else None,
            group_results=group_stats,
        )