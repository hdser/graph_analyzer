"""
Statistical Anomaly Detection Algorithms

Implements:
- Z-Score: Statistical z-score based outlier detection
- IQR: Interquartile Range based detection
"""

from typing import Dict, Any, Optional

import numpy as np
from scipy import stats as scipy_stats

from .base import (
    AnomalyAlgorithmBase,
    ParameterSpec,
    AlgorithmOutput,
)


class ZScoreAlgorithm(AnomalyAlgorithmBase):
    """
    Z-Score based anomaly detection.
    
    Computes z-score for each feature and aggregates across features.
    Points with high aggregated z-scores are marked as anomalies.
    
    Fully vectorized implementation for performance.
    """
    
    name = "zscore"
    display_name = "Z-Score"
    description = (
        "Statistical z-score based outlier detection. "
        "Fast and simple, best for normally distributed data. "
        "Computes z-scores per feature and aggregates them."
    )
    complexity = "O(n × d)"
    supports_multivariate = True
    requires_sklearn = False
    
    parameters = {
        "threshold": ParameterSpec(
            name="threshold",
            param_type="float",
            default=3.0,
            min_value=1.0,
            max_value=10.0,
            description="Z-score threshold for anomaly detection",
        ),
        "aggregation": ParameterSpec(
            name="aggregation",
            param_type="str",
            default="max",
            choices=["max", "mean", "l2", "weighted"],
            description="How to aggregate z-scores across features",
        ),
    }
    
    def fit_predict(
        self,
        X: np.ndarray,
        params: Dict[str, Any],
    ) -> AlgorithmOutput:
        """
        Run z-score based anomaly detection.
        
        Vectorized implementation using scipy.stats.zscore.
        """
        self._validate_input(X)
        
        threshold = params.get("threshold", 3.0)
        aggregation = params.get("aggregation", "max")
        
        n_samples, n_features = X.shape
        
        # Compute z-scores for all features at once (vectorized)
        # scipy.stats.zscore handles axis parameter efficiently
        z_scores = np.abs(scipy_stats.zscore(X, axis=0, nan_policy='omit'))
        
        # Replace any NaN z-scores with 0 (from constant columns)
        z_scores = np.nan_to_num(z_scores, nan=0.0)
        
        # Store per-feature scores
        per_metric_scores = {
            f"feature_{i}": z_scores[:, i]
            for i in range(n_features)
        }
        
        # Aggregate z-scores across features
        if n_features == 1:
            aggregated = z_scores[:, 0]
        elif aggregation == "max":
            aggregated = np.max(z_scores, axis=1)
        elif aggregation == "mean":
            aggregated = np.mean(z_scores, axis=1)
        elif aggregation == "l2":
            aggregated = np.sqrt(np.sum(z_scores ** 2, axis=1))
        elif aggregation == "weighted":
            # Default to equal weights if not provided
            weights = np.ones(n_features) / n_features
            aggregated = np.sum(z_scores * weights, axis=1)
        else:
            aggregated = np.max(z_scores, axis=1)
        
        # Determine anomalies
        anomaly_mask = aggregated > threshold
        
        return AlgorithmOutput(
            raw_scores=aggregated,
            anomaly_mask=anomaly_mask,
            threshold_used=threshold,
            per_metric_scores=per_metric_scores,
            extra_info={
                "aggregation": aggregation,
                "n_features": n_features,
            },
        )


class IQRAlgorithm(AnomalyAlgorithmBase):
    """
    Interquartile Range (IQR) based anomaly detection.
    
    Uses Tukey's method: outliers are points outside
    [Q1 - k*IQR, Q3 + k*IQR] where k is the multiplier.
    
    Robust to extreme values, good for skewed distributions.
    """
    
    name = "iqr"
    display_name = "Interquartile Range"
    description = (
        "IQR-based outlier detection using Tukey's method. "
        "Robust to extreme values, good for skewed distributions. "
        "Uses fences at Q1 - k*IQR and Q3 + k*IQR."
    )
    complexity = "O(n × d × log n)"
    supports_multivariate = True
    requires_sklearn = False
    
    parameters = {
        "multiplier": ParameterSpec(
            name="multiplier",
            param_type="float",
            default=1.5,
            min_value=1.0,
            max_value=5.0,
            description="IQR multiplier (1.5 = outlier, 3.0 = extreme)",
        ),
        "side": ParameterSpec(
            name="side",
            param_type="str",
            default="both",
            choices=["both", "high", "low"],
            description="Which tail(s) to detect as anomalies",
        ),
        "aggregation": ParameterSpec(
            name="aggregation",
            param_type="str",
            default="max",
            choices=["max", "mean", "any"],
            description="How to aggregate across features",
        ),
    }
    
    def fit_predict(
        self,
        X: np.ndarray,
        params: Dict[str, Any],
    ) -> AlgorithmOutput:
        """
        Run IQR-based anomaly detection.
        
        Vectorized implementation using numpy percentile.
        """
        self._validate_input(X)
        
        multiplier = params.get("multiplier", 1.5)
        side = params.get("side", "both")
        aggregation = params.get("aggregation", "max")
        
        n_samples, n_features = X.shape
        
        # Compute quartiles for all features at once (vectorized)
        q25 = np.percentile(X, 25, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        iqr = q75 - q25
        
        # Handle constant columns (IQR = 0)
        iqr_safe = np.where(iqr < 1e-10, 1.0, iqr)
        
        # Compute fences
        lower_fence = q25 - multiplier * iqr
        upper_fence = q75 + multiplier * iqr
        
        # Compute distance from fences (normalized by IQR)
        # Vectorized for all samples and features
        below_lower = np.maximum(0, lower_fence - X) / iqr_safe
        above_upper = np.maximum(0, X - upper_fence) / iqr_safe
        
        # Select which sides to consider
        if side == "high":
            distances = above_upper
        elif side == "low":
            distances = below_lower
        else:  # both
            distances = np.maximum(below_lower, above_upper)
        
        # Store per-feature scores
        per_metric_scores = {
            f"feature_{i}": distances[:, i]
            for i in range(n_features)
        }
        
        # Aggregate across features
        if aggregation == "max":
            aggregated = np.max(distances, axis=1)
            # Anomaly if any feature is outside fence
            anomaly_mask = np.any(distances > 0, axis=1)
        elif aggregation == "mean":
            aggregated = np.mean(distances, axis=1)
            # Anomaly if average distance > 0
            anomaly_mask = aggregated > 0
        elif aggregation == "any":
            # Count how many features are outliers
            aggregated = np.sum(distances > 0, axis=1).astype(float)
            anomaly_mask = aggregated > 0
        else:
            aggregated = np.max(distances, axis=1)
            anomaly_mask = np.any(distances > 0, axis=1)
        
        return AlgorithmOutput(
            raw_scores=aggregated,
            anomaly_mask=anomaly_mask,
            threshold_used=multiplier,
            per_metric_scores=per_metric_scores,
            extra_info={
                "side": side,
                "aggregation": aggregation,
                "iqr_values": iqr.tolist(),
                "lower_fence": lower_fence.tolist(),
                "upper_fence": upper_fence.tolist(),
            },
        )