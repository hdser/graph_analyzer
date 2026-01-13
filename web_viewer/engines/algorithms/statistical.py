"""
Statistical Anomaly Detection Algorithms

Simple statistical methods for anomaly detection.
"""

from typing import Dict, Any, Optional

import numpy as np

from .base import BaseAnomalyAlgorithm, AlgorithmType, AlgorithmOutput


class ZScoreAlgorithm(BaseAnomalyAlgorithm):
    """Z-Score based anomaly detection."""
    
    name = "zscore"
    display_name = "Z-Score"
    description = "Statistical z-score based detection. Fast and interpretable."
    algorithm_type = AlgorithmType.STATISTICAL
    complexity = "O(n)"
    
    def fit_predict(self, X: np.ndarray, params: Dict[str, Any]) -> AlgorithmOutput:
        """Compute z-scores for each sample."""
        threshold = params.get("threshold", 3.0)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Handle constant columns
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        
        # Compute z-scores
        z_scores = np.abs((X - mean) / std)
        
        # Per-metric scores
        per_metric_scores = {}
        for i in range(X.shape[1]):
            per_metric_scores[f"feature_{i}"] = z_scores[:, i]
        
        # For multivariate, take mean across features
        if X.shape[1] > 1:
            scores = np.mean(z_scores, axis=1)
        else:
            scores = z_scores.ravel()
        
        # Create anomaly mask
        anomaly_mask = scores > threshold
        
        return AlgorithmOutput(
            raw_scores=scores,
            anomaly_mask=anomaly_mask,
            threshold_used=threshold,
            per_metric_scores=per_metric_scores,
        )
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {"threshold": 3.0}
    
    @classmethod
    def get_param_descriptions(cls) -> Dict[str, str]:
        return {"threshold": "Z-score threshold for anomaly detection"}


class IQRAlgorithm(BaseAnomalyAlgorithm):
    """Interquartile Range (IQR) based anomaly detection."""
    
    name = "iqr"
    display_name = "IQR"
    description = "Interquartile range based detection. Robust to extreme values."
    algorithm_type = AlgorithmType.STATISTICAL
    complexity = "O(n log n)"
    
    def fit_predict(self, X: np.ndarray, params: Dict[str, Any]) -> AlgorithmOutput:
        """Compute IQR-based anomaly scores."""
        k = params.get("k", 1.5)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_features = X.shape[1]
        all_scores = np.zeros(X.shape)
        
        for i in range(n_features):
            col = X[:, i]
            q1 = np.percentile(col, 25)
            q3 = np.percentile(col, 75)
            iqr = q3 - q1
            
            if iqr == 0:
                all_scores[:, i] = 0
            else:
                lower = q1 - k * iqr
                upper = q3 + k * iqr
                
                below = np.maximum(0, lower - col) / iqr
                above = np.maximum(0, col - upper) / iqr
                all_scores[:, i] = below + above
        
        # Per-metric scores
        per_metric_scores = {}
        for i in range(n_features):
            per_metric_scores[f"feature_{i}"] = all_scores[:, i]
        
        # Mean score across features
        if n_features > 1:
            scores = np.mean(all_scores, axis=1)
        else:
            scores = all_scores.ravel()
        
        # Anomalies are those with score > 0
        anomaly_mask = scores > 0
        
        return AlgorithmOutput(
            raw_scores=scores,
            anomaly_mask=anomaly_mask,
            threshold_used=0.0,
            per_metric_scores=per_metric_scores,
        )
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {"k": 1.5}
    
    @classmethod
    def get_param_descriptions(cls) -> Dict[str, str]:
        return {"k": "IQR multiplier (1.5 for outliers, 3.0 for extreme)"}


class MahalanobisAlgorithm(BaseAnomalyAlgorithm):
    """Mahalanobis distance based anomaly detection."""
    
    name = "mahalanobis"
    display_name = "Mahalanobis"
    description = "Mahalanobis distance based detection. Accounts for feature correlations."
    algorithm_type = AlgorithmType.DISTANCE_BASED
    complexity = "O(n * d^2)"
    
    def fit_predict(self, X: np.ndarray, params: Dict[str, Any]) -> AlgorithmOutput:
        """Compute Mahalanobis distances."""
        regularization = params.get("regularization", 1e-6)
        threshold_percentile = params.get("threshold_percentile", 95)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Compute mean and covariance
        mean = np.mean(X, axis=0)
        
        if n_features == 1:
            var = np.var(X)
            if var == 0:
                scores = np.zeros(n_samples)
            else:
                scores = np.abs(X.ravel() - mean) / np.sqrt(var)
        else:
            # Multivariate case
            centered = X - mean
            cov = np.cov(X.T)
            
            # Regularize covariance matrix
            cov += np.eye(n_features) * regularization
            
            try:
                cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov)
            
            # Compute Mahalanobis distances
            scores = np.zeros(n_samples)
            for i in range(n_samples):
                diff = centered[i]
                scores[i] = np.sqrt(diff @ cov_inv @ diff)
        
        # Threshold based on percentile
        threshold = np.percentile(scores, threshold_percentile)
        anomaly_mask = scores > threshold
        
        return AlgorithmOutput(
            raw_scores=scores,
            anomaly_mask=anomaly_mask,
            threshold_used=threshold,
            per_metric_scores=None,
        )
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {"regularization": 1e-6, "threshold_percentile": 95}
    
    @classmethod
    def get_param_descriptions(cls) -> Dict[str, str]:
        return {
            "regularization": "Regularization for covariance matrix",
            "threshold_percentile": "Percentile threshold for anomaly detection",
        }


class ModifiedZScoreAlgorithm(BaseAnomalyAlgorithm):
    """Modified Z-Score using Median Absolute Deviation (MAD)."""
    
    name = "modified_zscore"
    display_name = "Modified Z-Score (MAD)"
    description = "Robust z-score using median and MAD. Resistant to outliers."
    algorithm_type = AlgorithmType.STATISTICAL
    complexity = "O(n log n)"
    
    def fit_predict(self, X: np.ndarray, params: Dict[str, Any]) -> AlgorithmOutput:
        """Compute modified z-scores using MAD."""
        threshold = params.get("threshold", 3.5)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_features = X.shape[1]
        all_scores = np.zeros(X.shape)
        
        for i in range(n_features):
            col = X[:, i]
            median = np.median(col)
            mad = np.median(np.abs(col - median))
            
            if mad == 0:
                mad = np.mean(np.abs(col - median))
            
            if mad == 0:
                all_scores[:, i] = 0
            else:
                all_scores[:, i] = np.abs(0.6745 * (col - median) / mad)
        
        # Per-metric scores
        per_metric_scores = {}
        for i in range(n_features):
            per_metric_scores[f"feature_{i}"] = all_scores[:, i]
        
        # Mean score across features
        if n_features > 1:
            scores = np.mean(all_scores, axis=1)
        else:
            scores = all_scores.ravel()
        
        anomaly_mask = scores > threshold
        
        return AlgorithmOutput(
            raw_scores=scores,
            anomaly_mask=anomaly_mask,
            threshold_used=threshold,
            per_metric_scores=per_metric_scores,
        )
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {"threshold": 3.5}
    
    @classmethod
    def get_param_descriptions(cls) -> Dict[str, str]:
        return {"threshold": "Modified z-score threshold for anomaly detection"}