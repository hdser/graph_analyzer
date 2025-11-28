"""
Distance-based Anomaly Detection Algorithms

Implements:
- Mahalanobis: Distance-based detection accounting for correlations
"""

import warnings
from typing import Dict, Any, Optional

import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import chi2

from .base import (
    AnomalyAlgorithmBase,
    ParameterSpec,
    AlgorithmOutput,
)

# Optional sklearn for robust covariance
try:
    from sklearn.covariance import MinCovDet, EmpiricalCovariance
    HAS_SKLEARN_COV = True
except ImportError:
    HAS_SKLEARN_COV = False


class MahalanobisAlgorithm(AnomalyAlgorithmBase):
    """
    Mahalanobis distance-based anomaly detection.
    
    Computes distance from centroid accounting for feature correlations.
    Points far from the centroid in Mahalanobis space are anomalies.
    
    Uses chi-squared distribution for principled threshold.
    """
    
    name = "mahalanobis"
    display_name = "Mahalanobis Distance"
    description = (
        "Distance-based detection accounting for correlations between features. "
        "Uses Mahalanobis distance which normalizes by covariance. "
        "Threshold based on chi-squared distribution for statistical rigor."
    )
    complexity = "O(n × d²)"
    supports_multivariate = True
    requires_sklearn = False
    
    parameters = {
        "alpha": ParameterSpec(
            name="alpha",
            param_type="float",
            default=0.99,
            min_value=0.9,
            max_value=0.9999,
            description="Confidence level for chi-squared threshold (0.99 = 99%)",
        ),
        "robust": ParameterSpec(
            name="robust",
            param_type="bool",
            default=False,
            description="Use robust covariance estimation (MinCovDet) - slower but handles outliers",
        ),
        "regularization": ParameterSpec(
            name="regularization",
            param_type="float",
            default=1e-5,
            min_value=1e-10,
            max_value=1e-1,
            description="Regularization for covariance matrix inversion",
        ),
        "support_fraction": ParameterSpec(
            name="support_fraction",
            param_type="float",
            default=0.75,
            min_value=0.5,
            max_value=1.0,
            description="Fraction of data used for MinCovDet (higher = more stable)",
        ),
    }
    
    def fit_predict(
        self,
        X: np.ndarray,
        params: Dict[str, Any],
    ) -> AlgorithmOutput:
        """
        Run Mahalanobis distance anomaly detection.
        
        Fully vectorized implementation using einsum.
        """
        self._validate_input(X)
        
        alpha = params.get("alpha", 0.99)
        robust = params.get("robust", False)
        regularization = params.get("regularization", 1e-5)
        support_fraction = params.get("support_fraction", 0.75)
        
        n_samples, n_features = X.shape
        
        # For single feature, fall back to z-score-like approach
        if n_features == 1:
            return self._single_feature_detection(X, alpha)
        
        # Compute mean and covariance with proper error handling
        mean, cov_matrix, robust_used = self._compute_covariance(
            X, robust, support_fraction
        )
        
        # Invert covariance matrix with regularization
        cov_inv = self._invert_covariance(cov_matrix, regularization)
        
        # Compute Mahalanobis distances (vectorized with einsum)
        diff = X - mean
        # Mahalanobis: sqrt(diff @ cov_inv @ diff.T) for each row
        # Using einsum for efficiency: 'ij,jk,ik->i'
        mahal_sq = np.einsum('ij,jk,ik->i', diff, cov_inv, diff)
        distances = np.sqrt(np.maximum(mahal_sq, 0))  # Ensure non-negative
        
        # Chi-squared threshold
        chi2_threshold = chi2.ppf(alpha, df=n_features)
        distance_threshold = np.sqrt(chi2_threshold)
        
        # Determine anomalies
        anomaly_mask = distances > distance_threshold
        
        # Compute per-feature contribution to distance
        per_metric_scores = self._compute_per_feature_contribution(
            diff, cov_inv, n_features
        )
        
        # Compute condition number safely
        try:
            cond_number = float(np.linalg.cond(cov_matrix))
            if np.isinf(cond_number) or np.isnan(cond_number):
                cond_number = -1.0
        except Exception:
            cond_number = -1.0
        
        return AlgorithmOutput(
            raw_scores=distances,
            anomaly_mask=anomaly_mask,
            threshold_used=distance_threshold,
            per_metric_scores=per_metric_scores,
            extra_info={
                "alpha": alpha,
                "chi2_threshold": float(chi2_threshold),
                "robust_covariance_used": robust_used,
                "n_features": n_features,
                "condition_number": cond_number,
                "regularization": regularization,
            },
        )
    
    def _compute_covariance(
        self,
        X: np.ndarray,
        robust: bool,
        support_fraction: float,
    ) -> tuple:
        """
        Compute mean and covariance matrix.
        
        Uses MinCovDet for robust estimation if available and requested.
        Falls back to empirical covariance if MinCovDet fails.
        """
        robust_used = False
        
        if robust and HAS_SKLEARN_COV:
            try:
                # Suppress warnings from MinCovDet
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    
                    # Use higher support_fraction for stability
                    mcd = MinCovDet(
                        random_state=42,
                        support_fraction=support_fraction,
                    )
                    mcd.fit(X)
                    mean = mcd.location_
                    cov_matrix = mcd.covariance_
                    robust_used = True
                    return mean, cov_matrix, robust_used
                    
            except Exception as e:
                # MinCovDet can fail with certain data distributions
                # Fall back to empirical covariance
                pass
        
        # Use sklearn's EmpiricalCovariance if available for consistency
        if HAS_SKLEARN_COV:
            try:
                emp_cov = EmpiricalCovariance()
                emp_cov.fit(X)
                mean = emp_cov.location_
                cov_matrix = emp_cov.covariance_
                return mean, cov_matrix, robust_used
            except Exception:
                pass
        
        # Pure numpy fallback
        mean = np.mean(X, axis=0)
        cov_matrix = np.cov(X.T)
        
        # Ensure 2D for small feature count
        if cov_matrix.ndim == 0:
            cov_matrix = np.array([[float(cov_matrix)]])
        elif cov_matrix.ndim == 1:
            cov_matrix = np.diag(cov_matrix)
        
        return mean, cov_matrix, robust_used
    
    def _invert_covariance(
        self,
        cov_matrix: np.ndarray,
        regularization: float,
    ) -> np.ndarray:
        """
        Invert covariance matrix with regularization.
        
        Uses Tikhonov regularization (adding to diagonal) to ensure
        numerical stability.
        """
        n = cov_matrix.shape[0]
        
        # Add regularization to diagonal (Tikhonov regularization)
        cov_reg = cov_matrix + regularization * np.eye(n)
        
        try:
            # Try Cholesky decomposition first (faster and more stable)
            L = np.linalg.cholesky(cov_reg)
            L_inv = np.linalg.inv(L)
            cov_inv = L_inv.T @ L_inv
        except np.linalg.LinAlgError:
            # Fall back to standard inverse
            try:
                cov_inv = np.linalg.inv(cov_reg)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse as last resort
                cov_inv = np.linalg.pinv(cov_reg)
        
        return cov_inv
    
    def _compute_per_feature_contribution(
        self,
        diff: np.ndarray,
        cov_inv: np.ndarray,
        n_features: int,
    ) -> Dict[str, np.ndarray]:
        """
        Compute per-feature contribution to Mahalanobis distance.
        
        Uses diagonal of (diff @ cov_inv * diff) for each feature.
        """
        # Contribution of each feature: (diff @ cov_inv) * diff for each dimension
        weighted_diff = diff @ cov_inv
        contributions = weighted_diff * diff
        
        per_metric_scores = {}
        for i in range(n_features):
            per_metric_scores[f"feature_{i}"] = np.sqrt(np.maximum(contributions[:, i], 0))
        
        return per_metric_scores
    
    def _single_feature_detection(
        self,
        X: np.ndarray,
        alpha: float,
    ) -> AlgorithmOutput:
        """
        Handle single-feature case using z-score approach.
        """
        x = X[:, 0]
        mean = np.mean(x)
        std = np.std(x)
        
        if std < 1e-10:
            # Constant column
            return AlgorithmOutput(
                raw_scores=np.zeros(len(x)),
                anomaly_mask=np.zeros(len(x), dtype=bool),
                threshold_used=0.0,
                extra_info={"single_feature_mode": True, "constant_column": True},
            )
        
        # Z-scores as "distances"
        z_scores = np.abs((x - mean) / std)
        
        # Use normal distribution quantile as threshold
        threshold = scipy_stats.norm.ppf((1 + alpha) / 2)
        
        anomaly_mask = z_scores > threshold
        
        return AlgorithmOutput(
            raw_scores=z_scores,
            anomaly_mask=anomaly_mask,
            threshold_used=float(threshold),
            per_metric_scores={"feature_0": z_scores},
            extra_info={
                "single_feature_mode": True,
                "alpha": alpha,
            },
        )