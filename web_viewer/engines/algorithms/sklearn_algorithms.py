"""
Sklearn-based Anomaly Detection Algorithms

Machine learning methods from scikit-learn.
"""

from typing import Dict, Any, Optional

import numpy as np

from .base import BaseAnomalyAlgorithm, AlgorithmType, AlgorithmOutput

# Check for sklearn
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.cluster import DBSCAN
    from sklearn.svm import OneClassSVM
    from sklearn.covariance import EllipticEnvelope
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class IsolationForestAlgorithm(BaseAnomalyAlgorithm):
    """Isolation Forest anomaly detection."""
    
    name = "isolation_forest"
    display_name = "Isolation Forest"
    description = "Tree-based anomaly detection. Works well on high-dimensional data."
    algorithm_type = AlgorithmType.MACHINE_LEARNING
    requires_sklearn = True
    complexity = "O(n log n)"
    
    def fit_predict(self, X: np.ndarray, params: Dict[str, Any]) -> AlgorithmOutput:
        """Run Isolation Forest."""
        contamination = params.get("contamination", 0.05)
        n_estimators = params.get("n_estimators", 100)
        max_samples = params.get("max_samples", "auto")
        random_state = params.get("random_state", 42)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )
        
        # Fit and predict
        labels = model.fit_predict(X)  # -1 for anomaly, 1 for normal
        
        # Get raw scores (more negative = more anomalous)
        raw_scores = -model.decision_function(X)
        
        # Normalize to 0-1 range
        score_min = raw_scores.min()
        score_max = raw_scores.max()
        if score_max > score_min:
            scores = (raw_scores - score_min) / (score_max - score_min)
        else:
            scores = np.zeros_like(raw_scores)
        
        anomaly_mask = labels == -1
        threshold = contamination
        
        return AlgorithmOutput(
            raw_scores=scores,
            anomaly_mask=anomaly_mask,
            threshold_used=threshold,
            per_metric_scores=None,
        )
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            "contamination": 0.05,
            "n_estimators": 100,
            "max_samples": "auto",
            "random_state": 42,
        }
    
    @classmethod
    def get_param_descriptions(cls) -> Dict[str, str]:
        return {
            "contamination": "Expected proportion of anomalies (0-0.5)",
            "n_estimators": "Number of trees",
            "max_samples": "Samples per tree ('auto' or integer)",
            "random_state": "Random seed for reproducibility",
        }


class LOFAlgorithm(BaseAnomalyAlgorithm):
    """Local Outlier Factor anomaly detection."""
    
    name = "lof"
    display_name = "Local Outlier Factor (LOF)"
    description = "Density-based anomaly detection. Good for local anomalies."
    algorithm_type = AlgorithmType.DENSITY_BASED
    requires_sklearn = True
    complexity = "O(n^2)"
    max_recommended_nodes = 50000
    
    def fit_predict(self, X: np.ndarray, params: Dict[str, Any]) -> AlgorithmOutput:
        """Run LOF."""
        n_neighbors = params.get("n_neighbors", 20)
        contamination = params.get("contamination", 0.05)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Adjust n_neighbors if necessary
        n_neighbors = min(n_neighbors, len(X) - 1)
        
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=False,
            n_jobs=-1,
        )
        
        labels = model.fit_predict(X)
        
        # Get negative outlier factor (more negative = more anomalous)
        raw_scores = -model.negative_outlier_factor_
        
        # Normalize
        score_min = raw_scores.min()
        score_max = raw_scores.max()
        if score_max > score_min:
            scores = (raw_scores - score_min) / (score_max - score_min)
        else:
            scores = np.zeros_like(raw_scores)
        
        anomaly_mask = labels == -1
        
        return AlgorithmOutput(
            raw_scores=scores,
            anomaly_mask=anomaly_mask,
            threshold_used=contamination,
            per_metric_scores=None,
        )
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {"n_neighbors": 20, "contamination": 0.05}
    
    @classmethod
    def get_param_descriptions(cls) -> Dict[str, str]:
        return {
            "n_neighbors": "Number of neighbors for density estimation",
            "contamination": "Expected proportion of anomalies",
        }


class DBSCANAlgorithm(BaseAnomalyAlgorithm):
    """DBSCAN clustering for anomaly detection."""
    
    name = "dbscan"
    display_name = "DBSCAN"
    description = "Density-based clustering. Points not in any cluster are anomalies."
    algorithm_type = AlgorithmType.DENSITY_BASED
    requires_sklearn = True
    complexity = "O(n^2)"
    
    def fit_predict(self, X: np.ndarray, params: Dict[str, Any]) -> AlgorithmOutput:
        """Run DBSCAN."""
        eps = params.get("eps", 0.5)
        min_samples = params.get("min_samples", 5)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = model.fit_predict(X)
        
        # -1 labels are noise/anomalies
        anomaly_mask = labels == -1
        
        # Score: 1 for anomaly, 0 for normal (simple binary)
        scores = anomaly_mask.astype(float)
        
        return AlgorithmOutput(
            raw_scores=scores,
            anomaly_mask=anomaly_mask,
            threshold_used=0.5,
            per_metric_scores=None,
        )
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {"eps": 0.5, "min_samples": 5}
    
    @classmethod
    def get_param_descriptions(cls) -> Dict[str, str]:
        return {
            "eps": "Maximum distance between points in a cluster",
            "min_samples": "Minimum points to form a dense region",
        }


class OneClassSVMAlgorithm(BaseAnomalyAlgorithm):
    """One-Class SVM anomaly detection."""
    
    name = "ocsvm"
    display_name = "One-Class SVM"
    description = "SVM-based anomaly detection. Good for high-dimensional data."
    algorithm_type = AlgorithmType.MACHINE_LEARNING
    requires_sklearn = True
    complexity = "O(n^2 * d) to O(n^3)"
    max_recommended_nodes = 10000
    
    def fit_predict(self, X: np.ndarray, params: Dict[str, Any]) -> AlgorithmOutput:
        """Run One-Class SVM."""
        kernel = params.get("kernel", "rbf")
        nu = params.get("nu", 0.05)
        gamma = params.get("gamma", "scale")
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        labels = model.fit_predict(X)
        
        # Get decision function scores
        raw_scores = -model.decision_function(X)
        
        # Normalize
        score_min = raw_scores.min()
        score_max = raw_scores.max()
        if score_max > score_min:
            scores = (raw_scores - score_min) / (score_max - score_min)
        else:
            scores = np.zeros_like(raw_scores)
        
        anomaly_mask = labels == -1
        
        return AlgorithmOutput(
            raw_scores=scores,
            anomaly_mask=anomaly_mask,
            threshold_used=nu,
            per_metric_scores=None,
        )
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {"kernel": "rbf", "nu": 0.05, "gamma": "scale"}
    
    @classmethod
    def get_param_descriptions(cls) -> Dict[str, str]:
        return {
            "kernel": "Kernel type (rbf, linear, poly)",
            "nu": "Upper bound on anomaly fraction",
            "gamma": "Kernel coefficient",
        }


class EllipticEnvelopeAlgorithm(BaseAnomalyAlgorithm):
    """Elliptic Envelope (robust covariance) anomaly detection."""
    
    name = "elliptic_envelope"
    display_name = "Elliptic Envelope"
    description = "Gaussian-based detection with robust covariance estimation."
    algorithm_type = AlgorithmType.STATISTICAL
    requires_sklearn = True
    complexity = "O(n * d^2)"
    
    def fit_predict(self, X: np.ndarray, params: Dict[str, Any]) -> AlgorithmOutput:
        """Run Elliptic Envelope."""
        contamination = params.get("contamination", 0.05)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        model = EllipticEnvelope(contamination=contamination, random_state=42)
        
        try:
            labels = model.fit_predict(X)
            raw_scores = -model.decision_function(X)
        except Exception:
            # Fallback if fitting fails
            labels = np.ones(len(X))
            raw_scores = np.zeros(len(X))
        
        # Normalize
        score_min = raw_scores.min()
        score_max = raw_scores.max()
        if score_max > score_min:
            scores = (raw_scores - score_min) / (score_max - score_min)
        else:
            scores = np.zeros_like(raw_scores)
        
        anomaly_mask = labels == -1
        
        return AlgorithmOutput(
            raw_scores=scores,
            anomaly_mask=anomaly_mask,
            threshold_used=contamination,
            per_metric_scores=None,
        )
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {"contamination": 0.05}
    
    @classmethod
    def get_param_descriptions(cls) -> Dict[str, str]:
        return {"contamination": "Expected proportion of anomalies"}