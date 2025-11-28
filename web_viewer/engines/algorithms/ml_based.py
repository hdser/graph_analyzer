"""
Machine Learning-based Anomaly Detection Algorithms

Implements:
- Isolation Forest: Tree-based anomaly detection
- LOF: Local Outlier Factor (density-based)
- DBSCAN: Clustering-based anomaly detection
"""

from typing import Dict, Any, Optional

import numpy as np

from .base import (
    AnomalyAlgorithmBase,
    ParameterSpec,
    AlgorithmOutput,
)

# These require sklearn
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class IsolationForestAlgorithm(AnomalyAlgorithmBase):
    """
    Isolation Forest anomaly detection.
    
    Isolates anomalies by randomly selecting features and split values.
    Anomalies are easier to isolate, requiring fewer splits.
    
    Excellent for high-dimensional data.
    """
    
    name = "isolation_forest"
    display_name = "Isolation Forest"
    description = (
        "Tree-based anomaly detection using random isolation. "
        "Anomalies are easier to isolate and require fewer splits. "
        "Excellent for high-dimensional data, handles non-linear patterns."
    )
    complexity = "O(n × t × log n)"
    supports_multivariate = True
    requires_sklearn = True
    
    parameters = {
        "n_estimators": ParameterSpec(
            name="n_estimators",
            param_type="int",
            default=100,
            min_value=10,
            max_value=1000,
            description="Number of isolation trees",
        ),
        "contamination": ParameterSpec(
            name="contamination",
            param_type="float",
            default=0.1,
            min_value=0.001,
            max_value=0.5,
            description="Expected proportion of outliers",
        ),
        "max_samples": ParameterSpec(
            name="max_samples",
            param_type="str",
            default="auto",
            choices=["auto", "256", "512", "1024", "all"],
            description="Number of samples for each tree (auto=min(256,n))",
        ),
        "random_state": ParameterSpec(
            name="random_state",
            param_type="int",
            default=42,
            min_value=0,
            max_value=99999,
            description="Random seed for reproducibility",
        ),
        "bootstrap": ParameterSpec(
            name="bootstrap",
            param_type="bool",
            default=False,
            description="Whether to use bootstrap sampling",
        ),
    }
    
    def fit_predict(
        self,
        X: np.ndarray,
        params: Dict[str, Any],
    ) -> AlgorithmOutput:
        """
        Run Isolation Forest anomaly detection.
        """
        self._validate_input(X)
        
        n_estimators = params.get("n_estimators", 100)
        contamination = params.get("contamination", 0.1)
        max_samples_str = params.get("max_samples", "auto")
        random_state = params.get("random_state", 42)
        bootstrap = params.get("bootstrap", False)
        
        # Parse max_samples
        if max_samples_str == "auto":
            max_samples = "auto"
        elif max_samples_str == "all":
            max_samples = X.shape[0]
        else:
            max_samples = int(max_samples_str)
        
        # Standardize features for better performance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit Isolation Forest
        clf = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            bootstrap=bootstrap,
            n_jobs=-1,
        )
        clf.fit(X_scaled)
        
        # Get anomaly scores
        # decision_function returns negative scores for anomalies
        decision_scores = clf.decision_function(X_scaled)
        raw_scores = -decision_scores  # Higher = more anomalous
        
        # Get binary predictions
        predictions = clf.predict(X_scaled)
        anomaly_mask = predictions == -1
        
        # Compute threshold (score at which prediction changes)
        if np.any(anomaly_mask) and np.any(~anomaly_mask):
            threshold = (np.min(raw_scores[anomaly_mask]) + 
                        np.max(raw_scores[~anomaly_mask])) / 2
        else:
            threshold = np.percentile(raw_scores, 100 * (1 - contamination))
        
        return AlgorithmOutput(
            raw_scores=raw_scores,
            anomaly_mask=anomaly_mask,
            threshold_used=threshold,
            extra_info={
                "n_estimators": n_estimators,
                "contamination": contamination,
                "max_samples": max_samples,
                "n_anomalies_detected": int(np.sum(anomaly_mask)),
            },
        )


class LOFAlgorithm(AnomalyAlgorithmBase):
    """
    Local Outlier Factor (LOF) anomaly detection.
    
    Measures local deviation of density with respect to neighbors.
    Points with substantially lower density than neighbors are anomalies.
    
    Best for detecting local deviations.
    """
    
    name = "lof"
    display_name = "Local Outlier Factor"
    description = (
        "Density-based local anomaly detection. "
        "Measures how isolated a point is relative to its neighbors. "
        "Best for detecting local deviations from clusters."
    )
    complexity = "O(n² × d)"
    supports_multivariate = True
    requires_sklearn = True
    
    parameters = {
        "n_neighbors": ParameterSpec(
            name="n_neighbors",
            param_type="int",
            default=20,
            min_value=2,
            max_value=200,
            description="Number of neighbors for density estimation",
        ),
        "contamination": ParameterSpec(
            name="contamination",
            param_type="float",
            default=0.1,
            min_value=0.001,
            max_value=0.5,
            description="Expected proportion of outliers",
        ),
        "algorithm": ParameterSpec(
            name="algorithm",
            param_type="str",
            default="auto",
            choices=["auto", "ball_tree", "kd_tree", "brute"],
            description="Algorithm for nearest neighbor search",
        ),
        "leaf_size": ParameterSpec(
            name="leaf_size",
            param_type="int",
            default=30,
            min_value=10,
            max_value=100,
            description="Leaf size for tree algorithms",
        ),
        "metric": ParameterSpec(
            name="metric",
            param_type="str",
            default="euclidean",
            choices=["euclidean", "manhattan", "minkowski", "chebyshev"],
            description="Distance metric",
        ),
    }
    
    def fit_predict(
        self,
        X: np.ndarray,
        params: Dict[str, Any],
    ) -> AlgorithmOutput:
        """
        Run LOF anomaly detection.
        
        Uses optimized tree-based algorithms for large datasets.
        """
        self._validate_input(X)
        
        n_neighbors = params.get("n_neighbors", 20)
        contamination = params.get("contamination", 0.1)
        algorithm = params.get("algorithm", "auto")
        leaf_size = params.get("leaf_size", 30)
        metric = params.get("metric", "euclidean")
        
        n_samples = X.shape[0]
        
        # Adjust n_neighbors if larger than dataset
        n_neighbors = min(n_neighbors, n_samples - 1)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit LOF
        clf = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            n_jobs=-1,
            novelty=False,  # Use for outlier detection
        )
        
        # fit_predict returns -1 for outliers, 1 for inliers
        predictions = clf.fit_predict(X_scaled)
        anomaly_mask = predictions == -1
        
        # Get LOF scores (negative_outlier_factor_ is negative, more negative = more anomalous)
        raw_scores = -clf.negative_outlier_factor_
        
        # Compute threshold
        if np.any(anomaly_mask) and np.any(~anomaly_mask):
            threshold = (np.min(raw_scores[anomaly_mask]) + 
                        np.max(raw_scores[~anomaly_mask])) / 2
        else:
            threshold = np.percentile(raw_scores, 100 * (1 - contamination))
        
        return AlgorithmOutput(
            raw_scores=raw_scores,
            anomaly_mask=anomaly_mask,
            threshold_used=threshold,
            extra_info={
                "n_neighbors": n_neighbors,
                "contamination": contamination,
                "algorithm_used": algorithm,
                "n_anomalies_detected": int(np.sum(anomaly_mask)),
            },
        )


class DBSCANAlgorithm(AnomalyAlgorithmBase):
    """
    DBSCAN-based anomaly detection.
    
    Points not belonging to any cluster (noise) are considered anomalies.
    Good for data with well-defined clusters.
    """
    
    name = "dbscan"
    display_name = "DBSCAN Clustering"
    description = (
        "Density-based clustering where noise points are anomalies. "
        "Points not belonging to any cluster are marked as outliers. "
        "Good for data with well-defined clusters of varying density."
    )
    complexity = "O(n × log n)"
    supports_multivariate = True
    requires_sklearn = True
    
    parameters = {
        "eps": ParameterSpec(
            name="eps",
            param_type="float",
            default=0.5,
            min_value=0.01,
            max_value=10.0,
            description="Maximum distance between points in a cluster",
        ),
        "min_samples": ParameterSpec(
            name="min_samples",
            param_type="int",
            default=5,
            min_value=2,
            max_value=100,
            description="Minimum points required to form a cluster",
        ),
        "algorithm": ParameterSpec(
            name="algorithm",
            param_type="str",
            default="auto",
            choices=["auto", "ball_tree", "kd_tree", "brute"],
            description="Algorithm for nearest neighbor search",
        ),
        "metric": ParameterSpec(
            name="metric",
            param_type="str",
            default="euclidean",
            choices=["euclidean", "manhattan", "minkowski", "chebyshev"],
            description="Distance metric",
        ),
    }
    
    def fit_predict(
        self,
        X: np.ndarray,
        params: Dict[str, Any],
    ) -> AlgorithmOutput:
        """
        Run DBSCAN-based anomaly detection.
        
        Optimized using NearestNeighbors for distance computation.
        """
        self._validate_input(X)
        
        eps = params.get("eps", 0.5)
        min_samples = params.get("min_samples", 5)
        algorithm = params.get("algorithm", "auto")
        metric = params.get("metric", "euclidean")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit DBSCAN
        clf = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            algorithm=algorithm,
            metric=metric,
            n_jobs=-1,
        )
        labels = clf.fit_predict(X_scaled)
        
        # -1 label means noise (anomaly)
        anomaly_mask = labels == -1
        
        # Compute anomaly scores based on distance to nearest core point
        scores = self._compute_scores(X_scaled, labels, clf, eps, min_samples)
        
        # Threshold is implicitly eps (distance-based)
        threshold = eps
        
        return AlgorithmOutput(
            raw_scores=scores,
            anomaly_mask=anomaly_mask,
            threshold_used=threshold,
            extra_info={
                "eps": eps,
                "min_samples": min_samples,
                "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
                "n_noise_points": int(np.sum(anomaly_mask)),
            },
        )
    
    def _compute_scores(
        self,
        X_scaled: np.ndarray,
        labels: np.ndarray,
        clf: DBSCAN,
        eps: float,
        min_samples: int,
    ) -> np.ndarray:
        """
        Compute anomaly scores based on distance to nearest core point.
        
        Uses NearestNeighbors for efficient distance computation.
        """
        n_samples = X_scaled.shape[0]
        scores = np.zeros(n_samples)
        
        anomaly_mask = labels == -1
        
        # Get core sample indices
        core_indices = clf.core_sample_indices_
        
        if len(core_indices) == 0:
            # No clusters formed - all points are anomalies
            return np.ones(n_samples)
        
        # Get core points
        core_points = X_scaled[core_indices]
        
        # Use NearestNeighbors for efficient distance computation
        nn = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=-1)
        nn.fit(core_points)
        
        # Compute distances for anomalous points only
        if np.any(anomaly_mask):
            anomaly_points = X_scaled[anomaly_mask]
            distances, _ = nn.kneighbors(anomaly_points)
            scores[anomaly_mask] = distances[:, 0]
        
        # Non-anomalous points get score 0
        scores[~anomaly_mask] = 0.0
        
        return scores