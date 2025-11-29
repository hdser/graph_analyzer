"""
Machine Learning-based Anomaly Detection Algorithms

Implements:
- Isolation Forest: Tree-based anomaly detection
- LOF: Local Outlier Factor (density-based)
- DBSCAN: Clustering-based anomaly detection
- PCA Reconstruction Error: Manifold-based anomaly detection
- One-Class SVM: Boundary-based anomaly detection
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
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM


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


class PCAReconstructionAlgorithm(AnomalyAlgorithmBase):
    """
    PCA Reconstruction Error anomaly detection.
    
    Uses Principal Component Analysis to project data to a lower-dimensional
    subspace, then reconstructs back to original space. The reconstruction
    error (difference between original and reconstructed) serves as the
    anomaly score.
    
    Nodes whose structural pattern doesn't fit the main "manifold" of the
    graph will have high reconstruction error.
    
    Best for detecting global structural anomalies in high-dimensional data.
    """
    
    name = "pca_reconstruction"
    display_name = "PCA Reconstruction Error"
    description = (
        "Detects anomalies by measuring reconstruction error after PCA projection. "
        "Nodes that don't fit the main data manifold have high reconstruction error. "
        "Excellent for high-dimensional structural patterns."
    )
    complexity = "O(n × d² + n × d × k)"
    supports_multivariate = True
    requires_sklearn = True
    
    parameters = {
        "n_components": ParameterSpec(
            name="n_components",
            param_type="str",
            default="0.95",
            choices=["0.50", "0.75", "0.90", "0.95", "0.99", "auto"],
            description="Variance ratio to retain (0.95 = 95%) or 'auto' for elbow method",
        ),
        "contamination": ParameterSpec(
            name="contamination",
            param_type="float",
            default=0.1,
            min_value=0.001,
            max_value=0.5,
            description="Expected proportion of outliers",
        ),
        "whiten": ParameterSpec(
            name="whiten",
            param_type="bool",
            default=False,
            description="Whiten components (normalize variance)",
        ),
        "random_state": ParameterSpec(
            name="random_state",
            param_type="int",
            default=42,
            min_value=0,
            max_value=99999,
            description="Random seed for reproducibility",
        ),
    }
    
    def fit_predict(
        self,
        X: np.ndarray,
        params: Dict[str, Any],
    ) -> AlgorithmOutput:
        """
        Run PCA reconstruction error anomaly detection.
        
        Algorithm:
        1. Standardize input data
        2. Fit PCA with specified n_components
        3. Transform data to reduced space
        4. Inverse transform back to original space
        5. Compute reconstruction error per sample
        6. Use contamination percentile as threshold
        """
        self._validate_input(X)
        
        n_components_str = params.get("n_components", "0.95")
        contamination = params.get("contamination", 0.1)
        whiten = params.get("whiten", False)
        random_state = params.get("random_state", 42)
        
        n_samples, n_features = X.shape
        
        # Standardize features - handle constant columns
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Replace any NaN/inf from scaling (e.g., constant columns with 0 std)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Determine n_components
        if n_components_str == "auto":
            # Use elbow method: find where explained variance ratio drops
            n_components = self._find_elbow_components(X_scaled, random_state)
        else:
            # Parse as variance ratio
            n_components = float(n_components_str)
        
        # Ensure n_components is valid
        max_components = min(n_samples, n_features)
        if isinstance(n_components, float) and n_components < 1.0:
            # It's a variance ratio - PCA will determine actual count
            # But ensure we have at least 1 component and not more than max
            pass
        else:
            n_components = min(int(n_components), max_components - 1)
            if n_components < 1:
                n_components = max(1, max_components - 1)
        
        # Handle edge case: if only 1 feature, can't do meaningful PCA
        if n_features == 1:
            # Fall back to simple variance-based scoring
            raw_scores = np.abs(X_scaled[:, 0])
            raw_scores = np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0)
            threshold = float(np.percentile(raw_scores, 100 * (1 - contamination)))
            anomaly_mask = raw_scores > threshold
            return AlgorithmOutput(
                raw_scores=raw_scores,
                anomaly_mask=anomaly_mask,
                threshold_used=threshold,
                per_metric_scores={"feature_0": raw_scores},
                extra_info={
                    "n_components_used": 1,
                    "n_features": 1,
                    "explained_variance_ratio": 1.0,
                    "single_feature_fallback": True,
                },
            )
        
        # Fit PCA
        pca = PCA(
            n_components=n_components,
            whiten=whiten,
            random_state=random_state,
        )
        
        # Transform to reduced space and back
        X_transformed = pca.fit_transform(X_scaled)
        X_reconstructed = pca.inverse_transform(X_transformed)
        
        # Compute reconstruction error (squared L2 norm per sample)
        reconstruction_error = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)
        
        # Also compute per-feature reconstruction error for diagnostics
        per_feature_error = (X_scaled - X_reconstructed) ** 2
        
        # Normalize scores (use sqrt for more interpretable scores)
        raw_scores = np.sqrt(np.maximum(reconstruction_error, 0))
        
        # Clean up any NaN/inf values that might have slipped through
        raw_scores = np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Compute threshold based on contamination
        threshold = float(np.percentile(raw_scores, 100 * (1 - contamination)))
        
        # Determine anomalies
        anomaly_mask = raw_scores > threshold
        
        # Compute per-metric scores with NaN handling
        per_metric_scores = {}
        for i in range(n_features):
            feature_scores = np.sqrt(np.maximum(per_feature_error[:, i], 0))
            feature_scores = np.nan_to_num(feature_scores, nan=0.0, posinf=0.0, neginf=0.0)
            per_metric_scores[f"feature_{i}"] = feature_scores
        
        # Get actual number of components used
        actual_n_components = pca.n_components_
        explained_variance_ratio = float(np.sum(pca.explained_variance_ratio_))
        
        # Ensure all extra_info values are JSON serializable
        mean_score = float(np.nan_to_num(np.mean(raw_scores), nan=0.0))
        max_score = float(np.nan_to_num(np.max(raw_scores), nan=0.0))
        
        return AlgorithmOutput(
            raw_scores=raw_scores,
            anomaly_mask=anomaly_mask,
            threshold_used=threshold,
            per_metric_scores=per_metric_scores,
            extra_info={
                "n_components_used": int(actual_n_components),
                "n_features": int(n_features),
                "explained_variance_ratio": explained_variance_ratio,
                "contamination": float(contamination),
                "whiten": bool(whiten),
                "n_anomalies_detected": int(np.sum(anomaly_mask)),
                "mean_reconstruction_error": mean_score,
                "max_reconstruction_error": max_score,
            },
        )
    
    def _find_elbow_components(
        self,
        X_scaled: np.ndarray,
        random_state: int,
    ) -> int:
        """
        Find optimal number of components using elbow method.
        
        Looks for the "elbow" in the explained variance curve.
        """
        n_samples, n_features = X_scaled.shape
        max_components = min(n_samples, n_features, 50)  # Cap at 50 for efficiency
        
        # Fit full PCA to get explained variance
        pca_full = PCA(n_components=max_components, random_state=random_state)
        pca_full.fit(X_scaled)
        
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        
        # Find elbow: where adding more components gives diminishing returns
        # Use the point where we reach 90% variance or the steepest drop
        for i, var in enumerate(cumsum):
            if var >= 0.90:
                return max(2, i + 1)  # At least 2 components
        
        # If we never reach 90%, use 75% of available components
        return max(2, int(max_components * 0.75))


class OneClassSVMAlgorithm(AnomalyAlgorithmBase):
    """
    One-Class SVM anomaly detection.
    
    Learns a decision boundary around "normal" data using Support Vector
    Machines. Points outside the boundary are classified as anomalies.
    
    Uses RBF kernel by default to capture complex, non-linear boundaries
    between normal and anomalous patterns.
    
    Best for dense graphs with complex patterns and correlated features.
    Can be slower than tree-based methods for large datasets.
    """
    
    name = "one_class_svm"
    display_name = "One-Class SVM"
    description = (
        "Boundary-based detection using Support Vector Machines with RBF kernel. "
        "Learns a decision boundary around normal data patterns. "
        "Best for complex, non-linear patterns but slower on large datasets."
    )
    complexity = "O(n² × d) to O(n³)"
    supports_multivariate = True
    requires_sklearn = True
    
    parameters = {
        "nu": ParameterSpec(
            name="nu",
            param_type="float",
            default=0.1,
            min_value=0.001,
            max_value=0.5,
            description="Upper bound on outlier fraction and lower bound on support vectors",
        ),
        "kernel": ParameterSpec(
            name="kernel",
            param_type="str",
            default="rbf",
            choices=["rbf", "linear", "poly", "sigmoid"],
            description="Kernel type for SVM",
        ),
        "gamma": ParameterSpec(
            name="gamma",
            param_type="str",
            default="scale",
            choices=["scale", "auto", "0.001", "0.01", "0.1", "1.0"],
            description="Kernel coefficient (scale=1/(n_features*var), auto=1/n_features)",
        ),
        "degree": ParameterSpec(
            name="degree",
            param_type="int",
            default=3,
            min_value=2,
            max_value=10,
            description="Degree for polynomial kernel (ignored by other kernels)",
        ),
        "shrinking": ParameterSpec(
            name="shrinking",
            param_type="bool",
            default=True,
            description="Use shrinking heuristic (faster training)",
        ),
        "max_iter": ParameterSpec(
            name="max_iter",
            param_type="int",
            default=10000,
            min_value=1000,
            max_value=100000,
            description="Maximum iterations (-1 for no limit, but we cap it)",
        ),
    }
    
    def fit_predict(
        self,
        X: np.ndarray,
        params: Dict[str, Any],
    ) -> AlgorithmOutput:
        """
        Run One-Class SVM anomaly detection.
        
        Algorithm:
        1. Standardize input data
        2. Fit OneClassSVM with specified parameters
        3. Get decision function scores (distance to boundary)
        4. Convert to anomaly scores (negative scores = anomaly)
        5. Use nu-based classification for binary labels
        """
        self._validate_input(X)
        
        nu = params.get("nu", 0.1)
        kernel = params.get("kernel", "rbf")
        gamma_str = params.get("gamma", "scale")
        degree = params.get("degree", 3)
        shrinking = params.get("shrinking", True)
        max_iter = params.get("max_iter", 10000)
        
        n_samples, n_features = X.shape
        
        # Standardize features (critical for SVM performance)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Replace any NaN/inf from scaling (e.g., constant columns with 0 std)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Parse gamma
        if gamma_str in ["scale", "auto"]:
            gamma = gamma_str
        else:
            gamma = float(gamma_str)
        
        # Fit One-Class SVM
        clf = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            shrinking=shrinking,
            max_iter=max_iter,
            cache_size=500,  # Increase cache for better performance
        )
        
        clf.fit(X_scaled)
        
        # Get predictions: +1 for inliers, -1 for outliers
        predictions = clf.predict(X_scaled)
        anomaly_mask = predictions == -1
        
        # Get decision function scores
        # Positive = inside boundary (normal), Negative = outside (anomaly)
        decision_scores = clf.decision_function(X_scaled)
        
        # Clean up any NaN values
        decision_scores = np.nan_to_num(decision_scores, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert to anomaly scores: higher = more anomalous
        # Negate so that points far outside boundary get high scores
        raw_scores = -decision_scores
        
        # Shift scores to be non-negative for easier interpretation
        min_score = float(np.min(raw_scores))
        if min_score < 0:
            raw_scores = raw_scores - min_score
        
        # Final NaN cleanup
        raw_scores = np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Compute threshold (score at decision boundary)
        # The decision boundary is at decision_function = 0
        threshold = -min_score if min_score < 0 else 0.0
        
        # Get support vector info
        n_support = len(clf.support_)
        support_fraction = n_support / n_samples
        
        return AlgorithmOutput(
            raw_scores=raw_scores,
            anomaly_mask=anomaly_mask,
            threshold_used=float(threshold),
            extra_info={
                "nu": float(nu),
                "kernel": str(kernel),
                "gamma": str(gamma_str),
                "n_support_vectors": int(n_support),
                "support_vector_fraction": float(support_fraction),
                "n_anomalies_detected": int(np.sum(anomaly_mask)),
                "converged": bool(clf.fit_status_ == 0) if hasattr(clf, 'fit_status_') else True,
                "n_iterations": int(clf.n_iter_) if hasattr(clf, 'n_iter_') else -1,
            },
        )