"""
Anomaly Detection Engine for Graph Metrics

Provides multiple anomaly detection algorithms with consistent interface:
- Z-Score: Statistical z-score based outlier detection
- IQR: Interquartile Range based detection
- Isolation Forest: Tree-based anomaly detection
- LOF: Local Outlier Factor (density-based)
- DBSCAN: Clustering-based anomaly detection
- Mahalanobis: Distance-based detection accounting for correlations
"""

import time
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.covariance import MinCovDet
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARNING] scikit-learn not installed. Some anomaly algorithms will be unavailable.")


class AnomalyAlgorithm(str, Enum):
    """Supported anomaly detection algorithms."""
    ZSCORE = "zscore"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    LOF = "lof"
    DBSCAN = "dbscan"
    MAHALANOBIS = "mahalanobis"


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    scores: Dict[str, float]           # node_id -> anomaly_score (0-1, higher = more anomalous)
    binary_labels: Dict[str, bool]     # node_id -> is_anomaly
    algorithm: str
    parameters: Dict[str, Any]
    metrics_used: List[str]
    threshold_used: float
    n_anomalies: int
    n_total: int
    computation_time: float
    statistics: Dict[str, Any]         # mean, std, percentiles of scores
    top_anomalies: List[Dict[str, Any]]  # Top N most anomalous nodes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'scores': self.scores,
            'binary_labels': self.binary_labels,
            'algorithm': self.algorithm,
            'parameters': self.parameters,
            'metrics_used': self.metrics_used,
            'threshold_used': self.threshold_used,
            'n_anomalies': self.n_anomalies,
            'n_total': self.n_total,
            'computation_time': self.computation_time,
            'statistics': self.statistics,
            'top_anomalies': self.top_anomalies
        }


class AnomalyEngine:
    """
    Anomaly detection engine for graph metrics.
    
    Supports multiple algorithms with consistent interface.
    All algorithms return scores normalized to [0, 1] range.
    """
    
    ALGORITHMS = {
        "zscore": {
            "name": "Z-Score",
            "description": "Statistical z-score based outlier detection. Fast and simple, best for normally distributed data.",
            "complexity": "O(n)",
            "multivariate": True,
            "parameters": {
                "threshold": {
                    "type": "float", 
                    "default": 3.0, 
                    "min": 1.0, 
                    "max": 5.0, 
                    "description": "Number of standard deviations from mean"
                }
            }
        },
        "iqr": {
            "name": "Interquartile Range",
            "description": "IQR-based outlier detection, robust to extreme values. Good for skewed distributions.",
            "complexity": "O(n log n)",
            "multivariate": True,
            "parameters": {
                "multiplier": {
                    "type": "float", 
                    "default": 1.5, 
                    "min": 1.0, 
                    "max": 3.0, 
                    "description": "IQR multiplier for fence (1.5 = outlier, 3.0 = extreme)"
                }
            }
        },
        "isolation_forest": {
            "name": "Isolation Forest",
            "description": "Tree-based anomaly detection. Excellent for high-dimensional data, handles non-linear patterns.",
            "complexity": "O(n log n)",
            "multivariate": True,
            "parameters": {
                "n_estimators": {
                    "type": "int", 
                    "default": 100, 
                    "min": 50, 
                    "max": 500, 
                    "description": "Number of isolation trees"
                },
                "contamination": {
                    "type": "float", 
                    "default": 0.1, 
                    "min": 0.01, 
                    "max": 0.5, 
                    "description": "Expected proportion of outliers"
                },
                "random_state": {
                    "type": "int", 
                    "default": 42, 
                    "min": 0, 
                    "max": 9999, 
                    "description": "Random seed for reproducibility"
                }
            }
        },
        "lof": {
            "name": "Local Outlier Factor",
            "description": "Density-based local anomaly detection. Best for detecting local deviations from neighbors.",
            "complexity": "O(n²)",
            "multivariate": True,
            "parameters": {
                "n_neighbors": {
                    "type": "int", 
                    "default": 20, 
                    "min": 5, 
                    "max": 100, 
                    "description": "Number of neighbors for density estimation"
                },
                "contamination": {
                    "type": "float", 
                    "default": 0.1, 
                    "min": 0.01, 
                    "max": 0.5, 
                    "description": "Expected proportion of outliers"
                }
            }
        },
        "dbscan": {
            "name": "DBSCAN Clustering",
            "description": "Density-based clustering where anomalies are noise points. Good for clustered data.",
            "complexity": "O(n log n)",
            "multivariate": True,
            "parameters": {
                "eps": {
                    "type": "float", 
                    "default": 0.5, 
                    "min": 0.1, 
                    "max": 5.0, 
                    "description": "Maximum distance between points in cluster"
                },
                "min_samples": {
                    "type": "int", 
                    "default": 5, 
                    "min": 2, 
                    "max": 50, 
                    "description": "Minimum points required to form a cluster"
                }
            }
        },
        "mahalanobis": {
            "name": "Mahalanobis Distance",
            "description": "Distance-based detection accounting for correlations between features. Best for correlated metrics.",
            "complexity": "O(n × d²)",
            "multivariate": True,
            "parameters": {
                "threshold": {
                    "type": "float", 
                    "default": 3.0, 
                    "min": 2.0, 
                    "max": 5.0, 
                    "description": "Distance threshold (chi-squared based)"
                },
                "robust": {
                    "type": "bool",
                    "default": True,
                    "description": "Use robust covariance estimation (MinCovDet)"
                }
            }
        }
    }

    def __init__(self, max_nodes_for_lof: int = 50000):
        """
        Initialize anomaly engine with performance limits.
        
        Args:
            max_nodes_for_lof: Maximum nodes for LOF algorithm (O(n²) complexity)
        """
        self.max_nodes_for_lof = max_nodes_for_lof
    
    @classmethod
    def get_available_algorithms(cls) -> Dict[str, Any]:
        """Return algorithm specifications for frontend."""
        available = {}
        for algo_id, algo_info in cls.ALGORITHMS.items():
            # Check if sklearn is needed
            if algo_id in ['isolation_forest', 'lof', 'dbscan'] and not HAS_SKLEARN:
                continue
            available[algo_id] = algo_info
        return available
    
    @classmethod
    def recommend_algorithm(cls, n_nodes: int, n_metrics: int) -> str:
        """
        Recommend algorithm based on data characteristics.
        
        Args:
            n_nodes: Number of nodes in graph
            n_metrics: Number of metrics to analyze
            
        Returns:
            Recommended algorithm name
        """
        if n_nodes > 100000:
            return "zscore"  # Fastest for large graphs
        elif n_metrics == 1:
            return "iqr"  # Best for single metric
        elif n_nodes > 50000:
            return "isolation_forest"  # Good balance
        elif n_metrics <= 5:
            return "isolation_forest"  # Good multivariate
        else:
            return "isolation_forest"  # Handles high dimensions
    
    def detect_anomalies(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        algorithm: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> AnomalyResult:
        """
        Run anomaly detection on specified metrics.
        
        Args:
            df: DataFrame with 'avatar' column and metric columns
            metrics: List of metric column names to analyze
            algorithm: Algorithm name (see ALGORITHMS)
            parameters: Algorithm-specific parameters (uses defaults if None)
        
        Returns:
            AnomalyResult with scores, labels, and statistics
            
        Raises:
            ValueError: If algorithm unknown or metrics not found
        """
        start_time = time.time()
        
        # Validate algorithm
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(self.ALGORITHMS.keys())}")
        
        # Check sklearn availability
        if algorithm in ['isolation_forest', 'lof', 'dbscan'] and not HAS_SKLEARN:
            raise ValueError(f"Algorithm {algorithm} requires scikit-learn. Please install it.")
        
        # Validate metrics
        missing_metrics = [m for m in metrics if m not in df.columns]
        if missing_metrics:
            raise ValueError(f"Metrics not found in data: {missing_metrics}")
        
        # Get default parameters
        algo_info = self.ALGORITHMS[algorithm]
        params = {}
        for param_name, param_info in algo_info['parameters'].items():
            params[param_name] = param_info['default']
        
        # Override with provided parameters
        if parameters:
            for key, value in parameters.items():
                if key in params:
                    params[key] = value
        
        # Check LOF size limit
        if algorithm == 'lof' and len(df) > self.max_nodes_for_lof:
            print(f"[ANOMALY] Warning: LOF with {len(df)} nodes may be slow. Consider using isolation_forest.")
        
        # Run detection
        detection_methods = {
            'zscore': self._zscore_detection,
            'iqr': self._iqr_detection,
            'isolation_forest': self._isolation_forest_detection,
            'lof': self._lof_detection,
            'dbscan': self._dbscan_detection,
            'mahalanobis': self._mahalanobis_detection
        }
        
        result = detection_methods[algorithm](df, metrics, params)
        result.computation_time = time.time() - start_time
        
        print(f"[ANOMALY] {algorithm} completed in {result.computation_time:.2f}s. "
              f"Found {result.n_anomalies}/{result.n_total} anomalies ({result.n_anomalies/result.n_total*100:.1f}%)")
        
        return result
    
    def _prepare_data(self, df: pd.DataFrame, metrics: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare data matrix for analysis.
        
        Returns:
            Tuple of (data matrix, avatar IDs)
        """
        # Get avatar IDs
        if 'avatar' in df.columns:
            avatars = df['avatar'].tolist()
        else:
            avatars = df.index.tolist()
        
        # Extract metric values
        X = df[metrics].values.astype(np.float64)
        
        # Handle NaN and Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X, avatars
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores to [0, 1] range where 1 = most anomalous.
        
        Uses min-max normalization with clipping.
        """
        scores = np.array(scores, dtype=np.float64)
        
        # Handle constant values
        if np.std(scores) < 1e-10:
            return np.zeros_like(scores)
        
        # Min-max normalization
        min_val = np.min(scores)
        max_val = np.max(scores)
        
        if max_val - min_val < 1e-10:
            return np.zeros_like(scores)
        
        normalized = (scores - min_val) / (max_val - min_val)
        
        # Clip to [0, 1]
        return np.clip(normalized, 0.0, 1.0)
    
    def _build_result(
        self,
        df: pd.DataFrame,
        avatars: List[str],
        scores: np.ndarray,
        anomaly_mask: np.ndarray,
        algorithm: str,
        params: Dict[str, Any],
        metrics: List[str],
        threshold: float = 0.5
    ) -> AnomalyResult:
        """Build AnomalyResult from detection output."""
        
        # Normalize scores
        normalized_scores = self._normalize_scores(scores)
        
        # Build dictionaries
        score_dict = {avatar: float(score) for avatar, score in zip(avatars, normalized_scores)}
        label_dict = {avatar: bool(label) for avatar, label in zip(avatars, anomaly_mask)}
        
        # Calculate statistics
        stats_dict = {
            'min': float(np.min(normalized_scores)),
            'max': float(np.max(normalized_scores)),
            'mean': float(np.mean(normalized_scores)),
            'std': float(np.std(normalized_scores)),
            'median': float(np.median(normalized_scores)),
            'p25': float(np.percentile(normalized_scores, 25)),
            'p75': float(np.percentile(normalized_scores, 75)),
            'p90': float(np.percentile(normalized_scores, 90)),
            'p95': float(np.percentile(normalized_scores, 95)),
            'p99': float(np.percentile(normalized_scores, 99))
        }
        
        # Get top anomalies (sorted by score descending)
        sorted_indices = np.argsort(normalized_scores)[::-1]
        top_n = min(20, len(sorted_indices))
        
        top_anomalies = []
        for idx in sorted_indices[:top_n]:
            avatar = avatars[idx]
            anomaly_info = {
                'id': avatar,
                'score': float(normalized_scores[idx]),
                'is_anomaly': bool(anomaly_mask[idx])
            }
            # Add metric values
            for metric in metrics:
                if metric in df.columns:
                    val = df.iloc[idx][metric] if 'avatar' in df.columns else df.loc[avatar, metric]
                    anomaly_info[metric] = float(val) if pd.notna(val) else 0.0
            top_anomalies.append(anomaly_info)
        
        return AnomalyResult(
            scores=score_dict,
            binary_labels=label_dict,
            algorithm=algorithm,
            parameters=params,
            metrics_used=metrics,
            threshold_used=threshold,
            n_anomalies=int(np.sum(anomaly_mask)),
            n_total=len(avatars),
            computation_time=0.0,  # Will be set by caller
            statistics=stats_dict,
            top_anomalies=top_anomalies
        )
    
    def _zscore_detection(
        self, 
        df: pd.DataFrame, 
        metrics: List[str], 
        params: Dict[str, Any]
    ) -> AnomalyResult:
        """
        Z-Score based anomaly detection.
        
        Computes z-score for each metric and uses maximum absolute z-score
        across metrics as the anomaly score.
        """
        threshold = params.get('threshold', 3.0)
        X, avatars = self._prepare_data(df, metrics)
        
        # Calculate z-scores for each metric
        if X.shape[1] == 1:
            # Single metric
            z_scores = np.abs(stats.zscore(X[:, 0], nan_policy='omit'))
            z_scores = np.nan_to_num(z_scores, nan=0.0)
        else:
            # Multiple metrics: use max z-score across all metrics
            all_z = np.zeros_like(X)
            for i in range(X.shape[1]):
                col_z = stats.zscore(X[:, i], nan_policy='omit')
                col_z = np.nan_to_num(col_z, nan=0.0)
                all_z[:, i] = np.abs(col_z)
            z_scores = np.max(all_z, axis=1)
        
        # Determine anomalies
        anomaly_mask = z_scores > threshold
        
        # Raw scores are the z-scores (will be normalized in _build_result)
        raw_scores = z_scores
        
        return self._build_result(df, avatars, raw_scores, anomaly_mask, 'zscore', params, metrics, threshold)
    
    def _iqr_detection(
        self, 
        df: pd.DataFrame, 
        metrics: List[str], 
        params: Dict[str, Any]
    ) -> AnomalyResult:
        """
        Interquartile Range (IQR) based anomaly detection.
        
        Robust to extreme values. Uses Tukey's method:
        - Outliers: value < Q1 - k*IQR or value > Q3 + k*IQR
        """
        multiplier = params.get('multiplier', 1.5)
        X, avatars = self._prepare_data(df, metrics)
        
        # Calculate IQR-based scores for each metric
        iqr_scores = np.zeros(X.shape[0])
        anomaly_counts = np.zeros(X.shape[0])
        
        for i in range(X.shape[1]):
            col = X[:, i]
            q1, q3 = np.percentile(col, [25, 75])
            iqr = q3 - q1
            
            if iqr < 1e-10:
                continue
            
            lower_fence = q1 - multiplier * iqr
            upper_fence = q3 + multiplier * iqr
            
            # Distance from fence (normalized by IQR)
            below_lower = np.maximum(0, lower_fence - col) / iqr
            above_upper = np.maximum(0, col - upper_fence) / iqr
            
            col_scores = np.maximum(below_lower, above_upper)
            iqr_scores += col_scores
            
            # Count anomalies per metric
            anomaly_counts += ((col < lower_fence) | (col > upper_fence)).astype(int)
        
        # Aggregate scores
        if X.shape[1] > 1:
            iqr_scores /= X.shape[1]  # Average across metrics
        
        # Anomaly if outside fence for any metric
        anomaly_mask = anomaly_counts > 0
        
        return self._build_result(df, avatars, iqr_scores, anomaly_mask, 'iqr', params, metrics, multiplier)
    
    def _isolation_forest_detection(
        self, 
        df: pd.DataFrame, 
        metrics: List[str], 
        params: Dict[str, Any]
    ) -> AnomalyResult:
        """
        Isolation Forest anomaly detection.
        
        Isolates anomalies by randomly selecting features and split values.
        Anomalies are easier to isolate, requiring fewer splits.
        """
        if not HAS_SKLEARN:
            raise ValueError("Isolation Forest requires scikit-learn")
        
        n_estimators = params.get('n_estimators', 100)
        contamination = params.get('contamination', 0.1)
        random_state = params.get('random_state', 42)
        
        X, avatars = self._prepare_data(df, metrics)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit Isolation Forest
        clf = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        clf.fit(X_scaled)
        
        # Get anomaly scores
        # decision_function returns negative scores for anomalies
        raw_scores = -clf.decision_function(X_scaled)
        
        # Get binary predictions
        predictions = clf.predict(X_scaled)
        anomaly_mask = predictions == -1
        
        return self._build_result(df, avatars, raw_scores, anomaly_mask, 'isolation_forest', params, metrics, contamination)
    
    def _lof_detection(
        self, 
        df: pd.DataFrame, 
        metrics: List[str], 
        params: Dict[str, Any]
    ) -> AnomalyResult:
        """
        Local Outlier Factor (LOF) anomaly detection.
        
        Measures local deviation of density with respect to neighbors.
        """
        if not HAS_SKLEARN:
            raise ValueError("LOF requires scikit-learn")
        
        n_neighbors = params.get('n_neighbors', 20)
        contamination = params.get('contamination', 0.1)
        
        X, avatars = self._prepare_data(df, metrics)
        
        # Adjust n_neighbors if larger than dataset
        n_neighbors = min(n_neighbors, len(X) - 1)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit LOF
        clf = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            n_jobs=-1,
            novelty=False  # Use for outlier detection (not novelty detection)
        )
        
        # fit_predict returns -1 for outliers, 1 for inliers
        predictions = clf.fit_predict(X_scaled)
        anomaly_mask = predictions == -1
        
        # Get LOF scores (negative_outlier_factor_ is negative, more negative = more anomalous)
        raw_scores = -clf.negative_outlier_factor_
        
        return self._build_result(df, avatars, raw_scores, anomaly_mask, 'lof', params, metrics, contamination)
    
    def _dbscan_detection(
        self, 
        df: pd.DataFrame, 
        metrics: List[str], 
        params: Dict[str, Any]
    ) -> AnomalyResult:
        """
        DBSCAN-based anomaly detection.
        
        Points not belonging to any cluster (noise) are considered anomalies.
        """
        if not HAS_SKLEARN:
            raise ValueError("DBSCAN requires scikit-learn")
        
        eps = params.get('eps', 0.5)
        min_samples = params.get('min_samples', 5)
        
        X, avatars = self._prepare_data(df, metrics)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit DBSCAN
        clf = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            n_jobs=-1
        )
        labels = clf.fit_predict(X_scaled)
        
        # -1 label means noise (anomaly)
        anomaly_mask = labels == -1
        
        # Calculate anomaly scores based on distance to nearest core point
        scores = np.zeros(len(X))
        
        if np.sum(~anomaly_mask) > 0:  # If there are clustered points
            # For each point, calculate minimum distance to a core point
            core_mask = np.zeros(len(X), dtype=bool)
            for i, label in enumerate(labels):
                if label >= 0:
                    # Check if it's a core point
                    neighbors = np.sum(np.linalg.norm(X_scaled - X_scaled[i], axis=1) <= eps)
                    if neighbors >= min_samples:
                        core_mask[i] = True
            
            if np.sum(core_mask) > 0:
                core_points = X_scaled[core_mask]
                for i in range(len(X)):
                    if anomaly_mask[i]:
                        # Distance to nearest core point
                        distances = np.linalg.norm(core_points - X_scaled[i], axis=1)
                        scores[i] = np.min(distances)
                    else:
                        scores[i] = 0.0
        else:
            # All points are anomalies (no clusters formed)
            scores = np.ones(len(X))
        
        return self._build_result(df, avatars, scores, anomaly_mask, 'dbscan', params, metrics, eps)
    
    def _mahalanobis_detection(
        self, 
        df: pd.DataFrame, 
        metrics: List[str], 
        params: Dict[str, Any]
    ) -> AnomalyResult:
        """
        Mahalanobis distance based anomaly detection.
        
        Accounts for correlations between features.
        """
        threshold = params.get('threshold', 3.0)
        use_robust = params.get('robust', True)
        
        X, avatars = self._prepare_data(df, metrics)
        
        # Need at least 2 features for Mahalanobis
        if X.shape[1] < 2:
            # Fall back to z-score for single feature
            return self._zscore_detection(df, metrics, {'threshold': threshold})
        
        # Calculate covariance and mean
        if use_robust and HAS_SKLEARN:
            try:
                # Use robust covariance estimation (MinCovDet)
                # This is less sensitive to outliers
                robust_cov = MinCovDet(random_state=42).fit(X)
                mean = robust_cov.location_
                cov_matrix = robust_cov.covariance_
            except Exception:
                # Fall back to empirical covariance
                mean = np.mean(X, axis=0)
                cov_matrix = np.cov(X.T)
        else:
            mean = np.mean(X, axis=0)
            cov_matrix = np.cov(X.T)
        
        # Ensure covariance matrix is invertible
        try:
            # Add small regularization
            cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6
            cov_inv = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse
            cov_inv = np.linalg.pinv(cov_matrix)
        
        # Calculate Mahalanobis distance for each point
        distances = np.zeros(len(X))
        for i in range(len(X)):
            diff = X[i] - mean
            distances[i] = np.sqrt(diff @ cov_inv @ diff)
        
        # Threshold based on chi-squared distribution
        # For p features, squared Mahalanobis distance follows chi-squared(p)
        p = X.shape[1]
        actual_threshold = threshold * np.sqrt(p)  # Scale threshold by dimensions
        
        anomaly_mask = distances > actual_threshold
        
        return self._build_result(df, avatars, distances, anomaly_mask, 'mahalanobis', params, metrics, threshold)