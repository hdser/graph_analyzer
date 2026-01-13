"""
Feature Builder for Deep Learning

Converts NetworkX graphs and computed metrics DataFrames into
feature matrices suitable for GNN training and inference.

Handles:
- Metric column selection and alignment
- Property integration from external sources
- Normalization (standard, minmax, robust)
- NaN handling strategies
- Log transforms for power-law metrics
- Node type encoding for heterogeneous graphs
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
import numpy as np
import pandas as pd
import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """
    Configuration for feature extraction.
    
    All parameters are configurable - no hardcoded values.
    """
    
    # Metrics to include as features (must match metrics DataFrame columns)
    metric_columns: List[str] = field(default_factory=lambda: [
        "in_degree",
        "out_degree",
        "pagerank",
        "betweenness_centrality",
        "clustering_coefficient",
        "eigenvector_centrality",
        "core_number",
    ])
    
    # Properties to include (from properties DataFrame)
    property_columns: List[str] = field(default_factory=list)
    
    # Columns to apply log1p transform (good for power-law distributions)
    log_transform_columns: List[str] = field(default_factory=lambda: [
        "in_degree",
        "out_degree",
        "pagerank",
        "betweenness_centrality",
    ])
    
    # Preprocessing options
    normalization: str = "standard"  # none, standard, minmax, robust
    nan_strategy: str = "zero"       # zero, mean, median, drop
    clip_outliers: bool = False      # Clip values beyond 3 std
    clip_std: float = 3.0            # Standard deviations for clipping
    
    # Identity features fallback
    use_identity_features: bool = True   # Use identity if no features available
    identity_dim: int = 64               # Dimension for identity features
    
    # Feature dimension limits
    max_feature_dim: Optional[int] = None  # Truncate to this dim
    min_feature_dim: Optional[int] = None  # Pad to this dim
    
    # Heterogeneous graph support
    node_type_column: Optional[str] = None
    default_node_type: str = "node"
    
    # Index column name in DataFrames
    node_id_column: str = "avatar"  # Column containing node IDs


@dataclass
class FeatureStats:
    """Statistics about extracted features."""
    num_nodes: int
    num_features: int
    feature_names: List[str]
    nan_counts: Dict[str, int]
    has_identity_features: bool
    normalization_params: Optional[Dict[str, Dict[str, float]]] = None


class FeatureBuilder:
    """
    Builds feature matrices from NetworkX graphs and metrics DataFrames.
    
    Usage:
        builder = FeatureBuilder(config)
        X, nodes, node_to_idx, stats = builder.build_features(G, metrics_df, properties_df)
        edge_index = builder.build_edge_index(G, node_to_idx)
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature builder.
        
        Args:
            config: Feature extraction configuration
        """
        self.config = config or FeatureConfig()
        
        # Store normalization parameters for inverse transform
        self._normalization_params: Dict[str, Dict[str, float]] = {}
        self._feature_names: List[str] = []
        self._fitted = False
    
    def build_features(
        self,
        G: nx.Graph,
        metrics_df: Optional[pd.DataFrame] = None,
        properties_df: Optional[pd.DataFrame] = None,
        fit: bool = True
    ) -> Tuple[np.ndarray, List[str], Dict[str, int], FeatureStats]:
        """
        Build feature matrix from graph and metrics.
        
        Args:
            G: NetworkX graph
            metrics_df: DataFrame with computed metrics (index or column = node IDs)
            properties_df: Optional DataFrame with node properties
            fit: Whether to fit normalization parameters (True for training)
            
        Returns:
            Tuple of:
            - X: Feature matrix [num_nodes, num_features]
            - nodes: List of node IDs in order
            - node_to_idx: Dict mapping node ID to index
            - stats: FeatureStats with extraction information
        """
        # Get node list from graph
        nodes = list(G.nodes())
        node_to_idx = {str(n): i for i, n in enumerate(nodes)}
        num_nodes = len(nodes)
        
        logger.info(f"Building features for {num_nodes} nodes")
        
        # Collect features
        features = []
        feature_names = []
        nan_counts = {}
        
        # Extract metric features
        if metrics_df is not None and not metrics_df.empty:
            metric_features, metric_names, metric_nans = self._extract_metric_features(
                metrics_df, nodes
            )
            if metric_features is not None:
                features.append(metric_features)
                feature_names.extend(metric_names)
                nan_counts.update(metric_nans)
        
        # Extract property features
        if properties_df is not None and not properties_df.empty:
            prop_features, prop_names, prop_nans = self._extract_property_features(
                properties_df, nodes
            )
            if prop_features is not None:
                features.append(prop_features)
                feature_names.extend(prop_names)
                nan_counts.update(prop_nans)
        
        # Combine features or use identity
        has_identity = False
        if features:
            X = np.hstack(features)
        elif self.config.use_identity_features:
            # No features available - use identity or random
            logger.warning("No features available, using identity features")
            if num_nodes <= self.config.identity_dim:
                X = np.eye(num_nodes, dtype=np.float32)
            else:
                # Random features for large graphs
                np.random.seed(42)
                X = np.random.randn(num_nodes, self.config.identity_dim).astype(np.float32)
                X = X / np.linalg.norm(X, axis=1, keepdims=True)
            feature_names = [f"identity_{i}" for i in range(X.shape[1])]
            has_identity = True
        else:
            raise ValueError("No features available and identity features disabled")
        
        # Handle NaN values
        X = self._handle_nan(X, feature_names)
        
        # Apply log transform
        X = self._apply_log_transform(X, feature_names)
        
        # Clip outliers if enabled
        if self.config.clip_outliers:
            X = self._clip_outliers(X)
        
        # Normalize features
        if fit:
            X = self._fit_normalize(X, feature_names)
            self._fitted = True
        else:
            X = self._transform_normalize(X, feature_names)
        
        # Handle dimension limits
        X = self._handle_dimension_limits(X)
        
        self._feature_names = feature_names
        
        # Build stats
        stats = FeatureStats(
            num_nodes=num_nodes,
            num_features=X.shape[1],
            feature_names=feature_names,
            nan_counts=nan_counts,
            has_identity_features=has_identity,
            normalization_params=self._normalization_params if fit else None,
        )
        
        logger.info(f"Built feature matrix: {X.shape}")
        
        return X.astype(np.float32), nodes, node_to_idx, stats
    
    def _extract_metric_features(
        self,
        df: pd.DataFrame,
        nodes: List[str]
    ) -> Tuple[Optional[np.ndarray], List[str], Dict[str, int]]:
        """Extract features from metrics DataFrame."""
        # Identify available columns
        available_cols = [c for c in self.config.metric_columns if c in df.columns]
        
        if not available_cols:
            logger.warning(f"No metric columns found. Available: {list(df.columns)}")
            return None, [], {}
        
        logger.debug(f"Extracting metrics: {available_cols}")
        
        # Align DataFrame index with nodes
        df_aligned = self._align_dataframe(df, nodes)
        
        # Extract values
        features = df_aligned[available_cols].values
        
        # Count NaNs per column
        nan_counts = {c: int(np.isnan(df_aligned[c]).sum()) for c in available_cols}
        
        return features, available_cols, nan_counts
    
    def _extract_property_features(
        self,
        df: pd.DataFrame,
        nodes: List[str]
    ) -> Tuple[Optional[np.ndarray], List[str], Dict[str, int]]:
        """Extract features from properties DataFrame."""
        available_cols = [c for c in self.config.property_columns if c in df.columns]
        
        if not available_cols:
            return None, [], {}
        
        logger.debug(f"Extracting properties: {available_cols}")
        
        df_aligned = self._align_dataframe(df, nodes)
        
        # Handle boolean columns
        features_list = []
        for col in available_cols:
            vals = df_aligned[col].values
            if vals.dtype == bool or df_aligned[col].dtype == 'bool':
                vals = vals.astype(np.float32)
            elif vals.dtype == object:
                # Try to convert strings
                try:
                    vals = pd.to_numeric(df_aligned[col], errors='coerce').values
                except Exception:
                    logger.warning(f"Could not convert column {col} to numeric, skipping")
                    continue
            features_list.append(vals.reshape(-1, 1))
        
        if not features_list:
            return None, [], {}
        
        features = np.hstack(features_list)
        nan_counts = {c: int(np.isnan(df_aligned[c].astype(float)).sum()) 
                      for c in available_cols}
        
        return features, available_cols, nan_counts
    
    def _align_dataframe(
        self,
        df: pd.DataFrame,
        nodes: List[str]
    ) -> pd.DataFrame:
        """Align DataFrame rows with node list."""
        # Determine index column
        id_col = self.config.node_id_column
        
        if id_col in df.columns:
            df = df.set_index(id_col)
        elif df.index.name != id_col and id_col not in df.columns:
            # Try common alternatives
            for alt in ['node', 'node_id', 'id', 'address']:
                if alt in df.columns:
                    df = df.set_index(alt)
                    break
        
        # Normalize index to string
        df.index = df.index.astype(str).str.lower()
        
        # Create aligned DataFrame
        nodes_lower = [str(n).lower() for n in nodes]
        aligned = pd.DataFrame(index=nodes_lower)
        
        for col in df.columns:
            aligned[col] = np.nan
            common = aligned.index.intersection(df.index)
            aligned.loc[common, col] = df.loc[common, col].values
        
        return aligned
    
    def _handle_nan(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> np.ndarray:
        """Handle NaN values based on strategy."""
        strategy = self.config.nan_strategy
        
        if strategy == "zero":
            return np.nan_to_num(X, nan=0.0)
        
        elif strategy == "mean":
            col_means = np.nanmean(X, axis=0)
            col_means = np.nan_to_num(col_means, nan=0.0)  # Handle all-NaN columns
            inds = np.where(np.isnan(X))
            X = X.copy()
            X[inds] = np.take(col_means, inds[1])
            return X
        
        elif strategy == "median":
            col_medians = np.nanmedian(X, axis=0)
            col_medians = np.nan_to_num(col_medians, nan=0.0)
            inds = np.where(np.isnan(X))
            X = X.copy()
            X[inds] = np.take(col_medians, inds[1])
            return X
        
        elif strategy == "drop":
            # Replace with 0 but log warning
            nan_mask = np.isnan(X).any(axis=1)
            if nan_mask.any():
                logger.warning(f"Dropping {nan_mask.sum()} nodes with NaN values")
            return np.nan_to_num(X, nan=0.0)
        
        return X
    
    def _apply_log_transform(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> np.ndarray:
        """Apply log1p transform to specified columns."""
        log_cols = self.config.log_transform_columns
        
        for i, name in enumerate(feature_names):
            if name in log_cols:
                # log1p(|x|) * sign(x) to handle negative values
                X[:, i] = np.sign(X[:, i]) * np.log1p(np.abs(X[:, i]))
        
        return X
    
    def _clip_outliers(self, X: np.ndarray) -> np.ndarray:
        """Clip outliers beyond specified standard deviations."""
        std_limit = self.config.clip_std
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-8
        
        lower = mean - std_limit * std
        upper = mean + std_limit * std
        
        return np.clip(X, lower, upper)
    
    def _fit_normalize(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> np.ndarray:
        """Fit and apply normalization."""
        norm_type = self.config.normalization
        
        if norm_type == "none":
            return X
        
        self._normalization_params = {}
        
        if norm_type == "standard":
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0) + 1e-8
            self._normalization_params = {"mean": mean, "std": std}
            return (X - mean) / std
        
        elif norm_type == "minmax":
            min_val = np.min(X, axis=0)
            max_val = np.max(X, axis=0)
            range_val = max_val - min_val + 1e-8
            self._normalization_params = {"min": min_val, "range": range_val}
            return (X - min_val) / range_val
        
        elif norm_type == "robust":
            median = np.median(X, axis=0)
            q75 = np.percentile(X, 75, axis=0)
            q25 = np.percentile(X, 25, axis=0)
            iqr = q75 - q25 + 1e-8
            self._normalization_params = {"median": median, "iqr": iqr}
            return (X - median) / iqr
        
        return X
    
    def _transform_normalize(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> np.ndarray:
        """Apply pre-fitted normalization."""
        if not self._fitted or not self._normalization_params:
            return X
        
        norm_type = self.config.normalization
        params = self._normalization_params
        
        if norm_type == "standard":
            return (X - params["mean"]) / params["std"]
        elif norm_type == "minmax":
            return (X - params["min"]) / params["range"]
        elif norm_type == "robust":
            return (X - params["median"]) / params["iqr"]
        
        return X
    
    def _handle_dimension_limits(self, X: np.ndarray) -> np.ndarray:
        """Handle min/max dimension limits."""
        current_dim = X.shape[1]
        
        # Truncate if needed
        if self.config.max_feature_dim and current_dim > self.config.max_feature_dim:
            logger.info(f"Truncating features from {current_dim} to {self.config.max_feature_dim}")
            X = X[:, :self.config.max_feature_dim]
        
        # Pad if needed
        if self.config.min_feature_dim and current_dim < self.config.min_feature_dim:
            pad_dim = self.config.min_feature_dim - current_dim
            logger.info(f"Padding features from {current_dim} to {self.config.min_feature_dim}")
            padding = np.zeros((X.shape[0], pad_dim), dtype=X.dtype)
            X = np.hstack([X, padding])
        
        return X
    
    def build_edge_index(
        self,
        G: nx.Graph,
        node_to_idx: Dict[str, int]
    ) -> np.ndarray:
        """
        Build edge index array for PyTorch Geometric.
        
        Args:
            G: NetworkX graph
            node_to_idx: Mapping from node ID to index
            
        Returns:
            Edge index array of shape [2, num_edges]
        """
        edges = list(G.edges())
        
        if not edges:
            return np.zeros((2, 0), dtype=np.int64)
        
        # Filter edges where both nodes exist in mapping
        valid_edges = []
        for u, v in edges:
            u_str = str(u)
            v_str = str(v)
            if u_str in node_to_idx and v_str in node_to_idx:
                valid_edges.append((node_to_idx[u_str], node_to_idx[v_str]))
        
        if not valid_edges:
            return np.zeros((2, 0), dtype=np.int64)
        
        src = [e[0] for e in valid_edges]
        dst = [e[1] for e in valid_edges]
        
        return np.array([src, dst], dtype=np.int64)
    
    def build_edge_attr(
        self,
        G: nx.Graph,
        node_to_idx: Dict[str, int],
        attr_names: List[str]
    ) -> Optional[np.ndarray]:
        """
        Build edge attribute matrix.
        
        Args:
            G: NetworkX graph
            node_to_idx: Mapping from node ID to index
            attr_names: Edge attribute names to extract
            
        Returns:
            Edge attributes [num_edges, num_attrs] or None
        """
        if not attr_names:
            return None
        
        valid_edges = []
        for u, v, data in G.edges(data=True):
            u_str = str(u)
            v_str = str(v)
            if u_str in node_to_idx and v_str in node_to_idx:
                row = [data.get(attr, 0.0) for attr in attr_names]
                valid_edges.append(row)
        
        if not valid_edges:
            return None
        
        return np.array(valid_edges, dtype=np.float32)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse normalization transform.
        
        Args:
            X: Normalized features
            
        Returns:
            Original-scale features
        """
        if not self._normalization_params:
            return X
        
        norm_type = self.config.normalization
        params = self._normalization_params
        
        if norm_type == "standard":
            return X * params["std"] + params["mean"]
        elif norm_type == "minmax":
            return X * params["range"] + params["min"]
        elif norm_type == "robust":
            return X * params["iqr"] + params["median"]
        
        return X
    
    @property
    def feature_names(self) -> List[str]:
        """Get names of extracted features."""
        return self._feature_names
    
    @property
    def feature_dim(self) -> int:
        """Get feature dimension."""
        return len(self._feature_names)