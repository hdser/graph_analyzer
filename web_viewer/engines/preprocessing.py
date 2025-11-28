"""
Preprocessing Pipeline for Anomaly Detection

Handles all data preparation:
- Metric validation and filtering
- Per-metric transforms (log, clip, etc.)
- NaN/Inf handling
- Global scaling
- Memory-efficient operations
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .anomaly_config import (
    MetricConfig, 
    MetricTransform, 
    NaNStrategy, 
    GlobalScaling,
)
from .result_builder import PreprocessingStats


class Preprocessor:
    """
    Preprocesses metrics for anomaly detection.
    
    All operations are vectorized for performance.
    Completely metric-agnostic - works with any numeric columns.
    """
    
    # Threshold for considering a column constant
    CONSTANT_THRESHOLD = 1e-10
    
    # Maximum unique values for warning about potential categorical
    CATEGORICAL_WARNING_THRESHOLD = 10
    
    def __init__(self, config: Optional[MetricConfig] = None):
        """
        Initialize preprocessor.
        
        Args:
            config: Metric configuration. If None, uses defaults.
        """
        self.config = config or MetricConfig()
        self._fitted = False
        self._fit_stats: Dict[str, Dict[str, Any]] = {}
        self._preprocessing_stats: Dict[str, PreprocessingStats] = {}
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        metrics: List[str],
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Fit preprocessor and transform data.
        
        Args:
            df: Input DataFrame
            metrics: List of metric column names to process
            
        Returns:
            Tuple of:
            - X: Processed data matrix (n_samples, n_features)
            - avatars: List of node IDs
            - metrics_used: List of metric names actually used (after filtering)
            
        Raises:
            ValueError: If no valid metrics after preprocessing
        """
        # Validate inputs
        self._validate_inputs(df, metrics)
        
        # Extract IDs
        avatars = self._extract_ids(df)
        
        # Process each metric
        processed_data = {}
        metrics_used = []
        
        for metric in metrics:
            transform = self.config.get_metric_transform(metric)
            
            # Skip if explicitly dropped
            if transform.drop:
                continue
            
            # Get column data
            if metric not in df.columns:
                warnings.warn(f"Metric '{metric}' not found in DataFrame, skipping")
                continue
            
            series = df[metric].copy()
            original_dtype = str(series.dtype)
            
            # Track original stats
            n_missing = series.isna().sum()
            n_inf = np.isinf(series.replace([np.inf, -np.inf], np.nan).dropna()).sum() if np.issubdtype(series.dtype, np.number) else 0
            n_zeros = (series == 0).sum()
            original_range = (float(series.min()) if len(series) > 0 else 0.0, 
                            float(series.max()) if len(series) > 0 else 0.0)
            
            transforms_applied = []
            
            # Convert to numeric
            if not np.issubdtype(series.dtype, np.number):
                series = pd.to_numeric(series, errors='coerce')
                transforms_applied.append('to_numeric')
            
            series = series.astype(np.float64)
            
            # Check for constant column
            if series.nunique(dropna=True) <= 1:
                warnings.warn(f"Metric '{metric}' is constant, skipping")
                continue
            
            # Warn about potential categorical
            if series.nunique(dropna=True) <= self.CATEGORICAL_WARNING_THRESHOLD:
                warnings.warn(
                    f"Metric '{metric}' has only {series.nunique()} unique values, "
                    "consider if this is appropriate for anomaly detection"
                )
            
            # Apply clipping
            if transform.clip_min is not None or transform.clip_max is not None:
                series = series.clip(lower=transform.clip_min, upper=transform.clip_max)
                transforms_applied.append(f'clip({transform.clip_min}, {transform.clip_max})')
            
            # Apply log transform
            if transform.log:
                # Ensure non-negative before log1p
                series = np.log1p(np.maximum(series, 0))
                transforms_applied.append('log1p')
            
            # Handle NaN/Inf
            series = self._handle_missing(series, transform)
            transforms_applied.append(f'nan_strategy({self.config.nan_strategy.value})')
            
            # Store fit statistics for this metric
            self._fit_stats[metric] = {
                'mean': float(series.mean()),
                'std': float(series.std()),
                'median': float(series.median()),
                'min': float(series.min()),
                'max': float(series.max()),
                'q25': float(series.quantile(0.25)),
                'q75': float(series.quantile(0.75)),
            }
            
            final_range = (float(series.min()), float(series.max()))
            
            # Store preprocessing stats
            self._preprocessing_stats[metric] = PreprocessingStats(
                original_dtype=original_dtype,
                n_missing=int(n_missing),
                n_inf=int(n_inf),
                n_zeros=int(n_zeros),
                transform_applied=transforms_applied,
                original_range=original_range,
                final_range=final_range,
            )
            
            processed_data[metric] = series.values
            metrics_used.append(metric)
        
        if not metrics_used:
            raise ValueError("No usable metrics after preprocessing")
        
        # Build data matrix
        X = np.column_stack([processed_data[m] for m in metrics_used])
        
        # Apply global scaling
        X = self._apply_global_scaling(X, metrics_used, fit=True)
        
        # Convert to desired dtype
        dtype = np.float32 if self.config.use_float32 else np.float64
        X = np.ascontiguousarray(X, dtype=dtype)
        
        self._fitted = True
        
        return X, avatars, metrics_used
    
    def transform(
        self,
        df: pd.DataFrame,
        metrics: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Transform data using fitted parameters.
        
        Must call fit_transform first.
        
        Args:
            df: Input DataFrame
            metrics: List of metric column names
            
        Returns:
            Tuple of (X, avatars)
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")
        
        avatars = self._extract_ids(df)
        
        processed_data = {}
        metrics_used = []
        
        for metric in metrics:
            if metric not in self._fit_stats:
                continue
            
            transform = self.config.get_metric_transform(metric)
            
            if metric not in df.columns:
                continue
            
            series = df[metric].copy().astype(np.float64)
            
            # Apply same transforms
            if transform.clip_min is not None or transform.clip_max is not None:
                series = series.clip(lower=transform.clip_min, upper=transform.clip_max)
            
            if transform.log:
                series = np.log1p(np.maximum(series, 0))
            
            series = self._handle_missing(series, transform)
            
            processed_data[metric] = series.values
            metrics_used.append(metric)
        
        X = np.column_stack([processed_data[m] for m in metrics_used])
        X = self._apply_global_scaling(X, metrics_used, fit=False)
        
        dtype = np.float32 if self.config.use_float32 else np.float64
        X = np.ascontiguousarray(X, dtype=dtype)
        
        return X, avatars
    
    def get_preprocessing_stats(self) -> Dict[str, PreprocessingStats]:
        """Get preprocessing statistics for all metrics."""
        return self._preprocessing_stats.copy()
    
    def get_fit_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get fitted statistics for all metrics."""
        return self._fit_stats.copy()
    
    def _validate_inputs(self, df: pd.DataFrame, metrics: List[str]) -> None:
        """Validate input DataFrame and metrics."""
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        if not metrics:
            raise ValueError("No metrics specified")
        
        # Check that at least some metrics exist
        existing = [m for m in metrics if m in df.columns]
        if not existing:
            raise ValueError(f"None of the specified metrics found in DataFrame. "
                           f"Available columns: {list(df.columns)}")
    
    def _extract_ids(self, df: pd.DataFrame) -> List[str]:
        """Extract node IDs from DataFrame."""
        id_col = self.config.id_column
        
        if id_col in df.columns:
            return df[id_col].astype(str).tolist()
        elif df.index.name == id_col:
            return df.index.astype(str).tolist()
        else:
            # Use index as IDs
            return df.index.astype(str).tolist()
    
    def _handle_missing(
        self, 
        series: pd.Series, 
        transform: MetricTransform
    ) -> pd.Series:
        """Handle NaN and Inf values."""
        # Replace Inf with NaN first
        series = series.replace([np.inf, -np.inf], np.nan)
        
        # Check for custom fill value
        if transform.fill_value is not None:
            return series.fillna(transform.fill_value)
        
        # Apply global strategy
        strategy = self.config.nan_strategy
        
        if strategy == NaNStrategy.ZERO:
            return series.fillna(0.0)
        elif strategy == NaNStrategy.MEAN:
            return series.fillna(series.mean())
        elif strategy == NaNStrategy.MEDIAN:
            return series.fillna(series.median())
        elif strategy == NaNStrategy.DROP:
            # Can't actually drop rows here, just fill with 0
            # The DROP strategy should be handled at a higher level
            return series.fillna(0.0)
        else:
            return series.fillna(0.0)
    
    def _apply_global_scaling(
        self,
        X: np.ndarray,
        metrics: List[str],
        fit: bool = True,
    ) -> np.ndarray:
        """Apply global scaling to data matrix."""
        scaling = self.config.global_scaling
        
        if scaling == GlobalScaling.NONE:
            return X
        
        if fit:
            # Compute and store scaling parameters
            self._scaling_params = {}
        
        if scaling == GlobalScaling.STANDARD:
            if fit:
                self._scaling_params['mean'] = np.mean(X, axis=0)
                self._scaling_params['std'] = np.std(X, axis=0)
                self._scaling_params['std'][self._scaling_params['std'] < self.CONSTANT_THRESHOLD] = 1.0
            
            return (X - self._scaling_params['mean']) / self._scaling_params['std']
        
        elif scaling == GlobalScaling.ROBUST:
            if fit:
                self._scaling_params['median'] = np.median(X, axis=0)
                q75 = np.percentile(X, 75, axis=0)
                q25 = np.percentile(X, 25, axis=0)
                self._scaling_params['iqr'] = q75 - q25
                self._scaling_params['iqr'][self._scaling_params['iqr'] < self.CONSTANT_THRESHOLD] = 1.0
            
            return (X - self._scaling_params['median']) / self._scaling_params['iqr']
        
        elif scaling == GlobalScaling.MINMAX:
            if fit:
                self._scaling_params['min'] = np.min(X, axis=0)
                self._scaling_params['max'] = np.max(X, axis=0)
                range_vals = self._scaling_params['max'] - self._scaling_params['min']
                range_vals[range_vals < self.CONSTANT_THRESHOLD] = 1.0
                self._scaling_params['range'] = range_vals
            
            return (X - self._scaling_params['min']) / self._scaling_params['range']
        
        return X


class ChunkedPreprocessor:
    """
    Memory-efficient preprocessor for large datasets.
    
    Uses streaming computation for statistics and processes
    data in chunks.
    """
    
    # Default chunk size
    DEFAULT_CHUNK_SIZE = 20000
    
    def __init__(
        self,
        config: Optional[MetricConfig] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        """
        Initialize chunked preprocessor.
        
        Args:
            config: Metric configuration
            chunk_size: Number of rows per chunk
        """
        self.config = config or MetricConfig()
        self.chunk_size = chunk_size
        self._fitted = False
        self._running_stats: Dict[str, _RunningStats] = {}
    
    def fit(self, df: pd.DataFrame, metrics: List[str]) -> None:
        """
        Fit preprocessor using streaming statistics.
        
        Uses Welford's algorithm for online mean/variance computation.
        """
        n_rows = len(df)
        n_chunks = (n_rows + self.chunk_size - 1) // self.chunk_size
        
        # Initialize running statistics
        for metric in metrics:
            if metric in df.columns:
                self._running_stats[metric] = _RunningStats()
        
        # Process chunks
        for i in range(n_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, n_rows)
            chunk = df.iloc[start_idx:end_idx]
            
            for metric in metrics:
                if metric not in chunk.columns:
                    continue
                
                values = chunk[metric].values.astype(np.float64)
                values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                
                self._running_stats[metric].update(values)
        
        self._fitted = True
    
    def transform_chunk(
        self,
        chunk: pd.DataFrame,
        metrics: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Transform a single chunk of data.
        
        Args:
            chunk: Data chunk
            metrics: Metric names
            
        Returns:
            Tuple of (X_chunk, avatars_chunk)
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit first.")
        
        preprocessor = Preprocessor(self.config)
        
        # Manually set fit stats from running statistics
        for metric, stats in self._running_stats.items():
            preprocessor._fit_stats[metric] = {
                'mean': stats.mean,
                'std': stats.std,
                'min': stats.min_val,
                'max': stats.max_val,
            }
        preprocessor._fitted = True
        
        return preprocessor.transform(chunk, metrics)


class _RunningStats:
    """
    Online statistics computation using Welford's algorithm.
    
    Computes mean and variance in a single pass with O(1) memory.
    """
    
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared differences
        self.min_val = float('inf')
        self.max_val = float('-inf')
    
    def update(self, values: np.ndarray) -> None:
        """Update statistics with new values."""
        for x in values:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2
            
            self.min_val = min(self.min_val, x)
            self.max_val = max(self.max_val, x)
    
    @property
    def variance(self) -> float:
        """Population variance."""
        if self.n < 2:
            return 0.0
        return self.M2 / self.n
    
    @property
    def std(self) -> float:
        """Population standard deviation."""
        return np.sqrt(self.variance)