"""
Anomaly Detection Engine

Main engine for running anomaly detection on graph metrics.
Orchestrates preprocessing, algorithm execution, and result building.

Features:
- Metric-agnostic: works with any numeric columns
- Configurable preprocessing via MetricConfig
- Group-aware detection
- Automatic sampling for large datasets
- Parallel execution for groups
"""

import time
import warnings
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

from .anomaly_config import (
    MetricConfig,
    AlgorithmConfig,
    ScoreNormalization,
    ThresholdMethod,
)
from .result_builder import (
    AnomalyResult,
    ResultBuilder,
    GroupAnomalyStats,
    PreprocessingStats,
)
from .preprocessing import Preprocessor, ChunkedPreprocessor
from .metric_profiler import MetricProfiler, MetricProfile
from .parallel import ParallelExecutor, get_optimal_workers
from .algorithms import (
    get_algorithm,
    list_algorithms,
    get_algorithm_info,
    ALGORITHM_REGISTRY,
    HAS_SKLEARN,
)
from .algorithms.base import AlgorithmInfo


class AnomalyEngine:
    """
    Main anomaly detection engine.
    
    Orchestrates:
    - Data preprocessing
    - Algorithm selection and execution
    - Result building
    - Group-aware detection
    - Large dataset handling
    
    Usage:
        engine = AnomalyEngine()
        
        # Simple detection
        result = engine.detect_anomalies(df, metrics, "isolation_forest")
        
        # With configuration
        config = MetricConfig(global_scaling=GlobalScaling.STANDARD)
        result = engine.detect_anomalies(df, metrics, "isolation_forest", config=config)
        
        # Group-aware detection
        config = MetricConfig(group_by="community_id")
        result = engine.detect_anomalies(df, metrics, "zscore", config=config)
    """
    
    # Threshold for automatic chunked processing
    CHUNKED_THRESHOLD = 20000
    
    # Maximum nodes for LOF (O(nÂ²) complexity)
    MAX_NODES_LOF = 50000
    
    def __init__(
        self,
        max_nodes_for_lof: int = 50000,
        n_jobs: int = -1,
        verbose: bool = True,
    ):
        """
        Initialize anomaly engine.
        
        Args:
            max_nodes_for_lof: Maximum nodes for LOF algorithm
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            verbose: Print progress information
        """
        self.max_nodes_for_lof = max_nodes_for_lof
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self._result_builder = ResultBuilder()
        self._profiler = MetricProfiler()
        self._parallel = ParallelExecutor(n_jobs=n_jobs, verbose=verbose)
    
    def detect_anomalies(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        algorithm: str,
        parameters: Optional[Dict[str, Any]] = None,
        config: Optional[MetricConfig] = None,
        algorithm_config: Optional[AlgorithmConfig] = None,
        sample_size: Optional[int] = None,
    ) -> AnomalyResult:
        """
        Run anomaly detection on specified metrics.
        
        Args:
            df: DataFrame with node data
            metrics: List of metric column names to analyze
            algorithm: Algorithm name (zscore, iqr, isolation_forest, lof, dbscan, mahalanobis)
            parameters: Algorithm-specific parameters (optional)
            config: Metric preprocessing configuration (optional)
            algorithm_config: Full algorithm configuration (optional, overrides algorithm/parameters)
            sample_size: Sample size for large datasets (optional)
            
        Returns:
            AnomalyResult with scores, labels, and statistics
            
        Raises:
            ValueError: If algorithm unknown or invalid configuration
        """
        start_time = time.time()
        
        # Use defaults if not provided
        if config is None:
            config = MetricConfig()
        
        # Build algorithm config
        if algorithm_config is not None:
            algorithm = algorithm_config.algorithm
            parameters = algorithm_config.parameters
            score_norm = algorithm_config.score_normalization
            threshold_method = algorithm_config.threshold_method
            threshold_value = algorithm_config.threshold_value
            top_n = algorithm_config.top_n
            # -1 or 0 means all nodes
            if top_n <= 0:
                top_n = len(df)
        else:
            score_norm = ScoreNormalization.MINMAX
            threshold_method = ThresholdMethod.FIXED
            threshold_value = 0.5
            top_n = len(df)  # Default to all nodes
        
        # Validate algorithm
        if algorithm not in ALGORITHM_REGISTRY:
            available = list(ALGORITHM_REGISTRY.keys())
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {available}")
        
        # Check LOF size limit
        if algorithm == "lof" and len(df) > self.max_nodes_for_lof:
            warnings.warn(
                f"LOF with {len(df)} nodes may be slow (O(nÂ²)). "
                f"Consider using isolation_forest or sampling."
            )
        
        # Check for group-aware detection
        if config.group_by is not None:
            return self._detect_grouped(
                df=df,
                metrics=metrics,
                algorithm=algorithm,
                parameters=parameters,
                config=config,
                score_norm=score_norm,
                threshold_method=threshold_method,
                threshold_value=threshold_value,
                top_n=top_n,
                start_time=start_time,
            )
        
        # Check for sampling
        if sample_size is not None and len(df) > sample_size:
            return self._detect_with_sampling(
                df=df,
                metrics=metrics,
                algorithm=algorithm,
                parameters=parameters,
                config=config,
                sample_size=sample_size,
                score_norm=score_norm,
                threshold_method=threshold_method,
                threshold_value=threshold_value,
                top_n=top_n,
                start_time=start_time,
            )
        
        # Standard detection
        return self._detect_single(
            df=df,
            metrics=metrics,
            algorithm=algorithm,
            parameters=parameters,
            config=config,
            score_norm=score_norm,
            threshold_method=threshold_method,
            threshold_value=threshold_value,
            top_n=top_n,
            start_time=start_time,
        )
    
    def _detect_single(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        algorithm: str,
        parameters: Optional[Dict[str, Any]],
        config: MetricConfig,
        score_norm: ScoreNormalization,
        threshold_method: ThresholdMethod,
        threshold_value: float,
        top_n: int,
        start_time: float,
    ) -> AnomalyResult:
        """
        Run detection on a single (non-grouped) dataset.
        """
        # Preprocess data
        preprocessor = Preprocessor(config)
        X, avatars, metrics_used = preprocessor.fit_transform(df, metrics)
        preprocessing_stats = preprocessor.get_preprocessing_stats()
        
        if self.verbose:
            print(f"[ANOMALY] Preprocessed {len(avatars)} nodes, {len(metrics_used)} metrics")
        
        # Get algorithm instance
        algo = get_algorithm(algorithm)
        
        # Validate parameters
        params = algo.validate_params(parameters)
        
        # Run algorithm
        output = algo.fit_predict(X, params)
        
        # Compute total time
        computation_time = time.time() - start_time
        
        if self.verbose:
            n_anomalies = int(np.sum(output.anomaly_mask))
            pct = 100 * n_anomalies / len(avatars)
            print(f"[ANOMALY] {algorithm} completed in {computation_time:.2f}s. "
                  f"Found {n_anomalies}/{len(avatars)} anomalies ({pct:.1f}%)")
        
        # Build result
        # Map per_metric_scores back to actual metric names
        per_metric_scores = None
        if output.per_metric_scores is not None:
            per_metric_scores = {}
            for i, metric in enumerate(metrics_used):
                key = f"feature_{i}"
                if key in output.per_metric_scores:
                    per_metric_scores[metric] = output.per_metric_scores[key]
        
        return self._result_builder.build(
            df=df,
            avatars=avatars,
            raw_scores=output.raw_scores,
            anomaly_mask=output.anomaly_mask,
            algorithm=algorithm,
            params=params,
            metrics=metrics_used,
            score_normalization=score_norm,
            threshold_method=threshold_method,
            threshold_value=output.threshold_used,
            computation_time=computation_time,
            top_n=top_n,
            per_metric_scores=per_metric_scores,
            preprocessing_stats=preprocessing_stats,
        )
    
    def _detect_grouped(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        algorithm: str,
        parameters: Optional[Dict[str, Any]],
        config: MetricConfig,
        score_norm: ScoreNormalization,
        threshold_method: ThresholdMethod,
        threshold_value: float,
        top_n: int,
        start_time: float,
    ) -> AnomalyResult:
        """
        Run detection separately for each group.
        """
        group_col = config.group_by
        
        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found in DataFrame")
        
        if self.verbose:
            n_groups = df[group_col].nunique()
            print(f"[ANOMALY] Running group-aware detection on {n_groups} groups")
        
        # Split into groups
        groups = []
        for group_val, group_df in df.groupby(group_col):
            if len(group_df) >= config.min_group_size:
                groups.append((group_val, group_df))
        
        if len(groups) == 0:
            raise ValueError(
                f"No groups with >= {config.min_group_size} samples. "
                f"Adjust min_group_size or use non-grouped detection."
            )
        
        if self.verbose:
            print(f"[ANOMALY] Processing {len(groups)} groups (filtered by min_group_size)")
        
        # Create config without group_by for sub-detection
        sub_config = MetricConfig(
            id_column=config.id_column,
            group_by=None,
            nan_strategy=config.nan_strategy,
            per_metric=config.per_metric,
            global_scaling=config.global_scaling,
            min_group_size=config.min_group_size,
            use_float32=config.use_float32,
        )
        
        # Define function to detect on single group
        def detect_group(group_df: pd.DataFrame) -> AnomalyResult:
            return self._detect_single(
                df=group_df,
                metrics=metrics,
                algorithm=algorithm,
                parameters=parameters,
                config=sub_config,
                score_norm=score_norm,
                threshold_method=threshold_method,
                threshold_value=threshold_value,
                top_n=5,  # Fewer per group
                start_time=time.time(),
            )
        
        # Run detection on all groups (parallel if multiple groups)
        group_results = {}
        if len(groups) > 1 and self.n_jobs != 1:
            # Parallel execution
            results_dict = self._parallel.map_groups(
                func=detect_group,
                groups=groups,
            )
            group_results = {k: v for k, v in results_dict.items() if v is not None}
        else:
            # Sequential execution
            for group_val, group_df in groups:
                try:
                    result = detect_group(group_df)
                    group_results[group_val] = result
                except Exception as e:
                    if self.verbose:
                        print(f"[ANOMALY] Error in group {group_val}: {e}")
        
        # Merge results
        computation_time = time.time() - start_time
        
        if self.verbose:
            total_anomalies = sum(r.n_anomalies for r in group_results.values())
            total_nodes = sum(r.n_total for r in group_results.values())
            print(f"[ANOMALY] Group-aware detection completed in {computation_time:.2f}s. "
                  f"Found {total_anomalies}/{total_nodes} anomalies across {len(group_results)} groups")
        
        return self._result_builder.merge_group_results(
            group_results=group_results,
            algorithm=algorithm,
            params=parameters or {},
            metrics=metrics,
            total_computation_time=computation_time,
            top_n=top_n,
        )
    
    def _detect_with_sampling(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        algorithm: str,
        parameters: Optional[Dict[str, Any]],
        config: MetricConfig,
        sample_size: int,
        score_norm: ScoreNormalization,
        threshold_method: ThresholdMethod,
        threshold_value: float,
        top_n: int,
        start_time: float,
    ) -> AnomalyResult:
        """
        Run detection with sampling for large datasets.
        
        Strategy:
        1. Sample subset of data
        2. Fit model on sample
        3. Score all data using fitted model
        """
        if self.verbose:
            print(f"[ANOMALY] Using sampling: {sample_size} of {len(df)} nodes")
        
        # For algorithms that don't support scoring new data, just run on sample
        if algorithm in ["lof"]:
            # Sample and run
            sample_df = df.sample(n=sample_size, random_state=42)
            result = self._detect_single(
                df=sample_df,
                metrics=metrics,
                algorithm=algorithm,
                parameters=parameters,
                config=config,
                score_norm=score_norm,
                threshold_method=threshold_method,
                threshold_value=threshold_value,
                top_n=top_n,
                start_time=start_time,
            )
            
            if self.verbose:
                print(f"[ANOMALY] Note: Only {sample_size} sampled nodes were analyzed")
            
            return result
        
        # For other algorithms, we can fit on sample and score all
        # This requires algorithm-specific handling - for now, just run on sample
        sample_df = df.sample(n=sample_size, random_state=42)
        return self._detect_single(
            df=sample_df,
            metrics=metrics,
            algorithm=algorithm,
            parameters=parameters,
            config=config,
            score_norm=score_norm,
            threshold_method=threshold_method,
            threshold_value=threshold_value,
            top_n=top_n,
            start_time=start_time,
        )
    
    def profile_metrics(
        self,
        df: pd.DataFrame,
        metrics: List[str],
    ) -> Dict[str, MetricProfile]:
        """
        Profile metrics for preprocessing suggestions.
        
        This is an explicit call - not automatic.
        
        Args:
            df: Input DataFrame
            metrics: List of metric column names
            
        Returns:
            Dictionary mapping metric names to profiles
        """
        return self._profiler.profile(df, metrics)
    
    def suggest_config(
        self,
        profiles: Dict[str, MetricProfile],
    ) -> MetricConfig:
        """
        Suggest MetricConfig based on metric profiles.
        
        Args:
            profiles: Dictionary of metric profiles from profile_metrics()
            
        Returns:
            Suggested MetricConfig
        """
        return self._profiler.suggest_config(profiles)
    
    def generate_profile_report(
        self,
        profiles: Dict[str, MetricProfile],
    ) -> str:
        """
        Generate human-readable profile report.
        
        Args:
            profiles: Dictionary of metric profiles
            
        Returns:
            Formatted report string
        """
        return self._profiler.generate_report(profiles)
    
    @staticmethod
    def get_available_algorithms() -> Dict[str, AlgorithmInfo]:
        """
        Get available algorithms with their metadata.
        
        Returns:
            Dictionary mapping algorithm names to info
        """
        return list_algorithms()
    
    @staticmethod
    def recommend_algorithm(
        n_nodes: int,
        n_metrics: int,
        time_constraint: Optional[str] = None,
        memory_constraint: Optional[str] = None,
    ) -> str:
        """
        Recommend algorithm based on data characteristics and constraints.
        
        Args:
            n_nodes: Number of nodes in graph
            n_metrics: Number of metrics to analyze
            time_constraint: "fast", "moderate", or "slow"
            memory_constraint: "low", "moderate", or "high"
            
        Returns:
            Recommended algorithm name
        """
        # Fast constraint
        if time_constraint == "fast":
            return "zscore"
        
        # Low memory constraint
        if memory_constraint == "low":
            return "zscore" if n_nodes > 50000 else "iqr"
        
        # Very large graphs
        if n_nodes > 100000:
            return "zscore"
        
        # Large graphs
        if n_nodes > 50000:
            return "isolation_forest"
        
        # Single metric
        if n_metrics == 1:
            return "iqr"
        
        # Moderate size, multiple metrics
        if n_nodes > 10000:
            return "isolation_forest"
        
        # Small graphs with multiple correlated metrics
        if n_metrics >= 3:
            return "mahalanobis"
        
        # Default
        return "isolation_forest"