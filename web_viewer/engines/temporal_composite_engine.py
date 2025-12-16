"""
Temporal Composite Engine

Engine for creating composite metrics that leverage temporal information including:
- Velocity (rate of change)
- Acceleration (change in velocity)
- Stability (consistency over time)
- Momentum (trend strength)
- Age-based weighting
- Relative to cohort metrics
"""

import time
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set

import numpy as np
import pandas as pd

from backend.models.temporal_composite import (
    TemporalOperation,
    CohortReference,
    CombineOperation,
    TemporalOperationConfig,
    TemporalCompositeConfig,
    TemporalMetricStatistics,
    TemporalCompositeResult,
    TemporalPreviewResult,
    TemporalPresetInfo,
    TemporalOperationInfo,
    AvailableOperationsResponse,
)
from backend.services.snapshot_storage import SnapshotStorage
from backend.services.snapshot_analysis_service import SnapshotAnalysisService


class TemporalCompositeEngine:
    """
    Engine for computing temporal composite metrics.
    
    Provides operations that combine current metric values with
    historical information to create time-aware metrics.
    """
    
    # Pre-built presets
    PRESETS = {
        "trust_growth": TemporalPresetInfo(
            preset_id="trust_growth",
            name="trust_growth_score",
            display_name="Trust Growth Score",
            description="Combines in-degree velocity with stability. High values indicate "
                       "nodes gaining trust consistently.",
            base_metric="in_degree",
            temporal_operation=TemporalOperation.VELOCITY,
            default_window=5,
            category="growth",
            use_case="Identify nodes that are growing in popularity"
        ),
        "influence_momentum": TemporalPresetInfo(
            preset_id="influence_momentum",
            name="influence_momentum",
            display_name="Influence Momentum",
            description="PageRank velocity weighted by stability. Shows nodes with "
                       "growing influence.",
            base_metric="pagerank",
            temporal_operation=TemporalOperation.MOMENTUM,
            default_window=5,
            category="influence",
            use_case="Find nodes becoming more influential over time"
        ),
        "stability_score": TemporalPresetInfo(
            preset_id="stability_score",
            name="network_stability",
            display_name="Network Stability Score",
            description="How consistent a node's degree has been over time. "
                       "High = stable position, Low = volatile.",
            base_metric="in_degree",
            temporal_operation=TemporalOperation.STABILITY,
            default_window=5,
            category="stability",
            use_case="Identify stable vs volatile nodes"
        ),
        "tenure_weighted_rank": TemporalPresetInfo(
            preset_id="tenure_weighted_rank",
            name="tenure_weighted_pagerank",
            display_name="Tenure-Weighted PageRank",
            description="PageRank weighted by how long the node has been in the network. "
                       "Rewards established nodes.",
            base_metric="pagerank",
            temporal_operation=TemporalOperation.AGE_WEIGHTED,
            default_window=0,
            category="influence",
            use_case="Reward longevity in the network"
        ),
        "anomaly_acceleration": TemporalPresetInfo(
            preset_id="anomaly_acceleration",
            name="anomaly_acceleration",
            display_name="Anomaly Acceleration",
            description="How quickly a node is becoming more anomalous. "
                       "High values indicate rapidly changing behavior.",
            base_metric="anomaly_score",
            temporal_operation=TemporalOperation.ACCELERATION,
            default_window=5,
            category="risk",
            use_case="Detect nodes becoming anomalous quickly"
        ),
    }
    
    # Operation metadata
    OPERATION_INFO = {
        TemporalOperation.VELOCITY: TemporalOperationInfo(
            operation=TemporalOperation.VELOCITY,
            name="Velocity",
            description="Rate of change of a metric over time",
            formula="(metric[t] - metric[t-n]) / n",
            min_window=2,
            default_window=5,
            output_range="(-∞, ∞)",
            interpretation="Positive = increasing, Negative = decreasing"
        ),
        TemporalOperation.ACCELERATION: TemporalOperationInfo(
            operation=TemporalOperation.ACCELERATION,
            name="Acceleration",
            description="Change in the rate of change (second derivative)",
            formula="velocity[t] - velocity[t-1]",
            min_window=3,
            default_window=5,
            output_range="(-∞, ∞)",
            interpretation="Positive = speeding up, Negative = slowing down"
        ),
        TemporalOperation.STABILITY: TemporalOperationInfo(
            operation=TemporalOperation.STABILITY,
            name="Stability",
            description="Consistency of metric values over time",
            formula="1 - (std(metric[t-n:t]) / mean(metric[t-n:t]))",
            min_window=3,
            default_window=5,
            output_range="[0, 1]",
            interpretation="1 = perfectly stable, 0 = highly volatile"
        ),
        TemporalOperation.VOLATILITY: TemporalOperationInfo(
            operation=TemporalOperation.VOLATILITY,
            name="Volatility",
            description="Coefficient of variation over time window",
            formula="std(metric[t-n:t]) / mean(metric[t-n:t])",
            min_window=3,
            default_window=5,
            output_range="[0, ∞)",
            interpretation="Higher = more volatile"
        ),
        TemporalOperation.MOMENTUM: TemporalOperationInfo(
            operation=TemporalOperation.MOMENTUM,
            name="Momentum",
            description="Exponentially weighted moving average trend",
            formula="Σ(decay^i * velocity[t-i])",
            min_window=3,
            default_window=5,
            output_range="(-∞, ∞)",
            interpretation="Strong positive/negative indicates sustained trend"
        ),
        TemporalOperation.AGE: TemporalOperationInfo(
            operation=TemporalOperation.AGE,
            name="Age",
            description="Number of snapshots since node first appeared",
            formula="current_block_index - first_seen_block_index",
            min_window=1,
            default_window=1,
            output_range="[0, num_snapshots]",
            interpretation="Higher = longer tenure in network"
        ),
        TemporalOperation.AGE_WEIGHTED: TemporalOperationInfo(
            operation=TemporalOperation.AGE_WEIGHTED,
            name="Age-Weighted",
            description="Metric value weighted by normalized node age",
            formula="metric * (1 + weight * normalized_age)",
            min_window=1,
            default_window=1,
            output_range="[0, ∞)",
            interpretation="Rewards established nodes"
        ),
    }
    
    def __init__(
        self,
        storage: Optional[SnapshotStorage] = None,
        analysis_service: Optional[SnapshotAnalysisService] = None
    ):
        """
        Initialize temporal composite engine.
        
        Args:
            storage: SnapshotStorage instance
            analysis_service: SnapshotAnalysisService for loading metrics
        """
        self.storage = storage or SnapshotStorage()
        self.analysis_service = analysis_service
        
        if self.analysis_service is None:
            from backend.services.snapshot_analysis_service import snapshot_analysis_service
            self.analysis_service = snapshot_analysis_service
        
        # Cache for metric values across snapshots
        self._metric_cache: Dict[str, pd.DataFrame] = {}
    
    def _generate_composite_id(self, config: TemporalCompositeConfig) -> str:
        """Generate unique ID for a temporal composite."""
        key = f"{config.name}_{config.base_metric}_{config.temporal_config.operation.value}_{config.target_block}"
        return hashlib.md5(key.encode()).hexdigest()[:12]
    
    def _load_metric_history(
        self,
        base_sql_file: str,
        metric: str,
        target_block: int,
        window_blocks: int
    ) -> Tuple[pd.DataFrame, List[int]]:
        """
        Load metric values for multiple snapshots leading up to target.
        
        Returns:
            Tuple of (DataFrame with columns: node_id, block_1, block_2, ..., block_n),
                     (list of block numbers used)
        """
        # Get all snapshots before and including target
        all_snapshots = self.storage.list_snapshots(base_sql_file)
        all_snapshots.sort(key=lambda s: s.block_number)
        
        # Find target index
        target_idx = None
        for i, snap in enumerate(all_snapshots):
            if snap.block_number == target_block:
                target_idx = i
                break
        
        if target_idx is None:
            raise ValueError(f"Target block {target_block} not found in snapshots")
        
        # Get window of snapshots
        start_idx = max(0, target_idx - window_blocks + 1)
        window_snapshots = all_snapshots[start_idx:target_idx + 1]
        
        if len(window_snapshots) < 2:
            raise ValueError(f"Need at least 2 snapshots for temporal analysis, found {len(window_snapshots)}")
        
        block_numbers = [s.block_number for s in window_snapshots]
        
        # Collect all node IDs across snapshots
        all_node_ids: Set[str] = set()
        metric_values: Dict[int, Dict[str, float]] = {}  # block -> {node_id: value}
        
        for snapshot in window_snapshots:
            result = self.analysis_service.get_metric_values(
                base_sql_file, snapshot.block_number, metric, include_values=True
            )
            
            if result and result.values:
                metric_values[snapshot.block_number] = result.values
                all_node_ids.update(result.values.keys())
            else:
                metric_values[snapshot.block_number] = {}
        
        # Build DataFrame
        rows = []
        for node_id in all_node_ids:
            row = {"node_id": node_id}
            for block in block_numbers:
                row[f"block_{block}"] = metric_values.get(block, {}).get(node_id)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df, block_numbers
    
    # =========================================================================
    # Core Temporal Operations
    # =========================================================================
    
    def compute_velocity(
        self,
        base_sql_file: str,
        metric: str,
        target_block: int,
        window_blocks: int = 5
    ) -> pd.Series:
        """
        Compute velocity (rate of change) for a metric.
        
        Formula: (current_value - past_value) / num_snapshots
        """
        df, blocks = self._load_metric_history(base_sql_file, metric, target_block, window_blocks)
        
        if len(blocks) < 2:
            raise ValueError("Need at least 2 snapshots for velocity")
        
        first_col = f"block_{blocks[0]}"
        last_col = f"block_{blocks[-1]}"
        
        # Get values
        first_values = df[first_col].astype(float)
        last_values = df[last_col].astype(float)
        
        # Compute velocity
        n_steps = len(blocks) - 1
        velocity = (last_values - first_values) / n_steps
        
        # Handle nodes that didn't exist in first snapshot
        velocity = velocity.fillna(0)
        
        result = pd.Series(velocity.values, index=df["node_id"])
        return result
    
    def compute_acceleration(
        self,
        base_sql_file: str,
        metric: str,
        target_block: int,
        window_blocks: int = 5
    ) -> pd.Series:
        """
        Compute acceleration (change in velocity).
        
        Requires computing velocity at two points and taking the difference.
        """
        df, blocks = self._load_metric_history(base_sql_file, metric, target_block, window_blocks)
        
        if len(blocks) < 3:
            raise ValueError("Need at least 3 snapshots for acceleration")
        
        # Compute velocity for first half and second half
        mid_idx = len(blocks) // 2
        
        first_half_start = f"block_{blocks[0]}"
        first_half_end = f"block_{blocks[mid_idx]}"
        second_half_start = f"block_{blocks[mid_idx]}"
        second_half_end = f"block_{blocks[-1]}"
        
        # First half velocity
        v1_num = df[first_half_end].astype(float) - df[first_half_start].astype(float)
        v1 = v1_num / mid_idx
        
        # Second half velocity
        v2_num = df[second_half_end].astype(float) - df[second_half_start].astype(float)
        v2 = v2_num / (len(blocks) - mid_idx - 1)
        
        # Acceleration
        acceleration = v2 - v1
        acceleration = acceleration.fillna(0)
        
        result = pd.Series(acceleration.values, index=df["node_id"])
        return result
    
    def compute_stability(
        self,
        base_sql_file: str,
        metric: str,
        target_block: int,
        window_blocks: int = 5
    ) -> pd.Series:
        """
        Compute stability (consistency over time).
        
        Formula: 1 - coefficient_of_variation
        Returns values in [0, 1] where 1 = perfectly stable.
        """
        df, blocks = self._load_metric_history(base_sql_file, metric, target_block, window_blocks)
        
        # Get all block columns
        block_cols = [f"block_{b}" for b in blocks]
        values_df = df[block_cols].astype(float)
        
        # Compute row-wise statistics
        row_mean = values_df.mean(axis=1)
        row_std = values_df.std(axis=1)
        
        # Coefficient of variation
        cv = row_std / row_mean.replace(0, np.nan)
        
        # Stability = 1 - CV (clamped to [0, 1])
        stability = 1 - cv.clip(0, 1)
        stability = stability.fillna(0)  # Nodes with constant 0 get stability 0
        
        result = pd.Series(stability.values, index=df["node_id"])
        return result
    
    def compute_volatility(
        self,
        base_sql_file: str,
        metric: str,
        target_block: int,
        window_blocks: int = 5
    ) -> pd.Series:
        """
        Compute volatility (coefficient of variation).
        """
        stability = self.compute_stability(base_sql_file, metric, target_block, window_blocks)
        volatility = 1 - stability
        return volatility
    
    def compute_momentum(
        self,
        base_sql_file: str,
        metric: str,
        target_block: int,
        window_blocks: int = 5,
        decay_factor: float = 0.9
    ) -> pd.Series:
        """
        Compute momentum (exponentially weighted trend).
        
        Recent changes weighted more heavily than older changes.
        """
        df, blocks = self._load_metric_history(base_sql_file, metric, target_block, window_blocks)
        
        if len(blocks) < 3:
            raise ValueError("Need at least 3 snapshots for momentum")
        
        # Compute changes between consecutive snapshots
        block_cols = [f"block_{b}" for b in blocks]
        values_df = df[block_cols].astype(float)
        
        changes = values_df.diff(axis=1).iloc[:, 1:]  # Skip first NaN column
        
        # Apply exponential weighting (most recent = highest weight)
        n_changes = changes.shape[1]
        weights = np.array([decay_factor ** (n_changes - 1 - i) for i in range(n_changes)])
        weights = weights / weights.sum()  # Normalize
        
        # Weighted sum of changes
        momentum = (changes.values * weights).sum(axis=1)
        momentum = np.nan_to_num(momentum, nan=0.0)
        
        result = pd.Series(momentum, index=df["node_id"])
        return result
    
    def compute_node_age(
        self,
        base_sql_file: str,
        target_block: int
    ) -> pd.Series:
        """
        Compute node age (snapshots since first appearance).
        """
        all_snapshots = self.storage.list_snapshots(base_sql_file)
        all_snapshots.sort(key=lambda s: s.block_number)
        
        # Find target index
        target_idx = None
        for i, snap in enumerate(all_snapshots):
            if snap.block_number == target_block:
                target_idx = i
                break
        
        if target_idx is None:
            raise ValueError(f"Target block {target_block} not found")
        
        # Track first seen for each node
        first_seen: Dict[str, int] = {}
        
        for i, snapshot in enumerate(all_snapshots[:target_idx + 1]):
            try:
                node_ids = self.storage.load_snapshot_node_ids(base_sql_file, snapshot.block_number)
                for nid in node_ids:
                    if nid not in first_seen:
                        first_seen[nid] = i
            except:
                continue
        
        # Compute age
        ages = {nid: target_idx - first_idx for nid, first_idx in first_seen.items()}
        
        return pd.Series(ages)
    
    def compute_age_weighted(
        self,
        base_sql_file: str,
        metric: str,
        target_block: int,
        age_weight: float = 0.1
    ) -> pd.Series:
        """
        Compute age-weighted metric.
        
        Formula: metric * (1 + age_weight * normalized_age)
        """
        # Get current metric values
        result = self.analysis_service.get_metric_values(
            base_sql_file, target_block, metric, include_values=True
        )
        
        if not result or not result.values:
            raise ValueError(f"Metric {metric} not found at block {target_block}")
        
        metric_series = pd.Series(result.values)
        
        # Get node ages
        ages = self.compute_node_age(base_sql_file, target_block)
        
        # Normalize ages to [0, 1]
        max_age = ages.max() if len(ages) > 0 and ages.max() > 0 else 1
        normalized_ages = ages / max_age
        
        # Align indices
        common_nodes = metric_series.index.intersection(ages.index)
        metric_aligned = metric_series.loc[common_nodes]
        ages_aligned = normalized_ages.loc[common_nodes]
        
        # Compute weighted metric
        weighted = metric_aligned * (1 + age_weight * ages_aligned)
        
        return weighted
    
    def compute_trend_strength(
        self,
        base_sql_file: str,
        metric: str,
        target_block: int,
        window: int = 5
    ) -> pd.Series:
        """
        Compute trend strength (R² of linear fit) for each node.
        
        Higher values indicate more consistent trend direction.
        """
        history = self._get_metric_history(base_sql_file, metric, target_block, window)
        
        if history.empty or len(history.columns) < 3:
            return pd.Series(dtype=float)
        
        def compute_r_squared(row):
            """Compute R² for a node's values over time."""
            values = row.dropna().values
            if len(values) < 3:
                return np.nan
            x = np.arange(len(values))
            # Linear regression
            slope, intercept = np.polyfit(x, values, 1)
            predicted = slope * x + intercept
            ss_res = np.sum((values - predicted) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            if ss_tot == 0:
                return 1.0  # Constant values = perfect fit
            return 1 - (ss_res / ss_tot)
        
        return history.apply(compute_r_squared, axis=1)
    
    def compute_tenure_ratio(
        self,
        base_sql_file: str,
        target_block: int
    ) -> pd.Series:
        """
        Compute tenure ratio: (blocks present) / (total blocks since first seen).
        
        Higher values indicate more consistent network participation.
        """
        snapshots = self.storage.list_snapshots(base_sql_file)
        snapshots.sort(key=lambda s: s.block_number)
        
        # Filter to snapshots up to target
        snapshots = [s for s in snapshots if s.block_number <= target_block]
        
        if not snapshots:
            return pd.Series(dtype=float)
        
        # Track first appearance and presence count for each node
        first_seen_idx = {}
        presence_count = {}
        
        for idx, snapshot in enumerate(snapshots):
            try:
                node_ids = self.storage.load_snapshot_node_ids(
                    base_sql_file, snapshot.block_number
                )
                for nid in node_ids:
                    if nid not in first_seen_idx:
                        first_seen_idx[nid] = idx
                    presence_count[nid] = presence_count.get(nid, 0) + 1
            except Exception:
                continue
        
        # Compute tenure ratio
        total_snapshots = len(snapshots)
        tenure_ratios = {}
        
        for nid, first_idx in first_seen_idx.items():
            possible_snapshots = total_snapshots - first_idx
            if possible_snapshots > 0:
                tenure_ratios[nid] = presence_count.get(nid, 0) / possible_snapshots
            else:
                tenure_ratios[nid] = 1.0
        
        return pd.Series(tenure_ratios)
    
    def compute_z_score_temporal(
        self,
        base_sql_file: str,
        metric: str,
        target_block: int,
        window: int = 5
    ) -> pd.Series:
        """
        Compute temporal Z-score: how many std devs current value is from historical mean.
        """
        history = self._get_metric_history(base_sql_file, metric, target_block, window)
        
        if history.empty:
            return pd.Series(dtype=float)
        
        # Get current values (last column)
        current = history.iloc[:, -1]
        
        # Compute historical mean and std (excluding current)
        historical = history.iloc[:, :-1] if len(history.columns) > 1 else history
        hist_mean = historical.mean(axis=1)
        hist_std = historical.std(axis=1)
        
        # Compute Z-score
        z_scores = (current - hist_mean) / hist_std.replace(0, np.nan)
        
        return z_scores.fillna(0)
    
    def compute_percentile_rank(
        self,
        base_sql_file: str,
        metric: str,
        target_block: int
    ) -> pd.Series:
        """
        Compute percentile rank of each node's current metric value.
        """
        result = self.analysis_service.get_metric_values(
            base_sql_file, target_block, metric, include_values=True
        )
        
        if not result or not result.values:
            return pd.Series(dtype=float)
        
        values = pd.Series(result.values)
        
        # Compute percentile rank (0-1)
        return values.rank(pct=True)
    
    def compute_growth_score(
        self,
        base_sql_file: str,
        metric: str,
        target_block: int,
        window: int = 5
    ) -> pd.Series:
        """
        Compute growth score combining velocity, stability, and age.
        
        Formula: growth_score = velocity * stability * age_factor
        Where:
        - velocity: normalized rate of change
        - stability: 1 - volatility (coefficient of variation)
        - age_factor: log(1 + age) normalized
        """
        # Compute components
        velocity = self.compute_velocity(base_sql_file, metric, target_block, window)
        stability = self.compute_stability(base_sql_file, metric, target_block, window)
        ages = self.compute_node_age(base_sql_file, target_block)
        
        # Normalize velocity to [-1, 1]
        if len(velocity) > 0 and velocity.abs().max() > 0:
            velocity_norm = velocity / velocity.abs().max()
        else:
            velocity_norm = velocity
        
        # Normalize ages using log
        if len(ages) > 0 and ages.max() > 0:
            age_factor = np.log1p(ages) / np.log1p(ages.max())
        else:
            age_factor = pd.Series(0, index=ages.index)
        
        # Align all series
        common_nodes = velocity_norm.index.intersection(stability.index).intersection(age_factor.index)
        
        if len(common_nodes) == 0:
            return pd.Series(dtype=float)
        
        v = velocity_norm.loc[common_nodes]
        s = stability.loc[common_nodes]
        a = age_factor.loc[common_nodes]
        
        # Compute growth score
        # Positive velocity with high stability = good growth
        # Negative velocity with high stability = bad growth
        growth_score = v * s * (0.5 + 0.5 * a)
        
        return growth_score
    
    # =========================================================================
    # Main Computation Method
    # =========================================================================
    
    def compute_temporal_composite(
        self,
        config: TemporalCompositeConfig
    ) -> TemporalCompositeResult:
        """
        Compute a temporal composite metric based on configuration.
        """
        start_time = time.time()
        composite_id = self._generate_composite_id(config)
        
        print(f"[TEMPORAL] Computing {config.name}: {config.temporal_config.operation.value} "
              f"on {config.base_metric}, window={config.temporal_config.window_blocks}")
        
        # Select operation
        operation = config.temporal_config.operation
        window = config.temporal_config.window_blocks
        
        try:
            if operation == TemporalOperation.VELOCITY:
                values = self.compute_velocity(
                    config.base_sql_file, config.base_metric,
                    config.target_block, window
                )
            elif operation == TemporalOperation.ACCELERATION:
                values = self.compute_acceleration(
                    config.base_sql_file, config.base_metric,
                    config.target_block, window
                )
            elif operation == TemporalOperation.STABILITY:
                values = self.compute_stability(
                    config.base_sql_file, config.base_metric,
                    config.target_block, window
                )
            elif operation == TemporalOperation.VOLATILITY:
                values = self.compute_volatility(
                    config.base_sql_file, config.base_metric,
                    config.target_block, window
                )
            elif operation == TemporalOperation.MOMENTUM:
                values = self.compute_momentum(
                    config.base_sql_file, config.base_metric,
                    config.target_block, window,
                    config.temporal_config.decay_factor
                )
            elif operation == TemporalOperation.TREND_STRENGTH:
                values = self.compute_trend_strength(
                    config.base_sql_file, config.base_metric,
                    config.target_block, window
                )
            elif operation == TemporalOperation.AGE:
                values = self.compute_node_age(config.base_sql_file, config.target_block)
            elif operation == TemporalOperation.AGE_WEIGHTED:
                values = self.compute_age_weighted(
                    config.base_sql_file, config.base_metric,
                    config.target_block, config.temporal_config.age_weight
                )
            elif operation == TemporalOperation.TENURE_RATIO:
                values = self.compute_tenure_ratio(
                    config.base_sql_file, config.target_block
                )
            elif operation == TemporalOperation.Z_SCORE_TEMPORAL:
                values = self.compute_z_score_temporal(
                    config.base_sql_file, config.base_metric,
                    config.target_block, window
                )
            elif operation == TemporalOperation.PERCENTILE_RANK:
                values = self.compute_percentile_rank(
                    config.base_sql_file, config.base_metric,
                    config.target_block
                )
            elif operation == TemporalOperation.GROWTH_SCORE:
                values = self.compute_growth_score(
                    config.base_sql_file, config.base_metric,
                    config.target_block, window
                )
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            # Optional combination with another metric
            if config.combine_with:
                values = self._combine_metrics(
                    values, config.base_sql_file, config.target_block,
                    config.combine_with, config.combine_operation,
                    config.combine_weights
                )
            
            # Normalize if requested
            if config.temporal_config.normalize_output:
                values = self._normalize(values)
            
            # Compute statistics
            arr = np.array(values.dropna())
            nodes_with_history = len(arr)
            nodes_without_history = len(values) - nodes_with_history
            
            stats = TemporalMetricStatistics(
                count=len(arr),
                min=float(np.min(arr)) if len(arr) > 0 else 0,
                max=float(np.max(arr)) if len(arr) > 0 else 0,
                mean=float(np.mean(arr)) if len(arr) > 0 else 0,
                std=float(np.std(arr)) if len(arr) > 0 else 0,
                median=float(np.median(arr)) if len(arr) > 0 else 0,
                nodes_with_history=nodes_with_history,
                nodes_without_history=nodes_without_history,
                q25=float(np.percentile(arr, 25)) if len(arr) > 0 else 0,
                q75=float(np.percentile(arr, 75)) if len(arr) > 0 else 0
            )
            
            # Compute histogram for visualization
            histogram_bins = []
            histogram_counts = []
            if len(arr) > 0:
                num_bins = min(50, max(10, int(np.sqrt(len(arr)))))
                counts, bins = np.histogram(arr, bins=num_bins)
                histogram_bins = [float(b) for b in bins]
                histogram_counts = [int(c) for c in counts]
            
            # Get top and bottom nodes
            top_nodes = []
            bottom_nodes = []
            if len(values.dropna()) > 0:
                sorted_values = values.dropna().sort_values(ascending=False)
                
                # Top 10 nodes
                for node_id, val in sorted_values.head(10).items():
                    top_nodes.append({
                        'node_id': str(node_id),
                        'value': float(val) if np.isfinite(val) else 0
                    })
                
                # Bottom 10 nodes
                for node_id, val in sorted_values.tail(10).items():
                    bottom_nodes.append({
                        'node_id': str(node_id),
                        'value': float(val) if np.isfinite(val) else 0
                    })
            
            # Build formula description
            formula = self._build_formula_description(config)
            
            # Get blocks used
            snapshots = self.storage.list_snapshots(config.base_sql_file)
            snapshots.sort(key=lambda s: s.block_number)
            target_idx = next((i for i, s in enumerate(snapshots) if s.block_number == config.target_block), 0)
            start_idx = max(0, target_idx - window + 1)
            blocks_used = [s.block_number for s in snapshots[start_idx:target_idx + 1]]
            
            elapsed = time.time() - start_time
            
            return TemporalCompositeResult(
                name=config.name,
                temporal_composite_id=composite_id,
                base_metric=config.base_metric,
                temporal_operation=operation,
                formula_description=formula,
                base_sql_file=config.base_sql_file,
                target_block=config.target_block,
                blocks_used=blocks_used,
                snapshots_used=len(blocks_used),
                node_count=len(values),
                statistics=stats,
                values=values.to_dict() if config.save else None,
                histogram_bins=histogram_bins,
                histogram_counts=histogram_counts,
                top_nodes=top_nodes,
                bottom_nodes=bottom_nodes,
                computation_time_seconds=elapsed,
                saved=config.save
            )
            
        except Exception as e:
            print(f"[TEMPORAL] Error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _combine_metrics(
        self,
        temporal_values: pd.Series,
        base_sql_file: str,
        target_block: int,
        other_metric: str,
        operation: CombineOperation,
        weights: List[float]
    ) -> pd.Series:
        """Combine temporal metric with another metric."""
        result = self.analysis_service.get_metric_values(
            base_sql_file, target_block, other_metric, include_values=True
        )
        
        if not result or not result.values:
            return temporal_values
        
        other_values = pd.Series(result.values)
        
        # Align indices
        common = temporal_values.index.intersection(other_values.index)
        a = temporal_values.loc[common]
        b = other_values.loc[common]
        
        if operation == CombineOperation.MULTIPLY:
            combined = a * b
        elif operation == CombineOperation.ADD:
            combined = a + b
        elif operation == CombineOperation.SUBTRACT:
            combined = a - b
        elif operation == CombineOperation.DIVIDE:
            combined = a / (b + 1e-10)
        elif operation == CombineOperation.AVERAGE:
            combined = (a + b) / 2
        elif operation == CombineOperation.MAXIMUM:
            combined = np.maximum(a, b)
        elif operation == CombineOperation.MINIMUM:
            combined = np.minimum(a, b)
        elif operation == CombineOperation.WEIGHTED_SUM:
            w1, w2 = weights[0], weights[1] if len(weights) > 1 else 1 - weights[0]
            combined = w1 * a + w2 * b
        else:
            combined = a * b
        
        return combined
    
    def _normalize(self, values: pd.Series) -> pd.Series:
        """Normalize values to [0, 1] range."""
        min_val = values.min()
        max_val = values.max()
        
        if max_val - min_val < 1e-10:
            return pd.Series(0.5, index=values.index)
        
        return (values - min_val) / (max_val - min_val)
    
    def _build_formula_description(self, config: TemporalCompositeConfig) -> str:
        """Build human-readable formula description."""
        op = config.temporal_config.operation.value
        metric = config.base_metric
        window = config.temporal_config.window_blocks
        
        formulas = {
            "velocity": f"d({metric})/dt over {window} snapshots",
            "acceleration": f"d²({metric})/dt² over {window} snapshots",
            "stability": f"1 - CV({metric}) over {window} snapshots",
            "volatility": f"CV({metric}) over {window} snapshots",
            "momentum": f"exp_weighted_trend({metric}, decay={config.temporal_config.decay_factor})",
            "age": "snapshots_since_first_seen",
            "age_weighted": f"{metric} × (1 + {config.temporal_config.age_weight} × norm_age)",
        }
        
        formula = formulas.get(op, f"{op}({metric})")
        
        if config.combine_with:
            formula = f"({formula}) {config.combine_operation.value} {config.combine_with}"
        
        if config.temporal_config.normalize_output:
            formula = f"normalize({formula})"
        
        return formula
    
    # =========================================================================
    # Preset Methods
    # =========================================================================
    
    def get_presets(self) -> List[TemporalPresetInfo]:
        """Get list of available presets."""
        return list(self.PRESETS.values())
    
    def apply_preset(
        self,
        preset_id: str,
        base_sql_file: str,
        target_block: int,
        window_blocks: Optional[int] = None,
        save: bool = True
    ) -> TemporalCompositeResult:
        """Apply a preset temporal composite."""
        if preset_id not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset_id}")
        
        preset = self.PRESETS[preset_id]
        
        config = TemporalCompositeConfig(
            name=preset.name,
            base_metric=preset.base_metric,
            temporal_config=TemporalOperationConfig(
                operation=preset.temporal_operation,
                window_blocks=window_blocks or preset.default_window
            ),
            base_sql_file=base_sql_file,
            target_block=target_block,
            save=save
        )
        
        return self.compute_temporal_composite(config)
    
    def get_available_operations(self) -> AvailableOperationsResponse:
        """Get information about all available temporal operations."""
        return AvailableOperationsResponse(
            operations=list(self.OPERATION_INFO.values()),
            rate_of_change=[
                TemporalOperation.VELOCITY.value,
                TemporalOperation.ACCELERATION.value,
                TemporalOperation.MOMENTUM.value
            ],
            stability=[
                TemporalOperation.STABILITY.value,
                TemporalOperation.VOLATILITY.value
            ],
            age_based=[
                TemporalOperation.AGE.value,
                TemporalOperation.AGE_WEIGHTED.value,
                TemporalOperation.TENURE_RATIO.value
            ],
            relative=[
                TemporalOperation.RELATIVE_TO_COHORT.value,
                TemporalOperation.PERCENTILE_RANK.value,
                TemporalOperation.Z_SCORE_TEMPORAL.value
            ]
        )


# Singleton instance
temporal_composite_engine = TemporalCompositeEngine()