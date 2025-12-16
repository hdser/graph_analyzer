"""
Timeseries Engine

Engine for computing timeseries metrics across historical snapshots including:
- Network-level metric aggregation over time
- Individual node trajectories
- Trend detection and analysis
- Distribution comparisons between snapshots
- Cohort analysis
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from backend.models.timeseries import (
    AggregationType,
    TrendDirection,
    TimeseriesPoint,
    TrajectoryPoint,
    NetworkSummaryPoint,
    TimeseriesStatistics,
    TrendAnalysis,
    DistributionComparison,
    TimeseriesData,
    NetworkTimeseriesData,
    NodeTrajectory,
    NodeTrajectoriesResponse,
    CohortDefinition,
    CohortStatistics,
    CohortTrajectory,
)
from backend.services.snapshot_storage import SnapshotStorage
from backend.services.snapshot_analysis_service import SnapshotAnalysisService


class TimeseriesEngine:
    """
    Engine for computing timeseries analysis across snapshots.
    
    Provides:
    - Network-level metric aggregation over time
    - Individual node trajectories
    - Trend detection
    - Distribution comparisons
    - Cohort analysis
    """
    
    def __init__(
        self,
        storage: Optional[SnapshotStorage] = None,
        analysis_service: Optional[SnapshotAnalysisService] = None
    ):
        """
        Initialize timeseries engine.
        
        Args:
            storage: SnapshotStorage instance
            analysis_service: SnapshotAnalysisService for loading metrics
        """
        self.storage = storage or SnapshotStorage()
        self.analysis_service = analysis_service
        
        # Lazy import to avoid circular imports
        if self.analysis_service is None:
            from backend.services.snapshot_analysis_service import snapshot_analysis_service
            self.analysis_service = snapshot_analysis_service
    
    # =========================================================================
    # Utility Functions
    # =========================================================================
    
    def _sanitize_float(self, value: float) -> float:
        """Sanitize float to be JSON-serializable (replace NaN/Inf with 0)."""
        if value is None or not np.isfinite(value):
            return 0.0
        return float(value)
    
    def _sanitize_statistics(self, stats: TimeseriesStatistics) -> TimeseriesStatistics:
        """Sanitize statistics to ensure all values are JSON-serializable."""
        return TimeseriesStatistics(
            count=stats.count,
            min=self._sanitize_float(stats.min),
            max=self._sanitize_float(stats.max),
            mean=self._sanitize_float(stats.mean),
            std=self._sanitize_float(stats.std),
            first_value=self._sanitize_float(stats.first_value),
            last_value=self._sanitize_float(stats.last_value),
            total_change=self._sanitize_float(stats.total_change),
            percent_change=self._sanitize_float(stats.percent_change)
        )
    
    # =========================================================================
    # Aggregation Functions
    # =========================================================================
    
    def _aggregate(self, values: np.ndarray, aggregation: AggregationType) -> float:
        """Apply aggregation function to values."""
        if len(values) == 0:
            return 0.0
        
        values = np.array(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        
        if len(values) == 0:
            return 0.0
        
        agg_funcs = {
            AggregationType.MEAN: np.mean,
            AggregationType.MEDIAN: np.median,
            AggregationType.SUM: np.sum,
            AggregationType.MIN: np.min,
            AggregationType.MAX: np.max,
            AggregationType.STD: np.std,
            AggregationType.COUNT: lambda x: len(x),
            AggregationType.P10: lambda x: np.percentile(x, 10),
            AggregationType.P25: lambda x: np.percentile(x, 25),
            AggregationType.P75: lambda x: np.percentile(x, 75),
            AggregationType.P90: lambda x: np.percentile(x, 90),
        }
        
        func = agg_funcs.get(aggregation, np.mean)
        return float(func(values))
    
    # =========================================================================
    # Network-Level Timeseries
    # =========================================================================
    
    def get_metric_timeseries(
        self,
        base_sql_file: str,
        metric: str,
        aggregation: AggregationType = AggregationType.MEAN,
        start_block: Optional[int] = None,
        end_block: Optional[int] = None,
        include_trend: bool = True
    ) -> TimeseriesData:
        """
        Get timeseries of a metric aggregated across all nodes.
        
        Args:
            base_sql_file: Base SQL file name
            metric: Metric name to track
            aggregation: How to aggregate across nodes
            start_block: Optional start block filter
            end_block: Optional end block filter
            include_trend: Whether to compute trend analysis
            
        Returns:
            TimeseriesData with aggregated values over time
        """
        print(f"[TIMESERIES] Computing {metric} ({aggregation.value}) for {base_sql_file}")
        start_time = time.time()
        
        # Get all snapshots
        snapshots = self.storage.list_snapshots(base_sql_file)
        snapshots.sort(key=lambda s: s.block_number)
        
        # Apply block filters
        if start_block:
            snapshots = [s for s in snapshots if s.block_number >= start_block]
        if end_block:
            snapshots = [s for s in snapshots if s.block_number <= end_block]
        
        if not snapshots:
            return TimeseriesData(
                base_sql_file=base_sql_file,
                metric=metric,
                aggregation=aggregation,
                data_points=[],
                statistics=TimeseriesStatistics(
                    count=0, min=0, max=0, mean=0, std=0,
                    first_value=0, last_value=0, total_change=0, percent_change=0
                ),
                snapshots_included=0
            )
        
        # Collect data points
        data_points = []
        
        for snapshot in snapshots:
            # Try to get metric values
            result = self.analysis_service.get_metric_values(
                base_sql_file, snapshot.block_number, metric, include_values=True
            )
            
            if result and result.values:
                values = np.array(list(result.values.values()), dtype=np.float64)
                agg_value = self._aggregate(values, aggregation)
                
                data_points.append(TimeseriesPoint(
                    block_number=snapshot.block_number,
                    timestamp=snapshot.block_timestamp,
                    value=agg_value,
                    node_count=snapshot.node_count,
                    sample_size=len(result.values)
                ))
            else:
                # Fallback: try loading directly from snapshot nodes via storage
                try:
                    nodes_data = self.storage.load_snapshot_nodes(base_sql_file, snapshot.block_number)
                    elements = nodes_data.get('elements', [])
                    
                    # Extract metric values from nodes
                    values = []
                    for el in elements:
                        node_data = el.get('data', {})
                        if metric in node_data:
                            val = node_data[metric]
                            if isinstance(val, (int, float)) and np.isfinite(val):
                                values.append(val)
                    
                    if values:
                        arr = np.array(values, dtype=np.float64)
                        agg_value = self._aggregate(arr, aggregation)
                        
                        data_points.append(TimeseriesPoint(
                            block_number=snapshot.block_number,
                            timestamp=snapshot.block_timestamp,
                            value=agg_value,
                            node_count=snapshot.node_count,
                            sample_size=len(values)
                        ))
                except Exception as e:
                    print(f"[TIMESERIES] Warning: Could not load metric {metric} from snapshot {snapshot.block_number}: {e}")
        
        if not data_points:
            return TimeseriesData(
                base_sql_file=base_sql_file,
                metric=metric,
                aggregation=aggregation,
                data_points=[],
                statistics=TimeseriesStatistics(
                    count=0, min=0, max=0, mean=0, std=0,
                    first_value=0, last_value=0, total_change=0, percent_change=0
                ),
                snapshots_included=0
            )
        
        # Compute statistics
        values = np.array([p.value for p in data_points])
        
        total_change = values[-1] - values[0]
        percent_change = (total_change / values[0] * 100) if values[0] != 0 else 0
        cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
        
        statistics = TimeseriesStatistics(
            count=len(values),
            min=self._sanitize_float(np.min(values)),
            max=self._sanitize_float(np.max(values)),
            mean=self._sanitize_float(np.mean(values)),
            std=self._sanitize_float(np.std(values)),
            first_value=self._sanitize_float(values[0]),
            last_value=self._sanitize_float(values[-1]),
            total_change=self._sanitize_float(total_change),
            percent_change=self._sanitize_float(percent_change),
            coefficient_of_variation=self._sanitize_float(cv) if cv else None
        )
        
        # Compute trend if requested
        trend = None
        if include_trend and len(data_points) >= 3:
            trend = self._compute_trend(data_points, metric, aggregation)
        
        elapsed = time.time() - start_time
        print(f"[TIMESERIES] Computed {len(data_points)} points in {elapsed:.2f}s")
        
        return TimeseriesData(
            base_sql_file=base_sql_file,
            metric=metric,
            aggregation=aggregation,
            data_points=data_points,
            statistics=statistics,
            trend=trend,
            snapshots_included=len(data_points),
            time_range_start=data_points[0].timestamp if data_points else None,
            time_range_end=data_points[-1].timestamp if data_points else None
        )
    
    def get_network_summary_timeseries(
        self,
        base_sql_file: str
    ) -> NetworkTimeseriesData:
        """
        Get network-level statistics over time.
        
        Returns node count, edge count, and density for each snapshot.
        """
        print(f"[TIMESERIES] Computing network summary for {base_sql_file}")
        
        snapshots = self.storage.list_snapshots(base_sql_file)
        snapshots.sort(key=lambda s: s.block_number)
        
        data_points = []
        
        for snapshot in snapshots:
            n = snapshot.node_count
            e = snapshot.edge_count
            density = e / (n * (n - 1)) if n > 1 else 0
            
            data_points.append(NetworkSummaryPoint(
                block_number=snapshot.block_number,
                timestamp=snapshot.block_timestamp,
                node_count=n,
                edge_count=e,
                density=density,
                avg_degree=2 * e / n if n > 0 else 0
            ))
        
        # Compute growth rates
        node_growth = 0.0
        edge_growth = 0.0
        
        if len(data_points) >= 2:
            node_changes = [
                data_points[i].node_count - data_points[i-1].node_count
                for i in range(1, len(data_points))
            ]
            edge_changes = [
                data_points[i].edge_count - data_points[i-1].edge_count
                for i in range(1, len(data_points))
            ]
            node_growth = float(np.mean(node_changes))
            edge_growth = float(np.mean(edge_changes))
        
        return NetworkTimeseriesData(
            base_sql_file=base_sql_file,
            data_points=data_points,
            node_growth_rate=node_growth,
            edge_growth_rate=edge_growth,
            snapshots_included=len(data_points),
            time_range_start=data_points[0].timestamp if data_points else None,
            time_range_end=data_points[-1].timestamp if data_points else None
        )
    
    # =========================================================================
    # Node-Level Trajectories
    # =========================================================================
    
    def get_node_trajectories(
        self,
        base_sql_file: str,
        node_ids: List[str],
        metric: str,
        include_statistics: bool = True,
        include_trend: bool = False
    ) -> NodeTrajectoriesResponse:
        """
        Get metric trajectories for specific nodes over time.
        
        Args:
            base_sql_file: Base SQL file name
            node_ids: List of node IDs to track
            metric: Metric to track
            include_statistics: Whether to compute statistics
            include_trend: Whether to compute trend analysis
            
        Returns:
            NodeTrajectoriesResponse with trajectories for each node
        """
        print(f"[TIMESERIES] Computing trajectories for {len(node_ids)} nodes, metric={metric}")
        start_time = time.time()
        
        # Get all snapshots
        snapshots = self.storage.list_snapshots(base_sql_file)
        snapshots.sort(key=lambda s: s.block_number)
        
        if not snapshots:
            return NodeTrajectoriesResponse(
                base_sql_file=base_sql_file,
                metric=metric,
                trajectories={},
                nodes_requested=len(node_ids),
                nodes_found=0,
                nodes_not_found=node_ids,
                block_numbers=[],
                timestamps=[]
            )
        
        # Initialize trajectories
        trajectories: Dict[str, List[TrajectoryPoint]] = {nid: [] for nid in node_ids}
        first_seen: Dict[str, int] = {}
        last_seen: Dict[str, int] = {}
        
        block_numbers = []
        timestamps = []
        
        # Process each snapshot
        for snapshot in snapshots:
            block_numbers.append(snapshot.block_number)
            timestamps.append(snapshot.block_timestamp)
            
            # Get metric values for this snapshot
            result = self.analysis_service.get_metric_values(
                base_sql_file, snapshot.block_number, metric, include_values=True
            )
            
            values_dict = result.values if result and result.values else {}
            
            # Get node existence for this snapshot
            try:
                node_ids_in_snapshot = self.storage.load_snapshot_node_ids(
                    base_sql_file, snapshot.block_number
                )
            except:
                node_ids_in_snapshot = set(values_dict.keys()) if values_dict else set()
            
            # Record trajectory points
            for nid in node_ids:
                exists = nid in node_ids_in_snapshot
                value = values_dict.get(nid) if exists and values_dict else None
                
                trajectories[nid].append(TrajectoryPoint(
                    block_number=snapshot.block_number,
                    timestamp=snapshot.block_timestamp,
                    value=value,
                    exists=exists
                ))
                
                if exists:
                    if nid not in first_seen:
                        first_seen[nid] = snapshot.block_number
                    last_seen[nid] = snapshot.block_number
        
        # Build NodeTrajectory objects
        result_trajectories: Dict[str, NodeTrajectory] = {}
        nodes_found = []
        nodes_not_found = []
        
        for nid in node_ids:
            points = trajectories[nid]
            present_count = sum(1 for p in points if p.exists)
            
            if present_count == 0:
                nodes_not_found.append(nid)
                continue
            
            nodes_found.append(nid)
            
            # Compute statistics on existing values
            values = [p.value for p in points if p.exists and p.value is not None]
            stats = None
            trend = None
            
            if include_statistics and values:
                arr = np.array(values)
                total_change = arr[-1] - arr[0] if len(arr) > 1 else 0
                percent_change = (total_change / arr[0] * 100) if arr[0] != 0 else 0
                
                stats = TimeseriesStatistics(
                    count=len(arr),
                    min=self._sanitize_float(np.min(arr)),
                    max=self._sanitize_float(np.max(arr)),
                    mean=self._sanitize_float(np.mean(arr)),
                    std=self._sanitize_float(np.std(arr)),
                    first_value=self._sanitize_float(arr[0]),
                    last_value=self._sanitize_float(arr[-1]),
                    total_change=self._sanitize_float(total_change),
                    percent_change=self._sanitize_float(percent_change)
                )
            
            if include_trend and len(values) >= 3:
                # Convert to TimeseriesPoints for trend computation
                ts_points = [
                    TimeseriesPoint(block_number=p.block_number, timestamp=p.timestamp, value=p.value)
                    for p in points if p.exists and p.value is not None
                ]
                if len(ts_points) >= 3:
                    trend = self._compute_trend(ts_points, metric, AggregationType.MEAN)
            
            result_trajectories[nid] = NodeTrajectory(
                node_id=nid,
                metric=metric,
                values=points,
                first_seen_block=first_seen.get(nid, 0),
                first_seen_timestamp=None,  # Could be computed
                last_seen_block=last_seen.get(nid, 0),
                last_seen_timestamp=None,
                snapshots_present=present_count,
                snapshots_missing=len(points) - present_count,
                statistics=stats,
                trend=trend
            )
        
        elapsed = time.time() - start_time
        print(f"[TIMESERIES] Computed {len(nodes_found)} trajectories in {elapsed:.2f}s")
        
        return NodeTrajectoriesResponse(
            base_sql_file=base_sql_file,
            metric=metric,
            trajectories=result_trajectories,
            nodes_requested=len(node_ids),
            nodes_found=len(nodes_found),
            nodes_not_found=nodes_not_found,
            block_numbers=block_numbers,
            timestamps=timestamps
        )
    
    # =========================================================================
    # Trend Analysis
    # =========================================================================
    
    def _compute_trend(
        self,
        data_points: List[TimeseriesPoint],
        metric: str,
        aggregation: AggregationType
    ) -> TrendAnalysis:
        """Compute trend analysis for a timeseries."""
        if len(data_points) < 3:
            raise ValueError("Need at least 3 data points for trend analysis")
        
        # Extract values and x-axis (use index as x)
        y = np.array([p.value for p in data_points], dtype=np.float64)
        x = np.arange(len(y), dtype=np.float64)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)
        r_squared = r_value ** 2
        
        # Determine trend direction
        if p_value > 0.05:
            direction = TrendDirection.STABLE
        elif abs(slope) < 0.01 * np.mean(np.abs(y)):  # Slope less than 1% of mean
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING
        
        # Check for volatility
        cv = np.std(y) / np.mean(y) if np.mean(y) != 0 else 0
        if cv > 0.5:  # High coefficient of variation
            direction = TrendDirection.VOLATILE
        
        # Compute changes
        absolute_change = self._sanitize_float(y[-1] - y[0])
        percent_change = self._sanitize_float((absolute_change / y[0]) * 100) if y[0] != 0 else 0.0
        
        # Compute volatility (std of differences)
        diffs = np.diff(y)
        volatility = self._sanitize_float(np.std(diffs)) if len(diffs) > 0 else 0.0
        
        # Compute max drawdown
        cummax = np.maximum.accumulate(y)
        drawdown = (cummax - y) / np.where(cummax == 0, 1, cummax)  # Avoid division by zero
        max_drawdown = self._sanitize_float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
        
        return TrendAnalysis(
            metric=metric,
            aggregation=aggregation,
            trend_direction=direction,
            slope=self._sanitize_float(slope),
            intercept=self._sanitize_float(intercept),
            r_squared=self._sanitize_float(r_squared),
            p_value=self._sanitize_float(p_value),
            is_significant=p_value < 0.05 if np.isfinite(p_value) else False,
            absolute_change=absolute_change,
            percent_change=percent_change,
            start_block=data_points[0].block_number,
            end_block=data_points[-1].block_number,
            num_points=len(data_points),
            volatility=volatility,
            max_drawdown=max_drawdown
        )
    
    def detect_metric_trend(
        self,
        base_sql_file: str,
        metric: str,
        aggregation: AggregationType = AggregationType.MEAN
    ) -> TrendAnalysis:
        """
        Compute trend analysis for a metric across all snapshots.
        """
        timeseries = self.get_metric_timeseries(
            base_sql_file, metric, aggregation, include_trend=True
        )
        
        if timeseries.trend is None:
            raise ValueError(f"Not enough data points for trend analysis on {metric}")
        
        return timeseries.trend
    
    # =========================================================================
    # Distribution Comparison
    # =========================================================================
    
    def compare_distributions(
        self,
        base_sql_file: str,
        metric: str,
        from_block: int,
        to_block: int
    ) -> DistributionComparison:
        """
        Compare metric distributions between two snapshots.
        
        Uses Kolmogorov-Smirnov test and computes various shift metrics.
        """
        from backend.models.timeseries import HistogramData
        
        print(f"[TIMESERIES] Comparing {metric} distributions: block {from_block} -> {to_block}")
        
        # Get metric values from both snapshots
        from_result = self.analysis_service.get_metric_values(
            base_sql_file, from_block, metric, include_values=True
        )
        to_result = self.analysis_service.get_metric_values(
            base_sql_file, to_block, metric, include_values=True
        )
        
        if from_result is None or from_result.values is None:
            raise ValueError(f"Metric {metric} not found in snapshot block {from_block}")
        if to_result is None or to_result.values is None:
            raise ValueError(f"Metric {metric} not found in snapshot block {to_block}")
        
        from_values = np.array(list(from_result.values.values()), dtype=np.float64)
        to_values = np.array(list(to_result.values.values()), dtype=np.float64)
        
        # Filter to finite values
        from_values = from_values[np.isfinite(from_values)]
        to_values = to_values[np.isfinite(to_values)]
        
        if len(from_values) == 0 or len(to_values) == 0:
            raise ValueError(f"No valid values for metric {metric}")
        
        # Compute histograms with shared bins for comparison
        all_values = np.concatenate([from_values, to_values])
        num_bins = min(50, max(10, int(np.sqrt(len(all_values)))))
        bins = np.histogram_bin_edges(all_values, bins=num_bins)
        
        from_counts, _ = np.histogram(from_values, bins=bins)
        to_counts, _ = np.histogram(to_values, bins=bins)
        
        from_histogram = HistogramData(
            bins=[float(b) for b in bins],
            counts=[int(c) for c in from_counts]
        )
        to_histogram = HistogramData(
            bins=[float(b) for b in bins],
            counts=[int(c) for c in to_counts]
        )
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = scipy_stats.ks_2samp(from_values, to_values)
        
        # Location shifts
        from_mean = float(np.mean(from_values))
        to_mean = float(np.mean(to_values))
        mean_shift = to_mean - from_mean
        mean_shift_pct = (mean_shift / from_mean * 100) if from_mean != 0 else 0
        
        from_median = float(np.median(from_values))
        to_median = float(np.median(to_values))
        median_shift = to_median - from_median
        median_shift_pct = (median_shift / from_median * 100) if from_median != 0 else 0
        
        # Spread changes
        from_std = float(np.std(from_values))
        to_std = float(np.std(to_values))
        std_change = to_std - from_std
        std_change_pct = (std_change / from_std * 100) if from_std != 0 else 0
        
        # Percentile changes
        percentiles = [10, 25, 50, 75, 90]
        percentile_changes = {}
        for p in percentiles:
            from_p = float(np.percentile(from_values, p))
            to_p = float(np.percentile(to_values, p))
            change_pct = ((to_p - from_p) / from_p * 100) if from_p != 0 else 0
            percentile_changes[f"p{p}"] = float(change_pct)
        
        # Shape changes
        from_skew = scipy_stats.skew(from_values)
        to_skew = scipy_stats.skew(to_values)
        from_kurt = scipy_stats.kurtosis(from_values)
        to_kurt = scipy_stats.kurtosis(to_values)
        
        # Get timestamps
        from_meta = self.storage.load_snapshot_metadata(base_sql_file, from_block)
        to_meta = self.storage.load_snapshot_metadata(base_sql_file, to_block)
        
        return DistributionComparison(
            metric=metric,
            from_block=from_block,
            to_block=to_block,
            from_timestamp=from_meta.block_timestamp if from_meta else None,
            to_timestamp=to_meta.block_timestamp if to_meta else None,
            from_count=len(from_values),
            to_count=len(to_values),
            from_histogram=from_histogram,
            to_histogram=to_histogram,
            ks_statistic=self._sanitize_float(ks_stat),
            ks_pvalue=self._sanitize_float(ks_pvalue),
            distributions_differ=ks_pvalue < 0.05 if np.isfinite(ks_pvalue) else False,
            mean_shift=self._sanitize_float(mean_shift),
            mean_shift_percent=self._sanitize_float(mean_shift_pct),
            median_shift=self._sanitize_float(median_shift),
            median_shift_percent=self._sanitize_float(median_shift_pct),
            std_change=self._sanitize_float(std_change),
            std_change_percent=self._sanitize_float(std_change_pct),
            percentile_changes={k: self._sanitize_float(v) for k, v in percentile_changes.items()},
            skewness_change=self._sanitize_float(to_skew - from_skew),
            kurtosis_change=self._sanitize_float(to_kurt - from_kurt)
        )
    
    # =========================================================================
    # Cohort Analysis
    # =========================================================================
    
    def get_cohort_trajectory(
        self,
        base_sql_file: str,
        cohort_definition: CohortDefinition,
        metric: str,
        aggregation: AggregationType = AggregationType.MEAN
    ) -> CohortTrajectory:
        """
        Track a cohort of nodes over time.
        
        A cohort is defined by when nodes first appeared or their initial metric values.
        """
        print(f"[TIMESERIES] Computing cohort trajectory: {cohort_definition.name}")
        
        snapshots = self.storage.list_snapshots(base_sql_file)
        snapshots.sort(key=lambda s: s.block_number)
        
        if not snapshots:
            raise ValueError(f"No snapshots found for {base_sql_file}")
        
        # Identify cohort members based on definition
        cohort_members = self._identify_cohort_members(
            base_sql_file, snapshots, cohort_definition
        )
        
        if not cohort_members:
            raise ValueError(f"No nodes match cohort definition: {cohort_definition.name}")
        
        initial_size = len(cohort_members)
        print(f"[TIMESERIES] Cohort '{cohort_definition.name}' has {initial_size} members")
        
        # Track cohort through time
        data_points = []
        
        for snapshot in snapshots:
            # Get current node IDs
            try:
                current_nodes = self.storage.load_snapshot_node_ids(
                    base_sql_file, snapshot.block_number
                )
            except:
                continue
            
            # How many cohort members still exist
            still_active = len(cohort_members & current_nodes)
            churned = initial_size - still_active
            
            # Get metric values for active cohort members
            result = self.analysis_service.get_metric_values(
                base_sql_file, snapshot.block_number, metric, include_values=True
            )
            
            metric_values = []
            if result and result.values:
                for nid in cohort_members:
                    if nid in result.values:
                        val = result.values[nid]
                        if np.isfinite(val):
                            metric_values.append(val)
            
            metric_mean = None
            metric_median = None
            metric_std = None
            
            if metric_values:
                arr = np.array(metric_values)
                metric_mean = float(np.mean(arr))
                metric_median = float(np.median(arr))
                metric_std = float(np.std(arr))
            
            data_points.append(CohortStatistics(
                block_number=snapshot.block_number,
                timestamp=snapshot.block_timestamp,
                cohort_size=initial_size,
                still_active=still_active,
                churned=churned,
                metric_mean=metric_mean,
                metric_median=metric_median,
                metric_std=metric_std
            ))
        
        final_size = data_points[-1].still_active if data_points else 0
        retention_rate = final_size / initial_size if initial_size > 0 else 0
        
        # Compute trend on metric values
        trend = None
        metric_points = [
            TimeseriesPoint(
                block_number=p.block_number,
                timestamp=p.timestamp,
                value=p.metric_mean
            )
            for p in data_points if p.metric_mean is not None
        ]
        
        if len(metric_points) >= 3:
            trend = self._compute_trend(metric_points, metric, aggregation)
        
        return CohortTrajectory(
            cohort_definition=cohort_definition,
            metric=metric,
            aggregation=aggregation,
            data_points=data_points,
            initial_size=initial_size,
            final_size=final_size,
            retention_rate=retention_rate,
            trend=trend
        )
    
    def _identify_cohort_members(
        self,
        base_sql_file: str,
        snapshots: list,
        definition: CohortDefinition
    ) -> set:
        """Identify nodes belonging to a cohort."""
        
        if definition.node_ids:
            # Explicit node list
            return set(definition.node_ids)
        
        if definition.cohort_type.value == "first_seen_block":
            # Nodes that first appeared in a block range
            cohort = set()
            seen_before = set()
            
            for snapshot in snapshots:
                try:
                    current_nodes = self.storage.load_snapshot_node_ids(
                        base_sql_file, snapshot.block_number
                    )
                except:
                    continue
                
                new_nodes = current_nodes - seen_before
                
                # Check if this snapshot is in the range
                in_range = True
                if definition.first_seen_block_start:
                    in_range = in_range and snapshot.block_number >= definition.first_seen_block_start
                if definition.first_seen_block_end:
                    in_range = in_range and snapshot.block_number <= definition.first_seen_block_end
                
                if in_range:
                    cohort.update(new_nodes)
                
                seen_before.update(current_nodes)
            
            return cohort
        
        # Default: use nodes from first snapshot
        if snapshots:
            try:
                return self.storage.load_snapshot_node_ids(
                    base_sql_file, snapshots[0].block_number
                )
            except:
                pass
        
        return set()


# Singleton instance
timeseries_engine = TimeseriesEngine()