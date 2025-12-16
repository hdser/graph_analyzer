"""
Snapshot Analysis Service

Service for running analysis on historical snapshots including:
- Computing full metrics for snapshots
- Running anomaly detection
- Storing and retrieving analysis results
"""

import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable

import numpy as np
import pandas as pd
import networkx as nx

from ..config import settings, HAS_ANOMALY
from ..models.snapshot_analysis import (
    SnapshotAnalysisConfig,
    SnapshotAnalysisResult,
    AnalysisProgressUpdate,
    AnalysisStatus,
    MetricStatistics,
    AnomalyResultSummary,
    AnalysisMetadata,
    AnalyzedSnapshotInfo,
    MetricValuesResponse,
)
from .snapshot_storage import SnapshotStorage

# Conditional imports
try:
    from engines.graph_metrics import GraphMetrics
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False
    print("[SNAPSHOT_ANALYSIS] Warning: GraphMetrics not available")

if HAS_ANOMALY:
    try:
        from engines.anomaly_engine import AnomalyEngine
        from engines.anomaly_config import AlgorithmConfig
    except ImportError:
        HAS_ANOMALY = False
        print("[SNAPSHOT_ANALYSIS] Warning: AnomalyEngine not available")


class SnapshotAnalysisService:
    """
    Service for analyzing historical snapshots.
    
    Provides:
    - Full metrics computation on snapshot data
    - Anomaly detection on snapshot data
    - Results storage and retrieval
    - Batch analysis with progress reporting
    """
    
    # Default metrics for anomaly detection
    DEFAULT_ANOMALY_METRICS = [
        "in_degree", "out_degree", "pagerank", 
        "clustering_coefficient", "betweenness_centrality"
    ]
    
    def __init__(self, storage: Optional[SnapshotStorage] = None):
        """
        Initialize the analysis service.
        
        Args:
            storage: SnapshotStorage instance (creates new if None)
        """
        self.storage = storage or SnapshotStorage()
        self._anomaly_engine = AnomalyEngine() if HAS_ANOMALY else None
        
    def _generate_analysis_id(self, snapshot_id: str, config: SnapshotAnalysisConfig) -> str:
        """Generate unique analysis ID based on snapshot and config."""
        config_str = f"{config.metrics_mode}_{config.run_anomaly_detection}_{config.anomaly_algorithm}"
        combined = f"{snapshot_id}_{config_str}_{int(time.time())}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]
    
    def _get_analysis_dir(self, base_sql_file: str, block_number: int) -> Path:
        """Get the analysis directory for a snapshot."""
        snapshot_dir = self.storage.get_snapshot_dir(base_sql_file, block_number)
        analysis_dir = snapshot_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        return analysis_dir
    
    # =========================================================================
    # Core Analysis Methods
    # =========================================================================
    
    def analyze_snapshot(
        self,
        base_sql_file: str,
        block_number: int,
        config: SnapshotAnalysisConfig,
        progress_callback: Optional[Callable[[AnalysisProgressUpdate], None]] = None
    ) -> SnapshotAnalysisResult:
        """
        Run analysis on a single snapshot.
        
        Args:
            base_sql_file: Base SQL file name
            block_number: Block number of snapshot
            config: Analysis configuration
            progress_callback: Optional callback for progress updates
            
        Returns:
            SnapshotAnalysisResult with all analysis data
        """
        start_time = time.time()
        snapshot_id = self.storage.get_snapshot_id(base_sql_file, block_number)
        analysis_id = self._generate_analysis_id(snapshot_id, config)
        
        print(f"[SNAPSHOT_ANALYSIS] Starting analysis: {snapshot_id}")
        print(f"[SNAPSHOT_ANALYSIS] Config: metrics_mode={config.metrics_mode}, "
              f"anomaly={config.run_anomaly_detection}")
        
        def report_progress(stage: str, percent: int, message: str, metric: str = None):
            if progress_callback:
                progress_callback(AnalysisProgressUpdate(
                    snapshot_id=snapshot_id,
                    status=AnalysisStatus.RUNNING,
                    stage=stage,
                    progress_percent=percent,
                    message=message,
                    current_metric=metric,
                    elapsed_seconds=time.time() - start_time
                ))
        
        try:
            # Verify snapshot exists
            if not self.storage.snapshot_exists(base_sql_file, block_number):
                raise ValueError(f"Snapshot not found: {snapshot_id}")
            
            report_progress("loading", 5, "Loading snapshot data...")
            
            # Load snapshot edges and existing metrics
            snapshot_data = self.storage.load_snapshot(base_sql_file, block_number)
            edges_df = pd.DataFrame(snapshot_data["edges"])
            
            # Load snapshot metadata
            metadata = self.storage.load_snapshot_metadata(base_sql_file, block_number)
            node_count = metadata.node_count if metadata else len(set(edges_df['source']) | set(edges_df['target']))
            edge_count = metadata.edge_count if metadata else len(edges_df)
            
            report_progress("loading", 15, f"Loaded {node_count} nodes, {edge_count} edges")
            
            # Build NetworkX graph
            report_progress("metrics", 20, "Building graph structure...")
            G = self._build_graph(edges_df)
            
            # Compute metrics
            metrics_computation_time = 0.0
            metrics_df = None
            metrics_computed = []
            metric_statistics = {}
            
            if HAS_METRICS:
                report_progress("metrics", 25, f"Computing metrics (mode: {config.metrics_mode})...")
                
                metrics_start = time.time()
                metrics_df, metrics_computed = self._compute_metrics(
                    G, config.metrics_mode, progress_callback, snapshot_id
                )
                metrics_computation_time = time.time() - metrics_start
                
                if metrics_df is not None and len(metrics_df) > 0:
                    report_progress("metrics", 60, f"Computing statistics for {len(metrics_computed)} metrics...")
                    metric_statistics = self._compute_metric_statistics(metrics_df, metrics_computed)
                
                print(f"[SNAPSHOT_ANALYSIS] Computed {len(metrics_computed)} metrics in {metrics_computation_time:.2f}s")
            else:
                print("[SNAPSHOT_ANALYSIS] Metrics computation not available")
            
            # Run anomaly detection
            anomaly_computation_time = 0.0
            anomaly_results = None
            
            if config.run_anomaly_detection and HAS_ANOMALY and metrics_df is not None:
                report_progress("anomaly", 70, "Running anomaly detection...")
                
                anomaly_start = time.time()
                anomaly_results = self._run_anomaly_detection(
                    metrics_df, config, metrics_computed
                )
                anomaly_computation_time = time.time() - anomaly_start
                
                if anomaly_results:
                    print(f"[SNAPSHOT_ANALYSIS] Found {anomaly_results.anomaly_count} anomalies "
                          f"in {anomaly_computation_time:.2f}s")
            
            # Save results if configured
            if config.save_results:
                report_progress("saving", 90, "Saving analysis results...")
                self._save_analysis_results(
                    base_sql_file, block_number, analysis_id,
                    config, metrics_df, metrics_computed,
                    metric_statistics, anomaly_results,
                    config.save_per_node_data
                )
            
            # Build result
            total_time = time.time() - start_time
            
            result = SnapshotAnalysisResult(
                snapshot_id=snapshot_id,
                base_sql_file=base_sql_file,
                block_number=block_number,
                block_timestamp=metadata.block_timestamp if metadata else None,
                analysis_id=analysis_id,
                analysis_timestamp=datetime.utcnow(),
                analysis_config=config,
                status=AnalysisStatus.COMPLETED,
                node_count=node_count,
                edge_count=edge_count,
                metrics_computed=metrics_computed,
                metric_statistics=metric_statistics,
                anomaly_results=anomaly_results,
                computation_time_seconds=total_time,
                metrics_computation_time=metrics_computation_time,
                anomaly_computation_time=anomaly_computation_time
            )
            
            report_progress("complete", 100, f"Analysis complete in {total_time:.1f}s")
            print(f"[SNAPSHOT_ANALYSIS] Completed {snapshot_id} in {total_time:.2f}s")
            
            return result
            
        except Exception as e:
            print(f"[SNAPSHOT_ANALYSIS] Error analyzing {snapshot_id}: {e}")
            import traceback
            traceback.print_exc()
            
            return SnapshotAnalysisResult(
                snapshot_id=snapshot_id,
                base_sql_file=base_sql_file,
                block_number=block_number,
                analysis_id=analysis_id,
                analysis_timestamp=datetime.utcnow(),
                analysis_config=config,
                status=AnalysisStatus.FAILED,
                node_count=0,
                edge_count=0,
                computation_time_seconds=time.time() - start_time,
                error_message=str(e)
            )
    
    def _build_graph(self, edges_df: pd.DataFrame) -> nx.DiGraph:
        """Build NetworkX DiGraph from edges DataFrame."""
        G = nx.DiGraph()
        
        for _, row in edges_df.iterrows():
            source = str(row.get('source', row.get('truster', '')))
            target = str(row.get('target', row.get('trustee', '')))
            if source and target:
                G.add_edge(source, target)
        
        return G
    
    def _compute_metrics(
        self,
        G: nx.DiGraph,
        metrics_mode: str,
        progress_callback: Optional[Callable] = None,
        snapshot_id: str = ""
    ) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """
        Compute graph metrics for the graph.
        
        Returns:
            Tuple of (metrics_df, list of computed metric names)
        """
        if not HAS_METRICS:
            return None, []
        
        try:
            calculator = GraphMetrics(G, metrics_mode=metrics_mode)
            metrics_df = calculator.compute_all()
            
            if metrics_df is not None and len(metrics_df) > 0:
                # Ensure avatar column
                if 'avatar' not in metrics_df.columns:
                    metrics_df['avatar'] = metrics_df.index.astype(str)
                
                metric_names = [col for col in metrics_df.columns if col != 'avatar']
                return metrics_df, metric_names
                
        except Exception as e:
            print(f"[SNAPSHOT_ANALYSIS] Metrics computation failed: {e}")
        
        return None, []
    
    def _compute_metric_statistics(
        self,
        metrics_df: pd.DataFrame,
        metric_names: List[str]
    ) -> Dict[str, MetricStatistics]:
        """Compute statistics for each metric."""
        statistics = {}
        
        for metric in metric_names:
            if metric not in metrics_df.columns:
                continue
            
            values = metrics_df[metric].dropna()
            
            if len(values) == 0:
                continue
            
            # Convert to numpy for calculations
            arr = np.array(values, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            
            if len(arr) == 0:
                continue
            
            try:
                stats = MetricStatistics(
                    metric_name=metric,
                    count=len(arr),
                    min=float(np.min(arr)),
                    max=float(np.max(arr)),
                    mean=float(np.mean(arr)),
                    std=float(np.std(arr)),
                    median=float(np.median(arr)),
                    q25=float(np.percentile(arr, 25)),
                    q75=float(np.percentile(arr, 75)),
                    skewness=float(self._compute_skewness(arr)) if len(arr) > 2 else None,
                    kurtosis=float(self._compute_kurtosis(arr)) if len(arr) > 3 else None
                )
                statistics[metric] = stats
            except Exception as e:
                print(f"[SNAPSHOT_ANALYSIS] Error computing stats for {metric}: {e}")
        
        return statistics
    
    def _compute_skewness(self, arr: np.ndarray) -> float:
        """Compute skewness of array."""
        n = len(arr)
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return 0.0
        return (n / ((n-1) * (n-2))) * np.sum(((arr - mean) / std) ** 3)
    
    def _compute_kurtosis(self, arr: np.ndarray) -> float:
        """Compute excess kurtosis of array."""
        n = len(arr)
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return 0.0
        return ((n * (n+1)) / ((n-1) * (n-2) * (n-3))) * \
               np.sum(((arr - mean) / std) ** 4) - \
               (3 * (n-1)**2) / ((n-2) * (n-3))
    
    def _run_anomaly_detection(
        self,
        metrics_df: pd.DataFrame,
        config: SnapshotAnalysisConfig,
        available_metrics: List[str]
    ) -> Optional[AnomalyResultSummary]:
        """Run anomaly detection on metrics data."""
        if not HAS_ANOMALY or self._anomaly_engine is None:
            return None
        
        try:
            # Determine which metrics to use
            metrics_to_use = config.anomaly_metrics if config.anomaly_metrics else []
            
            # Filter to available metrics
            if not metrics_to_use:
                # Use defaults if available
                metrics_to_use = [m for m in self.DEFAULT_ANOMALY_METRICS if m in available_metrics]
            else:
                metrics_to_use = [m for m in metrics_to_use if m in available_metrics]
            
            if len(metrics_to_use) < 1:
                print("[SNAPSHOT_ANALYSIS] No valid metrics for anomaly detection")
                return None
            
            print(f"[SNAPSHOT_ANALYSIS] Running {config.anomaly_algorithm} on {metrics_to_use}")
            
            # Prepare data
            # Filter to numeric columns only
            numeric_metrics = []
            for m in metrics_to_use:
                if m in metrics_df.columns:
                    col = metrics_df[m]
                    if np.issubdtype(col.dtype, np.number):
                        numeric_metrics.append(m)
            
            if len(numeric_metrics) < 1:
                return None
            
            # Build feature matrix
            X = metrics_df[numeric_metrics].values.astype(np.float64)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            node_ids = metrics_df['avatar'].astype(str).tolist() if 'avatar' in metrics_df.columns \
                       else metrics_df.index.astype(str).tolist()
            
            # Run detection
            start = time.time()
            result = self._anomaly_engine.detect(
                data=X,
                algorithm=config.anomaly_algorithm,
                parameters=config.anomaly_parameters,
                node_ids=node_ids
            )
            computation_time = time.time() - start
            
            if result is None:
                return None
            
            # Get scores array
            scores = np.array([result.scores.get(nid, 0.0) for nid in node_ids])
            
            # Identify anomalies
            anomaly_ids = [nid for nid, is_anom in result.binary_labels.items() if is_anom]
            
            # Sort by score to get top anomalies
            score_pairs = [(nid, result.scores.get(nid, 0.0)) for nid in node_ids]
            score_pairs.sort(key=lambda x: x[1], reverse=True)
            top_anomaly_ids = [nid for nid, _ in score_pairs[:100] if nid in anomaly_ids]
            
            return AnomalyResultSummary(
                algorithm=config.anomaly_algorithm,
                metrics_used=numeric_metrics,
                parameters=config.anomaly_parameters,
                total_nodes=len(node_ids),
                anomaly_count=len(anomaly_ids),
                anomaly_percentage=100.0 * len(anomaly_ids) / len(node_ids) if node_ids else 0.0,
                threshold_method=config.anomaly_threshold_method,
                threshold_value=float(result.threshold_info.value) if result.threshold_info else 0.5,
                score_min=float(np.min(scores)),
                score_max=float(np.max(scores)),
                score_mean=float(np.mean(scores)),
                score_std=float(np.std(scores)),
                top_anomaly_ids=top_anomaly_ids,
                computation_time_seconds=computation_time
            )
            
        except Exception as e:
            print(f"[SNAPSHOT_ANALYSIS] Anomaly detection failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # =========================================================================
    # Storage Methods
    # =========================================================================
    
    def _save_analysis_results(
        self,
        base_sql_file: str,
        block_number: int,
        analysis_id: str,
        config: SnapshotAnalysisConfig,
        metrics_df: Optional[pd.DataFrame],
        metrics_computed: List[str],
        metric_statistics: Dict[str, MetricStatistics],
        anomaly_results: Optional[AnomalyResultSummary],
        save_per_node_data: bool
    ) -> None:
        """Save analysis results to disk."""
        analysis_dir = self._get_analysis_dir(base_sql_file, block_number)
        
        # Save full metrics DataFrame
        if save_per_node_data and metrics_df is not None and len(metrics_df) > 0:
            metrics_path = analysis_dir / "full_metrics.parquet"
            metrics_df.to_parquet(metrics_path, compression="snappy")
            print(f"[SNAPSHOT_ANALYSIS] Saved metrics to {metrics_path}")
        
        # Save metric statistics
        stats_path = analysis_dir / "metric_statistics.json"
        stats_dict = {name: stat.model_dump() for name, stat in metric_statistics.items()}
        with open(stats_path, 'w') as f:
            json.dump(stats_dict, f, indent=2, default=str)
        
        # Save anomaly results
        if anomaly_results:
            anomaly_path = analysis_dir / "anomaly_results.json"
            with open(anomaly_path, 'w') as f:
                json.dump(anomaly_results.model_dump(), f, indent=2, default=str)
        
        # Save metadata
        meta = AnalysisMetadata(
            analysis_id=analysis_id,
            snapshot_id=self.storage.get_snapshot_id(base_sql_file, block_number),
            base_sql_file=base_sql_file,
            block_number=block_number,
            config=config,
            status=AnalysisStatus.COMPLETED,
            metrics_computed=metrics_computed,
            has_anomaly_results=anomaly_results is not None,
            anomaly_count=anomaly_results.anomaly_count if anomaly_results else None,
            analysis_started=datetime.utcnow(),
            analysis_completed=datetime.utcnow(),
            computation_time_seconds=0.0  # Will be updated
        )
        
        meta_path = analysis_dir / "analysis_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta.model_dump(), f, indent=2, default=str)
        
        print(f"[SNAPSHOT_ANALYSIS] Saved analysis metadata to {meta_path}")
    
    def load_analysis_results(
        self,
        base_sql_file: str,
        block_number: int
    ) -> Optional[SnapshotAnalysisResult]:
        """Load stored analysis results for a snapshot."""
        analysis_dir = self._get_analysis_dir(base_sql_file, block_number)
        meta_path = analysis_dir / "analysis_meta.json"
        
        if not meta_path.exists():
            return None
        
        try:
            # Load metadata
            with open(meta_path, 'r') as f:
                meta_dict = json.load(f)
            
            # Load metric statistics
            stats_path = analysis_dir / "metric_statistics.json"
            metric_statistics = {}
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    stats_dict = json.load(f)
                metric_statistics = {
                    name: MetricStatistics(**stat) 
                    for name, stat in stats_dict.items()
                }
            
            # Load anomaly results
            anomaly_results = None
            anomaly_path = analysis_dir / "anomaly_results.json"
            if anomaly_path.exists():
                with open(anomaly_path, 'r') as f:
                    anomaly_dict = json.load(f)
                anomaly_results = AnomalyResultSummary(**anomaly_dict)
            
            # Build result
            config = SnapshotAnalysisConfig(**meta_dict.get('config', {}))
            
            return SnapshotAnalysisResult(
                snapshot_id=meta_dict['snapshot_id'],
                base_sql_file=base_sql_file,
                block_number=block_number,
                analysis_id=meta_dict['analysis_id'],
                analysis_timestamp=datetime.fromisoformat(meta_dict['analysis_started']),
                analysis_config=config,
                status=AnalysisStatus(meta_dict['status']),
                node_count=0,  # Not stored in meta
                edge_count=0,
                metrics_computed=meta_dict.get('metrics_computed', []),
                metric_statistics=metric_statistics,
                anomaly_results=anomaly_results,
                computation_time_seconds=meta_dict.get('computation_time_seconds', 0.0)
            )
            
        except Exception as e:
            print(f"[SNAPSHOT_ANALYSIS] Error loading analysis results: {e}")
            return None
    
    def load_analysis_metrics(
        self,
        base_sql_file: str,
        block_number: int
    ) -> Optional[pd.DataFrame]:
        """Load the full metrics DataFrame from a stored analysis."""
        analysis_dir = self._get_analysis_dir(base_sql_file, block_number)
        metrics_path = analysis_dir / "full_metrics.parquet"
        
        if not metrics_path.exists():
            return None
        
        try:
            return pd.read_parquet(metrics_path)
        except Exception as e:
            print(f"[SNAPSHOT_ANALYSIS] Error loading metrics: {e}")
            return None
    
    def get_metric_values(
        self,
        base_sql_file: str,
        block_number: int,
        metric_name: str,
        include_values: bool = True
    ) -> Optional[MetricValuesResponse]:
        """Get values for a specific metric from stored analysis."""
        # First try analysis directory
        metrics_df = self.load_analysis_metrics(base_sql_file, block_number)
        
        if metrics_df is None:
            # Try loading from snapshot's stored metrics parquet file
            try:
                snapshot_dir = self.storage.get_snapshot_dir(base_sql_file, block_number)
                metrics_path = snapshot_dir / "metrics.parquet"
                
                if metrics_path.exists():
                    metrics_df = pd.read_parquet(metrics_path)
                    id_col = 'avatar' if 'avatar' in metrics_df.columns else metrics_df.columns[0]
                    metrics_df[id_col] = metrics_df[id_col].astype(str)
                    metrics_df = metrics_df.rename(columns={id_col: 'avatar'})
            except Exception as e:
                print(f"[SNAPSHOT_ANALYSIS] Warning loading metrics parquet: {e}")
        
        if metrics_df is None:
            # Final fallback: try loading from snapshot nodes via load_snapshot_nodes
            try:
                nodes_data = self.storage.load_snapshot_nodes(base_sql_file, block_number)
                elements = nodes_data.get('elements', [])
                
                if elements:
                    rows = []
                    for el in elements:
                        node_data = el.get('data', {})
                        if metric_name in node_data:
                            val = node_data.get(metric_name)
                            if isinstance(val, (int, float)) and np.isfinite(val):
                                node_id = node_data.get('id', node_data.get('avatar', ''))
                                rows.append({'avatar': str(node_id), metric_name: float(val)})
                    
                    if rows:
                        metrics_df = pd.DataFrame(rows)
            except Exception as e:
                print(f"[SNAPSHOT_ANALYSIS] Warning loading metric from nodes: {e}")
        
        if metrics_df is None or metric_name not in metrics_df.columns:
            return None
        
        # Get values
        values = metrics_df[metric_name].dropna()
        arr = np.array(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        
        if len(arr) == 0:
            return None
        
        # Compute statistics
        stats = MetricStatistics(
            metric_name=metric_name,
            count=len(arr),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            median=float(np.median(arr)),
            q25=float(np.percentile(arr, 25)),
            q75=float(np.percentile(arr, 75))
        )
        
        # Compute histogram
        hist_counts, bin_edges = np.histogram(arr, bins=30)
        
        # Build values dict if requested
        values_dict = None
        if include_values:
            id_col = 'avatar' if 'avatar' in metrics_df.columns else metrics_df.index
            if 'avatar' in metrics_df.columns:
                values_dict = dict(zip(
                    metrics_df['avatar'].astype(str),
                    metrics_df[metric_name].astype(float)
                ))
            else:
                values_dict = dict(zip(
                    metrics_df.index.astype(str),
                    metrics_df[metric_name].astype(float)
                ))
            # Clean NaN values
            values_dict = {k: v for k, v in values_dict.items() if np.isfinite(v)}
        
        return MetricValuesResponse(
            snapshot_id=self.storage.get_snapshot_id(base_sql_file, block_number),
            metric_name=metric_name,
            node_count=len(arr),
            statistics=stats,
            values=values_dict,
            histogram_bins=[float(b) for b in bin_edges],
            histogram_counts=[int(c) for c in hist_counts]
        )
    
    def has_analysis(self, base_sql_file: str, block_number: int) -> bool:
        """Check if a snapshot has stored analysis results."""
        analysis_dir = self._get_analysis_dir(base_sql_file, block_number)
        meta_path = analysis_dir / "analysis_meta.json"
        return meta_path.exists()
    
    def list_analyzed_snapshots(self, base_sql_file: str) -> List[AnalyzedSnapshotInfo]:
        """List all snapshots that have analysis results."""
        snapshots = self.storage.list_snapshots(base_sql_file)
        analyzed = []
        
        for snapshot in snapshots:
            if self.has_analysis(base_sql_file, snapshot.block_number):
                result = self.load_analysis_results(base_sql_file, snapshot.block_number)
                if result:
                    analyzed.append(AnalyzedSnapshotInfo(
                        snapshot_id=snapshot.snapshot_id,
                        block_number=snapshot.block_number,
                        block_timestamp=snapshot.block_timestamp,
                        has_analysis=True,
                        analysis_timestamp=result.analysis_timestamp,
                        metrics_computed=result.metrics_computed,
                        has_anomaly_results=result.anomaly_results is not None,
                        anomaly_count=result.anomaly_results.anomaly_count if result.anomaly_results else None,
                        node_count=snapshot.node_count,
                        edge_count=snapshot.edge_count
                    ))
        
        return analyzed
    
    def delete_analysis(self, base_sql_file: str, block_number: int) -> bool:
        """Delete stored analysis results for a snapshot."""
        import shutil
        
        analysis_dir = self._get_analysis_dir(base_sql_file, block_number)
        
        if not analysis_dir.exists():
            return False
        
        try:
            shutil.rmtree(analysis_dir)
            print(f"[SNAPSHOT_ANALYSIS] Deleted analysis for {base_sql_file} block {block_number}")
            return True
        except Exception as e:
            print(f"[SNAPSHOT_ANALYSIS] Error deleting analysis: {e}")
            return False


# Singleton instance
snapshot_analysis_service = SnapshotAnalysisService()