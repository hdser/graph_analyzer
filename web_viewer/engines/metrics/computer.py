"""
Metric Computer / Engine

Main orchestrator for computing graph metrics.
Handles metric resolution, computation, and aggregation.
"""

import time
import logging
import multiprocessing
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
import networkx as nx

from .registry import (
    METRIC_REGISTRY,
    MetricDefinition,
    list_all_metrics,
    list_categories,
    list_presets,
)
from .resolver import MetricResolver
from .algorithms import ALGORITHM_CLASSES, get_algorithm_class

logger = logging.getLogger(__name__)


class MetricComputer:
    """
    Computes metrics for a graph based on resolved metric definitions.
    
    This is the low-level computation engine that executes individual
    metric algorithms and aggregates results.
    """
    
    def __init__(self, n_jobs: int = None):
        """
        Initialize metric computer.
        
        Args:
            n_jobs: Number of parallel workers (None = auto)
        """
        self.n_jobs = n_jobs or max(1, multiprocessing.cpu_count() - 1)
        self._algorithm_cache: Dict[str, Any] = {}
    
    def compute(
        self,
        G: nx.DiGraph,
        definitions: List[MetricDefinition],
        metric_parameters: Optional[Dict[str, Dict[str, Any]]] = None,
        converters: List[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Compute metrics for all nodes in the graph.
        
        Args:
            G: NetworkX directed graph
            definitions: List of MetricDefinition objects to compute
            metric_parameters: Per-metric parameter overrides {metric_name: {param: value}}
            converters: List of trusted seed nodes (for trust algorithms)
            **kwargs: Additional parameters passed to algorithms
            
        Returns:
            DataFrame with node IDs as index and metric columns
        """
        nodes = list(G.nodes())
        n = len(nodes)
        
        if n == 0:
            print("[COMPUTER] Empty graph, returning empty DataFrame")
            return pd.DataFrame()
        
        # Create undirected version
        U = G.to_undirected()
        is_connected = nx.is_connected(U)
        
        print(f"[COMPUTER] ╔{'═'*60}╗")
        print(f"[COMPUTER] ║ STARTING METRIC COMPUTATION")
        print(f"[COMPUTER] ╠{'═'*60}╣")
        print(f"[COMPUTER] ║ Graph: {n:,} nodes, {G.number_of_edges():,} edges")
        print(f"[COMPUTER] ║ Connected: {is_connected}")
        print(f"[COMPUTER] ║ Definitions to compute: {len(definitions)}")
        print(f"[COMPUTER] ╚{'═'*60}╝")
        
        # Log definitions we received
        print(f"[COMPUTER] Metrics queue:")
        for i, defn in enumerate(definitions, 1):
            print(f"[COMPUTER]   {i}. {defn.name} (class={defn.algorithm_class}, cost={defn.cost})")
        
        # Initialize results
        metrics = {node: {} for node in nodes}
        computed_count = 0
        skipped_count = 0
        failed_metrics = []
        
        start_time = time.time()
        
        # Compute each metric
        for i, defn in enumerate(definitions, 1):
            metric_start = time.time()
            
            print(f"\n[COMPUTER] ┌─ [{i}/{len(definitions)}] {defn.name} ─────────────────────")
            print(f"[COMPUTER] │ Algorithm class: {defn.algorithm_class}")
            print(f"[COMPUTER] │ Cost: {defn.cost}, Source: {getattr(defn, 'source', 'networkx')}")
            
            # Get or create algorithm instance
            algorithm = self._get_algorithm(defn)
            if algorithm is None:
                print(f"[COMPUTER] │ ✗ FAILED: Algorithm class not found!")
                print(f"[COMPUTER] │ Available: {list(ALGORITHM_CLASSES.keys())[:5]}...")
                print(f"[COMPUTER] └─────────────────────────────────────────────────")
                skipped_count += 1
                failed_metrics.append((defn.name, "algorithm_not_found"))
                continue
            
            print(f"[COMPUTER] │ ✓ Algorithm loaded: {type(algorithm).__name__}")
            
            # Check if algorithm can be computed
            can_compute = algorithm.can_compute(n, is_connected)
            print(f"[COMPUTER] │ can_compute(n={n}, connected={is_connected}) = {can_compute}")
            
            if not can_compute:
                # Get more info about why
                alg_max_nodes = getattr(algorithm, 'max_nodes', None)
                alg_requires_connected = getattr(algorithm, 'requires_connected', False)
                print(f"[COMPUTER] │ ✗ SKIPPED: Cannot compute for this graph")
                print(f"[COMPUTER] │   Algorithm max_nodes: {alg_max_nodes}")
                print(f"[COMPUTER] │   Algorithm requires_connected: {alg_requires_connected}")
                print(f"[COMPUTER] └─────────────────────────────────────────────────")
                skipped_count += 1
                failed_metrics.append((defn.name, f"cannot_compute (max_nodes={alg_max_nodes}, requires_connected={alg_requires_connected})"))
                continue
            
            # Get parameters for this metric
            params = metric_parameters.get(defn.name, {}) if metric_parameters else {}
            if params:
                print(f"[COMPUTER] │ Parameters: {params}")
            
            # Compute metric
            try:
                print(f"[COMPUTER] │ Computing...")
                
                result = algorithm.compute(
                    G=G,
                    U=U,
                    nodes=nodes,
                    n_jobs=self.n_jobs,
                    converters=converters,
                    computed_metrics=metrics,
                    parameters=params,
                    **kwargs
                )
                
                # Check result
                if result is None:
                    print(f"[COMPUTER] │ ✗ FAILED: Algorithm returned None")
                    print(f"[COMPUTER] └─────────────────────────────────────────────────")
                    skipped_count += 1
                    failed_metrics.append((defn.name, "returned_none"))
                    continue
                
                result_nodes = len(result)
                result_with_data = sum(1 for v in result.values() if v)
                
                # Sample some results
                sample_data = None
                if result:
                    sample_node = next(iter(result))
                    sample_data = result[sample_node]
                
                # Merge results
                merged_count = 0
                for node, node_metrics in result.items():
                    if node in metrics and node_metrics:
                        metrics[node].update(node_metrics)
                        merged_count += 1
                
                elapsed = time.time() - metric_start
                computed_count += 1
                
                print(f"[COMPUTER] │ ✓ SUCCESS in {elapsed:.2f}s")
                print(f"[COMPUTER] │   Nodes with results: {result_with_data}/{result_nodes}")
                print(f"[COMPUTER] │   Merged: {merged_count} nodes")
                if sample_data:
                    print(f"[COMPUTER] │   Sample output: {sample_data}")
                print(f"[COMPUTER] └─────────────────────────────────────────────────")
                
            except Exception as e:
                import traceback
                print(f"[COMPUTER] │ ✗ EXCEPTION: {e}")
                print(f"[COMPUTER] │ Traceback:")
                for line in traceback.format_exc().split('\n')[-5:]:
                    if line.strip():
                        print(f"[COMPUTER] │   {line}")
                print(f"[COMPUTER] └─────────────────────────────────────────────────")
                skipped_count += 1
                failed_metrics.append((defn.name, str(e)))
        
        # Convert to DataFrame
        print(f"\n[COMPUTER] Converting results to DataFrame...")
        df = pd.DataFrame.from_dict(metrics, orient='index')
        df.index.name = 'avatar'
        df = df.reset_index()
        
        # Clean data types
        for col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        
        total_time = time.time() - start_time
        
        print(f"\n[COMPUTER] ╔{'═'*60}╗")
        print(f"[COMPUTER] ║ COMPUTATION COMPLETE")
        print(f"[COMPUTER] ╠{'═'*60}╣")
        print(f"[COMPUTER] ║ Total time: {total_time:.2f}s")
        print(f"[COMPUTER] ║ Computed: {computed_count} metrics")
        print(f"[COMPUTER] ║ Skipped: {skipped_count} metrics")
        print(f"[COMPUTER] ║ Output: {len(df)} rows × {len(df.columns)} columns")
        print(f"[COMPUTER] ║ Columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
        print(f"[COMPUTER] ╚{'═'*60}╝")
        
        if failed_metrics:
            print(f"\n[COMPUTER] ⚠ Failed/Skipped metrics:")
            for name, reason in failed_metrics:
                print(f"[COMPUTER]   • {name}: {reason}")
        
        return df
    
    def _get_algorithm(self, defn: MetricDefinition):
        """Get or create algorithm instance for a metric definition."""
        if defn.name in self._algorithm_cache:
            return self._algorithm_cache[defn.name]
        
        algorithm_class = get_algorithm_class(defn.algorithm_class)
        
        if algorithm_class is None:
            print(f"[COMPUTER] Algorithm class '{defn.algorithm_class}' not found in ALGORITHM_CLASSES")
            return None
        
        algorithm = algorithm_class()
        self._algorithm_cache[defn.name] = algorithm
        return algorithm


class MetricEngine:
    """
    High-level API for metric computation.
    
    Combines resolver and computer to provide a simple interface
    for computing graph metrics with various selection options.
    """
    
    def __init__(
        self,
        graph: nx.DiGraph,
        n_jobs: int = None,
    ):
        """
        Initialize metric engine.
        
        Args:
            graph: NetworkX directed graph
            n_jobs: Number of parallel workers
        """
        self.G = graph
        self.U = graph.to_undirected()
        self.n = graph.number_of_nodes()
        self.m = graph.number_of_edges()
        self.n_jobs = n_jobs or max(1, multiprocessing.cpu_count() - 1)
        
        self.resolver = MetricResolver()
        self.computer = MetricComputer(n_jobs=self.n_jobs)
        
        self._print_graph_info()
    
    def _print_graph_info(self):
        """Print graph statistics."""
        print(f"\n[ENGINE] ══════════════════════════════════════════════════════")
        print(f"[ENGINE] METRIC ENGINE INITIALIZED")
        print(f"[ENGINE] ══════════════════════════════════════════════════════")
        print(f"[ENGINE] Graph: {self.n:,} nodes, {self.m:,} edges")
        if self.n > 0:
            print(f"[ENGINE] Avg degree: {2 * self.m / self.n:.2f}")
        if self.n > 1:
            density = self.m / (self.n * (self.n - 1))
            print(f"[ENGINE] Density: {density:.6f}")
        print(f"[ENGINE] Connected: {nx.is_connected(self.U)}")
        print(f"[ENGINE] Workers: {self.n_jobs} (CPUs: {multiprocessing.cpu_count()})")
        print(f"[ENGINE] ══════════════════════════════════════════════════════")
    
    def compute(
        self,
        preset: Optional[str] = None,
        categories: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        exclude_metrics: Optional[List[str]] = None,
        skip_expensive: bool = False,
        metric_parameters: Optional[Dict[str, Dict[str, Any]]] = None,
        converters: List[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Compute metrics for the graph.
        """
        print(f"\n[ENGINE] compute() called:")
        print(f"[ENGINE]   preset={preset}")
        print(f"[ENGINE]   categories={categories}")
        print(f"[ENGINE]   metrics={metrics}")
        print(f"[ENGINE]   exclude_metrics={exclude_metrics}")
        print(f"[ENGINE]   skip_expensive={skip_expensive}")
        
        # Resolve metrics
        print(f"[ENGINE] Resolving metrics...")
        definitions = self.resolver.resolve(
            preset=preset,
            categories=categories,
            metrics=metrics,
            exclude_metrics=exclude_metrics,
            skip_expensive=skip_expensive,
            max_nodes=self.n,
            require_connected=nx.is_connected(self.U),
        )
        
        print(f"[ENGINE] Resolver returned {len(definitions)} metric definitions")
        
        if not definitions:
            print(f"[ENGINE] ⚠ No metrics to compute!")
            if metrics:
                print(f"[ENGINE] Checking requested metrics in registry:")
                for m in metrics:
                    exists = m in METRIC_REGISTRY
                    if exists:
                        defn = METRIC_REGISTRY[m]
                        print(f"[ENGINE]   {m}: EXISTS (class={defn.algorithm_class}, max_nodes={defn.max_nodes})")
                    else:
                        print(f"[ENGINE]   {m}: NOT FOUND")
            return pd.DataFrame({'avatar': list(self.G.nodes())})
        
        # Log what we're computing
        print(f"\n[ENGINE] ────────────────────────────────────────────────────")
        print(f"[ENGINE] METRICS TO COMPUTE ({len(definitions)} total):")
        by_category = {}
        for d in definitions:
            if d.category not in by_category:
                by_category[d.category] = []
            by_category[d.category].append(d.name)
        
        for cat, metric_names in sorted(by_category.items()):
            print(f"[ENGINE]   {cat}: {', '.join(metric_names)}")
        print(f"[ENGINE] ────────────────────────────────────────────────────")
        
        # Compute
        return self.computer.compute(
            G=self.G,
            definitions=definitions,
            metric_parameters=metric_parameters,
            converters=converters,
            **kwargs
        )
    
    def preview(
        self,
        preset: Optional[str] = None,
        categories: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Preview what metrics would be computed."""
        return self.resolver.get_metrics_info(
            preset=preset,
            categories=categories,
            metrics=metrics,
        )
    
    @staticmethod
    def available_metrics() -> List[Dict[str, Any]]:
        """Get list of all available metrics."""
        return list_all_metrics()
    
    @staticmethod
    def available_categories() -> List[Dict[str, Any]]:
        """Get list of all categories."""
        return list_categories()
    
    @staticmethod
    def available_presets() -> List[Dict[str, Any]]:
        """Get list of all presets."""
        return list_presets()