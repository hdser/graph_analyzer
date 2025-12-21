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
        converters: List[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Compute metrics for all nodes in the graph.
        
        Args:
            G: NetworkX directed graph
            definitions: List of MetricDefinition objects to compute
            converters: List of trusted seed nodes (for trust algorithms)
            **kwargs: Additional parameters passed to algorithms
            
        Returns:
            DataFrame with node IDs as index and metric columns
        """
        nodes = list(G.nodes())
        n = len(nodes)
        
        if n == 0:
            logger.warning("Empty graph, returning empty DataFrame")
            return pd.DataFrame()
        
        # Create undirected version
        U = G.to_undirected()
        is_connected = nx.is_connected(U)
        
        logger.info(f"Computing {len(definitions)} metrics for {n} nodes")
        logger.info(f"Graph connected: {is_connected}")
        
        # Initialize results
        metrics = {node: {} for node in nodes}
        computed_count = 0
        skipped_count = 0
        
        start_time = time.time()
        
        # Compute each metric
        for i, defn in enumerate(definitions, 1):
            metric_start = time.time()
            
            # Get or create algorithm instance
            algorithm = self._get_algorithm(defn)
            if algorithm is None:
                logger.warning(f"No algorithm found for {defn.name}")
                skipped_count += 1
                continue
            
            # Check if algorithm can be computed
            if not algorithm.can_compute(n, is_connected):
                logger.debug(f"Skipping {defn.name}: cannot compute for this graph")
                skipped_count += 1
                continue
            
            # Compute metric
            try:
                logger.debug(f"[{i}/{len(definitions)}] Computing {defn.name}...")
                
                result = algorithm.compute(
                    G=G,
                    U=U,
                    nodes=nodes,
                    n_jobs=self.n_jobs,
                    converters=converters,
                    computed_metrics=metrics,  # Pass already computed metrics for dependencies
                    **kwargs
                )
                
                # Merge results
                for node, node_metrics in result.items():
                    if node in metrics:
                        metrics[node].update(node_metrics)
                
                elapsed = time.time() - metric_start
                computed_count += 1
                logger.debug(f"  ✓ {defn.name} computed in {elapsed:.2f}s")
                
            except Exception as e:
                logger.warning(f"  ✗ {defn.name} failed: {e}")
                skipped_count += 1
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(metrics, orient='index')
        df.index.name = 'avatar'
        df = df.reset_index()
        
        # Clean data types
        for col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        
        total_time = time.time() - start_time
        logger.info(f"Computation complete: {computed_count} metrics in {total_time:.2f}s")
        logger.info(f"Skipped: {skipped_count} metrics")
        
        return df
    
    def _get_algorithm(self, defn: MetricDefinition):
        """Get or create algorithm instance for a metric definition."""
        if defn.name in self._algorithm_cache:
            return self._algorithm_cache[defn.name]
        
        algorithm_class = get_algorithm_class(defn.algorithm_class)
        if algorithm_class is None:
            return None
        
        algorithm = algorithm_class()
        self._algorithm_cache[defn.name] = algorithm
        return algorithm


class MetricEngine:
    """
    High-level API for metric computation.
    
    Combines resolver and computer to provide a simple interface
    for computing graph metrics with various selection options.
    
    Usage:
        engine = MetricEngine(graph)
        df = engine.compute(metrics=["pagerank", "eigentrust"])
        # or
        df = engine.compute(categories=["centrality", "trust"])
        # or
        df = engine.compute(preset="essential")
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
        logger.info("=" * 70)
        logger.info("METRIC ENGINE INITIALIZED")
        logger.info("=" * 70)
        logger.info(f"Graph Statistics:")
        logger.info(f"  • Nodes: {self.n:,}")
        logger.info(f"  • Edges: {self.m:,}")
        if self.n > 0:
            logger.info(f"  • Avg degree: {2 * self.m / self.n:.2f}")
        if self.n > 1:
            logger.info(f"  • Density: {self.m / (self.n * (self.n - 1)):.6f}")
        logger.info(f"  • Is connected: {nx.is_connected(self.U)}")
        logger.info(f"Parallel Processing:")
        logger.info(f"  • CPU cores: {multiprocessing.cpu_count()}")
        logger.info(f"  • Workers: {self.n_jobs}")
        logger.info("=" * 70)
    
    def compute(
        self,
        preset: Optional[str] = None,
        categories: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        exclude_metrics: Optional[List[str]] = None,
        skip_expensive: bool = False,
        converters: List[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Compute metrics for the graph.
        
        Args:
            preset: Preset name (basic, essential, moderate, comprehensive, all)
            categories: List of category names
            metrics: List of individual metric names
            exclude_metrics: Metrics to exclude
            skip_expensive: Skip metrics with cost='very_high'
            converters: Trusted seed nodes for trust algorithms
            **kwargs: Additional parameters for algorithms
            
        Returns:
            DataFrame with metric values for each node
        """
        # Resolve metrics
        definitions = self.resolver.resolve(
            preset=preset,
            categories=categories,
            metrics=metrics,
            exclude_metrics=exclude_metrics,
            skip_expensive=skip_expensive,
            max_nodes=self.n,
            require_connected=nx.is_connected(self.U),
        )
        
        if not definitions:
            logger.warning("No metrics to compute")
            return pd.DataFrame({'avatar': list(self.G.nodes())})
        
        # Log what we're computing
        logger.info("")
        logger.info("METRICS TO COMPUTE")
        logger.info("-" * 40)
        by_category = {}
        for d in definitions:
            if d.category not in by_category:
                by_category[d.category] = []
            by_category[d.category].append(d.name)
        
        for cat, metric_names in sorted(by_category.items()):
            logger.info(f"  {cat}: {', '.join(metric_names)}")
        logger.info("-" * 40)
        
        # Compute
        return self.computer.compute(
            G=self.G,
            definitions=definitions,
            converters=converters,
            **kwargs
        )
    
    def preview(
        self,
        preset: Optional[str] = None,
        categories: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Preview what metrics would be computed without running computation.
        
        Returns dict with metric counts, categories, and estimated cost.
        """
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