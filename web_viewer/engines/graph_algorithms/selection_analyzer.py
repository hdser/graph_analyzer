"""
Selection Analyzer

Analyze a selection of nodes: compute statistics, compare to full graph.

Location: web_viewer/engines/graph_algorithms/selection_analyzer.py
"""

import time
from typing import List, Dict, Any, Optional, Set

import networkx as nx
import numpy as np


class SelectionAnalyzer:
    """Analyze selected nodes in a graph."""
    
    def __init__(self, graph: nx.DiGraph):
        self.G = graph
        self._U = None
    
    @property
    def undirected(self) -> nx.Graph:
        """Lazy undirected graph."""
        if self._U is None:
            self._U = self.G.to_undirected()
        return self._U
    
    def analyze_selection(
        self,
        node_ids: List[str],
        metrics: Optional[List[str]] = None,
        compare_to_full: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a selection of nodes.
        
        Args:
            node_ids: List of selected node IDs
            metrics: List of metrics to compute (default: basic stats)
            compare_to_full: Include comparison to full graph
            
        Returns:
            Result dict with selection statistics
        """
        start = time.time()
        
        # Filter to valid nodes
        valid_nodes = [n for n in node_ids if n in self.G]
        
        if not valid_nodes:
            return {
                "node_count": 0,
                "valid_count": 0,
                "invalid_count": len(node_ids),
                "statistics": {},
                "comparison": {},
                "computation_time_ms": (time.time() - start) * 1000,
                "success": False,
                "message": "No valid nodes in selection",
            }
        
        invalid_count = len(node_ids) - len(valid_nodes)
        node_set = set(valid_nodes)
        
        # Extract induced subgraph
        subgraph = self.G.subgraph(valid_nodes)
        
        # Compute selection statistics
        stats = self._compute_statistics(subgraph, node_set)
        
        # Compare to full graph if requested
        comparison = {}
        if compare_to_full:
            full_stats = self._compute_statistics(self.G, set(self.G.nodes()))
            comparison = self._compare_statistics(stats, full_stats)
        
        return {
            "node_count": len(node_ids),
            "valid_count": len(valid_nodes),
            "invalid_count": invalid_count,
            "statistics": stats,
            "comparison": comparison,
            "computation_time_ms": (time.time() - start) * 1000,
            "success": True,
        }
    
    def _compute_statistics(
        self,
        graph: nx.DiGraph,
        node_set: Set[str]
    ) -> Dict[str, Any]:
        """Compute statistics for a graph."""
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        
        if n == 0:
            return {
                "node_count": 0,
                "edge_count": 0,
                "density": 0,
                "avg_degree": 0,
            }
        
        stats = {
            "node_count": n,
            "edge_count": m,
            "density": nx.density(graph),
        }
        
        # Degree statistics
        in_degrees = [d for _, d in graph.in_degree()]
        out_degrees = [d for _, d in graph.out_degree()]
        total_degrees = [d for _, d in graph.degree()]
        
        stats["avg_in_degree"] = np.mean(in_degrees) if in_degrees else 0
        stats["avg_out_degree"] = np.mean(out_degrees) if out_degrees else 0
        stats["avg_degree"] = np.mean(total_degrees) if total_degrees else 0
        stats["max_in_degree"] = max(in_degrees) if in_degrees else 0
        stats["max_out_degree"] = max(out_degrees) if out_degrees else 0
        stats["max_degree"] = max(total_degrees) if total_degrees else 0
        
        # Degree variance
        stats["degree_variance"] = np.var(total_degrees) if total_degrees else 0
        
        # Connectivity
        try:
            undirected = graph.to_undirected()
            components = list(nx.connected_components(undirected))
            stats["num_components"] = len(components)
            stats["largest_component_size"] = max(len(c) for c in components)
            stats["is_connected"] = len(components) == 1
        except Exception:
            stats["num_components"] = None
            stats["largest_component_size"] = None
            stats["is_connected"] = None
        
        # Clustering (on undirected)
        try:
            clustering = nx.clustering(graph.to_undirected())
            stats["avg_clustering"] = np.mean(list(clustering.values())) if clustering else 0
        except Exception:
            stats["avg_clustering"] = None
        
        return stats
    
    def _compare_statistics(
        self,
        selection_stats: Dict[str, Any],
        full_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare selection statistics to full graph."""
        comparison = {}
        
        # Percentage comparisons
        if full_stats.get("node_count", 0) > 0:
            comparison["node_percentage"] = (
                selection_stats["node_count"] / full_stats["node_count"] * 100
            )
        
        if full_stats.get("edge_count", 0) > 0:
            comparison["edge_percentage"] = (
                selection_stats["edge_count"] / full_stats["edge_count"] * 100
            )
        
        # Ratio comparisons for averages
        for key in ["avg_degree", "avg_in_degree", "avg_out_degree", "avg_clustering", "density"]:
            if key in selection_stats and key in full_stats:
                sel_val = selection_stats.get(key)
                full_val = full_stats.get(key)
                
                if sel_val is not None and full_val is not None and full_val > 0:
                    comparison[f"{key}_ratio"] = sel_val / full_val
                elif sel_val is not None and full_val is not None:
                    comparison[f"{key}_ratio"] = None
        
        return comparison
    
    def get_boundary_nodes(
        self,
        node_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Find boundary nodes: selection nodes with edges to non-selection nodes.
        
        Args:
            node_ids: List of selected node IDs
            
        Returns:
            Result dict with boundary nodes
        """
        start = time.time()
        
        node_set = set(n for n in node_ids if n in self.G)
        
        if not node_set:
            return {
                "boundary_nodes": [],
                "internal_nodes": [],
                "external_neighbors": [],
                "computation_time_ms": (time.time() - start) * 1000,
                "success": False,
            }
        
        boundary = []
        internal = []
        external_neighbors = set()
        
        for node in node_set:
            neighbors = set(self.G.predecessors(node)) | set(self.G.successors(node))
            external = neighbors - node_set
            
            if external:
                boundary.append(node)
                external_neighbors.update(external)
            else:
                internal.append(node)
        
        return {
            "boundary_nodes": boundary,
            "internal_nodes": internal,
            "external_neighbors": list(external_neighbors),
            "boundary_count": len(boundary),
            "internal_count": len(internal),
            "computation_time_ms": (time.time() - start) * 1000,
            "success": True,
        }
    
    def get_connecting_edges(
        self,
        node_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Get edges connecting selection to rest of graph.
        
        Args:
            node_ids: List of selected node IDs
            
        Returns:
            Result dict with connecting edges
        """
        start = time.time()
        
        node_set = set(n for n in node_ids if n in self.G)
        
        if not node_set:
            return {
                "incoming_edges": [],
                "outgoing_edges": [],
                "computation_time_ms": (time.time() - start) * 1000,
                "success": False,
            }
        
        incoming = []
        outgoing = []
        
        for node in node_set:
            # Incoming from outside
            for pred in self.G.predecessors(node):
                if pred not in node_set:
                    incoming.append({"source": pred, "target": node})
            
            # Outgoing to outside
            for succ in self.G.successors(node):
                if succ not in node_set:
                    outgoing.append({"source": node, "target": succ})
        
        return {
            "incoming_edges": incoming,
            "outgoing_edges": outgoing,
            "incoming_count": len(incoming),
            "outgoing_count": len(outgoing),
            "total_connecting": len(incoming) + len(outgoing),
            "computation_time_ms": (time.time() - start) * 1000,
            "success": True,
        }
    
    def get_selection_metrics(
        self,
        node_ids: List[str],
        metric_values: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Get metric statistics for selected nodes from precomputed metrics.
        
        Args:
            node_ids: List of selected node IDs
            metric_values: Dict of metric_name -> {node_id: value}
            
        Returns:
            Result dict with metric statistics for selection
        """
        start = time.time()
        
        valid_nodes = [n for n in node_ids if n in self.G]
        
        if not valid_nodes:
            return {
                "metrics": {},
                "computation_time_ms": (time.time() - start) * 1000,
                "success": False,
            }
        
        node_set = set(valid_nodes)
        result_metrics = {}
        
        for metric_name, values in metric_values.items():
            # Get values for selected nodes
            selection_values = [
                values[n] for n in valid_nodes 
                if n in values and values[n] is not None
            ]
            
            if selection_values:
                result_metrics[metric_name] = {
                    "count": len(selection_values),
                    "mean": np.mean(selection_values),
                    "std": np.std(selection_values),
                    "min": min(selection_values),
                    "max": max(selection_values),
                    "median": np.median(selection_values),
                    "sum": sum(selection_values),
                }
        
        return {
            "metrics": result_metrics,
            "node_count": len(valid_nodes),
            "computation_time_ms": (time.time() - start) * 1000,
            "success": True,
        }