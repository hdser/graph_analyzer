"""
Path Metric Algorithms

Shortest path and reachability analysis.
"""

from typing import Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import networkx as nx
import numpy as np

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


def _compute_node_paths(G, node):
    """Compute path metrics for a single node."""
    result = {}
    try:
        lengths = dict(nx.single_source_shortest_path_length(G, node))
        if len(lengths) > 1:
            vals = [l for l in lengths.values() if l > 0]
            if vals:
                result['avg_shortest_path'] = float(np.mean(vals))
                result['median_shortest_path'] = float(np.median(vals))
                result['max_shortest_path'] = int(np.max(vals))
                result['path_variance'] = float(np.var(vals))
                result['path_sum'] = int(sum(vals))
                result['reachable_nodes'] = len(vals)
            else:
                result.update({
                    'avg_shortest_path': 0, 'median_shortest_path': 0,
                    'max_shortest_path': 0, 'path_variance': 0,
                    'path_sum': 0, 'reachable_nodes': 0
                })
        else:
            result.update({
                'avg_shortest_path': 0, 'median_shortest_path': 0,
                'max_shortest_path': 0, 'path_variance': 0,
                'path_sum': 0, 'reachable_nodes': 0
            })
    except Exception:
        result.update({
            'avg_shortest_path': 0, 'median_shortest_path': 0,
            'max_shortest_path': 0, 'path_variance': 0,
            'path_sum': 0, 'reachable_nodes': 0
        })
    return node, result


class ShortestPathsAlgorithm(BaseMetricAlgorithm):
    """Compute shortest path statistics with parallel processing."""
    
    name = "shortest_paths"
    category = "paths"
    description = "Shortest path statistics (avg, median, max)"
    cost = "high"
    
    def __init__(self, n_jobs: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.n_jobs = n_jobs
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, n_jobs: int = None, **kwargs) -> Dict[str, Dict[str, Any]]:
        n_jobs = n_jobs or self.n_jobs
        
        try:
            result = {}
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                func = partial(_compute_node_paths, G)
                futures = {executor.submit(func, node): node for node in nodes}
                
                for future in as_completed(futures):
                    node, path_metrics = future.result()
                    result[node] = path_metrics
            
            return result
        except Exception as e:
            logger.warning(f"Shortest paths computation failed: {e}")
            return {node: {} for node in nodes}


class HopPathsAlgorithm(BaseMetricAlgorithm):
    """Compute direct and 2-hop path counts."""
    
    name = "hop_paths"
    category = "paths"
    description = "Direct and 2-hop path counts"
    cost = "medium"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        result = {}
        
        for node in nodes:
            try:
                # Direct paths (1-hop)
                paths_length_1 = len(list(G.successors(node)))
                
                # 2-hop paths
                paths_2 = set()
                for neighbor in G.successors(node):
                    for second_hop in G.successors(neighbor):
                        if second_hop != node:
                            paths_2.add(second_hop)
                
                result[node] = {
                    "paths_length_1": paths_length_1,
                    "paths_length_2_targets": len(paths_2),
                }
            except Exception:
                result[node] = {"paths_length_1": 0, "paths_length_2_targets": 0}
        
        return result


class EccentricityAlgorithm(BaseMetricAlgorithm):
    """Compute eccentricity (maximum distance to any other node)."""
    
    name = "eccentricity"
    category = "paths"
    description = "Maximum distance to any other node"
    cost = "high"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            if nx.is_connected(U):
                ecc = nx.eccentricity(U)
            else:
                # Use largest connected component
                largest_cc = max(nx.connected_components(U), key=len)
                subgraph = U.subgraph(largest_cc)
                ecc = nx.eccentricity(subgraph)
            
            result = {}
            for node in nodes:
                result[node] = {"eccentricity": ecc.get(node, 0)}
            return result
        except Exception as e:
            logger.warning(f"Eccentricity failed: {e}")
            return {node: {} for node in nodes}


def _compute_node_wiener(U, node):
    """Compute Wiener contribution for a single node."""
    try:
        lengths = nx.single_source_shortest_path_length(U, node)
        return node, sum(lengths.values())
    except Exception:
        return node, 0


class WienerContributionAlgorithm(BaseMetricAlgorithm):
    """Compute node's contribution to Wiener index."""
    
    name = "wiener_contribution"
    category = "paths"
    description = "Node's contribution to Wiener index"
    cost = "high"
    max_nodes = 500
    graph_type = "undirected"
    
    def __init__(self, n_jobs: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.n_jobs = n_jobs
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, n_jobs: int = None, **kwargs) -> Dict[str, Dict[str, Any]]:
        n_nodes = U.number_of_nodes()
        if n_nodes > self.max_nodes:
            return {node: {} for node in nodes}
        
        n_jobs = n_jobs or self.n_jobs
        
        try:
            result = {}
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                func = partial(_compute_node_wiener, U)
                futures = {executor.submit(func, node): node for node in nodes}
                
                for future in as_completed(futures):
                    node, wiener = future.result()
                    result[node] = {"wiener_contribution": wiener}
            
            return result
        except Exception as e:
            logger.warning(f"Wiener contribution failed: {e}")
            return {node: {} for node in nodes}