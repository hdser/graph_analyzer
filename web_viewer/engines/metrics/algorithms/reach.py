"""
Reach Metric Algorithms

N-hop reachability metrics.
"""

from typing import Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import networkx as nx

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


def _compute_node_reach(G, node, max_hops):
    """Compute BFS reach for a single node."""
    result = {}
    visited = {node}
    current = {node}
    total = 0
    n = G.number_of_nodes()
    
    for hop in range(1, max_hops + 1):
        next_level = set()
        for parent in current:
            neighbors = set(G.successors(parent))
            next_level.update(neighbors - visited)
        
        if next_level:
            visited.update(next_level)
            result[f'reach_hop_{hop}'] = len(next_level)
            total += len(next_level)
        else:
            result[f'reach_hop_{hop}'] = 0
        
        current = next_level
    
    result['total_reach'] = total
    result['network_penetration'] = total / (n - 1) if n > 1 else 0
    
    return node, result


class ReachAlgorithm(BaseMetricAlgorithm):
    """Compute N-hop reachability and network penetration."""
    
    name = "reach"
    category = "reach"
    description = "N-hop reachability and network penetration"
    cost = "high"
    
    def __init__(self, max_hops: int = 6, n_jobs: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.max_hops = max_hops
        self.n_jobs = n_jobs
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, n_jobs: int = None, **kwargs) -> Dict[str, Dict[str, Any]]:
        n_jobs = n_jobs or self.n_jobs
        max_hops = self.max_hops
        
        try:
            result = {}
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                func = partial(_compute_node_reach, G, max_hops=max_hops)
                futures = {executor.submit(func, node): node for node in nodes}
                
                for future in as_completed(futures):
                    node, reach_metrics = future.result()
                    result[node] = reach_metrics
            
            return result
        except Exception as e:
            logger.warning(f"Reach computation failed: {e}")
            return {node: {} for node in nodes}