"""
Dominance Metric Algorithms

Dominance relationships.
"""

from typing import Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import networkx as nx

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


def _compute_node_dominance(G, node, n):
    """Compute dominance for a single node."""
    try:
        descendants = nx.descendants(G, node)
        dominated = len(descendants)
        return node, {
            'dominated_nodes_count': dominated,
            'dominance_ratio': dominated / (n - 1) if n > 1 else 0
        }
    except Exception:
        return node, {'dominated_nodes_count': 0, 'dominance_ratio': 0}


class DominanceAlgorithm(BaseMetricAlgorithm):
    """Compute dominated nodes count and ratio."""
    
    name = "dominance"
    category = "dominance"
    description = "Dominated nodes count and ratio"
    cost = "high"
    
    def __init__(self, n_jobs: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.n_jobs = n_jobs
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, n_jobs: int = None, **kwargs) -> Dict[str, Dict[str, Any]]:
        n_jobs = n_jobs or self.n_jobs
        n = G.number_of_nodes()
        
        try:
            result = {}
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                func = partial(_compute_node_dominance, G, n=n)
                futures = {executor.submit(func, node): node for node in nodes}
                
                for future in as_completed(futures):
                    node, dom_metrics = future.result()
                    result[node] = dom_metrics
            
            return result
        except Exception as e:
            logger.warning(f"Dominance computation failed: {e}")
            return {node: {} for node in nodes}