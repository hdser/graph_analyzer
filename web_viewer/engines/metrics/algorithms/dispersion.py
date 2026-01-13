"""
Dispersion Metric Algorithms

Neighborhood spread patterns.
"""

from typing import Dict, Any
import logging

import networkx as nx
import numpy as np

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


class DispersionAlgorithm(BaseMetricAlgorithm):
    """Compute dispersion of node's neighborhood."""
    
    name = "dispersion"
    category = "dispersion"
    description = "Dispersion of node's neighborhood"
    cost = "very_high"
    max_nodes = 100
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        n_nodes = U.number_of_nodes()
        
        # Only compute for a sample if too many nodes
        sample_size = min(self.max_nodes, len(nodes))
        sampled = nodes[:sample_size]
        
        result = {}
        
        for node in sampled:
            try:
                disp = nx.dispersion(U, node)
                if disp:
                    result[node] = {
                        "avg_dispersion": float(np.mean(list(disp.values()))),
                        "max_dispersion": float(np.max(list(disp.values()))),
                    }
                else:
                    result[node] = {"avg_dispersion": 0, "max_dispersion": 0}
            except Exception:
                result[node] = {"avg_dispersion": 0, "max_dispersion": 0}
        
        # Fill in empty results for non-sampled nodes
        for node in nodes:
            if node not in result:
                result[node] = {}
        
        return result