"""
Reciprocity Metric Algorithms

Mutual connection patterns.
"""

from typing import Dict, Any
import logging

import networkx as nx

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


class ReciprocityAlgorithm(BaseMetricAlgorithm):
    """Compute mutual connection statistics."""
    
    name = "reciprocity"
    category = "reciprocity"
    description = "Mutual connection statistics"
    cost = "low"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        result = {}
        
        for node in nodes:
            out_neighbors = set(G.successors(node))
            in_neighbors = set(G.predecessors(node))
            mutual = out_neighbors & in_neighbors
            
            result[node] = {
                "mutual_count": len(mutual),
                "mutual_ratio": len(mutual) / len(out_neighbors) if out_neighbors else 0,
                "mutual_received_ratio": len(mutual) / len(in_neighbors) if in_neighbors else 0,
                "one_way_out": len(out_neighbors - mutual),
                "one_way_in": len(in_neighbors - mutual),
            }
        
        return result