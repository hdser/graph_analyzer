"""
Efficiency Metric Algorithms

Communication efficiency metrics.
"""

from typing import Dict, Any
import logging

import networkx as nx

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


class LocalEfficiencyAlgorithm(BaseMetricAlgorithm):
    """Compute local efficiency of node's neighborhood."""
    
    name = "local_efficiency"
    category = "efficiency"
    description = "Local efficiency of node's neighborhood"
    cost = "high"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            local_eff = nx.local_efficiency(U)
            
            result = {}
            for node in nodes:
                result[node] = {"local_efficiency": local_eff}
            return result
        except Exception as e:
            logger.warning(f"Local efficiency failed: {e}")
            return {node: {} for node in nodes}