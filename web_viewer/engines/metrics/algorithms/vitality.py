"""
Vitality Metric Algorithms

Node removal impact metrics.
"""

from typing import Dict, Any
import logging

import networkx as nx

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


class ClosenessVitalityAlgorithm(BaseMetricAlgorithm):
    """Compute change in Wiener index when node removed."""
    
    name = "closeness_vitality"
    category = "vitality"
    description = "Change in Wiener index when node removed"
    cost = "very_high"
    max_nodes = 500
    graph_type = "undirected"
    requires_connected = True
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        n_nodes = U.number_of_nodes()
        if n_nodes > self.max_nodes:
            return {node: {} for node in nodes}
        
        if not nx.is_connected(U):
            logger.debug("Skipping closeness vitality: graph not connected")
            return {node: {} for node in nodes}
        
        try:
            vitality = nx.closeness_vitality(U)
            
            result = {}
            for node in nodes:
                result[node] = {"closeness_vitality": vitality.get(node, 0)}
            return result
        except Exception as e:
            logger.warning(f"Closeness vitality failed: {e}")
            return {node: {} for node in nodes}