"""
Distance Metric Algorithms

Network distance measures.
"""

from typing import Dict, Any
import logging

import networkx as nx

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


class GraphDistancesAlgorithm(BaseMetricAlgorithm):
    """Compute graph radius, diameter, center and periphery membership."""
    
    name = "graph_distances"
    category = "distances"
    description = "Graph radius, diameter, center and periphery membership"
    cost = "high"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            if nx.is_connected(U):
                radius = nx.radius(U)
                diameter = nx.diameter(U)
                center = set(nx.center(U))
                periphery = set(nx.periphery(U))
            else:
                # Use largest connected component
                largest_cc = max(nx.connected_components(U), key=len)
                subgraph = U.subgraph(largest_cc)
                radius = nx.radius(subgraph)
                diameter = nx.diameter(subgraph)
                center = set(nx.center(subgraph))
                periphery = set(nx.periphery(subgraph))
            
            result = {}
            for node in nodes:
                result[node] = {
                    "graph_radius": radius,
                    "graph_diameter": diameter,
                    "is_center": 1 if node in center else 0,
                    "is_periphery": 1 if node in periphery else 0,
                }
            return result
        except Exception as e:
            logger.warning(f"Graph distances failed: {e}")
            return {node: {} for node in nodes}