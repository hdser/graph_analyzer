"""
Graph Coloring Metric Algorithms

Graph coloring related metrics.
"""

from typing import Dict, Any, List
import logging

import networkx as nx

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


class GreedyColorAlgorithm(BaseMetricAlgorithm):
    """
    Compute greedy graph coloring.
    
    Assigns colors to nodes using greedy coloring and computes
    color-related metrics for each node.
    """
    
    name = "greedy_color"
    category = "graph_coloring"
    description = "Greedy graph coloring metrics"
    cost = "low"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, parameters=None, **kwargs) -> Dict[str, Dict[str, Any]]:
        params = parameters or {}
        strategy = params.get('strategy', 'largest_first')
        
        try:
            # Compute greedy coloring
            coloring = nx.coloring.greedy_color(U, strategy=strategy)
            
            # Count nodes in each color class
            color_counts = {}
            for node, color in coloring.items():
                color_counts[color] = color_counts.get(color, 0) + 1
            
            # Total number of colors used
            num_colors = len(color_counts)
            
            result = {}
            for node in nodes:
                if node not in U:
                    result[node] = {
                        "greedy_color": -1,
                        "color_class_size": 0,
                        "chromatic_estimate": 0,
                    }
                    continue
                
                color = coloring.get(node, -1)
                
                result[node] = {
                    "greedy_color": color,
                    "color_class_size": color_counts.get(color, 0),
                    "chromatic_estimate": num_colors,
                }
            
            return result
        except Exception as e:
            logger.warning(f"Greedy coloring failed: {e}")
            return {node: {} for node in nodes}