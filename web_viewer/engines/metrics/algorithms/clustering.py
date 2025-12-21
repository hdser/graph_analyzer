"""
Clustering Metric Algorithms

Local connectivity and triangle formation metrics.
"""

from typing import Dict, Any
import logging

import networkx as nx

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


class ClusteringCoefficientAlgorithm(BaseMetricAlgorithm):
    """Compute local clustering coefficient."""
    
    name = "clustering_coefficient"
    category = "clustering"
    description = "Local clustering coefficient"
    cost = "medium"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            clustering = nx.clustering(U)
            clustering_dir = nx.clustering(G)
            
            result = {}
            for node in nodes:
                result[node] = {
                    "clustering_coefficient": clustering.get(node, 0),
                    "clustering_coefficient_directed": clustering_dir.get(node, 0),
                }
            return result
        except Exception as e:
            logger.warning(f"Clustering coefficient failed: {e}")
            return {node: {} for node in nodes}


class TrianglesAlgorithm(BaseMetricAlgorithm):
    """Compute triangle count."""
    
    name = "triangles"
    category = "clustering"
    description = "Number of triangles involving node"
    cost = "medium"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            triangles = nx.triangles(U)
            triangles_dir = nx.triangles(G)
            
            result = {}
            for node in nodes:
                result[node] = {
                    "triangle_count": triangles.get(node, 0),
                    "triangle_count_directed": triangles_dir.get(node, 0),
                }
            return result
        except Exception as e:
            logger.warning(f"Triangles failed: {e}")
            return {node: {} for node in nodes}


class SquareClusteringAlgorithm(BaseMetricAlgorithm):
    """Compute square clustering coefficient."""
    
    name = "square_clustering"
    category = "clustering"
    description = "Square clustering coefficient"
    cost = "medium"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            square = nx.square_clustering(U)
            
            result = {}
            for node in nodes:
                result[node] = {"square_clustering": square.get(node, 0)}
            return result
        except Exception as e:
            logger.warning(f"Square clustering failed: {e}")
            return {node: {} for node in nodes}


class LocalTransitivityAlgorithm(BaseMetricAlgorithm):
    """Compute local transitivity (same as clustering coefficient)."""
    
    name = "local_transitivity"
    category = "clustering"
    description = "Local transitivity (same as clustering coefficient)"
    cost = "medium"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            clustering = nx.clustering(U)
            
            result = {}
            for node in nodes:
                result[node] = {"local_transitivity": clustering.get(node, 0)}
            return result
        except Exception as e:
            logger.warning(f"Local transitivity failed: {e}")
            return {node: {} for node in nodes}