"""
Clustering Metric Algorithms

Local connectivity and triangle formation metrics.
"""

from typing import Dict, Any
import logging
from itertools import combinations

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
            
            # For directed graphs, we need to compute differently
            # Count triangles in the underlying undirected graph
            triangles_dir = {}
            for node in G.nodes():
                # Get all neighbors (both in and out)
                all_neighbors = set(G.predecessors(node)) | set(G.successors(node))
                # Count triangles
                count = 0
                neighbors_list = list(all_neighbors)
                for i, n1 in enumerate(neighbors_list):
                    for n2 in neighbors_list[i+1:]:
                        if G.has_edge(n1, n2) or G.has_edge(n2, n1):
                            count += 1
                triangles_dir[node] = count
            
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


# =============================================================================
# NEW CLUSTERING ALGORITHMS
# =============================================================================

class CliqueCountAlgorithm(BaseMetricAlgorithm):
    """
    Count cliques containing each node.
    
    Computes the number of maximal cliques containing each node.
    """
    
    name = "clique_count"
    category = "clustering"
    description = "Number of maximal cliques containing node"
    cost = "high"
    graph_type = "undirected"
    # max_nodes is handled by registry/resolver - no internal limit
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            # Count cliques for each node
            clique_counts = {node: 0 for node in U.nodes()}
            max_clique_sizes = {node: 0 for node in U.nodes()}
            
            for clique in nx.find_cliques(U):
                clique_size = len(clique)
                for node in clique:
                    clique_counts[node] += 1
                    if clique_size > max_clique_sizes[node]:
                        max_clique_sizes[node] = clique_size
            
            result = {}
            for node in nodes:
                result[node] = {
                    "clique_count": clique_counts.get(node, 0),
                    "max_clique_size": max_clique_sizes.get(node, 0),
                }
            return result
        except Exception as e:
            logger.warning(f"Clique count failed: {e}")
            return {node: {} for node in nodes}


class AverageNeighborClusteringAlgorithm(BaseMetricAlgorithm):
    """
    Compute average clustering coefficient of neighbors.
    
    Measures how tightly connected a node's neighbors are to each other.
    """
    
    name = "average_neighbor_clustering"
    category = "clustering"
    description = "Average clustering coefficient of neighbors"
    cost = "medium"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            # First compute clustering for all nodes
            clustering = nx.clustering(U)
            
            result = {}
            for node in nodes:
                if node not in U or U.degree(node) == 0:
                    result[node] = {"avg_neighbor_clustering": 0}
                    continue
                
                neighbors = list(U.neighbors(node))
                if len(neighbors) == 0:
                    result[node] = {"avg_neighbor_clustering": 0}
                    continue
                
                # Compute average clustering of neighbors
                neighbor_clustering = [clustering.get(n, 0) for n in neighbors]
                avg_clustering = sum(neighbor_clustering) / len(neighbor_clustering)
                
                result[node] = {"avg_neighbor_clustering": avg_clustering}
            
            return result
        except Exception as e:
            logger.warning(f"Average neighbor clustering failed: {e}")
            return {node: {} for node in nodes}