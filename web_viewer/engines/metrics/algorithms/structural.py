"""
Structural Metric Algorithms

Structural holes and robustness metrics.
"""

from typing import Dict, Any
import logging

import networkx as nx
import numpy as np

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


class StructuralHolesAlgorithm(BaseMetricAlgorithm):
    """Compute Burt's structural holes metrics."""
    
    name = "structural_holes"
    category = "structural"
    description = "Burt's structural holes (constraint, effective size)"
    cost = "medium"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            constraint = nx.constraint(U)
            effective = nx.effective_size(U)
            
            result = {}
            for node in nodes:
                c = constraint.get(node, 0)
                e = effective.get(node, 0)
                deg = U.degree(node)
                redundancy = 1 - (e / deg) if deg > 0 else 0
                
                result[node] = {
                    "constraint": c,
                    "effective_size": e,
                    "redundancy": redundancy,
                }
            return result
        except Exception as e:
            logger.warning(f"Structural holes failed: {e}")
            return {node: {} for node in nodes}


class ArticulationPointsAlgorithm(BaseMetricAlgorithm):
    """Compute whether node is an articulation point."""
    
    name = "articulation_points"
    category = "structural"
    description = "Whether node is an articulation point"
    cost = "medium"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            articulation = set(nx.articulation_points(U))
            logger.debug(f"Found {len(articulation)} articulation points")
            
            result = {}
            for node in nodes:
                result[node] = {"is_articulation_point": 1 if node in articulation else 0}
            return result
        except Exception as e:
            logger.warning(f"Articulation points failed: {e}")
            return {node: {} for node in nodes}


class BridgesAlgorithm(BaseMetricAlgorithm):
    """Compute number of bridge edges incident to node."""
    
    name = "bridges"
    category = "structural"
    description = "Number of bridge edges incident to node"
    cost = "medium"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            bridges = set(nx.bridges(U))
            logger.debug(f"Found {len(bridges)} bridges")
            
            result = {}
            for node in nodes:
                bridge_count = sum(1 for u, v in bridges if u == node or v == node)
                result[node] = {"bridge_count": bridge_count}
            return result
        except Exception as e:
            logger.warning(f"Bridges failed: {e}")
            return {node: {} for node in nodes}


class NeighborDegreeStatsAlgorithm(BaseMetricAlgorithm):
    """Compute statistics of neighbor degrees."""
    
    name = "neighbor_degree_stats"
    category = "structural"
    description = "Statistics of neighbor degrees"
    cost = "medium"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            avg_neighbor_deg_undirected = nx.average_neighbor_degree(U)
            avg_neighbor_deg_directed = nx.average_neighbor_degree(G)
            
            result = {}
            for node in nodes:
                neighbors = list(U.neighbors(node))
                if neighbors:
                    degs = [U.degree(n) for n in neighbors]
                    result[node] = {
                        "avg_neighbor_degree": float(np.mean(degs)),
                        "min_neighbor_degree": int(np.min(degs)),
                        "max_neighbor_degree": int(np.max(degs)),
                        "std_neighbor_degree": float(np.std(degs)),
                        "avg_neighbor_degree_undirected": avg_neighbor_deg_undirected.get(node, 0),
                        "avg_neighbor_degree_directed": avg_neighbor_deg_directed.get(node, 0),
                    }
                else:
                    result[node] = {
                        "avg_neighbor_degree": 0,
                        "min_neighbor_degree": 0,
                        "max_neighbor_degree": 0,
                        "std_neighbor_degree": 0,
                        "avg_neighbor_degree_undirected": 0,
                        "avg_neighbor_degree_directed": 0,
                    }
            return result
        except Exception as e:
            logger.warning(f"Neighbor degree stats failed: {e}")
            return {node: {} for node in nodes}


class BiconnectedComponentAlgorithm(BaseMetricAlgorithm):
    """Compute biconnected component membership."""
    
    name = "biconnected_component"
    category = "structural"
    description = "Biconnected component membership"
    cost = "medium"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            # Get biconnected components
            components = list(nx.biconnected_components(U))
            
            # Map nodes to component IDs and sizes
            node_to_comp = {}
            comp_sizes = {}
            for idx, comp in enumerate(components):
                comp_sizes[idx] = len(comp)
                for n in comp:
                    # A node can be in multiple biconnected components if it's an articulation point
                    # We assign to the largest component
                    if n not in node_to_comp or comp_sizes[idx] > comp_sizes.get(node_to_comp[n], 0):
                        node_to_comp[n] = idx
            
            result = {}
            for node in nodes:
                comp_id = node_to_comp.get(node, -1)
                result[node] = {
                    "biconnected_component_id": comp_id,
                    "biconnected_component_size": comp_sizes.get(comp_id, 0),
                }
            return result
        except Exception as e:
            logger.warning(f"Biconnected component failed: {e}")
            return {node: {} for node in nodes}