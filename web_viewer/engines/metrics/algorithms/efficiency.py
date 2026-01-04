"""
Efficiency Metric Algorithms

Communication efficiency metrics measuring how effectively information
can be transmitted through the network.
"""

from typing import Dict, Any
import logging

import networkx as nx
import numpy as np

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


class LocalEfficiencyAlgorithm(BaseMetricAlgorithm):
    """
    Compute local efficiency of node's neighborhood.
    
    Local efficiency measures the efficiency of information transfer
    among the neighbors of a node, indicating how well-connected
    the neighborhood is.
    """
    
    name = "local_efficiency"
    category = "efficiency"
    description = "Local efficiency of node's neighborhood"
    cost = "high"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            # Compute local efficiency for each node
            result = {}
            for node in nodes:
                try:
                    neighbors = list(U.neighbors(node))
                    if len(neighbors) < 2:
                        # Need at least 2 neighbors for local efficiency
                        local_eff = 0.0
                    else:
                        # Get subgraph of neighbors
                        subgraph = U.subgraph(neighbors)
                        local_eff = nx.global_efficiency(subgraph)
                    
                    result[node] = {"local_efficiency": local_eff}
                except Exception:
                    result[node] = {"local_efficiency": 0.0}
            
            return result
            
        except Exception as e:
            logger.warning(f"Local efficiency failed: {e}")
            return {node: {} for node in nodes}


class GlobalEfficiencyContributionAlgorithm(BaseMetricAlgorithm):
    """
    Compute each node's contribution to global efficiency.
    
    Measures how much the global efficiency of the network would decrease
    if the node (and its edges) were removed. Higher values indicate
    nodes that are critical for network communication.
    
    This is computationally expensive as it requires recomputing
    global efficiency for each node removal.
    
    Computes:
    - global_efficiency_contribution: Decrease in global efficiency when removed
    - global_efficiency_ratio: Ratio of efficiency decrease to original efficiency
    """
    
    name = "global_efficiency_contribution"
    category = "efficiency"
    description = "Node's contribution to global network efficiency"
    cost = "very_high"
    graph_type = "undirected"
    # max_nodes is handled by registry/resolver - no internal limit
    # Internal sampling is used for performance on large graphs
    sampling_threshold = 2000  # Sample nodes if graph exceeds this for performance
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            n = len(nodes)
            
            # Calculate base global efficiency
            base_efficiency = nx.global_efficiency(U)
            
            if base_efficiency == 0:
                # Network is already disconnected
                return {node: {
                    "global_efficiency_contribution": 0.0,
                    "global_efficiency_ratio": 0.0,
                } for node in nodes}
            
            result = {}
            
            # For large graphs, sample nodes for performance
            if n > self.sampling_threshold:
                sample_size = min(1000, n)
                sampled = np.random.choice(nodes, sample_size, replace=False)
                logger.debug(f"Sampling {sample_size} of {n} nodes for efficiency contribution")
            else:
                sampled = nodes
            
            sampled_set = set(sampled)
            
            for node in nodes:
                if node in sampled_set:
                    try:
                        # Create graph without this node
                        U_minus = U.copy()
                        U_minus.remove_node(node)
                        
                        if U_minus.number_of_nodes() > 0:
                            new_efficiency = nx.global_efficiency(U_minus)
                            contribution = base_efficiency - new_efficiency
                            ratio = contribution / base_efficiency if base_efficiency > 0 else 0
                        else:
                            contribution = base_efficiency
                            ratio = 1.0
                        
                        result[node] = {
                            "global_efficiency_contribution": contribution,
                            "global_efficiency_ratio": ratio,
                        }
                    except Exception:
                        result[node] = {
                            "global_efficiency_contribution": 0.0,
                            "global_efficiency_ratio": 0.0,
                        }
                else:
                    # For non-sampled nodes, estimate based on degree
                    degree = U.degree(node)
                    avg_degree = 2 * U.number_of_edges() / n if n > 0 else 0
                    estimated_ratio = (degree / avg_degree) / n if avg_degree > 0 and n > 0 else 0
                    result[node] = {
                        "global_efficiency_contribution": base_efficiency * estimated_ratio,
                        "global_efficiency_ratio": estimated_ratio,
                    }
            
            logger.debug(f"Global efficiency contribution computed, base efficiency={base_efficiency:.4f}")
            return result
            
        except Exception as e:
            logger.warning(f"Global efficiency contribution failed: {e}")
            return {node: {} for node in nodes}


class NodeEfficiencyAlgorithm(BaseMetricAlgorithm):
    """
    Compute per-node efficiency metrics.
    
    For each node, computes how efficiently it can reach/be reached
    by all other nodes in the network.
    
    Computes:
    - node_efficiency: Average inverse distance from this node to all others
    - node_efficiency_in: Average inverse distance from all others to this node (directed)
    """
    
    name = "node_efficiency"
    category = "efficiency"
    description = "Per-node communication efficiency"
    cost = "high"
    # max_nodes is handled by registry/resolver - no internal limit
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            n = len(nodes)
            if n <= 1:
                return {node: {
                    "node_efficiency": 0.0,
                    "node_efficiency_in": 0.0,
                } for node in nodes}
            
            result = {}
            
            # For undirected efficiency
            for node in nodes:
                try:
                    # Compute shortest paths from this node (undirected)
                    distances_u = nx.single_source_shortest_path_length(U, node)
                    
                    # Sum of inverse distances (excluding self)
                    inv_sum = sum(1.0 / d for d in distances_u.values() if d > 0)
                    efficiency = inv_sum / (n - 1) if n > 1 else 0
                    
                    # For directed: paths coming TO this node
                    # Use reverse graph for incoming paths
                    distances_in = {}
                    G_rev = G.reverse(copy=False)
                    try:
                        distances_in = nx.single_source_shortest_path_length(G_rev, node)
                    except Exception:
                        pass
                    
                    inv_sum_in = sum(1.0 / d for d in distances_in.values() if d > 0)
                    efficiency_in = inv_sum_in / (n - 1) if n > 1 else 0
                    
                    result[node] = {
                        "node_efficiency": efficiency,
                        "node_efficiency_in": efficiency_in,
                    }
                except Exception:
                    result[node] = {
                        "node_efficiency": 0.0,
                        "node_efficiency_in": 0.0,
                    }
            
            return result
            
        except Exception as e:
            logger.warning(f"Node efficiency failed: {e}")
            return {node: {} for node in nodes}


class RobustnessEfficiencyAlgorithm(BaseMetricAlgorithm):
    """
    Compute efficiency-based robustness metrics.
    
    Measures how the network's efficiency would be affected if
    specific attack strategies were employed. Useful for identifying
    critical infrastructure nodes.
    
    Computes:
    - efficiency_criticality: Normalized impact on efficiency if removed
    - efficiency_redundancy: Measure of alternate paths available
    """
    
    name = "robustness_efficiency"
    category = "efficiency"
    description = "Efficiency-based robustness analysis"
    cost = "high"
    graph_type = "undirected"
    # max_nodes is handled by registry/resolver - no internal limit
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            n = len(nodes)
            
            # Base metrics
            base_efficiency = nx.global_efficiency(U)
            
            result = {}
            
            for node in nodes:
                try:
                    neighbors = list(U.neighbors(node))
                    degree = len(neighbors)
                    
                    # Count number of edges between neighbors (redundancy)
                    if degree >= 2:
                        neighbor_edges = U.subgraph(neighbors).number_of_edges()
                        max_possible = degree * (degree - 1) / 2
                        redundancy = neighbor_edges / max_possible if max_possible > 0 else 0
                    else:
                        redundancy = 0.0
                    
                    # Criticality estimate based on betweenness-like metric
                    # Without full removal, estimate based on local structure
                    criticality = (degree / n) * (1 - redundancy) if n > 0 else 0
                    
                    result[node] = {
                        "efficiency_criticality": criticality,
                        "efficiency_redundancy": redundancy,
                    }
                except Exception:
                    result[node] = {
                        "efficiency_criticality": 0.0,
                        "efficiency_redundancy": 0.0,
                    }
            
            return result
            
        except Exception as e:
            logger.warning(f"Robustness efficiency failed: {e}")
            return {node: {} for node in nodes}


class RoutingEfficiencyAlgorithm(BaseMetricAlgorithm):
    """
    Compute routing efficiency metrics.
    
    Measures how efficiently information can be routed through each node,
    considering both local and global routing properties.
    
    Computes:
    - routing_efficiency: Node's effectiveness as a routing point
    - path_diversity: Number of distinct shortest paths through node
    """
    
    name = "routing_efficiency"
    category = "efficiency"
    description = "Routing efficiency through each node"
    cost = "high"
    # max_nodes is handled by registry/resolver - no internal limit
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            n = len(nodes)
            
            # Use betweenness as base for routing importance
            betweenness = nx.betweenness_centrality(U, normalized=True)
            
            result = {}
            
            for node in nodes:
                try:
                    neighbors = list(U.neighbors(node))
                    degree = len(neighbors)
                    
                    # Routing efficiency based on betweenness and connectivity
                    bc = betweenness.get(node, 0)
                    
                    # Path diversity: estimate based on neighbor connectivity
                    if degree >= 2:
                        subgraph = U.subgraph(neighbors)
                        neighbor_edges = subgraph.number_of_edges()
                        # More edges between neighbors = fewer unique paths through node
                        path_div = degree * (1 - neighbor_edges / (degree * (degree - 1) / 2)) if degree > 1 else 0
                    else:
                        path_div = 0
                    
                    # Combined routing efficiency
                    routing_eff = bc * (1 + path_div / n) if n > 0 else bc
                    
                    result[node] = {
                        "routing_efficiency": routing_eff,
                        "path_diversity": path_div,
                    }
                except Exception:
                    result[node] = {
                        "routing_efficiency": 0.0,
                        "path_diversity": 0.0,
                    }
            
            return result
            
        except Exception as e:
            logger.warning(f"Routing efficiency failed: {e}")
            return {node: {} for node in nodes}