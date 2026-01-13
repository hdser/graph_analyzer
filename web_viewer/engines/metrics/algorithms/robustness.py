"""
Robustness Metric Algorithms

Network robustness and resilience metrics.

NOTE: max_nodes limits are handled by the registry/resolver, not internally.
"""

from typing import Dict, Any, List
import logging

import networkx as nx

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


class NodeConnectivityAlgorithm(BaseMetricAlgorithm):
    """
    Compute local node connectivity.
    
    For each node, computes the minimum number of nodes that must be
    removed to disconnect the node from the rest of the graph.
    """
    
    name = "node_connectivity"
    category = "robustness"
    description = "Local node connectivity"
    cost = "high"
    graph_type = "undirected"
    # max_nodes is handled by registry/resolver - no internal limit
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            result = {}
            
            # Get graph connectivity as baseline
            try:
                graph_connectivity = nx.node_connectivity(U)
            except Exception:
                graph_connectivity = 0
            
            for node in nodes:
                if node not in U or U.degree(node) == 0:
                    result[node] = {"node_connectivity": 0}
                    continue
                
                # For each node, compute min connectivity to any other node
                min_connectivity = float('inf')
                neighbors = set(U.neighbors(node))
                
                # Sample a few non-neighbor nodes to estimate
                non_neighbors = [n for n in U.nodes() if n != node and n not in neighbors]
                sample = non_neighbors[:20]  # Limit for performance
                
                for other in sample:
                    try:
                        conn = nx.node_connectivity(U, node, other)
                        min_connectivity = min(min_connectivity, conn)
                    except Exception:
                        pass
                
                if min_connectivity == float('inf'):
                    min_connectivity = 0
                
                result[node] = {"node_connectivity": min_connectivity}
            
            return result
        except Exception as e:
            logger.warning(f"Node connectivity failed: {e}")
            return {node: {} for node in nodes}


class EdgeConnectivityAlgorithm(BaseMetricAlgorithm):
    """
    Compute local edge connectivity.
    
    For each node, computes the minimum number of edges that must be
    removed to disconnect the node from the rest of the graph.
    """
    
    name = "edge_connectivity"
    category = "robustness"
    description = "Local edge connectivity"
    cost = "high"
    graph_type = "undirected"
    # max_nodes is handled by registry/resolver - no internal limit
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            result = {}
            
            for node in nodes:
                if node not in U or U.degree(node) == 0:
                    result[node] = {"edge_connectivity": 0}
                    continue
                
                # For each node, compute min edge connectivity to any other node
                min_connectivity = float('inf')
                neighbors = set(U.neighbors(node))
                
                # Sample non-neighbor nodes
                non_neighbors = [n for n in U.nodes() if n != node and n not in neighbors]
                sample = non_neighbors[:20]
                
                for other in sample:
                    try:
                        conn = nx.edge_connectivity(U, node, other)
                        min_connectivity = min(min_connectivity, conn)
                    except Exception:
                        pass
                
                if min_connectivity == float('inf'):
                    min_connectivity = 0
                
                result[node] = {"edge_connectivity": min_connectivity}
            
            return result
        except Exception as e:
            logger.warning(f"Edge connectivity failed: {e}")
            return {node: {} for node in nodes}


class ResilienceScoreAlgorithm(BaseMetricAlgorithm):
    """
    Compute node resilience score.
    
    Measures the impact on network connectivity when a node is removed.
    Higher score means the node is more critical for network resilience.
    """
    
    name = "resilience_score"
    category = "robustness"
    description = "Node resilience impact score"
    cost = "high"
    graph_type = "undirected"
    # max_nodes is handled by registry/resolver - no internal limit
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            # Compute original largest component size
            if nx.is_connected(U):
                original_lcc_size = U.number_of_nodes()
            else:
                components = list(nx.connected_components(U))
                original_lcc_size = max(len(c) for c in components) if components else 0
            
            result = {}
            
            for node in nodes:
                if node not in U:
                    result[node] = {
                        "resilience_score": 0,
                        "lcc_reduction": 0,
                    }
                    continue
                
                # Create graph without this node
                nodes_without = [n for n in U.nodes() if n != node]
                subgraph = U.subgraph(nodes_without)
                
                # Compute new largest component size
                if subgraph.number_of_nodes() == 0:
                    new_lcc_size = 0
                elif nx.is_connected(subgraph):
                    new_lcc_size = subgraph.number_of_nodes()
                else:
                    components = list(nx.connected_components(subgraph))
                    new_lcc_size = max(len(c) for c in components) if components else 0
                
                # Resilience score is the reduction in LCC size
                lcc_reduction = original_lcc_size - new_lcc_size
                resilience_score = lcc_reduction / original_lcc_size if original_lcc_size > 0 else 0
                
                result[node] = {
                    "resilience_score": resilience_score,
                    "lcc_reduction": lcc_reduction,
                }
            
            return result
        except Exception as e:
            logger.warning(f"Resilience score failed: {e}")
            return {node: {} for node in nodes}