"""
Community Metric Algorithms

Community and core structure detection.
Includes Louvain, label propagation, and community quality metrics.
"""

from typing import Dict, Any, Optional, List
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from collections import defaultdict

import networkx as nx
import networkx.algorithms.community as nx_comm

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


class LouvainCommunityAlgorithm(BaseMetricAlgorithm):
    """Compute Louvain community detection."""
    
    name = "louvain_community"
    category = "community"
    description = "Louvain community detection"
    cost = "medium"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, parameters=None, **kwargs) -> Dict[str, Dict[str, Any]]:
        # Get parameters with defaults
        params = parameters or {}
        resolution = params.get('resolution', 1.0)
        seed = params.get('seed', 42)
        
        try:
            communities = nx_comm.louvain_communities(U, resolution=resolution, threshold=1e-10, seed=seed)
            
            comm_map = {}
            comm_sizes = {}
            for idx, comm in enumerate(communities):
                comm_sizes[idx] = len(comm)
                for node in comm:
                    comm_map[node] = idx
            
            logger.debug(f"Found {len(communities)} communities")
            
            result = {}
            for node in nodes:
                cid = comm_map.get(node, -1)
                result[node] = {
                    "community_id": cid,
                    "community_size": comm_sizes.get(cid, 0),
                }
            return result
        except Exception as e:
            logger.warning(f"Louvain community detection failed: {e}")
            return {node: {} for node in nodes}


class CoreNumberAlgorithm(BaseMetricAlgorithm):
    """Compute k-core membership number."""
    
    name = "core_number"
    category = "community"
    description = "K-core membership number"
    cost = "low"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            core = nx.core_number(U)
            
            result = {}
            for node in nodes:
                result[node] = {"core_number": core.get(node, 0)}
            return result
        except Exception as e:
            logger.warning(f"Core number failed: {e}")
            return {node: {} for node in nodes}


class OnionLayerAlgorithm(BaseMetricAlgorithm):
    """Compute onion decomposition layer."""
    
    name = "onion_layer"
    category = "community"
    description = "Onion decomposition layer"
    cost = "medium"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            onion = nx.onion_layers(U)
            
            result = {}
            for node in nodes:
                result[node] = {"onion_layer": onion.get(node, 0)}
            return result
        except Exception as e:
            logger.warning(f"Onion layers failed: {e}")
            return {node: {} for node in nodes}


def _compute_node_lrc(G, node):
    """Compute local reaching centrality for a single node."""
    try:
        return node, nx.local_reaching_centrality(G, node)
    except Exception:
        return node, 0


class LocalReachingCentralityAlgorithm(BaseMetricAlgorithm):
    """Compute local reaching centrality with parallel processing."""
    
    name = "local_reaching_centrality"
    category = "community"
    description = "Proportion of network reachable from node"
    cost = "high"
    # max_nodes is handled by registry/resolver - no internal limit
    
    def __init__(self, n_jobs: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.n_jobs = n_jobs
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, n_jobs: int = None, **kwargs) -> Dict[str, Dict[str, Any]]:
        n_jobs = n_jobs or self.n_jobs
        
        try:
            result = {}
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                func = partial(_compute_node_lrc, G)
                futures = {executor.submit(func, node): node for node in nodes}
                
                for future in as_completed(futures):
                    node, lrc = future.result()
                    result[node] = {"local_reaching_centrality": lrc}
            
            return result
        except Exception as e:
            logger.warning(f"Local reaching centrality failed: {e}")
            return {node: {} for node in nodes}


# =============================================================================
# NEW COMMUNITY ALGORITHMS
# =============================================================================

class LabelPropagationAlgorithm(BaseMetricAlgorithm):
    """
    Label propagation community detection.
    
    Fast semi-synchronous label propagation algorithm.
    """
    
    name = "label_propagation"
    category = "community"
    description = "Label propagation community detection"
    cost = "low"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            communities = list(nx_comm.label_propagation_communities(U))
            
            comm_map = {}
            comm_sizes = {}
            for idx, comm in enumerate(communities):
                comm_sizes[idx] = len(comm)
                for node in comm:
                    comm_map[node] = idx
            
            logger.debug(f"Label propagation found {len(communities)} communities")
            
            result = {}
            for node in nodes:
                cid = comm_map.get(node, -1)
                result[node] = {
                    "label_prop_community_id": cid,
                    "label_prop_community_size": comm_sizes.get(cid, 0),
                }
            return result
        except Exception as e:
            logger.warning(f"Label propagation failed: {e}")
            return {node: {} for node in nodes}


class AsyncLabelPropagationAlgorithm(BaseMetricAlgorithm):
    """
    Asynchronous label propagation community detection.
    
    Asynchronous version of label propagation.
    """
    
    name = "async_label_propagation"
    category = "community"
    description = "Asynchronous label propagation community detection"
    cost = "low"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, parameters=None, **kwargs) -> Dict[str, Dict[str, Any]]:
        params = parameters or {}
        seed = params.get('seed', 42)
        
        try:
            communities = list(nx_comm.asyn_lpa_communities(U, seed=seed))
            
            comm_map = {}
            comm_sizes = {}
            for idx, comm in enumerate(communities):
                comm_sizes[idx] = len(comm)
                for node in comm:
                    comm_map[node] = idx
            
            result = {}
            for node in nodes:
                cid = comm_map.get(node, -1)
                result[node] = {
                    "async_lpa_community_id": cid,
                    "async_lpa_community_size": comm_sizes.get(cid, 0),
                }
            return result
        except Exception as e:
            logger.warning(f"Async label propagation failed: {e}")
            return {node: {} for node in nodes}


class GreedyModularityCommunityAlgorithm(BaseMetricAlgorithm):
    """
    Greedy modularity optimization community detection.
    
    Uses Clauset-Newman-Moore greedy modularity maximization.
    """
    
    name = "greedy_modularity"
    category = "community"
    description = "Greedy modularity community detection"
    cost = "medium"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, parameters=None, **kwargs) -> Dict[str, Dict[str, Any]]:
        params = parameters or {}
        resolution = params.get('resolution', 1.0)
        
        try:
            communities = list(nx_comm.greedy_modularity_communities(U, resolution=resolution))
            
            comm_map = {}
            comm_sizes = {}
            for idx, comm in enumerate(communities):
                comm_sizes[idx] = len(comm)
                for node in comm:
                    comm_map[node] = idx
            
            # Calculate modularity
            modularity = nx_comm.modularity(U, communities)
            
            result = {}
            for node in nodes:
                cid = comm_map.get(node, -1)
                result[node] = {
                    "greedy_mod_community_id": cid,
                    "greedy_mod_community_size": comm_sizes.get(cid, 0),
                    "graph_modularity": modularity,
                }
            return result
        except Exception as e:
            logger.warning(f"Greedy modularity community detection failed: {e}")
            return {node: {} for node in nodes}


class ParticipationCoefficientAlgorithm(BaseMetricAlgorithm):
    """
    Compute participation coefficient.
    
    Measures how evenly distributed a node's connections are across communities.
    High value = connections spread across many communities.
    """
    
    name = "participation_coefficient"
    category = "community"
    description = "Participation coefficient (inter-community connectivity)"
    cost = "medium"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, computed_metrics=None, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            # First, detect communities using Louvain
            communities = list(nx_comm.louvain_communities(U, seed=42))
            
            # Build node -> community mapping
            node_community = {}
            for idx, comm in enumerate(communities):
                for node in comm:
                    node_community[node] = idx
            
            result = {}
            for node in nodes:
                if node not in U or U.degree(node) == 0:
                    result[node] = {"participation_coefficient": 0}
                    continue
                
                k_i = U.degree(node)
                
                # Count connections to each community
                comm_connections = defaultdict(int)
                for neighbor in U.neighbors(node):
                    comm = node_community.get(neighbor, -1)
                    comm_connections[comm] += 1
                
                # Compute participation coefficient
                # P = 1 - sum((k_is / k_i)^2) for all communities s
                participation = 0
                for comm, k_is in comm_connections.items():
                    participation += (k_is / k_i) ** 2
                
                participation = 1 - participation
                result[node] = {"participation_coefficient": participation}
            
            return result
        except Exception as e:
            logger.warning(f"Participation coefficient failed: {e}")
            return {node: {} for node in nodes}


class WithinModuleDegreeAlgorithm(BaseMetricAlgorithm):
    """
    Compute within-module degree z-score.
    
    Measures how well-connected a node is within its community
    relative to other nodes in the same community.
    """
    
    name = "within_module_degree"
    category = "community"
    description = "Within-module degree z-score"
    cost = "medium"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            # Detect communities
            communities = list(nx_comm.louvain_communities(U, seed=42))
            
            # Build node -> community mapping
            node_community = {}
            for idx, comm in enumerate(communities):
                for node in comm:
                    node_community[node] = idx
            
            # Compute within-module degree for each node
            within_degree = {}
            for node in U.nodes():
                my_comm = node_community.get(node)
                count = 0
                for neighbor in U.neighbors(node):
                    if node_community.get(neighbor) == my_comm:
                        count += 1
                within_degree[node] = count
            
            # Compute z-score for each community
            comm_stats = defaultdict(list)
            for node, degree in within_degree.items():
                comm = node_community.get(node)
                comm_stats[comm].append(degree)
            
            # Compute mean and std for each community
            comm_mean = {}
            comm_std = {}
            import numpy as np
            for comm, degrees in comm_stats.items():
                comm_mean[comm] = np.mean(degrees)
                comm_std[comm] = np.std(degrees) if len(degrees) > 1 else 1.0
            
            result = {}
            for node in nodes:
                if node not in U:
                    result[node] = {"within_module_degree_z": 0}
                    continue
                
                comm = node_community.get(node)
                mean = comm_mean.get(comm, 0)
                std = comm_std.get(comm, 1)
                
                if std > 0:
                    z_score = (within_degree.get(node, 0) - mean) / std
                else:
                    z_score = 0
                
                result[node] = {"within_module_degree_z": z_score}
            
            return result
        except Exception as e:
            logger.warning(f"Within-module degree failed: {e}")
            return {node: {} for node in nodes}