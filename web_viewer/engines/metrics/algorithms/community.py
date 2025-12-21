"""
Community Metric Algorithms

Community and core structure detection.
"""

from typing import Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

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
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            communities = nx_comm.louvain_communities(U, threshold=1e-10, seed=42)
            
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
    max_nodes = 500
    
    def __init__(self, n_jobs: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.n_jobs = n_jobs
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, n_jobs: int = None, **kwargs) -> Dict[str, Dict[str, Any]]:
        n_nodes = G.number_of_nodes()
        if n_nodes > self.max_nodes:
            return {node: {} for node in nodes}
        
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