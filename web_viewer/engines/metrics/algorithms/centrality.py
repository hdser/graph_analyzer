"""
Centrality Metric Algorithms

Node importance and influence measures.
Includes standard NetworkX centralities plus custom implementations.
"""

from typing import Dict, Any, Optional, List
import logging
import random
from collections import defaultdict

import networkx as nx
import numpy as np
import scipy.sparse as sp

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


class DegreeCentralityAlgorithm(BaseMetricAlgorithm):
    """Compute normalized degree centrality."""
    
    name = "degree_centrality"
    category = "centrality"
    description = "Normalized degree centrality (in, out, undirected)"
    cost = "low"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            in_deg = nx.in_degree_centrality(G)
            out_deg = nx.out_degree_centrality(G)
            deg_undirected = nx.degree_centrality(U)
            
            result = {}
            for node in nodes:
                result[node] = {
                    "in_degree_centrality": in_deg.get(node, 0),
                    "out_degree_centrality": out_deg.get(node, 0),
                    "degree_centrality_undirected": deg_undirected.get(node, 0),
                }
            return result
        except Exception as e:
            logger.warning(f"Degree centrality failed: {e}")
            return {node: {} for node in nodes}


class ClosenessCentralityAlgorithm(BaseMetricAlgorithm):
    """Compute closeness centrality."""
    
    name = "closeness_centrality"
    category = "centrality"
    description = "Closeness centrality (directed and undirected)"
    cost = "medium"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            closeness = nx.closeness_centrality(G)
            closeness_in = nx.closeness_centrality(G.reverse())
            closeness_undirected = nx.closeness_centrality(U)
            
            result = {}
            for node in nodes:
                result[node] = {
                    "closeness_centrality": closeness.get(node, 0),
                    "closeness_centrality_in": closeness_in.get(node, 0),
                    "closeness_centrality_undirected": closeness_undirected.get(node, 0),
                }
            return result
        except Exception as e:
            logger.warning(f"Closeness centrality failed: {e}")
            return {node: {} for node in nodes}


class BetweennessCentralityAlgorithm(BaseMetricAlgorithm):
    """Compute betweenness centrality."""
    
    name = "betweenness_centrality"
    category = "centrality"
    description = "Betweenness centrality (directed and undirected)"
    cost = "high"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, parameters=None, **kwargs) -> Dict[str, Dict[str, Any]]:
        # Get parameters with defaults
        params = parameters or {}
        normalized = params.get('normalized', True)
        endpoints = params.get('endpoints', False)
        
        try:
            betweenness = nx.betweenness_centrality(G, normalized=normalized, endpoints=endpoints)
            betweenness_undirected = nx.betweenness_centrality(U, normalized=normalized, endpoints=endpoints)
            
            result = {}
            for node in nodes:
                result[node] = {
                    "betweenness_centrality": betweenness.get(node, 0),
                    "betweenness_centrality_undirected": betweenness_undirected.get(node, 0),
                }
            return result
        except Exception as e:
            logger.warning(f"Betweenness centrality failed: {e}")
            return {node: {} for node in nodes}


class EigenvectorCentralityAlgorithm(BaseMetricAlgorithm):
    """Compute eigenvector centrality."""
    
    name = "eigenvector_centrality"
    category = "centrality"
    description = "Eigenvector centrality (directed and undirected)"
    cost = "medium"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, parameters=None, **kwargs) -> Dict[str, Dict[str, Any]]:
        # Get parameters with defaults
        params = parameters or {}
        max_iter = params.get('max_iter', 1000)
        tol = params.get('tol', 1.0e-6)
        
        result = {}
        
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=max_iter, tol=tol)
        except Exception as e:
            logger.warning(f"Directed eigenvector centrality failed: {e}")
            eigenvector = {}
        
        try:
            eigenvector_undirected = nx.eigenvector_centrality(U, max_iter=max_iter, tol=tol)
        except Exception as e:
            logger.warning(f"Undirected eigenvector centrality failed: {e}")
            eigenvector_undirected = {}
        
        for node in nodes:
            result[node] = {
                "eigenvector_centrality": eigenvector.get(node, 0),
                "eigenvector_centrality_undirected": eigenvector_undirected.get(node, 0),
            }
        return result


class KatzCentralityAlgorithm(BaseMetricAlgorithm):
    """Compute Katz centrality with safe alpha."""
    
    name = "katz_centrality"
    category = "centrality"
    description = "Katz centrality with safe alpha computation"
    cost = "medium"
    
    def _safe_alpha(self, G: nx.DiGraph) -> float:
        """Compute safe alpha for Katz centrality."""
        try:
            adj = nx.to_scipy_sparse_array(G, format='csr')
            eigenvals = sp.linalg.eigs(adj.astype(float), k=1, which='LM', return_eigenvectors=False)
            largest = abs(eigenvals[0])
            alpha = 0.9 / largest if largest > 0 else 0.01
            return alpha
        except Exception:
            return 0.01
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, parameters=None, **kwargs) -> Dict[str, Dict[str, Any]]:
        # Get parameters with defaults
        params = parameters or {}
        alpha = params.get('alpha', None)  # None means auto-calculate
        beta = params.get('beta', 1.0)
        max_iter = params.get('max_iter', 1000)
        tol = params.get('tol', 1.0e-6)
        
        result = {}
        
        try:
            if alpha is None:
                alpha = self._safe_alpha(G)
            katz = nx.katz_centrality(G, alpha=alpha, beta=beta, max_iter=max_iter, tol=tol)
        except Exception as e:
            logger.warning(f"Directed Katz centrality failed: {e}")
            katz = {}
        
        try:
            if params.get('alpha', None) is None:  # Use same auto-calculated alpha
                alpha_u = self._safe_alpha(U)
            else:
                alpha_u = alpha
            katz_undirected = nx.katz_centrality(U, alpha=alpha_u, beta=beta, max_iter=max_iter, tol=tol)
        except Exception as e:
            logger.warning(f"Undirected Katz centrality failed: {e}")
            katz_undirected = {}
        
        for node in nodes:
            result[node] = {
                "katz_centrality": katz.get(node, 0),
                "katz_centrality_undirected": katz_undirected.get(node, 0),
            }
        return result


class PageRankAlgorithm(BaseMetricAlgorithm):
    """Compute PageRank."""
    
    name = "pagerank"
    category = "centrality"
    description = "Google PageRank score"
    cost = "low"
    
    def __init__(self, alpha: float = 0.85, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, parameters=None, **kwargs) -> Dict[str, Dict[str, Any]]:
        # Get parameters with defaults
        params = parameters or {}
        alpha = params.get('alpha', self.alpha)
        max_iter = params.get('max_iter', 100)
        tol = params.get('tol', 1.0e-6)
        
        try:
            pagerank = nx.pagerank(G, alpha=alpha, max_iter=max_iter, tol=tol)
            pagerank_undirected = nx.pagerank(U, alpha=alpha, max_iter=max_iter, tol=tol)
            
            result = {}
            for node in nodes:
                result[node] = {
                    "pagerank": pagerank.get(node, 0),
                    "pagerank_undirected": pagerank_undirected.get(node, 0),
                }
            return result
        except Exception as e:
            logger.warning(f"PageRank failed: {e}")
            return {node: {} for node in nodes}


class HITSAlgorithm(BaseMetricAlgorithm):
    """Compute HITS hub and authority scores."""
    
    name = "hits"
    category = "centrality"
    description = "HITS hub and authority scores"
    cost = "medium"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            hubs, authorities = nx.hits(G, max_iter=100, tol=1e-6)
            
            result = {}
            for node in nodes:
                result[node] = {
                    "hub_score": hubs.get(node, 0),
                    "authority_score": authorities.get(node, 0),
                }
            return result
        except Exception as e:
            logger.warning(f"HITS failed: {e}")
            return {node: {} for node in nodes}


class HarmonicCentralityAlgorithm(BaseMetricAlgorithm):
    """Compute harmonic centrality."""
    
    name = "harmonic_centrality"
    category = "centrality"
    description = "Harmonic centrality (sum of inverse distances)"
    cost = "medium"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            harmonic = nx.harmonic_centrality(G)
            harmonic_undirected = nx.harmonic_centrality(U)
            
            result = {}
            for node in nodes:
                result[node] = {
                    "harmonic_centrality": harmonic.get(node, 0),
                    "harmonic_centrality_undirected": harmonic_undirected.get(node, 0),
                }
            return result
        except Exception as e:
            logger.warning(f"Harmonic centrality failed: {e}")
            return {node: {} for node in nodes}


class LoadCentralityAlgorithm(BaseMetricAlgorithm):
    """Compute load centrality."""
    
    name = "load_centrality"
    category = "centrality"
    description = "Load centrality (traffic flow through node)"
    cost = "high"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            load = nx.load_centrality(G)
            load_undirected = nx.load_centrality(U)
            
            result = {}
            for node in nodes:
                result[node] = {
                    "load_centrality": load.get(node, 0),
                    "load_centrality_undirected": load_undirected.get(node, 0),
                }
            return result
        except Exception as e:
            logger.warning(f"Load centrality failed: {e}")
            return {node: {} for node in nodes}


class SubgraphCentralityAlgorithm(BaseMetricAlgorithm):
    """Compute subgraph centrality."""
    
    name = "subgraph_centrality"
    category = "centrality"
    description = "Subgraph centrality based on closed walks"
    cost = "medium"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            subgraph = nx.subgraph_centrality(U)
            
            result = {}
            for node in nodes:
                result[node] = {"subgraph_centrality": subgraph.get(node, 0)}
            return result
        except Exception as e:
            logger.warning(f"Subgraph centrality failed: {e}")
            return {node: {} for node in nodes}


class SecondOrderCentralityAlgorithm(BaseMetricAlgorithm):
    """Compute second order centrality."""
    
    name = "second_order_centrality"
    category = "centrality"
    description = "Second order centrality (random walk variance)"
    cost = "high"
    graph_type = "undirected"
    # max_nodes is handled by registry/resolver - no internal limit
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            second_order = nx.second_order_centrality(U)
            
            result = {}
            for node in nodes:
                result[node] = {"second_order_centrality": second_order.get(node, 0)}
            return result
        except Exception as e:
            logger.warning(f"Second order centrality failed: {e}")
            return {node: {} for node in nodes}


class PercolationCentralityAlgorithm(BaseMetricAlgorithm):
    """Compute percolation centrality."""
    
    name = "percolation_centrality"
    category = "centrality"
    description = "Percolation centrality with random states"
    cost = "medium"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            random.seed(42)
            states = {node: random.random() for node in G.nodes()}
            percolation = nx.percolation_centrality(G, states=states)
            
            result = {}
            for node in nodes:
                result[node] = {"percolation_centrality": percolation.get(node, 0)}
            return result
        except Exception as e:
            logger.warning(f"Percolation centrality failed: {e}")
            return {node: {} for node in nodes}


class TrophicLevelAlgorithm(BaseMetricAlgorithm):
    """Compute trophic levels."""
    
    name = "trophic_level"
    category = "centrality"
    description = "Trophic level in directed graph hierarchy"
    cost = "medium"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            trophic = nx.trophic_levels(G)
            
            result = {}
            for node in nodes:
                result[node] = {"trophic_level": trophic.get(node, 0)}
            return result
        except Exception as e:
            logger.warning(f"Trophic levels failed: {e}")
            return {node: {} for node in nodes}


class CurrentFlowCentralityAlgorithm(BaseMetricAlgorithm):
    """Compute current flow betweenness and closeness centrality."""
    
    name = "current_flow_centrality"
    category = "centrality"
    description = "Current flow betweenness and closeness centrality"
    cost = "very_high"
    graph_type = "undirected"
    requires_connected = True
    # max_nodes is handled by registry/resolver - no internal limit
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        if not nx.is_connected(U):
            logger.debug("Skipping current flow: graph not connected")
            return {node: {} for node in nodes}
        
        try:
            cf_between = nx.current_flow_betweenness_centrality(U)
            cf_close = nx.current_flow_closeness_centrality(U)
            
            result = {}
            for node in nodes:
                result[node] = {
                    "current_flow_betweenness": cf_between.get(node, 0),
                    "current_flow_closeness": cf_close.get(node, 0),
                }
            return result
        except Exception as e:
            logger.warning(f"Current flow centrality failed: {e}")
            return {node: {} for node in nodes}


class InformationCentralityAlgorithm(BaseMetricAlgorithm):
    """Compute information centrality."""
    
    name = "information_centrality"
    category = "centrality"
    description = "Information centrality based on information flow"
    cost = "very_high"
    graph_type = "undirected"
    requires_connected = True
    # max_nodes is handled by registry/resolver - no internal limit
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        if not nx.is_connected(U):
            return {node: {} for node in nodes}
        
        try:
            info = nx.information_centrality(U)
            
            result = {}
            for node in nodes:
                result[node] = {"information_centrality": info.get(node, 0)}
            return result
        except Exception as e:
            logger.warning(f"Information centrality failed: {e}")
            return {node: {} for node in nodes}


class CommunicabilityBetweennessAlgorithm(BaseMetricAlgorithm):
    """Compute communicability betweenness centrality."""
    
    name = "communicability_betweenness"
    category = "centrality"
    description = "Communicability betweenness centrality"
    cost = "very_high"
    graph_type = "undirected"
    # max_nodes is handled by registry/resolver - no internal limit
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            comm_between = nx.communicability_betweenness_centrality(U)
            
            result = {}
            for node in nodes:
                result[node] = {"communicability_betweenness": comm_between.get(node, 0)}
            return result
        except Exception as e:
            logger.warning(f"Communicability betweenness failed: {e}")
            return {node: {} for node in nodes}


class VoteRankAlgorithm(BaseMetricAlgorithm):
    """Compute VoteRank influence measure."""
    
    name = "voterank"
    category = "centrality"
    description = "VoteRank influence measure"
    cost = "medium"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            voterank_list = nx.voterank(U)
            voterank_dict = {node: len(voterank_list) - i for i, node in enumerate(voterank_list)}
            
            result = {}
            for node in nodes:
                result[node] = {"voterank": voterank_dict.get(node, 0)}
            return result
        except Exception as e:
            logger.warning(f"VoteRank failed: {e}")
            return {node: {} for node in nodes}


class EdgeBetweennessSumAlgorithm(BaseMetricAlgorithm):
    """Compute sum of edge betweenness for incident edges."""
    
    name = "edge_betweenness_sum"
    category = "centrality"
    description = "Sum of edge betweenness for incident edges"
    cost = "high"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            edge_between = nx.edge_betweenness_centrality(G)
            
            result = {}
            for node in nodes:
                edges = [(u, v) for u, v in edge_between.keys() if u == node or v == node]
                result[node] = {"edge_betweenness_sum": sum(edge_between[e] for e in edges)}
            return result
        except Exception as e:
            logger.warning(f"Edge betweenness failed: {e}")
            return {node: {} for node in nodes}


# =============================================================================
# NEW CENTRALITY ALGORITHMS
# =============================================================================

class LaplacianCentralityAlgorithm(BaseMetricAlgorithm):
    """
    Compute Laplacian centrality based on Laplacian matrix.
    
    Laplacian centrality measures the drop in Laplacian energy when a node is removed.
    """
    
    name = "laplacian_centrality"
    category = "centrality"
    description = "Laplacian centrality based on Laplacian energy"
    cost = "high"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            # Compute Laplacian centrality for each node
            # Based on the drop in Laplacian energy when node is removed
            n = U.number_of_nodes()
            if n == 0:
                return {node: {} for node in nodes}
            
            # Get original Laplacian energy
            L = nx.laplacian_matrix(U).toarray()
            eigenvalues = np.linalg.eigvalsh(L)
            original_energy = np.sum(eigenvalues ** 2)
            
            result = {}
            nodes_list = list(U.nodes())
            
            for node in nodes:
                if node not in U:
                    result[node] = {"laplacian_centrality": 0}
                    continue
                
                # Create subgraph without this node
                subgraph = U.subgraph([n for n in nodes_list if n != node])
                if subgraph.number_of_nodes() == 0:
                    result[node] = {"laplacian_centrality": original_energy}
                    continue
                
                # Compute energy of subgraph
                L_sub = nx.laplacian_matrix(subgraph).toarray()
                sub_eigenvalues = np.linalg.eigvalsh(L_sub)
                sub_energy = np.sum(sub_eigenvalues ** 2)
                
                # Laplacian centrality is the drop in energy
                centrality = (original_energy - sub_energy) / original_energy if original_energy > 0 else 0
                result[node] = {"laplacian_centrality": max(0, centrality)}
            
            return result
        except Exception as e:
            logger.warning(f"Laplacian centrality failed: {e}")
            return {node: {} for node in nodes}


class LeverageCentralityAlgorithm(BaseMetricAlgorithm):
    """
    Compute leverage centrality.
    
    Measures how much a node's degree exceeds its neighbors' degrees.
    """
    
    name = "leverage_centrality"
    category = "centrality"
    description = "Leverage centrality based on neighbor degrees"
    cost = "medium"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            result = {}
            
            for node in nodes:
                if node not in U or U.degree(node) == 0:
                    result[node] = {"leverage_centrality": 0}
                    continue
                
                k_i = U.degree(node)
                neighbors = list(U.neighbors(node))
                
                if len(neighbors) == 0:
                    result[node] = {"leverage_centrality": 0}
                    continue
                
                # Compute leverage centrality
                leverage_sum = 0
                for neighbor in neighbors:
                    k_j = U.degree(neighbor)
                    if k_i + k_j > 0:
                        leverage_sum += (k_i - k_j) / (k_i + k_j)
                
                leverage = leverage_sum / k_i
                result[node] = {"leverage_centrality": leverage}
            
            return result
        except Exception as e:
            logger.warning(f"Leverage centrality failed: {e}")
            return {node: {} for node in nodes}


class SemiLocalCentralityAlgorithm(BaseMetricAlgorithm):
    """
    Compute semi-local centrality.
    
    Captures local influence by considering 2-hop neighborhood structure.
    """
    
    name = "semi_local_centrality"
    category = "centrality"
    description = "Semi-local centrality based on 2-hop neighborhood"
    cost = "low"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            result = {}
            
            for node in nodes:
                if node not in U:
                    result[node] = {"semi_local_centrality": 0}
                    continue
                
                # Get neighbors
                neighbors = list(U.neighbors(node))
                
                # Compute Q(v) = sum of N(u) for all neighbors u
                # where N(u) is the number of nearest and next-nearest neighbors of u
                q_score = 0
                for neighbor in neighbors:
                    # Count neighbors of neighbor (2nd hop from node)
                    second_hop = set(U.neighbors(neighbor))
                    q_score += len(second_hop)
                
                result[node] = {"semi_local_centrality": q_score}
            
            return result
        except Exception as e:
            logger.warning(f"Semi-local centrality failed: {e}")
            return {node: {} for node in nodes}


class DecayCentralityAlgorithm(BaseMetricAlgorithm):
    """
    Compute decay centrality.
    
    Distance-weighted influence measure where influence decays with distance.
    """
    
    name = "decay_centrality"
    category = "centrality"
    description = "Decay centrality with distance-weighted influence"
    cost = "medium"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, parameters=None, **kwargs) -> Dict[str, Dict[str, Any]]:
        params = parameters or {}
        delta = params.get('delta', 0.5)  # Decay factor
        max_distance = params.get('max_distance', 5)  # Maximum distance to consider
        
        try:
            result = {}
            
            for node in nodes:
                if node not in G:
                    result[node] = {"decay_centrality": 0}
                    continue
                
                # Get shortest path lengths from this node
                try:
                    lengths = nx.single_source_shortest_path_length(G, node, cutoff=max_distance)
                except Exception:
                    lengths = {}
                
                # Compute decay centrality
                centrality = 0
                for target, distance in lengths.items():
                    if target != node and distance > 0:
                        centrality += delta ** distance
                
                result[node] = {"decay_centrality": centrality}
            
            return result
        except Exception as e:
            logger.warning(f"Decay centrality failed: {e}")
            return {node: {} for node in nodes}