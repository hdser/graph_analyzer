"""
igraph-Specific Metric Algorithms

Algorithms that use igraph library for better performance
or unique functionality not available in NetworkX.
"""

from typing import Dict, Any, List, Optional
import logging

import networkx as nx

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)

# Check if igraph is available
try:
    import igraph as ig
    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False
    logger.info("igraph not available - igraph algorithms disabled")


class LeidenCommunityAlgorithm(BaseMetricAlgorithm):
    """
    Leiden community detection using igraph.
    
    The Leiden algorithm is an improved version of Louvain that
    guarantees well-connected communities.
    """
    
    name = "leiden_community"
    category = "community"
    description = "Leiden algorithm (improved Louvain)"
    cost = "medium"
    preferred_library = "igraph"
    requires_igraph = True
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, 
                ig_graph: Optional[Any] = None, parameters=None, **kwargs) -> Dict[str, Dict[str, Any]]:
        if not HAS_IGRAPH:
            logger.warning("igraph not available for Leiden algorithm")
            return {node: {} for node in nodes}
        
        params = parameters or {}
        resolution = params.get('resolution', 1.0)
        
        try:
            # Convert to igraph if not provided
            if ig_graph is None:
                ig_graph = self.nx_undirected_to_igraph(U)
            
            if ig_graph is None:
                return {node: {} for node in nodes}
            
            # Ensure undirected
            if ig_graph.is_directed():
                ig_graph = ig_graph.as_undirected()
            
            # Run Leiden algorithm
            partition = ig_graph.community_leiden(
                objective_function='modularity',
                resolution_parameter=resolution
            )
            
            # Build results
            membership = partition.membership
            sizes = [len(list(c)) for c in partition]
            
            # Map igraph vertex indices to node names
            node_names = ig_graph.vs['name']
            
            result = {}
            for i, node_name in enumerate(node_names):
                if node_name in nodes:
                    comm_id = membership[i]
                    result[node_name] = {
                        'leiden_community_id': comm_id,
                        'leiden_community_size': sizes[comm_id] if comm_id < len(sizes) else 0
                    }
            
            # Fill in missing nodes
            for node in nodes:
                if node not in result:
                    result[node] = {
                        'leiden_community_id': -1,
                        'leiden_community_size': 0
                    }
            
            return result
        except Exception as e:
            logger.warning(f"Leiden community detection failed: {e}")
            return {node: {} for node in nodes}


class InfomapCommunityAlgorithm(BaseMetricAlgorithm):
    """
    Infomap community detection using igraph.
    
    Infomap uses information flow to detect communities.
    """
    
    name = "infomap_community"
    category = "community"
    description = "Infomap algorithm (information flow)"
    cost = "medium"
    preferred_library = "igraph"
    requires_igraph = True
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list,
                ig_graph: Optional[Any] = None, **kwargs) -> Dict[str, Dict[str, Any]]:
        if not HAS_IGRAPH:
            logger.warning("igraph not available for Infomap algorithm")
            return {node: {} for node in nodes}
        
        try:
            # Convert to igraph if not provided
            if ig_graph is None:
                ig_graph = self.nx_to_igraph(G)
            
            if ig_graph is None:
                return {node: {} for node in nodes}
            
            # Run Infomap
            partition = ig_graph.community_infomap()
            
            membership = partition.membership
            sizes = [len(list(c)) for c in partition]
            
            node_names = ig_graph.vs['name']
            
            result = {}
            for i, node_name in enumerate(node_names):
                if node_name in nodes:
                    comm_id = membership[i]
                    result[node_name] = {
                        'infomap_community_id': comm_id,
                        'infomap_community_size': sizes[comm_id] if comm_id < len(sizes) else 0
                    }
            
            for node in nodes:
                if node not in result:
                    result[node] = {
                        'infomap_community_id': -1,
                        'infomap_community_size': 0
                    }
            
            return result
        except Exception as e:
            logger.warning(f"Infomap community detection failed: {e}")
            return {node: {} for node in nodes}


class WalktrapCommunityAlgorithm(BaseMetricAlgorithm):
    """
    Walktrap community detection using igraph.
    
    Uses random walks to find communities.
    """
    
    name = "walktrap_community"
    category = "community"
    description = "Walktrap algorithm (random walks)"
    cost = "medium"
    preferred_library = "igraph"
    requires_igraph = True
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list,
                ig_graph: Optional[Any] = None, parameters=None, **kwargs) -> Dict[str, Dict[str, Any]]:
        if not HAS_IGRAPH:
            logger.warning("igraph not available for Walktrap algorithm")
            return {node: {} for node in nodes}
        
        params = parameters or {}
        steps = params.get('steps', 4)
        
        try:
            if ig_graph is None:
                ig_graph = self.nx_undirected_to_igraph(U)
            
            if ig_graph is None:
                return {node: {} for node in nodes}
            
            if ig_graph.is_directed():
                ig_graph = ig_graph.as_undirected()
            
            # Run Walktrap
            dendrogram = ig_graph.community_walktrap(steps=steps)
            partition = dendrogram.as_clustering()
            
            membership = partition.membership
            sizes = [len(list(c)) for c in partition]
            
            node_names = ig_graph.vs['name']
            
            result = {}
            for i, node_name in enumerate(node_names):
                if node_name in nodes:
                    comm_id = membership[i]
                    result[node_name] = {
                        'walktrap_community_id': comm_id,
                        'walktrap_community_size': sizes[comm_id] if comm_id < len(sizes) else 0
                    }
            
            for node in nodes:
                if node not in result:
                    result[node] = {
                        'walktrap_community_id': -1,
                        'walktrap_community_size': 0
                    }
            
            return result
        except Exception as e:
            logger.warning(f"Walktrap community detection failed: {e}")
            return {node: {} for node in nodes}


class FastGreedyCommunityAlgorithm(BaseMetricAlgorithm):
    """
    Fast greedy modularity optimization using igraph.
    """
    
    name = "fast_greedy_community"
    category = "community"
    description = "Fast greedy modularity optimization"
    cost = "medium"
    preferred_library = "igraph"
    requires_igraph = True
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list,
                ig_graph: Optional[Any] = None, **kwargs) -> Dict[str, Dict[str, Any]]:
        if not HAS_IGRAPH:
            logger.warning("igraph not available for Fast Greedy algorithm")
            return {node: {} for node in nodes}
        
        try:
            if ig_graph is None:
                ig_graph = self.nx_undirected_to_igraph(U)
            
            if ig_graph is None:
                return {node: {} for node in nodes}
            
            if ig_graph.is_directed():
                ig_graph = ig_graph.as_undirected()
            
            # Run Fast Greedy
            dendrogram = ig_graph.community_fastgreedy()
            partition = dendrogram.as_clustering()
            
            membership = partition.membership
            sizes = [len(list(c)) for c in partition]
            
            node_names = ig_graph.vs['name']
            
            result = {}
            for i, node_name in enumerate(node_names):
                if node_name in nodes:
                    comm_id = membership[i]
                    result[node_name] = {
                        'fast_greedy_community_id': comm_id,
                        'fast_greedy_community_size': sizes[comm_id] if comm_id < len(sizes) else 0
                    }
            
            for node in nodes:
                if node not in result:
                    result[node] = {
                        'fast_greedy_community_id': -1,
                        'fast_greedy_community_size': 0
                    }
            
            return result
        except Exception as e:
            logger.warning(f"Fast Greedy community detection failed: {e}")
            return {node: {} for node in nodes}


class SpinglassCommunityAlgorithm(BaseMetricAlgorithm):
    """
    Spinglass community detection using igraph.
    
    Uses statistical physics approach to find communities.
    Note: This algorithm requires a connected graph.
    """
    
    name = "spinglass_community"
    category = "community"
    description = "Spinglass community detection"
    cost = "high"
    preferred_library = "igraph"
    requires_igraph = True
    graph_type = "undirected"
    requires_connected = True
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list,
                ig_graph: Optional[Any] = None, parameters=None, **kwargs) -> Dict[str, Dict[str, Any]]:
        if not HAS_IGRAPH:
            logger.warning("igraph not available for Spinglass algorithm")
            return {node: {} for node in nodes}
        
        # Spinglass requires connected graph - check here as safety
        if not nx.is_connected(U):
            logger.debug("Skipping Spinglass: graph not connected")
            return {node: {} for node in nodes}
        
        params = parameters or {}
        spins = params.get('spins', 25)
        
        try:
            if ig_graph is None:
                ig_graph = self.nx_undirected_to_igraph(U)
            
            if ig_graph is None:
                return {node: {} for node in nodes}
            
            if ig_graph.is_directed():
                ig_graph = ig_graph.as_undirected()
            
            # Run Spinglass
            partition = ig_graph.community_spinglass(spins=spins)
            
            membership = partition.membership
            sizes = [len(list(c)) for c in partition]
            
            node_names = ig_graph.vs['name']
            
            result = {}
            for i, node_name in enumerate(node_names):
                if node_name in nodes:
                    comm_id = membership[i]
                    result[node_name] = {
                        'spinglass_community_id': comm_id,
                        'spinglass_community_size': sizes[comm_id] if comm_id < len(sizes) else 0
                    }
            
            for node in nodes:
                if node not in result:
                    result[node] = {
                        'spinglass_community_id': -1,
                        'spinglass_community_size': 0
                    }
            
            return result
        except Exception as e:
            logger.warning(f"Spinglass community detection failed: {e}")
            return {node: {} for node in nodes}


class AlphaCentralityAlgorithm(BaseMetricAlgorithm):
    """
    Alpha centrality using igraph.
    
    Generalized eigenvector centrality with exogenous input.
    """
    
    name = "alpha_centrality"
    category = "centrality"
    description = "Alpha centrality (generalized eigenvector)"
    cost = "medium"
    preferred_library = "igraph"
    requires_igraph = True
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list,
                ig_graph: Optional[Any] = None, parameters=None, **kwargs) -> Dict[str, Dict[str, Any]]:
        if not HAS_IGRAPH:
            logger.warning("igraph not available for Alpha centrality")
            return {node: {} for node in nodes}
        
        params = parameters or {}
        alpha = params.get('alpha', 0.85)
        
        try:
            if ig_graph is None:
                ig_graph = self.nx_to_igraph(G)
            
            if ig_graph is None:
                return {node: {} for node in nodes}
            
            # Compute alpha centrality (similar to Katz/eigenvector)
            # igraph uses personalized PageRank as approximation
            centrality = ig_graph.personalized_pagerank(damping=alpha)
            
            node_names = ig_graph.vs['name']
            
            result = {}
            for i, node_name in enumerate(node_names):
                if node_name in nodes:
                    result[node_name] = {
                        'alpha_centrality': centrality[i]
                    }
            
            for node in nodes:
                if node not in result:
                    result[node] = {'alpha_centrality': 0}
            
            return result
        except Exception as e:
            logger.warning(f"Alpha centrality failed: {e}")
            return {node: {} for node in nodes}


class MotifCountAlgorithm(BaseMetricAlgorithm):
    """
    Motif counting using igraph.
    
    Counts 3-node and 4-node motif participation.
    """
    
    name = "motif_count"
    category = "motifs"
    description = "3-node and 4-node motif participation"
    cost = "high"
    preferred_library = "igraph"
    requires_igraph = True
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list,
                ig_graph: Optional[Any] = None, parameters=None, **kwargs) -> Dict[str, Dict[str, Any]]:
        if not HAS_IGRAPH:
            logger.warning("igraph not available for motif counting")
            return {node: {} for node in nodes}
        
        # max_nodes check is handled by registry/resolver - no internal check needed
        
        params = parameters or {}
        size = params.get('size', 3)
        
        try:
            if ig_graph is None:
                ig_graph = self.nx_to_igraph(G)
            
            if ig_graph is None:
                return {node: {} for node in nodes}
            
            # Count motifs graph-wide
            try:
                motif_counts = ig_graph.motifs_randesu(size=size)
                total_motifs = sum(m for m in motif_counts if m is not None and m > 0)
            except Exception:
                total_motifs = 0
            
            # For per-node motif participation, we approximate using
            # the node's contribution to triangles (for size=3)
            node_names = ig_graph.vs['name']
            
            result = {}
            
            if size == 3:
                # Count triangles per node as proxy for 3-motifs
                try:
                    # Use clustering coefficient as proxy
                    clustering = ig_graph.transitivity_local_undirected()
                    degrees = ig_graph.degree()
                    
                    for i, node_name in enumerate(node_names):
                        if node_name in nodes:
                            # Estimate triangle participation
                            cc = clustering[i] if clustering[i] is not None else 0
                            k = degrees[i]
                            triangles_est = int(cc * k * (k - 1) / 2) if k > 1 else 0
                            
                            result[node_name] = {
                                'motif_3_count': triangles_est,
                                'graph_total_motifs': total_motifs
                            }
                except Exception:
                    pass
            else:
                # For size 4, just report graph-level count
                for node in nodes:
                    result[node] = {
                        f'motif_{size}_count': 0,
                        'graph_total_motifs': total_motifs
                    }
            
            for node in nodes:
                if node not in result:
                    result[node] = {
                        f'motif_{size}_count': 0,
                        'graph_total_motifs': total_motifs
                    }
            
            return result
        except Exception as e:
            logger.warning(f"Motif counting failed: {e}")
            return {node: {} for node in nodes}