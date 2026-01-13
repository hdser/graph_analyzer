"""
Spectral Metric Algorithms

Spectral graph theory metrics based on graph Laplacian and adjacency matrices.
"""

from typing import Dict, Any, List
import logging

import networkx as nx
import numpy as np

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


class FiedlerVectorAlgorithm(BaseMetricAlgorithm):
    """
    Compute Fiedler vector component.
    
    The Fiedler vector is the eigenvector corresponding to the second
    smallest eigenvalue of the Laplacian matrix. It's useful for
    spectral partitioning.
    """
    
    name = "fiedler_vector"
    category = "spectral"
    description = "Fiedler vector component (spectral partitioning)"
    cost = "high"
    graph_type = "undirected"
    requires_connected = True
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        # max_nodes check is handled by registry/resolver - no internal check needed
        
        if not nx.is_connected(U):
            # Use largest connected component
            largest_cc = max(nx.connected_components(U), key=len)
            U = U.subgraph(largest_cc).copy()
        
        try:
            # Compute Fiedler vector using algebraic connectivity
            fiedler = nx.fiedler_vector(U, normalized=True)
            
            # Map to node IDs
            nodes_list = list(U.nodes())
            fiedler_dict = {nodes_list[i]: fiedler[i] for i in range(len(nodes_list))}
            
            result = {}
            for node in nodes:
                if node in fiedler_dict:
                    result[node] = {
                        "fiedler_component": fiedler_dict[node],
                        "fiedler_partition": 1 if fiedler_dict[node] >= 0 else 0,
                    }
                else:
                    result[node] = {
                        "fiedler_component": 0,
                        "fiedler_partition": -1,
                    }
            
            return result
        except Exception as e:
            logger.warning(f"Fiedler vector failed: {e}")
            return {node: {} for node in nodes}


class SpectralCentralityAlgorithm(BaseMetricAlgorithm):
    """
    Compute spectral centrality metrics.
    
    Uses eigenvalue decomposition of the adjacency matrix to compute
    centrality measures.
    """
    
    name = "spectral_centrality"
    category = "spectral"
    description = "Spectral centrality from adjacency eigenvalues"
    cost = "high"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        # max_nodes check is handled by registry/resolver - no internal check needed
        
        try:
            # Get adjacency matrix
            A = nx.adjacency_matrix(U).toarray().astype(float)
            
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(A)
            
            # Sort by eigenvalue magnitude (descending)
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Spectral centrality based on top eigenvector
            top_eigenvector = np.abs(eigenvectors[:, 0])
            
            # Normalize
            if np.max(top_eigenvector) > 0:
                top_eigenvector = top_eigenvector / np.max(top_eigenvector)
            
            # Map to nodes
            nodes_list = list(U.nodes())
            
            result = {}
            for node in nodes:
                if node in U:
                    node_idx = nodes_list.index(node)
                    result[node] = {
                        "spectral_centrality": float(top_eigenvector[node_idx]),
                    }
                else:
                    result[node] = {"spectral_centrality": 0}
            
            return result
        except Exception as e:
            logger.warning(f"Spectral centrality failed: {e}")
            return {node: {} for node in nodes}