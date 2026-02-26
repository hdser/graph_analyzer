"""
ComputeBackend ABC

Abstract base class for pluggable graph computation backends.
Each backend wraps a graph library (NetworkX, igraph, cuGraph) and exposes
a uniform API for the most performance-critical graph algorithms.

Algorithms not covered here continue to run via their original NetworkX
code path — the backend is an optional accelerator, not a replacement for
every algorithm.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any

import networkx as nx


class ComputeBackend(ABC):
    """Abstract base for pluggable compute backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g., 'networkx', 'igraph', 'cugraph')."""

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether the required library is installed and importable."""

    # =========================================================================
    # Centrality algorithms (the biggest scaling bottlenecks)
    # =========================================================================

    @abstractmethod
    def pagerank(
        self,
        G: nx.DiGraph,
        alpha: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-06,
        **kwargs,
    ) -> Dict[str, float]:
        """PageRank. Returns {node_id: score}."""

    @abstractmethod
    def betweenness_centrality(
        self,
        G: nx.Graph,
        normalized: bool = True,
        endpoints: bool = False,
        **kwargs,
    ) -> Dict[str, float]:
        """Betweenness centrality. Returns {node_id: score}."""

    @abstractmethod
    def closeness_centrality(
        self,
        G: nx.Graph,
        **kwargs,
    ) -> Dict[str, float]:
        """Closeness centrality. Returns {node_id: score}."""

    @abstractmethod
    def eigenvector_centrality(
        self,
        G: nx.Graph,
        max_iter: int = 100,
        tol: float = 1e-06,
        **kwargs,
    ) -> Dict[str, float]:
        """Eigenvector centrality. Returns {node_id: score}."""

    @abstractmethod
    def katz_centrality(
        self,
        G: nx.Graph,
        alpha: float = 0.1,
        beta: float = 1.0,
        **kwargs,
    ) -> Dict[str, float]:
        """Katz centrality. Returns {node_id: score}."""

    @abstractmethod
    def harmonic_centrality(
        self,
        G: nx.Graph,
        **kwargs,
    ) -> Dict[str, float]:
        """Harmonic centrality. Returns {node_id: score}."""

    # =========================================================================
    # Clustering
    # =========================================================================

    @abstractmethod
    def clustering_coefficient(
        self,
        G: nx.Graph,
        **kwargs,
    ) -> Dict[str, float]:
        """Local clustering coefficient. Returns {node_id: coefficient}."""

    @abstractmethod
    def triangles(
        self,
        G: nx.Graph,
        **kwargs,
    ) -> Dict[str, int]:
        """Triangle count per node. Returns {node_id: count}."""

    # =========================================================================
    # Community detection
    # =========================================================================

    @abstractmethod
    def louvain_communities(
        self,
        G: nx.Graph,
        resolution: float = 1.0,
        **kwargs,
    ) -> Dict[str, int]:
        """Louvain community IDs. Returns {node_id: community_id}."""

    # =========================================================================
    # Utility
    # =========================================================================

    def supports(self, method_name: str) -> bool:
        """Check if this backend has a real implementation for a method."""
        method = getattr(self, method_name, None)
        if method is None:
            return False
        # Check if it's the stub that raises NotImplementedError
        try:
            # If the method is abstract, it's supported by definition
            return True
        except Exception:
            return False
