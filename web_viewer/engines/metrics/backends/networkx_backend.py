"""
NetworkX Compute Backend

Default backend — wraps NetworkX functions with the ComputeBackend API.
No additional dependencies. Works for any graph size but O(n²) or worse
for betweenness, closeness, etc. on large graphs.
"""

from typing import Dict, Any

import networkx as nx

from .base import ComputeBackend


class NetworkXBackend(ComputeBackend):
    """NetworkX-based compute backend (default)."""

    @property
    def name(self) -> str:
        return "networkx"

    @property
    def is_available(self) -> bool:
        return True  # Always available

    # -- Centrality --------------------------------------------------------

    def pagerank(self, G, alpha=0.85, max_iter=100, tol=1e-06, **kw):
        return nx.pagerank(G, alpha=alpha, max_iter=max_iter, tol=tol)

    def betweenness_centrality(self, G, normalized=True, endpoints=False, **kw):
        return nx.betweenness_centrality(
            G, normalized=normalized, endpoints=endpoints
        )

    def closeness_centrality(self, G, **kw):
        return nx.closeness_centrality(G)

    def eigenvector_centrality(self, G, max_iter=100, tol=1e-06, **kw):
        try:
            return nx.eigenvector_centrality(G, max_iter=max_iter, tol=tol)
        except nx.PowerIterationFailedConvergence:
            return nx.eigenvector_centrality_numpy(G)

    def katz_centrality(self, G, alpha=0.1, beta=1.0, **kw):
        try:
            return nx.katz_centrality(G, alpha=alpha, beta=beta)
        except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
            return nx.katz_centrality_numpy(G, alpha=alpha, beta=beta)

    def harmonic_centrality(self, G, **kw):
        return nx.harmonic_centrality(G)

    # -- Clustering --------------------------------------------------------

    def clustering_coefficient(self, G, **kw):
        return nx.clustering(G)

    def triangles(self, G, **kw):
        # NetworkX triangles only works on undirected graphs
        if G.is_directed():
            G = G.to_undirected()
        return nx.triangles(G)

    # -- Community ---------------------------------------------------------

    def louvain_communities(self, G, resolution=1.0, **kw):
        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(G, resolution=resolution)
            result = {}
            for cid, members in enumerate(communities):
                for node in members:
                    result[node] = cid
            return result
        except Exception:
            return {node: 0 for node in G.nodes()}
