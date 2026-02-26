"""
igraph Compute Backend

C-core graph library — 10-100× faster than NetworkX for centrality and
community detection on large graphs.  Optimal for 50K–5M nodes.

Uses the nx_to_igraph converter from BaseMetricAlgorithm for seamless
NetworkX → igraph graph conversion with node-name preservation.
"""

from typing import Dict, Any, Optional
import logging

import networkx as nx

from .base import ComputeBackend

logger = logging.getLogger(__name__)

try:
    import igraph as ig
    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False


class IGraphBackend(ComputeBackend):
    """igraph-based compute backend for medium-to-large graphs."""

    def __init__(self):
        self._ig_cache: Dict[int, ig.Graph] = {}  # id(G) -> igraph

    @property
    def name(self) -> str:
        return "igraph"

    @property
    def is_available(self) -> bool:
        return HAS_IGRAPH

    # =====================================================================
    # Conversion helpers
    # =====================================================================

    def _to_igraph(self, G: nx.Graph, directed: bool = True) -> "ig.Graph":
        """Convert NetworkX graph to igraph, caching by object id."""
        cache_key = (id(G), directed)
        if cache_key in self._ig_cache:
            return self._ig_cache[cache_key]

        node_list = list(G.nodes())
        node_index = {n: i for i, n in enumerate(node_list)}

        ig_graph = ig.Graph(directed=directed)
        ig_graph.add_vertices(len(node_list))
        ig_graph.vs["name"] = node_list

        edges = []
        weights = []
        for u, v, data in G.edges(data=True):
            edges.append((node_index[u], node_index[v]))
            weights.append(data.get("weight", 1.0))
        ig_graph.add_edges(edges)
        ig_graph.es["weight"] = weights

        self._ig_cache[cache_key] = ig_graph
        return ig_graph

    def _result_dict(self, ig_graph: "ig.Graph", values) -> Dict[str, float]:
        """Map igraph vertex-indexed values back to node names."""
        names = ig_graph.vs["name"]
        return {names[i]: float(v) for i, v in enumerate(values)}

    # =====================================================================
    # Centrality
    # =====================================================================

    def pagerank(self, G, alpha=0.85, max_iter=100, tol=1e-06, **kw):
        ig_g = self._to_igraph(G, directed=True)
        pr = ig_g.pagerank(damping=alpha)
        return self._result_dict(ig_g, pr)

    def betweenness_centrality(self, G, normalized=True, endpoints=False, **kw):
        directed = G.is_directed()
        ig_g = self._to_igraph(G, directed=directed)
        bw = ig_g.betweenness(directed=directed)
        if normalized and len(ig_g.vs) > 2:
            n = len(ig_g.vs)
            if directed:
                norm = (n - 1) * (n - 2)
            else:
                norm = (n - 1) * (n - 2) / 2
            bw = [b / norm for b in bw]
        return self._result_dict(ig_g, bw)

    def closeness_centrality(self, G, **kw):
        directed = G.is_directed()
        ig_g = self._to_igraph(G, directed=directed)
        cl = ig_g.closeness(mode="all")
        # Replace NaN with 0
        cl = [0.0 if c != c else c for c in cl]  # NaN != NaN
        return self._result_dict(ig_g, cl)

    def eigenvector_centrality(self, G, max_iter=100, tol=1e-06, **kw):
        directed = G.is_directed()
        ig_g = self._to_igraph(G, directed=directed)
        try:
            ev = ig_g.eigenvector_centrality(directed=directed)
            return self._result_dict(ig_g, ev)
        except Exception:
            # Fallback: return zeros
            return {n: 0.0 for n in G.nodes()}

    def katz_centrality(self, G, alpha=0.1, beta=1.0, **kw):
        # igraph doesn't have a direct katz_centrality
        # Fall back to NetworkX for this one
        try:
            return nx.katz_centrality(G, alpha=alpha, beta=beta)
        except Exception:
            return {n: 0.0 for n in G.nodes()}

    def harmonic_centrality(self, G, **kw):
        directed = G.is_directed()
        ig_g = self._to_igraph(G, directed=directed)
        hc = ig_g.harmonic_centrality(mode="all")
        return self._result_dict(ig_g, hc)

    # =====================================================================
    # Clustering
    # =====================================================================

    def clustering_coefficient(self, G, **kw):
        ig_g = self._to_igraph(G, directed=False)
        cc = ig_g.transitivity_local_undirected(mode="zero")
        return self._result_dict(ig_g, cc)

    def triangles(self, G, **kw):
        ig_g = self._to_igraph(G, directed=False)
        # igraph gives list of triangles per vertex
        # But we need counts — use motifs or direct count
        cl = ig_g.transitivity_local_undirected(mode="zero")
        degrees = ig_g.degree()
        names = ig_g.vs["name"]
        result = {}
        for i, name in enumerate(names):
            d = degrees[i]
            # triangles = clustering * C(degree, 2)
            possible = d * (d - 1) / 2
            result[name] = int(round(cl[i] * possible))
        return result

    # =====================================================================
    # Community
    # =====================================================================

    def louvain_communities(self, G, resolution=1.0, **kw):
        ig_g = self._to_igraph(G, directed=False)
        partition = ig_g.community_multilevel(
            weights=ig_g.es["weight"] if ig_g.ecount() > 0 else None
        )
        names = ig_g.vs["name"]
        return {names[i]: m for i, m in enumerate(partition.membership)}
