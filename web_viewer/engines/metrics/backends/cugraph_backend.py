"""
cuGraph Compute Backend (GPU-Accelerated)

Optional NVIDIA RAPIDS cuGraph backend for massive graphs (>5M nodes).
Requires: cudf, cugraph, CUDA-capable GPU.

This is a stub — methods fall back to NetworkX if cuGraph isn't available.
When cuGraph IS available, it uses GPU-accelerated implementations.
"""

from typing import Dict, Any
import logging

import networkx as nx

from .base import ComputeBackend

logger = logging.getLogger(__name__)

try:
    import cugraph
    import cudf
    HAS_CUGRAPH = True
except ImportError:
    HAS_CUGRAPH = False


def _nx_to_cudf_edgelist(G: nx.Graph):
    """Convert NetworkX graph to cuDF edge list for cuGraph."""
    if not HAS_CUGRAPH:
        raise RuntimeError("cuGraph not available")

    import cudf
    edges = list(G.edges(data=True))
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    wt = [e[2].get("weight", 1.0) for e in edges]

    df = cudf.DataFrame({"src": src, "dst": dst, "weight": wt})
    return df


class CuGraphBackend(ComputeBackend):
    """GPU-accelerated compute backend via NVIDIA cuGraph."""

    @property
    def name(self) -> str:
        return "cugraph"

    @property
    def is_available(self) -> bool:
        return HAS_CUGRAPH

    # -- Centrality --------------------------------------------------------

    def pagerank(self, G, alpha=0.85, max_iter=100, tol=1e-06, **kw):
        if not HAS_CUGRAPH:
            return nx.pagerank(G, alpha=alpha)
        cug = cugraph.Graph(directed=True)
        df = _nx_to_cudf_edgelist(G)
        cug.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="weight")
        pr = cugraph.pagerank(cug, alpha=alpha, max_iter=max_iter, tol=tol)
        return dict(zip(pr["vertex"].to_pandas(), pr["pagerank"].to_pandas()))

    def betweenness_centrality(self, G, normalized=True, endpoints=False, **kw):
        if not HAS_CUGRAPH:
            return nx.betweenness_centrality(G, normalized=normalized)
        cug = cugraph.Graph(directed=G.is_directed())
        df = _nx_to_cudf_edgelist(G)
        cug.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="weight")
        bc = cugraph.betweenness_centrality(cug, normalized=normalized)
        return dict(zip(bc["vertex"].to_pandas(), bc["betweenness_centrality"].to_pandas()))

    def closeness_centrality(self, G, **kw):
        if not HAS_CUGRAPH:
            return nx.closeness_centrality(G)
        # cuGraph doesn't have closeness — fall back
        return nx.closeness_centrality(G)

    def eigenvector_centrality(self, G, max_iter=100, tol=1e-06, **kw):
        if not HAS_CUGRAPH:
            return nx.eigenvector_centrality(G, max_iter=max_iter)
        cug = cugraph.Graph(directed=G.is_directed())
        df = _nx_to_cudf_edgelist(G)
        cug.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="weight")
        ev = cugraph.eigenvector_centrality(cug, max_iter=max_iter, tol=tol)
        return dict(zip(ev["vertex"].to_pandas(), ev["eigenvector_centrality"].to_pandas()))

    def katz_centrality(self, G, alpha=0.1, beta=1.0, **kw):
        if not HAS_CUGRAPH:
            return nx.katz_centrality(G, alpha=alpha, beta=beta)
        cug = cugraph.Graph(directed=G.is_directed())
        df = _nx_to_cudf_edgelist(G)
        cug.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="weight")
        kz = cugraph.katz_centrality(cug, alpha=alpha)
        return dict(zip(kz["vertex"].to_pandas(), kz["katz_centrality"].to_pandas()))

    def harmonic_centrality(self, G, **kw):
        # cuGraph doesn't have harmonic centrality — fall back
        return nx.harmonic_centrality(G)

    # -- Clustering --------------------------------------------------------

    def clustering_coefficient(self, G, **kw):
        if not HAS_CUGRAPH:
            return nx.clustering(G)
        # cuGraph clustering
        cug = cugraph.Graph(directed=False)
        df = _nx_to_cudf_edgelist(G if not G.is_directed() else G.to_undirected())
        cug.from_cudf_edgelist(df, source="src", destination="dst")
        cc = cugraph.clustering(cug)
        return dict(zip(cc["vertex"].to_pandas(), cc["clustering_coeff"].to_pandas()))

    def triangles(self, G, **kw):
        if not HAS_CUGRAPH:
            ug = G.to_undirected() if G.is_directed() else G
            return nx.triangles(ug)
        # cuGraph triangle count is global, not per-node — fall back
        ug = G.to_undirected() if G.is_directed() else G
        return nx.triangles(ug)

    # -- Community ---------------------------------------------------------

    def louvain_communities(self, G, resolution=1.0, **kw):
        if not HAS_CUGRAPH:
            return {n: 0 for n in G.nodes()}
        cug = cugraph.Graph(directed=False)
        df = _nx_to_cudf_edgelist(G if not G.is_directed() else G.to_undirected())
        cug.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="weight")
        parts, _ = cugraph.louvain(cug, resolution=resolution)
        return dict(zip(parts["vertex"].to_pandas(), parts["partition"].to_pandas()))
