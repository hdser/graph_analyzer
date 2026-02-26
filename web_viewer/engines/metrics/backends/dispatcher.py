"""
Compute Dispatcher

Selects the optimal compute backend based on graph size and library
availability.  Mirrors the layout backend selection pattern from
layout_backends.py.

Thresholds:
  - <50K nodes:  NetworkX  (pure Python, always available)
  - 50K–5M:      igraph    (C core, ~10-100× faster)
  - >5M:         cuGraph   (GPU, ~100-1000× faster, requires CUDA)
"""

import logging
from typing import Optional

from .base import ComputeBackend
from .networkx_backend import NetworkXBackend
from .igraph_backend import IGraphBackend, HAS_IGRAPH
from .cugraph_backend import CuGraphBackend, HAS_CUGRAPH

logger = logging.getLogger(__name__)

# Size thresholds for automatic backend selection
IGRAPH_THRESHOLD = 50_000
CUGRAPH_THRESHOLD = 5_000_000


class ComputeDispatcher:
    """
    Selects the best available compute backend for a given graph.

    Usage:
        dispatcher = ComputeDispatcher()
        backend = dispatcher.select(node_count=100_000)
        scores = backend.pagerank(G)
    """

    def __init__(self):
        self._networkx = NetworkXBackend()
        self._igraph = IGraphBackend() if HAS_IGRAPH else None
        self._cugraph = CuGraphBackend() if HAS_CUGRAPH else None

    def select(
        self,
        node_count: int,
        preferred: Optional[str] = None,
    ) -> ComputeBackend:
        """
        Select the optimal backend.

        Args:
            node_count: Number of nodes in the graph.
            preferred: Force a specific backend ('networkx', 'igraph', 'cugraph').
                       Falls back gracefully if not available.

        Returns:
            ComputeBackend instance.
        """
        # Explicit preference
        if preferred:
            backend = self._get_by_name(preferred)
            if backend and backend.is_available:
                logger.info(f"[DISPATCHER] Using preferred backend: {backend.name}")
                return backend
            logger.warning(
                f"[DISPATCHER] Preferred backend '{preferred}' not available, "
                f"falling back to auto-selection"
            )

        # Auto-select by graph size
        if node_count > CUGRAPH_THRESHOLD and self._cugraph and self._cugraph.is_available:
            logger.info(
                f"[DISPATCHER] {node_count:,} nodes > {CUGRAPH_THRESHOLD:,} → cuGraph"
            )
            return self._cugraph

        if node_count > IGRAPH_THRESHOLD and self._igraph and self._igraph.is_available:
            logger.info(
                f"[DISPATCHER] {node_count:,} nodes > {IGRAPH_THRESHOLD:,} → igraph"
            )
            return self._igraph

        logger.info(
            f"[DISPATCHER] {node_count:,} nodes → NetworkX"
        )
        return self._networkx

    def _get_by_name(self, name: str) -> Optional[ComputeBackend]:
        """Look up a backend by name string."""
        mapping = {
            "networkx": self._networkx,
            "igraph": self._igraph,
            "cugraph": self._cugraph,
        }
        return mapping.get(name)

    def available_backends(self):
        """List available backend names."""
        names = ["networkx"]
        if self._igraph and self._igraph.is_available:
            names.append("igraph")
        if self._cugraph and self._cugraph.is_available:
            names.append("cugraph")
        return names
