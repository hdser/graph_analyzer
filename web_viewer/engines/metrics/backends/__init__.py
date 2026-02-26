"""
Compute Backends

Pluggable graph computation backends for multi-library metric execution.
Follows the same pattern as layout_backends.py.

Backend selection:
  - NetworkX: default, works for all graph sizes but slow >50K nodes
  - igraph:   preferred for 50K-5M nodes (C core, much faster)
  - cuGraph:  optional GPU-accelerated, for >5M nodes
"""

from .base import ComputeBackend
from .networkx_backend import NetworkXBackend
from .igraph_backend import IGraphBackend, HAS_IGRAPH
from .cugraph_backend import CuGraphBackend, HAS_CUGRAPH
from .dispatcher import ComputeDispatcher

__all__ = [
    "ComputeBackend",
    "NetworkXBackend",
    "IGraphBackend",
    "CuGraphBackend",
    "ComputeDispatcher",
    "HAS_IGRAPH",
    "HAS_CUGRAPH",
    "get_available_backends",
    "get_backend_info",
]


def get_available_backends():
    """Get dict of available compute backends."""
    backends = {"networkx": NetworkXBackend()}

    ig = IGraphBackend()
    if ig.is_available:
        backends["igraph"] = ig

    cg = CuGraphBackend()
    if cg.is_available:
        backends["cugraph"] = cg

    return backends


def get_backend_info():
    """Get metadata about all backends for logging/display."""
    return [
        {"name": "NetworkX", "available": True, "note": "default"},
        {"name": "igraph", "available": HAS_IGRAPH, "note": "50K-5M nodes"},
        {"name": "cuGraph", "available": HAS_CUGRAPH, "note": "GPU, >5M nodes"},
    ]
