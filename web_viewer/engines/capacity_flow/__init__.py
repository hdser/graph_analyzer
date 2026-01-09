"""
Capacity Flow Package

Capacity graph construction and max flow computation for Circles trust networks.

Location: web_viewer/engines/capacity_flow/__init__.py
"""
from .models import (
    NodeType,
    EdgeType,
    CapacityNode,
    CapacityEdge,
    TrustRelation,
    TokenBalance,
    GroupInfo,
    CapacityGraphData,
    FlowEdge,
    FlowPath,
    CapacityGraphStats,
    FlowResult,
    INFINITY_CAPACITY,
)

from .address_mapper import AddressMapper

from .capacity_graph_builder import (
    CapacityGraph,
    CapacityGraphBuilder,
)

from .flow_decomposer import (
    FlowDecomposer,
    decompose_flow,
)

from .path_simplifier import (
    PathSimplifier,
    simplify_paths,
)

from .flow_engine import CapacityFlowEngine

from .backends import (
    BaseFlowBackend,
    NetworkXBackend,
    ORToolsBackend,
    get_backend,
    get_available_backends,
    get_best_backend,
    get_backend_info,
)


__all__ = [
    # Models
    "NodeType",
    "EdgeType",
    "CapacityNode",
    "CapacityEdge",
    "TrustRelation",
    "TokenBalance",
    "GroupInfo",
    "CapacityGraphData",
    "FlowEdge",
    "FlowPath",
    "CapacityGraphStats",
    "FlowResult",
    "INFINITY_CAPACITY",
    # Address mapping
    "AddressMapper",
    # Graph building
    "CapacityGraph",
    "CapacityGraphBuilder",
    # Flow computation
    "FlowDecomposer",
    "decompose_flow",
    "PathSimplifier",
    "simplify_paths",
    "CapacityFlowEngine",
    # Backends
    "BaseFlowBackend",
    "NetworkXBackend",
    "ORToolsBackend",
    "get_backend",
    "get_available_backends",
    "get_best_backend",
    "get_backend_info",
]