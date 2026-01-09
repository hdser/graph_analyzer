"""
Capacity Flow Data Models

Data classes for capacity graph construction and flow computation.

Location: web_viewer/engines/capacity_flow/models.py
"""
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
from enum import Enum


# Infinity capacity for unlimited edges
INFINITY_CAPACITY = 2**62


class NodeType(Enum):
    """Types of nodes in capacity graph."""
    AVATAR = "avatar"
    TOKEN_POOL = "token_pool"
    GROUP = "group"
    ROUTER = "router"
    VIRTUAL_SINK = "virtual_sink"


class EdgeType(Enum):
    """Types of edges in capacity graph."""
    BALANCE = "balance"      # Avatar -> TokenPool (holds tokens)
    TRUST = "trust"          # TokenPool -> Avatar (can receive)
    MINT = "mint"            # Group -> Avatar (group minting)
    DIRECT = "direct"        # Direct transfer


@dataclass
class CapacityNode:
    """Node in capacity graph."""
    id: str
    node_type: NodeType
    address: Optional[str] = None
    token_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.node_type.value,
            "address": self.address,
            "token_id": self.token_id,
        }


@dataclass
class CapacityEdge:
    """Edge in capacity graph."""
    source: str
    target: str
    capacity: int
    edge_type: EdgeType
    token_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "capacity": self.capacity,
            "type": self.edge_type.value,
            "token_id": self.token_id,
        }


@dataclass
class TrustRelation:
    """Trust relationship between avatars."""
    truster: str      # Who is trusting
    trustee: str      # Whose token is trusted
    limit: int = 0    # Trust limit (0 = unlimited)
    expiry: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "truster": self.truster,
            "trustee": self.trustee,
            "limit": self.limit,
            "expiry": self.expiry,
        }


@dataclass
class TokenBalance:
    """Token balance for an avatar."""
    holder: str       # Who holds the tokens
    token_id: str     # Token identifier (usually issuer address)
    balance: int      # Amount held
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "holder": self.holder,
            "token_id": self.token_id,
            "balance": self.balance,
        }


@dataclass
class GroupInfo:
    """Group information."""
    address: str
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "address": self.address,
            "name": self.name,
        }


@dataclass
class CapacityGraphData:
    """Input data for building capacity graph."""
    trusts: List[TrustRelation] = field(default_factory=list)
    balances: List[TokenBalance] = field(default_factory=list)
    groups: Set[str] = field(default_factory=set)
    router_address: Optional[str] = None


@dataclass
class FlowEdge:
    """Edge with flow value."""
    source: str
    target: str
    flow: int
    capacity: int
    token_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "flow": self.flow,
            "capacity": self.capacity,
            "token_id": self.token_id,
        }


@dataclass
class FlowPath:
    """Single flow path with tokens."""
    nodes: List[str]
    edges: List[Dict[str, str]]
    tokens: List[str]
    flow: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "tokens": self.tokens,
            "flow": self.flow,
        }


@dataclass
class CapacityGraphStats:
    """Statistics about capacity graph."""
    num_nodes: int = 0
    num_edges: int = 0
    num_avatars: int = 0
    num_token_pools: int = 0
    num_groups: int = 0
    num_balance_edges: int = 0
    num_trust_edges: int = 0
    num_mint_edges: int = 0
    build_time_ms: float = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_avatars": self.num_avatars,
            "num_token_pools": self.num_token_pools,
            "num_groups": self.num_groups,
            "num_balance_edges": self.num_balance_edges,
            "num_trust_edges": self.num_trust_edges,
            "num_mint_edges": self.num_mint_edges,
            "build_time_ms": self.build_time_ms,
        }


@dataclass
class FlowResult:
    """Result of max flow computation."""
    success: bool = True
    max_flow: int = 0
    source: str = ""
    sink: str = ""
    paths: List[FlowPath] = field(default_factory=list)
    token_flows: Dict[str, int] = field(default_factory=dict)
    computation_time_ms: float = 0
    backend: str = ""
    algorithm: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "success": self.success,
            "max_flow": self.max_flow,
            "source": self.source,
            "sink": self.sink,
            "paths": [p.to_dict() for p in self.paths],
            "token_flows": self.token_flows,
            "computation_time_ms": self.computation_time_ms,
            "backend": self.backend,
            "algorithm": self.algorithm,
            "path_count": len(self.paths),
        }
        if self.error:
            result["error"] = self.error
        return result