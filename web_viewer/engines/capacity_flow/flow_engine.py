"""
Capacity Flow Engine

Main orchestrator for capacity graph and max flow operations.

Location: web_viewer/engines/capacity_flow/flow_engine.py
"""
import time
import logging
from typing import Dict, List, Set, Optional, Tuple, Any

from .models import (
    TrustRelation,
    TokenBalance,
    CapacityGraphData,
    CapacityGraphStats,
    FlowPath,
    FlowResult,
)
from .address_mapper import AddressMapper
from .capacity_graph_builder import CapacityGraph, CapacityGraphBuilder
from .flow_decomposer import FlowDecomposer
from .path_simplifier import PathSimplifier
from .backends import get_backend, get_best_backend, get_available_backends

logger = logging.getLogger(__name__)

# Default router address for Circles protocol
DEFAULT_ROUTER_ADDRESS = "0xdc287474114cc0551a81ddc2eb51783fbf34802f"


class CapacityFlowEngine:
    """
    Main engine for capacity graph and max flow operations.
    
    Coordinates:
    - Graph building from trust/balance data
    - Max flow computation via backends
    - Flow decomposition into paths
    - Path simplification for display
    """
    
    def __init__(self, router_address: Optional[str] = None):
        self.router_address = router_address or DEFAULT_ROUTER_ADDRESS
        
        # Components
        self.mapper: Optional[AddressMapper] = None
        self.graph: Optional[CapacityGraph] = None
        self.builder: Optional[CapacityGraphBuilder] = None
        
        # State
        self._stats: Optional[CapacityGraphStats] = None
        self._groups: Set[str] = set()
        self._build_time: float = 0
        self._is_ready: bool = False
    
    def is_ready(self) -> bool:
        """Check if graph is built and ready."""
        return self._is_ready and self.graph is not None
    
    def get_stats(self) -> Optional[CapacityGraphStats]:
        """Get graph statistics."""
        return self._stats
    
    def get_build_time(self) -> float:
        """Get last build time in seconds."""
        return self._build_time
    
    def get_graph(self) -> Optional[CapacityGraph]:
        """Get the capacity graph for visualization."""
        return self.graph
    
    def get_mapper(self) -> Optional[AddressMapper]:
        """Get the address mapper for ID/address resolution."""
        return self.mapper
    
    def build_graph(
        self,
        trusts: List[TrustRelation],
        balances: List[TokenBalance],
        groups: Optional[Set[str]] = None
    ) -> CapacityGraphStats:
        """
        Build capacity graph from trust and balance data.
        
        Args:
            trusts: List of trust relations
            balances: List of token balances
            groups: Set of group addresses
            
        Returns:
            Graph statistics
        """
        start_time = time.time()
        
        self._groups = groups or set()
        
        # Create builder
        self.builder = CapacityGraphBuilder(router_address=self.router_address)
        
        # Prepare data
        data = CapacityGraphData(
            trusts=trusts,
            balances=balances,
            groups=self._groups,
            router_address=self.router_address,
        )
        
        # Build graph
        self.graph, self._stats = self.builder.build(data)
        self.mapper = self.builder.mapper
        
        self._build_time = time.time() - start_time
        self._is_ready = True
        
        logger.info(f"Built capacity graph in {self._build_time:.2f}s")
        
        return self._stats
    
    def compute_max_flow(
        self,
        source: str,
        sink: str,
        backend: Optional[str] = None,
        algorithm: Optional[str] = None,
        cutoff: Optional[int] = None,
        decompose_paths: bool = True,
        simplify_paths: bool = True,
        max_paths: Optional[int] = None
    ) -> FlowResult:
        """
        Compute maximum flow between source and sink.
        
        Args:
            source: Source address
            sink: Sink address
            backend: Backend name (auto-select if None)
            algorithm: Algorithm name (backend default if None)
            cutoff: Maximum flow cutoff
            decompose_paths: Whether to decompose into paths
            simplify_paths: Whether to simplify paths
            max_paths: Maximum paths to extract
            
        Returns:
            FlowResult with max flow and paths
        """
        start_time = time.time()
        
        # Validate state
        if not self.is_ready():
            return FlowResult(
                success=False,
                error="Capacity graph not built",
                source=source,
                sink=sink,
            )
        
        # Map addresses to IDs
        source_id = self.mapper.get_id(source)
        sink_id = self.mapper.get_id(sink)
        
        if source_id is None:
            return FlowResult(
                success=False,
                error=f"Source address not found: {source}",
                source=source,
                sink=sink,
            )
        
        if sink_id is None:
            return FlowResult(
                success=False,
                error=f"Sink address not found: {sink}",
                source=source,
                sink=sink,
            )
        
        # Handle source == sink (token conversion)
        actual_sink_id = sink_id
        if source == sink:
            # Create virtual sink
            target_tokens = self._get_trusted_tokens(source)
            actual_sink_id = self.builder.create_virtual_sink(source, target_tokens)
            if actual_sink_id is None:
                return FlowResult(
                    success=False,
                    error="Failed to create virtual sink for token conversion",
                    source=source,
                    sink=sink,
                )
        
        # Get backend
        if backend:
            flow_backend = get_backend(backend)
            if flow_backend is None:
                return FlowResult(
                    success=False,
                    error=f"Backend not available: {backend}",
                    source=source,
                    sink=sink,
                )
        else:
            flow_backend = get_best_backend()
            if flow_backend is None:
                return FlowResult(
                    success=False,
                    error="No flow backends available",
                    source=source,
                    sink=sink,
                )
        
        # Prepare edges for backend
        edges = []
        edge_tokens: Dict[Tuple[str, str], str] = {}
        
        for edge in self.graph.edges:
            edges.append((edge.source, edge.target, edge.capacity))
            if edge.token_id:
                edge_tokens[(edge.source, edge.target)] = edge.token_id
        
        # Compute max flow
        try:
            flow_value, flow_dict = flow_backend.compute_max_flow(
                edges=edges,
                source=source_id,
                sink=actual_sink_id,
                algorithm=algorithm,
                cutoff=cutoff,
            )
        except Exception as e:
            logger.error(f"Max flow computation failed: {e}")
            return FlowResult(
                success=False,
                error=str(e),
                source=source,
                sink=sink,
            )
        
        # Decompose into paths if requested
        paths: List[FlowPath] = []
        token_flows: Dict[str, int] = {}
        
        if decompose_paths and flow_value > 0:
            decomposer = FlowDecomposer(max_paths=max_paths or 1000)
            paths, token_flows = decomposer.decompose(
                flow_dict=flow_dict,
                source=source_id,
                sink=actual_sink_id,
                edge_tokens=edge_tokens,
            )
            
            # Simplify paths if requested
            if simplify_paths and paths:
                simplifier = PathSimplifier(
                    mapper=self.mapper,
                    router_address=self.router_address,
                    groups=self._groups,
                )
                paths = simplifier.simplify(paths)
        
        computation_time = (time.time() - start_time) * 1000
        
        return FlowResult(
            success=True,
            max_flow=flow_value,
            source=source,
            sink=sink,
            paths=paths,
            token_flows=token_flows,
            computation_time_ms=computation_time,
            backend=flow_backend.name,
            algorithm=algorithm or flow_backend.DEFAULT_ALGORITHM,
        )
    
    def get_node_capacity(
        self,
        address: str,
        direction: str = "both"
    ) -> Dict[str, Any]:
        """
        Get capacity information for a node.
        
        Args:
            address: Node address
            direction: "in", "out", or "both"
            
        Returns:
            Capacity info dictionary
        """
        if not self.is_ready():
            return {"error": "Graph not built"}
        
        node_id = self.mapper.get_id(address)
        if node_id is None:
            return {"error": f"Address not found: {address}"}
        
        result = {
            "address": address,
            "node_id": node_id,
        }
        
        if direction in ("in", "both"):
            in_edges = self.graph.get_incoming_edges(node_id)
            result["incoming"] = {
                "count": len(in_edges),
                "total_capacity": sum(e.capacity for e in in_edges),
            }
        
        if direction in ("out", "both"):
            out_edges = self.graph.get_outgoing_edges(node_id)
            result["outgoing"] = {
                "count": len(out_edges),
                "total_capacity": sum(e.capacity for e in out_edges),
            }
        
        return result
    
    def _get_trusted_tokens(self, address: str) -> Set[str]:
        """Get set of tokens trusted by an address."""
        tokens = set()
        node_id = self.mapper.get_id(address)
        
        if node_id is None:
            return tokens
        
        # Look at incoming trust edges
        for edge in self.graph.get_incoming_edges(node_id):
            if edge.token_id:
                tokens.add(edge.token_id)
        
        return tokens
    
    def clear(self) -> None:
        """Clear all state."""
        self.graph = None
        self.mapper = None
        self.builder = None
        self._stats = None
        self._groups.clear()
        self._build_time = 0
        self._is_ready = False