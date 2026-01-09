"""
Capacity Graph Builder

Constructs capacity graphs from trust relations and token balances.

Location: web_viewer/engines/capacity_flow/capacity_graph_builder.py
"""
import time
import logging
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

from .models import (
    NodeType,
    EdgeType,
    CapacityNode,
    CapacityEdge,
    TrustRelation,
    TokenBalance,
    CapacityGraphData,
    CapacityGraphStats,
    INFINITY_CAPACITY,
)
from .address_mapper import AddressMapper, TOKEN_POOL_PREFIX

logger = logging.getLogger(__name__)


class CapacityGraph:
    """
    In-memory capacity graph structure.
    
    Stores nodes and edges with efficient adjacency lookups.
    """
    
    def __init__(self):
        self.nodes: Dict[str, CapacityNode] = {}
        self.edges: List[CapacityEdge] = []
        self.outgoing: Dict[str, List[CapacityEdge]] = defaultdict(list)
        self.incoming: Dict[str, List[CapacityEdge]] = defaultdict(list)
        
        # Quick lookups
        self._groups: Set[str] = set()
        self._token_pools: Set[str] = set()
        self._avatars: Set[str] = set()
    
    def add_node(self, node: CapacityNode) -> None:
        """Add node to graph."""
        self.nodes[node.id] = node
        
        if node.node_type == NodeType.GROUP:
            self._groups.add(node.id)
        elif node.node_type == NodeType.TOKEN_POOL:
            self._token_pools.add(node.id)
        elif node.node_type == NodeType.AVATAR:
            self._avatars.add(node.id)
    
    def add_edge(self, edge: CapacityEdge) -> None:
        """Add edge to graph."""
        self.edges.append(edge)
        self.outgoing[edge.source].append(edge)
        self.incoming[edge.target].append(edge)
    
    def get_node(self, node_id: str) -> Optional[CapacityNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)
    
    def has_node(self, node_id: str) -> bool:
        """Check if node exists."""
        return node_id in self.nodes
    
    def get_outgoing_edges(self, node_id: str) -> List[CapacityEdge]:
        """Get outgoing edges from node."""
        return self.outgoing.get(node_id, [])
    
    def get_incoming_edges(self, node_id: str) -> List[CapacityEdge]:
        """Get incoming edges to node."""
        return self.incoming.get(node_id, [])
    
    def is_group(self, node_id: str) -> bool:
        """Check if node is a group."""
        return node_id in self._groups
    
    def is_token_pool(self, node_id: str) -> bool:
        """Check if node is a token pool."""
        return node_id in self._token_pools
    
    def get_stats(self) -> CapacityGraphStats:
        """Get graph statistics."""
        stats = CapacityGraphStats()
        stats.num_nodes = len(self.nodes)
        stats.num_edges = len(self.edges)
        stats.num_avatars = len(self._avatars)
        stats.num_token_pools = len(self._token_pools)
        stats.num_groups = len(self._groups)
        
        for edge in self.edges:
            if edge.edge_type == EdgeType.BALANCE:
                stats.num_balance_edges += 1
            elif edge.edge_type == EdgeType.TRUST:
                stats.num_trust_edges += 1
            elif edge.edge_type == EdgeType.MINT:
                stats.num_mint_edges += 1
        
        return stats
    
    def clear(self) -> None:
        """Clear graph."""
        self.nodes.clear()
        self.edges.clear()
        self.outgoing.clear()
        self.incoming.clear()
        self._groups.clear()
        self._token_pools.clear()
        self._avatars.clear()


class CapacityGraphBuilder:
    """
    Builds capacity graphs from trust/balance data.
    
    The graph construction follows these phases:
    1. Initialize avatar and group nodes
    2. Add balance edges (Avatar -> TokenPool)
    3. Add trust edges (TokenPool -> Avatar)
    4. Add minting edges (Group -> Avatar)
    """
    
    def __init__(self, router_address: Optional[str] = None):
        self.router_address = router_address
        self.mapper = AddressMapper()
        self.graph = CapacityGraph()
        
        # Tracking
        self._token_pool_map: Dict[str, str] = {}  # token_address -> pool_id
        self._trust_lookup: Dict[str, Set[str]] = defaultdict(set)  # token -> set of trusters
    
    def build(self, data: CapacityGraphData) -> Tuple[CapacityGraph, CapacityGraphStats]:
        """
        Build capacity graph from input data.
        
        Returns:
            Tuple of (CapacityGraph, CapacityGraphStats)
        """
        start_time = time.time()
        
        # Reset state
        self.mapper.clear()
        self.graph.clear()
        self._token_pool_map.clear()
        self._trust_lookup.clear()
        
        # Phase 1: Initialize nodes
        self._init_nodes(data)
        
        # Phase 2: Build trust lookup
        self._build_trust_lookup(data.trusts)
        
        # Phase 3: Add balance edges
        self._add_balance_edges(data.balances)
        
        # Phase 4: Add trust edges
        self._add_trust_edges()
        
        # Phase 5: Add minting edges for groups
        self._add_mint_edges(data.groups)
        
        # Get stats
        stats = self.graph.get_stats()
        stats.build_time_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Built capacity graph: {stats.num_avatars} avatars, "
            f"{stats.num_token_pools} pools, {stats.num_edges} edges "
            f"in {stats.build_time_ms:.1f}ms"
        )
        
        return self.graph, stats
    
    def _init_nodes(self, data: CapacityGraphData) -> None:
        """Phase 1: Initialize avatar and group nodes."""
        # Collect all addresses from trusts
        addresses = set()
        for trust in data.trusts:
            addresses.add(trust.truster)
            addresses.add(trust.trustee)
        
        # Add holders from balances
        for balance in data.balances:
            addresses.add(balance.holder)
        
        # Remove router if present
        if self.router_address:
            addresses.discard(self.router_address)
        
        # Create nodes
        for addr in addresses:
            if addr in data.groups:
                node_type = NodeType.GROUP
            else:
                node_type = NodeType.AVATAR
            
            node_id = self.mapper.get_or_create_id(addr)
            node = CapacityNode(
                id=node_id,
                node_type=node_type,
                address=addr,
            )
            self.graph.add_node(node)
    
    def _build_trust_lookup(self, trusts: List[TrustRelation]) -> None:
        """Build lookup: which addresses trust which tokens."""
        for trust in trusts:
            # Skip router
            if self.router_address:
                if trust.truster == self.router_address:
                    continue
                if trust.trustee == self.router_address:
                    continue
            
            # truster trusts trustee's token
            # Token ID is the trustee's address
            token_id = trust.trustee
            self._trust_lookup[token_id].add(trust.truster)
    
    def _get_or_create_token_pool(self, token_address: str) -> str:
        """Get or create token pool node for a token."""
        if token_address in self._token_pool_map:
            return self._token_pool_map[token_address]
        
        pool_id = self.mapper.create_token_pool_id(token_address)
        node = CapacityNode(
            id=pool_id,
            node_type=NodeType.TOKEN_POOL,
            token_id=token_address,
        )
        self.graph.add_node(node)
        self._token_pool_map[token_address] = pool_id
        
        return pool_id
    
    def _add_balance_edges(self, balances: List[TokenBalance]) -> None:
        """Phase 2: Add balance edges (Avatar -> TokenPool)."""
        for balance in balances:
            if balance.balance <= 0:
                continue
            
            # Skip router
            if self.router_address and balance.holder == self.router_address:
                continue
            
            # Get holder node ID
            holder_id = self.mapper.get_id(balance.holder)
            if holder_id is None:
                continue
            
            # Get or create token pool
            pool_id = self._get_or_create_token_pool(balance.token_id)
            
            # Add edge: holder -> pool with capacity = balance
            edge = CapacityEdge(
                source=holder_id,
                target=pool_id,
                capacity=balance.balance,
                edge_type=EdgeType.BALANCE,
                token_id=balance.token_id,
            )
            self.graph.add_edge(edge)
    
    def _add_trust_edges(self) -> None:
        """Phase 3: Add trust edges (TokenPool -> Avatar)."""
        for pool_id in list(self.graph._token_pools):
            pool_node = self.graph.get_node(pool_id)
            if pool_node is None:
                continue
            
            token_address = pool_node.token_id
            if token_address is None:
                continue
            
            # Find who trusts this token
            trusters = self._trust_lookup.get(token_address, set())
            
            for truster_addr in trusters:
                # Skip router
                if self.router_address and truster_addr == self.router_address:
                    continue
                
                truster_id = self.mapper.get_id(truster_addr)
                if truster_id is None:
                    continue
                
                # Add edge: pool -> truster with infinite capacity
                edge = CapacityEdge(
                    source=pool_id,
                    target=truster_id,
                    capacity=INFINITY_CAPACITY,
                    edge_type=EdgeType.TRUST,
                    token_id=token_address,
                )
                self.graph.add_edge(edge)
    
    def _add_mint_edges(self, groups: Set[str]) -> None:
        """Phase 4: Add minting edges (Group -> Avatar)."""
        for group_addr in groups:
            group_id = self.mapper.get_id(group_addr)
            if group_id is None:
                continue
            
            # Group's token is the group address itself
            group_token = group_addr
            
            # Find who trusts the group token
            trusters = self._trust_lookup.get(group_token, set())
            
            for truster_addr in trusters:
                # Skip router and other groups
                if self.router_address and truster_addr == self.router_address:
                    continue
                if truster_addr in groups:
                    continue
                
                truster_id = self.mapper.get_id(truster_addr)
                if truster_id is None:
                    continue
                
                # Add edge: group -> truster with infinite capacity
                edge = CapacityEdge(
                    source=group_id,
                    target=truster_id,
                    capacity=INFINITY_CAPACITY,
                    edge_type=EdgeType.MINT,
                    token_id=group_token,
                )
                self.graph.add_edge(edge)
    
    def create_virtual_sink(
        self,
        source_address: str,
        target_tokens: Set[str]
    ) -> Optional[str]:
        """
        Create virtual sink for source == sink case.
        
        When source equals sink, we need a virtual sink that accepts
        only the tokens the source can use for conversion.
        """
        source_id = self.mapper.get_id(source_address)
        if source_id is None:
            return None
        
        # Create virtual sink
        vsink_id = self.mapper.create_virtual_sink_id(source_address)
        vsink_node = CapacityNode(
            id=vsink_id,
            node_type=NodeType.VIRTUAL_SINK,
            address=source_address,
        )
        self.graph.add_node(vsink_node)
        
        # Add edges from token pools to virtual sink
        for pool_id in self.graph._token_pools:
            pool_node = self.graph.get_node(pool_id)
            if pool_node is None or pool_node.token_id is None:
                continue
            
            # Only tokens in target_tokens that source also trusts
            if pool_node.token_id in target_tokens:
                if source_address in self._trust_lookup.get(pool_node.token_id, set()):
                    edge = CapacityEdge(
                        source=pool_id,
                        target=vsink_id,
                        capacity=INFINITY_CAPACITY,
                        edge_type=EdgeType.TRUST,
                        token_id=pool_node.token_id,
                    )
                    self.graph.add_edge(edge)
        
        return vsink_id