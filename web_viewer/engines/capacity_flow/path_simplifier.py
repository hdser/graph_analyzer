"""
Path Simplifier

Simplifies flow paths for display.

Location: web_viewer/engines/capacity_flow/path_simplifier.py
"""
import logging
from typing import Dict, List, Optional, Set

from .models import FlowPath
from .address_mapper import AddressMapper, TOKEN_POOL_PREFIX, VIRTUAL_SINK_PREFIX

logger = logging.getLogger(__name__)

# Circles protocol router address
DEFAULT_ROUTER_ADDRESS = "0xdc287474114cc0551a81ddc2eb51783fbf34802f"


class PathSimplifier:
    """
    Simplifies flow paths for display.
    
    Operations:
    - Collapse token pool nodes
    - Remove virtual sink nodes
    - Convert IDs to addresses
    - Insert router nodes where needed
    """
    
    def __init__(
        self,
        mapper: AddressMapper,
        router_address: Optional[str] = None,
        groups: Optional[Set[str]] = None
    ):
        self.mapper = mapper
        self.router_address = router_address or DEFAULT_ROUTER_ADDRESS
        self.groups = groups or set()
    
    def simplify(self, paths: List[FlowPath]) -> List[FlowPath]:
        """
        Simplify list of paths.
        
        Args:
            paths: Raw flow paths with internal IDs
            
        Returns:
            Simplified paths with addresses
        """
        return [self._simplify_path(path) for path in paths]
    
    def _simplify_path(self, path: FlowPath) -> FlowPath:
        """Simplify single path."""
        # Convert to addresses and filter special nodes
        simplified_nodes = []
        tokens = []
        
        for node_id in path.nodes:
            # Skip token pool nodes but extract token
            if node_id.startswith(TOKEN_POOL_PREFIX):
                token_addr = self.mapper.get_address(node_id)
                if token_addr:
                    tokens.append(token_addr)
                continue
            
            # Skip virtual sink nodes
            if node_id.startswith(VIRTUAL_SINK_PREFIX):
                continue
            
            # Convert to address
            address = self.mapper.get_address(node_id)
            if address:
                simplified_nodes.append(address)
            else:
                # Keep as-is if no mapping
                simplified_nodes.append(node_id)
        
        # Build edges with potential router insertion
        edges = self._build_edges(simplified_nodes)
        
        return FlowPath(
            nodes=simplified_nodes,
            edges=edges,
            tokens=tokens or path.tokens,
            flow=path.flow,
        )
    
    def _build_edges(self, nodes: List[str]) -> List[Dict[str, str]]:
        """
        Build edge list, inserting router where needed.
        
        Router is inserted between Avatar -> Group transitions.
        """
        edges = []
        
        for i in range(len(nodes) - 1):
            source = nodes[i]
            target = nodes[i + 1]
            
            # Check if we need router between avatar and group
            source_is_avatar = source not in self.groups
            target_is_group = target in self.groups
            
            if source_is_avatar and target_is_group and self.router_address:
                # Insert router: avatar -> router -> group
                edges.append({"source": source, "target": self.router_address})
                edges.append({"source": self.router_address, "target": target})
            else:
                edges.append({"source": source, "target": target})
        
        return edges
    
    def get_simplified_nodes(self, path: FlowPath) -> List[str]:
        """Get just the simplified node addresses."""
        simplified = []
        
        for node_id in path.nodes:
            if node_id.startswith(TOKEN_POOL_PREFIX):
                continue
            if node_id.startswith(VIRTUAL_SINK_PREFIX):
                continue
            
            address = self.mapper.get_address(node_id)
            if address:
                simplified.append(address)
        
        return simplified


def simplify_paths(
    paths: List[FlowPath],
    mapper: AddressMapper,
    router_address: Optional[str] = None,
    groups: Optional[Set[str]] = None
) -> List[FlowPath]:
    """
    Convenience function to simplify paths.
    
    Args:
        paths: Raw flow paths
        mapper: Address mapper
        router_address: Router address for insertion
        groups: Set of group addresses
        
    Returns:
        Simplified paths
    """
    simplifier = PathSimplifier(mapper, router_address, groups)
    return simplifier.simplify(paths)