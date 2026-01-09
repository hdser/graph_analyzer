"""
Address Mapper

Bidirectional mapping between blockchain addresses and integer IDs.

Location: web_viewer/engines/capacity_flow/address_mapper.py
"""
from typing import Dict, Optional, Set


# Special prefixes for synthetic nodes
TOKEN_POOL_PREFIX = "tpool-"
VIRTUAL_SINK_PREFIX = "vsink-"


class AddressMapper:
    """
    Maps blockchain addresses to sequential integer IDs and back.
    
    Integer IDs are more efficient for graph algorithms, while
    addresses are needed for display and API responses.
    """
    
    def __init__(self):
        self._address_to_id: Dict[str, str] = {}
        self._id_to_address: Dict[str, str] = {}
        self._next_id: int = 0
        self._token_pool_counter: int = 0
        self._virtual_sink_counter: int = 0
    
    def get_or_create_id(self, address: str) -> str:
        """Get existing ID or create new one for address."""
        if address in self._address_to_id:
            return self._address_to_id[address]
        
        node_id = str(self._next_id)
        self._next_id += 1
        
        self._address_to_id[address] = node_id
        self._id_to_address[node_id] = address
        
        return node_id
    
    def create_token_pool_id(self, token_address: str) -> str:
        """Create a token pool node ID for a token."""
        pool_id = f"{TOKEN_POOL_PREFIX}{self._token_pool_counter}"
        self._token_pool_counter += 1
        
        # Store mapping from pool_id to token address
        self._id_to_address[pool_id] = token_address
        
        return pool_id
    
    def create_virtual_sink_id(self, source_address: str) -> str:
        """Create a virtual sink node ID."""
        sink_id = f"{VIRTUAL_SINK_PREFIX}{self._virtual_sink_counter}"
        self._virtual_sink_counter += 1
        
        self._id_to_address[sink_id] = source_address
        
        return sink_id
    
    def get_id(self, address: str) -> Optional[str]:
        """Get ID for address, or None if not mapped."""
        return self._address_to_id.get(address)
    
    def get_address(self, node_id: str) -> Optional[str]:
        """Get address for ID, or None if not mapped."""
        return self._id_to_address.get(node_id)
    
    def has_address(self, address: str) -> bool:
        """Check if address is mapped."""
        return address in self._address_to_id
    
    def has_id(self, node_id: str) -> bool:
        """Check if ID exists."""
        return node_id in self._id_to_address
    
    @staticmethod
    def is_token_pool(node_id: str) -> bool:
        """Check if node ID is a token pool."""
        return node_id.startswith(TOKEN_POOL_PREFIX)
    
    @staticmethod
    def is_virtual_sink(node_id: str) -> bool:
        """Check if node ID is a virtual sink."""
        return node_id.startswith(VIRTUAL_SINK_PREFIX)
    
    @staticmethod
    def get_token_from_pool_id(pool_id: str) -> Optional[str]:
        """Extract token index from pool ID."""
        if pool_id.startswith(TOKEN_POOL_PREFIX):
            return pool_id[len(TOKEN_POOL_PREFIX):]
        return None
    
    def get_all_addresses(self) -> Set[str]:
        """Get all mapped addresses."""
        return set(self._address_to_id.keys())
    
    def get_all_ids(self) -> Set[str]:
        """Get all node IDs."""
        return set(self._id_to_address.keys())
    
    def clear(self) -> None:
        """Clear all mappings."""
        self._address_to_id.clear()
        self._id_to_address.clear()
        self._next_id = 0
        self._token_pool_counter = 0
        self._virtual_sink_counter = 0
    
    def __len__(self) -> int:
        return len(self._id_to_address)