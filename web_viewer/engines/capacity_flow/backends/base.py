"""
Base Flow Backend

Abstract base class for max flow computation backends.

Location: web_viewer/engines/capacity_flow/backends/base.py
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any


class BaseFlowBackend(ABC):
    """
    Abstract base class for flow computation backends.
    
    Each backend implements max flow using different algorithms
    or libraries (NetworkX, OR-Tools, etc.).
    """
    
    name: str = "base"
    DEFAULT_ALGORITHM: str = "default"
    
    @abstractmethod
    def compute_max_flow(
        self,
        edges: List[Tuple[str, str, int]],
        source: str,
        sink: str,
        algorithm: Optional[str] = None,
        cutoff: Optional[int] = None
    ) -> Tuple[int, Dict[str, Dict[str, int]]]:
        """
        Compute maximum flow from source to sink.
        
        Args:
            edges: List of (source, target, capacity) tuples
            source: Source node ID
            sink: Sink node ID
            algorithm: Algorithm name (backend-specific)
            cutoff: Maximum flow cutoff
            
        Returns:
            Tuple of (flow_value, flow_dict)
            where flow_dict[u][v] = flow on edge (u, v)
        """
        pass
    
    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check if this backend is available."""
        pass
    
    @classmethod
    @abstractmethod
    def supported_algorithms(cls) -> List[str]:
        """Return list of supported algorithms."""
        pass
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get backend information."""
        return {
            "name": cls.name,
            "available": cls.is_available(),
            "default_algorithm": cls.DEFAULT_ALGORITHM,
            "algorithms": cls.supported_algorithms() if cls.is_available() else [],
        }