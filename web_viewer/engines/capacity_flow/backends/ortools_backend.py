"""
OR-Tools Flow Backend

Max flow computation using Google OR-Tools.

Location: web_viewer/engines/capacity_flow/backends/ortools_backend.py
"""
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from .base import BaseFlowBackend

logger = logging.getLogger(__name__)

# Try to import OR-Tools
try:
    from ortools.graph.python import max_flow
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False
    logger.debug("OR-Tools not available")


class ORToolsBackend(BaseFlowBackend):
    """
    OR-Tools based max flow backend.
    
    Uses Google's optimized SimpleMaxFlow solver.
    Generally faster than NetworkX for large graphs.
    """
    
    name = "ortools"
    DEFAULT_ALGORITHM = "simple_max_flow"
    
    # Maximum capacity to avoid overflow
    MAX_CAPACITY = 2**62
    
    def compute_max_flow(
        self,
        edges: List[Tuple[str, str, int]],
        source: str,
        sink: str,
        algorithm: Optional[str] = None,
        cutoff: Optional[int] = None
    ) -> Tuple[int, Dict[str, Dict[str, int]]]:
        """Compute max flow using OR-Tools."""
        
        if not HAS_ORTOOLS:
            logger.error("OR-Tools not installed")
            return 0, {}
        
        # Create node index mapping
        node_to_idx: Dict[str, int] = {}
        idx_to_node: Dict[int, str] = {}
        next_idx = 0
        
        def get_node_idx(node: str) -> int:
            nonlocal next_idx
            if node not in node_to_idx:
                node_to_idx[node] = next_idx
                idx_to_node[next_idx] = node
                next_idx += 1
            return node_to_idx[node]
        
        # Ensure source and sink are indexed
        source_idx = get_node_idx(source)
        sink_idx = get_node_idx(sink)
        
        # Create solver
        smf = max_flow.SimpleMaxFlow()
        
        # Add edges
        edge_indices = []
        for src, tgt, cap in edges:
            src_idx = get_node_idx(src)
            tgt_idx = get_node_idx(tgt)
            
            # Clamp capacity
            clamped_cap = min(cap, self.MAX_CAPACITY)
            
            arc_idx = smf.add_arc_with_capacity(src_idx, tgt_idx, clamped_cap)
            edge_indices.append((arc_idx, src, tgt))
        
        # Solve
        status = smf.solve(source_idx, sink_idx)
        
        if status != smf.OPTIMAL:
            logger.warning(f"OR-Tools solver status: {status}")
            return 0, {}
        
        # Extract flow
        flow_value = smf.optimal_flow()
        
        # Build flow dict
        flow_dict: Dict[str, Dict[str, int]] = defaultdict(dict)
        
        for arc_idx, src, tgt in edge_indices:
            flow = smf.flow(arc_idx)
            if flow > 0:
                flow_dict[src][tgt] = int(flow)
        
        # Apply cutoff if specified
        if cutoff is not None and flow_value > cutoff:
            flow_value = cutoff
        
        return int(flow_value), dict(flow_dict)
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if OR-Tools is installed."""
        return HAS_ORTOOLS
    
    @classmethod
    def supported_algorithms(cls) -> List[str]:
        """Return supported algorithms."""
        return ["simple_max_flow"]