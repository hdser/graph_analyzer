"""
Flow Decomposer

Decomposes max flow into individual paths.

Location: web_viewer/engines/capacity_flow/flow_decomposer.py
"""
import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

from .models import FlowPath, CapacityEdge

logger = logging.getLogger(__name__)


class FlowDecomposer:
    """
    Decomposes a flow into individual source-to-sink paths.
    
    Uses greedy path extraction: repeatedly find a path from
    source to sink with positive flow, extract minimum flow
    along the path, and update residual flow.
    """
    
    def __init__(self, max_paths: int = 1000, max_iterations: int = 10000):
        self.max_paths = max_paths
        self.max_iterations = max_iterations
    
    def decompose(
        self,
        flow_dict: Dict[str, Dict[str, int]],
        source: str,
        sink: str,
        edge_tokens: Optional[Dict[Tuple[str, str], str]] = None
    ) -> Tuple[List[FlowPath], Dict[str, int]]:
        """
        Decompose flow into paths.
        
        Args:
            flow_dict: Flow on each edge {u: {v: flow}}
            source: Source node
            sink: Sink node
            edge_tokens: Optional mapping of edges to token IDs
            
        Returns:
            Tuple of (paths, token_flows)
        """
        if not flow_dict:
            return [], {}
        
        edge_tokens = edge_tokens or {}
        
        # Create residual flow copy
        residual: Dict[str, Dict[str, int]] = defaultdict(dict)
        for u, targets in flow_dict.items():
            for v, flow in targets.items():
                if flow > 0:
                    residual[u][v] = flow
        
        paths: List[FlowPath] = []
        token_flows: Dict[str, int] = defaultdict(int)
        iterations = 0
        
        while len(paths) < self.max_paths and iterations < self.max_iterations:
            iterations += 1
            
            # Find path from source to sink
            path_result = self._find_path(residual, source, sink)
            if path_result is None:
                break
            
            path_nodes, min_flow = path_result
            
            if min_flow <= 0:
                break
            
            # Extract tokens along path
            tokens = []
            edges = []
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i + 1]
                edges.append({"source": u, "target": v})
                
                token = edge_tokens.get((u, v))
                if token:
                    tokens.append(token)
                    token_flows[token] += min_flow
            
            # Create path object
            flow_path = FlowPath(
                nodes=path_nodes,
                edges=edges,
                tokens=tokens,
                flow=min_flow,
            )
            paths.append(flow_path)
            
            # Update residual
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i + 1]
                residual[u][v] -= min_flow
                if residual[u][v] <= 0:
                    del residual[u][v]
                    if not residual[u]:
                        del residual[u]
        
        logger.debug(f"Decomposed into {len(paths)} paths in {iterations} iterations")
        
        return paths, dict(token_flows)
    
    def _find_path(
        self,
        residual: Dict[str, Dict[str, int]],
        source: str,
        sink: str
    ) -> Optional[Tuple[List[str], int]]:
        """
        Find path from source to sink using DFS.
        
        Returns:
            Tuple of (path_nodes, min_flow) or None
        """
        if source not in residual:
            return None
        
        # DFS to find path
        visited: Set[str] = set()
        parent: Dict[str, str] = {}
        stack = [source]
        
        while stack:
            node = stack.pop()
            
            if node == sink:
                # Reconstruct path
                path = [sink]
                current = sink
                while current != source:
                    current = parent[current]
                    path.append(current)
                path.reverse()
                
                # Find min flow along path
                min_flow = float('inf')
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    flow = residual.get(u, {}).get(v, 0)
                    min_flow = min(min_flow, flow)
                
                if min_flow == float('inf') or min_flow <= 0:
                    return None
                
                return path, int(min_flow)
            
            if node in visited:
                continue
            visited.add(node)
            
            for neighbor, flow in residual.get(node, {}).items():
                if flow > 0 and neighbor not in visited:
                    parent[neighbor] = node
                    stack.append(neighbor)
        
        return None


def decompose_flow(
    flow_dict: Dict[str, Dict[str, int]],
    source: str,
    sink: str,
    edge_tokens: Optional[Dict[Tuple[str, str], str]] = None,
    max_paths: int = 1000
) -> Tuple[List[FlowPath], Dict[str, int]]:
    """
    Convenience function to decompose flow.
    
    Args:
        flow_dict: Flow on each edge
        source: Source node
        sink: Sink node
        edge_tokens: Edge to token mapping
        max_paths: Maximum paths to extract
        
    Returns:
        Tuple of (paths, token_flows)
    """
    decomposer = FlowDecomposer(max_paths=max_paths)
    return decomposer.decompose(flow_dict, source, sink, edge_tokens)