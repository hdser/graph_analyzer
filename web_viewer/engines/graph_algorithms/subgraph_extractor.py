"""
Subgraph Extractor

Extract subgraphs from a graph: neighborhoods, ego graphs,
induced subgraphs, and connected components.

Location: web_viewer/engines/graph_algorithms/subgraph_extractor.py
"""

import time
from typing import List, Dict, Any, Optional, Set

import networkx as nx


class SubgraphExtractor:
    """Extract subgraphs from a graph."""
    
    MAX_HOPS = 5
    
    def __init__(self, graph: nx.DiGraph):
        self.G = graph
        self._U = None
    
    @property
    def undirected(self) -> nx.Graph:
        """Lazy undirected graph."""
        if self._U is None:
            self._U = self.G.to_undirected()
        return self._U
    
    def _get_graph(self, directed: bool):
        """Get directed or undirected graph."""
        return self.G if directed else self.undirected
    
    def _subgraph_to_dict(
        self,
        nodes: Set[str],
        graph: nx.Graph,
        start_time: float,
        center: Optional[str] = None,
        mode: str = ""
    ) -> Dict[str, Any]:
        """Convert subgraph nodes to result dict with edges."""
        node_list = list(nodes)
        
        # Get edges within subgraph
        edges = []
        for u, v in graph.edges():
            if u in nodes and v in nodes:
                edges.append({"source": u, "target": v})
        
        return {
            "nodes": node_list,
            "edges": edges,
            "node_count": len(node_list),
            "edge_count": len(edges),
            "center": center,
            "mode": mode,
            "computation_time_ms": (time.time() - start_time) * 1000,
            "success": True,
        }
    
    def neighborhood(
        self,
        node_id: str,
        hops: int = 1,
        directed: bool = False
    ) -> Dict[str, Any]:
        """
        Extract N-hop neighborhood around a node.
        
        Args:
            node_id: Center node ID
            hops: Number of hops (radius)
            directed: Use directed edges
            
        Returns:
            Result dict with neighborhood nodes and edges
        """
        start = time.time()
        graph = self._get_graph(directed)
        hops = min(hops, self.MAX_HOPS)
        
        if node_id not in graph:
            return {
                "nodes": [],
                "edges": [],
                "node_count": 0,
                "edge_count": 0,
                "center": node_id,
                "mode": "neighborhood",
                "computation_time_ms": (time.time() - start) * 1000,
                "success": False,
                "message": f"Node '{node_id}' not found",
            }
        
        # BFS to find nodes within N hops
        nodes = {node_id}
        frontier = {node_id}
        
        for _ in range(hops):
            next_frontier = set()
            for n in frontier:
                next_frontier.update(graph.neighbors(n))
            nodes.update(next_frontier)
            frontier = next_frontier - nodes | next_frontier
        
        return self._subgraph_to_dict(nodes, graph, start, node_id, "neighborhood")
    
    def ego_graph(
        self,
        node_id: str,
        radius: int = 1,
        directed: bool = False
    ) -> Dict[str, Any]:
        """
        Extract ego graph centered on a node.
        
        Args:
            node_id: Center node ID
            radius: Ego graph radius
            directed: Use directed edges
            
        Returns:
            Result dict with ego graph nodes and edges
        """
        start = time.time()
        graph = self._get_graph(directed)
        radius = min(radius, self.MAX_HOPS)
        
        if node_id not in graph:
            return {
                "nodes": [],
                "edges": [],
                "node_count": 0,
                "edge_count": 0,
                "center": node_id,
                "mode": "ego",
                "computation_time_ms": (time.time() - start) * 1000,
                "success": False,
                "message": f"Node '{node_id}' not found",
            }
        
        ego = nx.ego_graph(graph, node_id, radius=radius)
        nodes = set(ego.nodes())
        
        return self._subgraph_to_dict(nodes, graph, start, node_id, "ego")
    
    def induced_subgraph(
        self,
        node_ids: List[str],
        directed: bool = True
    ) -> Dict[str, Any]:
        """
        Extract induced subgraph from a set of nodes.
        
        Args:
            node_ids: List of node IDs to include
            directed: Use directed edges
            
        Returns:
            Result dict with induced subgraph
        """
        start = time.time()
        graph = self._get_graph(directed)
        
        # Filter to existing nodes
        valid_nodes = set(n for n in node_ids if n in graph)
        
        if not valid_nodes:
            return {
                "nodes": [],
                "edges": [],
                "node_count": 0,
                "edge_count": 0,
                "mode": "induced",
                "computation_time_ms": (time.time() - start) * 1000,
                "success": False,
                "message": "No valid nodes found",
            }
        
        return self._subgraph_to_dict(valid_nodes, graph, start, mode="induced")
    
    def connected_component(
        self,
        node_id: str,
        directed: bool = False
    ) -> Dict[str, Any]:
        """
        Extract the connected component containing a node.
        
        Args:
            node_id: Node ID
            directed: If True, find weakly connected component
            
        Returns:
            Result dict with component nodes and edges
        """
        start = time.time()
        graph = self._get_graph(directed)
        
        if node_id not in graph:
            return {
                "nodes": [],
                "edges": [],
                "node_count": 0,
                "edge_count": 0,
                "center": node_id,
                "mode": "component",
                "computation_time_ms": (time.time() - start) * 1000,
                "success": False,
                "message": f"Node '{node_id}' not found",
            }
        
        if directed:
            # Find weakly connected component for directed graphs
            for component in nx.weakly_connected_components(self.G):
                if node_id in component:
                    nodes = component
                    break
        else:
            nodes = nx.node_connected_component(graph, node_id)
        
        return self._subgraph_to_dict(set(nodes), graph, start, node_id, "component")
    
    def k_hop_subgraph(
        self,
        node_ids: List[str],
        hops: int = 1,
        directed: bool = False
    ) -> Dict[str, Any]:
        """
        Extract subgraph within K hops of any node in the set.
        
        Args:
            node_ids: List of center node IDs
            hops: Number of hops
            directed: Use directed edges
            
        Returns:
            Result dict with k-hop subgraph
        """
        start = time.time()
        graph = self._get_graph(directed)
        hops = min(hops, self.MAX_HOPS)
        
        # Start with given nodes
        valid_seeds = set(n for n in node_ids if n in graph)
        
        if not valid_seeds:
            return {
                "nodes": [],
                "edges": [],
                "node_count": 0,
                "edge_count": 0,
                "mode": "k_hop",
                "computation_time_ms": (time.time() - start) * 1000,
                "success": False,
                "message": "No valid seed nodes found",
            }
        
        # BFS from all seeds
        nodes = set(valid_seeds)
        frontier = set(valid_seeds)
        
        for _ in range(hops):
            next_frontier = set()
            for n in frontier:
                next_frontier.update(graph.neighbors(n))
            nodes.update(next_frontier)
            frontier = next_frontier
        
        return self._subgraph_to_dict(nodes, graph, start, mode="k_hop")