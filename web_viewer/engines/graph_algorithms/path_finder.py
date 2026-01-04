"""
Path Finder

Find paths between nodes using various algorithms.

Location: web_viewer/engines/graph_algorithms/path_finder.py
"""

import time
from typing import List, Dict, Any, Optional

import networkx as nx


class PathFinder:
    """Find paths between nodes in a graph."""
    
    MAX_PATHS = 1000  # Increased from 100
    MAX_CUTOFF = 20   # Increased from 15
    
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
    
    def _build_path_result(
        self, 
        path: List[str], 
        index: int = 0,
        weight_attr: Optional[str] = None
    ) -> Dict[str, Any]:
        """Convert node list to path result dict."""
        if not path:
            return {"nodes": [], "edges": [], "length": 0, "weight": 0}
        
        edges = []
        weight = 0.0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edges.append({"source": u, "target": v})
            
            if weight_attr and self.G.has_edge(u, v):
                weight += self.G[u][v].get(weight_attr, 1.0)
            else:
                weight += 1.0
        
        return {
            "nodes": path,
            "edges": edges,
            "length": len(path) - 1,
            "weight": weight,
            "index": index,
        }
    
    def _result(
        self,
        source: str,
        target: str,
        algorithm: str,
        paths: List[Dict],
        start_time: float,
        message: str = "",
        truncated: bool = False,
        success: bool = True
    ) -> Dict[str, Any]:
        """Build standard result dict."""
        return {
            "source": source,
            "target": target,
            "algorithm": algorithm,
            "paths": paths,
            "path_count": len(paths),
            "computation_time_ms": (time.time() - start_time) * 1000,
            "message": message,
            "truncated": truncated,
            "success": success,
        }
    
    def shortest_path(
        self,
        source: str,
        target: str,
        directed: bool = True
    ) -> Dict[str, Any]:
        """
        Find single shortest path (unweighted BFS).
        
        Args:
            source: Source node ID
            target: Target node ID
            directed: Use directed edges
            
        Returns:
            Result dict with path
        """
        start = time.time()
        graph = self._get_graph(directed)
        
        # Debug: Check if node exists and show sample nodes
        if source not in graph:
            # Get sample of actual node IDs for debugging
            sample_nodes = list(graph.nodes())[:5]
            # Check for similar nodes (case-insensitive match)
            source_lower = source.lower()
            similar = [n for n in list(graph.nodes())[:1000] if str(n).lower() == source_lower]
            extra_info = f" Similar: {similar}" if similar else ""
            return self._result(source, target, "shortest_path", [], start,
                              f"Source '{source}' not found. Graph has {graph.number_of_nodes()} nodes. Sample: {sample_nodes}.{extra_info}", 
                              success=False)
        if target not in graph:
            return self._result(source, target, "shortest_path", [], start,
                              f"Target '{target}' not found", success=False)
        
        try:
            path = nx.shortest_path(graph, source, target)
            paths = [self._build_path_result(path)]
            return self._result(source, target, "shortest_path", paths, start,
                              f"Found path of length {len(path) - 1}")
        except nx.NetworkXNoPath:
            return self._result(source, target, "shortest_path", [], start,
                              f"No path exists between {source} and {target}")
    
    def all_shortest_paths(
        self,
        source: str,
        target: str,
        directed: bool = True,
        max_paths: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Find all shortest paths of equal length.
        
        Args:
            source: Source node ID
            target: Target node ID
            directed: Use directed edges
            max_paths: Maximum paths to return
            
        Returns:
            Result dict with all shortest paths
        """
        start = time.time()
        graph = self._get_graph(directed)
        max_paths = max_paths or self.MAX_PATHS
        
        if source not in graph:
            return self._result(source, target, "all_shortest_paths", [], start,
                              f"Source '{source}' not found", success=False)
        if target not in graph:
            return self._result(source, target, "all_shortest_paths", [], start,
                              f"Target '{target}' not found", success=False)
        
        try:
            all_paths = list(nx.all_shortest_paths(graph, source, target))
            truncated = len(all_paths) > max_paths
            
            if truncated:
                all_paths = all_paths[:max_paths]
            
            paths = [self._build_path_result(p, i) for i, p in enumerate(all_paths)]
            msg = f"Found {len(paths)} shortest paths"
            if truncated:
                msg += f" (limited to {max_paths})"
            
            return self._result(source, target, "all_shortest_paths", paths, start,
                              msg, truncated=truncated)
        except nx.NetworkXNoPath:
            return self._result(source, target, "all_shortest_paths", [], start,
                              f"No path exists between {source} and {target}")
    
    def k_shortest_paths(
        self,
        source: str,
        target: str,
        k: int = 5,
        directed: bool = True,
        weight: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find k shortest simple paths (Yen's algorithm).
        
        Args:
            source: Source node ID
            target: Target node ID
            k: Number of paths to find
            directed: Use directed edges
            weight: Edge weight attribute
            
        Returns:
            Result dict with k shortest paths
        """
        start = time.time()
        graph = self._get_graph(directed)
        
        if source not in graph:
            return self._result(source, target, "k_shortest_paths", [], start,
                              f"Source '{source}' not found", success=False)
        if target not in graph:
            return self._result(source, target, "k_shortest_paths", [], start,
                              f"Target '{target}' not found", success=False)
        
        try:
            gen = nx.shortest_simple_paths(graph, source, target, weight=weight)
            all_paths = []
            
            for i, path in enumerate(gen):
                if i >= k:
                    break
                all_paths.append(path)
            
            paths = [self._build_path_result(p, i, weight) for i, p in enumerate(all_paths)]
            return self._result(source, target, "k_shortest_paths", paths, start,
                              f"Found {len(paths)} shortest paths")
        except nx.NetworkXNoPath:
            return self._result(source, target, "k_shortest_paths", [], start,
                              f"No path exists between {source} and {target}")
    
    def all_simple_paths(
        self,
        source: str,
        target: str,
        cutoff: Optional[int] = None,
        directed: bool = True,
        max_paths: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Find all simple paths up to a cutoff length.
        
        Args:
            source: Source node ID
            target: Target node ID
            cutoff: Maximum path length
            directed: Use directed edges
            max_paths: Maximum paths to return
            
        Returns:
            Result dict with all simple paths
        """
        start = time.time()
        graph = self._get_graph(directed)
        cutoff = min(cutoff or self.MAX_CUTOFF, self.MAX_CUTOFF)
        max_paths = max_paths or self.MAX_PATHS
        
        if source not in graph:
            return self._result(source, target, "all_simple_paths", [], start,
                              f"Source '{source}' not found", success=False)
        if target not in graph:
            return self._result(source, target, "all_simple_paths", [], start,
                              f"Target '{target}' not found", success=False)
        
        try:
            gen = nx.all_simple_paths(graph, source, target, cutoff=cutoff)
            all_paths = []
            truncated = False
            
            for i, path in enumerate(gen):
                if i >= max_paths:
                    truncated = True
                    break
                all_paths.append(path)
            
            paths = [self._build_path_result(p, i) for i, p in enumerate(all_paths)]
            msg = f"Found {len(paths)} paths (cutoff={cutoff})"
            if truncated:
                msg += f" (limited to {max_paths})"
            
            return self._result(source, target, "all_simple_paths", paths, start,
                              msg, truncated=truncated)
        except nx.NetworkXNoPath:
            return self._result(source, target, "all_simple_paths", [], start,
                              f"No path exists between {source} and {target}")
    
    def dijkstra(
        self,
        source: str,
        target: str,
        weight: str = "weight",
        directed: bool = True
    ) -> Dict[str, Any]:
        """
        Find weighted shortest path using Dijkstra's algorithm.
        
        Args:
            source: Source node ID
            target: Target node ID
            weight: Edge weight attribute
            directed: Use directed edges
            
        Returns:
            Result dict with weighted shortest path
        """
        start = time.time()
        graph = self._get_graph(directed)
        
        if source not in graph:
            return self._result(source, target, "dijkstra", [], start,
                              f"Source '{source}' not found", success=False)
        if target not in graph:
            return self._result(source, target, "dijkstra", [], start,
                              f"Target '{target}' not found", success=False)
        
        try:
            path = nx.dijkstra_path(graph, source, target, weight=weight)
            paths = [self._build_path_result(path, 0, weight)]
            return self._result(source, target, "dijkstra", paths, start,
                              f"Found path with weight {paths[0]['weight']:.2f}")
        except nx.NetworkXNoPath:
            return self._result(source, target, "dijkstra", [], start,
                              f"No path exists between {source} and {target}")
    
    def node_disjoint_paths(
        self,
        source: str,
        target: str,
        directed: bool = True
    ) -> Dict[str, Any]:
        """
        Find node-disjoint paths (no shared intermediate nodes).
        
        Args:
            source: Source node ID
            target: Target node ID
            directed: Use directed edges
            
        Returns:
            Result dict with node-disjoint paths
        """
        start = time.time()
        graph = self._get_graph(directed)
        
        if source not in graph:
            return self._result(source, target, "node_disjoint_paths", [], start,
                              f"Source '{source}' not found", success=False)
        if target not in graph:
            return self._result(source, target, "node_disjoint_paths", [], start,
                              f"Target '{target}' not found", success=False)
        
        try:
            all_paths = list(nx.node_disjoint_paths(graph, source, target))
            paths = [self._build_path_result(p, i) for i, p in enumerate(all_paths)]
            return self._result(source, target, "node_disjoint_paths", paths, start,
                              f"Found {len(paths)} node-disjoint paths")
        except nx.NetworkXNoPath:
            return self._result(source, target, "node_disjoint_paths", [], start,
                              f"No path exists between {source} and {target}")
    
    def edge_disjoint_paths(
        self,
        source: str,
        target: str,
        directed: bool = True
    ) -> Dict[str, Any]:
        """
        Find edge-disjoint paths (no shared edges).
        
        Args:
            source: Source node ID
            target: Target node ID
            directed: Use directed edges
            
        Returns:
            Result dict with edge-disjoint paths
        """
        start = time.time()
        graph = self._get_graph(directed)
        
        if source not in graph:
            return self._result(source, target, "edge_disjoint_paths", [], start,
                              f"Source '{source}' not found", success=False)
        if target not in graph:
            return self._result(source, target, "edge_disjoint_paths", [], start,
                              f"Target '{target}' not found", success=False)
        
        try:
            all_paths = list(nx.edge_disjoint_paths(graph, source, target))
            paths = [self._build_path_result(p, i) for i, p in enumerate(all_paths)]
            return self._result(source, target, "edge_disjoint_paths", paths, start,
                              f"Found {len(paths)} edge-disjoint paths")
        except nx.NetworkXNoPath:
            return self._result(source, target, "edge_disjoint_paths", [], start,
                              f"No path exists between {source} and {target}")
    
    def path_exists(
        self,
        source: str,
        target: str,
        directed: bool = True
    ) -> Dict[str, Any]:
        """
        Check if a path exists between two nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            directed: Use directed edges
            
        Returns:
            Result dict with exists boolean
        """
        start = time.time()
        graph = self._get_graph(directed)
        
        if source not in graph or target not in graph:
            return {
                "source": source,
                "target": target,
                "exists": False,
                "computation_time_ms": (time.time() - start) * 1000,
            }
        
        exists = nx.has_path(graph, source, target)
        return {
            "source": source,
            "target": target,
            "exists": exists,
            "computation_time_ms": (time.time() - start) * 1000,
        }
    
    def shortest_path_length(
        self,
        source: str,
        target: str,
        directed: bool = True,
        weight: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get shortest path length without computing full path.
        
        Args:
            source: Source node ID
            target: Target node ID
            directed: Use directed edges
            weight: Edge weight attribute (for weighted length)
            
        Returns:
            Result dict with length
        """
        start = time.time()
        graph = self._get_graph(directed)
        
        if source not in graph or target not in graph:
            return {
                "source": source,
                "target": target,
                "length": None,
                "exists": False,
                "computation_time_ms": (time.time() - start) * 1000,
            }
        
        try:
            if weight:
                length = nx.dijkstra_path_length(graph, source, target, weight=weight)
            else:
                length = nx.shortest_path_length(graph, source, target)
            
            return {
                "source": source,
                "target": target,
                "length": length,
                "exists": True,
                "weighted": weight is not None,
                "computation_time_ms": (time.time() - start) * 1000,
            }
        except nx.NetworkXNoPath:
            return {
                "source": source,
                "target": target,
                "length": None,
                "exists": False,
                "computation_time_ms": (time.time() - start) * 1000,
            }