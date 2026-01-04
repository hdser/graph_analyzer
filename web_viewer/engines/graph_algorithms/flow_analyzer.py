"""
Flow Analyzer

Network flow analysis: maximum flow and minimum cut.

Location: web_viewer/engines/graph_algorithms/flow_analyzer.py
"""

import time
from typing import Dict, Any, Optional

import networkx as nx


class FlowAnalyzer:
    """Analyze network flow between nodes."""
    
    def __init__(self, graph: nx.DiGraph):
        self.G = graph
    
    def maximum_flow(
        self,
        source: str,
        target: str,
        capacity: str = "weight"
    ) -> Dict[str, Any]:
        """
        Compute maximum flow between source and target.
        
        Args:
            source: Source node ID
            target: Target node ID (sink)
            capacity: Edge attribute for capacity (empty string = unit capacity)
            
        Returns:
            Result dict with flow value and flow edges
        """
        start = time.time()
        
        if source not in self.G:
            return {
                "source": source,
                "target": target,
                "flow_value": 0,
                "flow_edges": [],
                "computation_time_ms": (time.time() - start) * 1000,
                "success": False,
                "message": f"Source '{source}' not found",
            }
        
        if target not in self.G:
            return {
                "source": source,
                "target": target,
                "flow_value": 0,
                "flow_edges": [],
                "computation_time_ms": (time.time() - start) * 1000,
                "success": False,
                "message": f"Target '{target}' not found",
            }
        
        try:
            # ALWAYS create a copy with explicit capacities to avoid infinite capacity issues
            # NetworkX treats missing capacity attribute as INFINITE, not 1
            G_flow = nx.DiGraph()
            G_flow.add_nodes_from(self.G.nodes())
            
            use_attr = capacity and capacity.strip()
            
            for u, v, data in self.G.edges(data=True):
                if use_attr and capacity in data:
                    # Use the specified attribute value
                    cap_value = float(data[capacity])
                    if cap_value <= 0:
                        cap_value = 1.0  # Ensure positive capacity
                else:
                    # Default to unit capacity
                    cap_value = 1.0
                G_flow.add_edge(u, v, capacity=cap_value)
            
            flow_value, flow_dict = nx.maximum_flow(
                G_flow, source, target, capacity='capacity'
            )
            
            # Build flow edges list
            flow_edges = []
            for u, targets in flow_dict.items():
                for v, flow in targets.items():
                    if flow > 0:
                        flow_edges.append({
                            "source": u,
                            "target": v,
                            "flow": flow,
                        })
            
            cap_msg = f" (using '{capacity}')" if use_attr else " (unit capacity)"
            return {
                "source": source,
                "target": target,
                "flow_value": flow_value,
                "flow_edges": flow_edges,
                "computation_time_ms": (time.time() - start) * 1000,
                "success": True,
                "message": f"Maximum flow: {flow_value}{cap_msg}",
            }
            
        except nx.NetworkXUnbounded:
            return {
                "source": source,
                "target": target,
                "flow_value": float('inf'),
                "flow_edges": [],
                "computation_time_ms": (time.time() - start) * 1000,
                "success": False,
                "message": "Flow is unbounded (infinite capacity path exists)",
            }
        except nx.NetworkXError as e:
            return {
                "source": source,
                "target": target,
                "flow_value": 0,
                "flow_edges": [],
                "computation_time_ms": (time.time() - start) * 1000,
                "success": False,
                "message": str(e),
            }
        except Exception as e:
            return {
                "source": source,
                "target": target,
                "flow_value": 0,
                "flow_edges": [],
                "computation_time_ms": (time.time() - start) * 1000,
                "success": False,
                "message": f"Error: {str(e)}",
            }
    
    def minimum_cut(
        self,
        source: str,
        target: str,
        capacity: str = "weight"
    ) -> Dict[str, Any]:
        """
        Find minimum cut between source and target.
        
        Args:
            source: Source node ID
            target: Target node ID
            capacity: Edge attribute for capacity (empty string = unit capacity)
            
        Returns:
            Result dict with cut value and cut edges
        """
        start = time.time()
        
        if source not in self.G:
            return {
                "source": source,
                "target": target,
                "cut_value": 0,
                "cut_edges": [],
                "partition": [[], []],
                "computation_time_ms": (time.time() - start) * 1000,
                "success": False,
                "message": f"Source '{source}' not found",
            }
        
        if target not in self.G:
            return {
                "source": source,
                "target": target,
                "cut_value": 0,
                "cut_edges": [],
                "partition": [[], []],
                "computation_time_ms": (time.time() - start) * 1000,
                "success": False,
                "message": f"Target '{target}' not found",
            }
        
        try:
            # ALWAYS create a copy with explicit capacities to avoid infinite capacity issues
            G_flow = nx.DiGraph()
            G_flow.add_nodes_from(self.G.nodes())
            
            use_attr = capacity and capacity.strip()
            
            for u, v, data in self.G.edges(data=True):
                if use_attr and capacity in data:
                    cap_value = float(data[capacity])
                    if cap_value <= 0:
                        cap_value = 1.0
                else:
                    cap_value = 1.0
                G_flow.add_edge(u, v, capacity=cap_value)
            
            cut_value, partition = nx.minimum_cut(
                G_flow, source, target, capacity='capacity'
            )
            
            reachable, non_reachable = partition
            
            # Find cut edges (edges from reachable to non-reachable)
            cut_edges = []
            for u in reachable:
                for v in self.G.successors(u):
                    if v in non_reachable:
                        cut_edges.append({"source": u, "target": v})
            
            cap_msg = f" (using '{capacity}')" if use_attr else " (unit capacity)"
            return {
                "source": source,
                "target": target,
                "cut_value": cut_value,
                "cut_edges": cut_edges,
                "partition": [list(reachable), list(non_reachable)],
                "computation_time_ms": (time.time() - start) * 1000,
                "success": True,
                "message": f"Minimum cut: {cut_value} ({len(cut_edges)} edges){cap_msg}",
            }
            
        except nx.NetworkXUnbounded:
            return {
                "source": source,
                "target": target,
                "cut_value": float('inf'),
                "cut_edges": [],
                "partition": [[], []],
                "computation_time_ms": (time.time() - start) * 1000,
                "success": False,
                "message": "Cut is unbounded (infinite capacity path exists)",
            }
        except nx.NetworkXError as e:
            return {
                "source": source,
                "target": target,
                "cut_value": 0,
                "cut_edges": [],
                "partition": [[], []],
                "computation_time_ms": (time.time() - start) * 1000,
                "success": False,
                "message": str(e),
            }
        except Exception as e:
            return {
                "source": source,
                "target": target,
                "cut_value": 0,
                "cut_edges": [],
                "partition": [[], []],
                "computation_time_ms": (time.time() - start) * 1000,
                "success": False,
                "message": f"Error: {str(e)}",
            }
    
    def edge_connectivity(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute edge connectivity between nodes or for entire graph.
        
        Args:
            source: Source node ID (optional)
            target: Target node ID (optional)
            
        Returns:
            Result dict with connectivity value
        """
        start = time.time()
        
        try:
            if source and target:
                if source not in self.G or target not in self.G:
                    return {
                        "source": source,
                        "target": target,
                        "connectivity": 0,
                        "computation_time_ms": (time.time() - start) * 1000,
                        "success": False,
                        "message": "Source or target not found",
                    }
                connectivity = nx.edge_connectivity(self.G, source, target)
            else:
                connectivity = nx.edge_connectivity(self.G)
            
            return {
                "source": source,
                "target": target,
                "connectivity": connectivity,
                "computation_time_ms": (time.time() - start) * 1000,
                "success": True,
            }
            
        except nx.NetworkXError as e:
            return {
                "source": source,
                "target": target,
                "connectivity": 0,
                "computation_time_ms": (time.time() - start) * 1000,
                "success": False,
                "message": str(e),
            }
    
    def node_connectivity(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute node connectivity between nodes or for entire graph.
        
        Args:
            source: Source node ID (optional)
            target: Target node ID (optional)
            
        Returns:
            Result dict with connectivity value
        """
        start = time.time()
        
        try:
            if source and target:
                if source not in self.G or target not in self.G:
                    return {
                        "source": source,
                        "target": target,
                        "connectivity": 0,
                        "computation_time_ms": (time.time() - start) * 1000,
                        "success": False,
                        "message": "Source or target not found",
                    }
                connectivity = nx.node_connectivity(self.G, source, target)
            else:
                connectivity = nx.node_connectivity(self.G)
            
            return {
                "source": source,
                "target": target,
                "connectivity": connectivity,
                "computation_time_ms": (time.time() - start) * 1000,
                "success": True,
            }
            
        except nx.NetworkXError as e:
            return {
                "source": source,
                "target": target,
                "connectivity": 0,
                "computation_time_ms": (time.time() - start) * 1000,
                "success": False,
                "message": str(e),
            }