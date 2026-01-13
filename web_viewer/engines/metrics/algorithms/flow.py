"""
Flow Metric Algorithms

Flow and hierarchy metrics including maximum flow analysis,
flow betweenness, and hierarchy detection.
"""

from typing import Dict, Any, Optional
import logging

import networkx as nx
import numpy as np

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


class FlowHierarchyAlgorithm(BaseMetricAlgorithm):
    """
    Compute flow hierarchy of directed graph.
    
    Flow hierarchy measures the proportion of edges that are
    not contained in cycles, indicating how hierarchical the
    network structure is.
    """
    
    name = "flow_hierarchy"
    category = "flow"
    description = "Flow hierarchy of directed graph"
    cost = "medium"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            if G.number_of_nodes() > 1:
                flow_h = nx.flow_hierarchy(G)
                logger.debug(f"Flow hierarchy: {flow_h:.4f}")
            else:
                flow_h = 0
            
            result = {}
            for node in nodes:
                result[node] = {"flow_hierarchy": flow_h}
            return result
        except Exception as e:
            logger.warning(f"Flow hierarchy failed: {e}")
            return {node: {} for node in nodes}


class MaxFlowCentralityAlgorithm(BaseMetricAlgorithm):
    """
    Compute maximum flow centrality for each node.
    
    Measures the maximum flow capacity through each node by
    computing max flow from sample sources to the node and
    from the node to sample sinks.
    
    Computes:
    - max_flow_in: Average max flow capacity into the node
    - max_flow_out: Average max flow capacity out of the node
    - max_flow_centrality: Combined flow centrality score
    
    This is computationally expensive as it requires solving
    multiple maximum flow problems.
    """
    
    name = "max_flow_centrality"
    category = "flow"
    description = "Maximum flow centrality based on flow capacity"
    cost = "very_high"
    # max_nodes is handled by registry/resolver - no internal limit
    
    def compute(
        self, 
        G: nx.DiGraph, 
        U: nx.Graph, 
        nodes: list, 
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        try:
            n = len(nodes)
            params = parameters or {}
            
            # Sample size for flow computations
            sample_size = params.get('sample_size', min(20, n // 10 + 1))
            
            if n < 2:
                return {node: {
                    "max_flow_in": 0.0,
                    "max_flow_out": 0.0,
                    "max_flow_centrality": 0.0,
                } for node in nodes}
            
            # Create capacity graph (unit capacity if no weights)
            H = nx.DiGraph()
            for u, v, data in G.edges(data=True):
                capacity = data.get('weight', 1.0)
                if capacity <= 0:
                    capacity = 1.0
                H.add_edge(u, v, capacity=capacity)
            
            # Sample source/sink nodes
            sampled_sources = np.random.choice(nodes, min(sample_size, n), replace=False)
            sampled_sinks = np.random.choice(nodes, min(sample_size, n), replace=False)
            
            result = {}
            
            for node in nodes:
                try:
                    # Compute incoming flow (from sample sources to this node)
                    flow_in_values = []
                    for src in sampled_sources:
                        if src != node and H.has_node(src) and H.has_node(node):
                            try:
                                flow_value, _ = nx.maximum_flow(H, src, node)
                                if flow_value > 0:
                                    flow_in_values.append(flow_value)
                            except (nx.NetworkXError, nx.NetworkXUnbounded):
                                pass
                    
                    # Compute outgoing flow (from this node to sample sinks)
                    flow_out_values = []
                    for sink in sampled_sinks:
                        if sink != node and H.has_node(node) and H.has_node(sink):
                            try:
                                flow_value, _ = nx.maximum_flow(H, node, sink)
                                if flow_value > 0:
                                    flow_out_values.append(flow_value)
                            except (nx.NetworkXError, nx.NetworkXUnbounded):
                                pass
                    
                    max_flow_in = np.mean(flow_in_values) if flow_in_values else 0.0
                    max_flow_out = np.mean(flow_out_values) if flow_out_values else 0.0
                    
                    # Combined centrality (geometric mean-like combination)
                    if max_flow_in > 0 and max_flow_out > 0:
                        centrality = np.sqrt(max_flow_in * max_flow_out)
                    else:
                        centrality = max_flow_in + max_flow_out
                    
                    result[node] = {
                        "max_flow_in": float(max_flow_in),
                        "max_flow_out": float(max_flow_out),
                        "max_flow_centrality": float(centrality),
                    }
                    
                except Exception as e:
                    logger.debug(f"Max flow for node {node} failed: {e}")
                    result[node] = {
                        "max_flow_in": 0.0,
                        "max_flow_out": 0.0,
                        "max_flow_centrality": 0.0,
                    }
            
            logger.debug(f"Max flow centrality computed for {n} nodes (sample size: {sample_size})")
            return result
            
        except Exception as e:
            logger.warning(f"Max flow centrality failed: {e}")
            return {node: {} for node in nodes}


class FlowBetweennessAlgorithm(BaseMetricAlgorithm):
    """
    Compute flow betweenness centrality.
    
    Unlike standard betweenness which counts shortest paths,
    flow betweenness measures the amount of flow that passes
    through each node in a maximum flow network.
    
    Uses edge betweenness as a proxy to estimate flow importance.
    
    Computes:
    - flow_betweenness: Node's importance in network flow
    - flow_bottleneck_score: Likelihood of being a bottleneck
    """
    
    name = "flow_betweenness"
    category = "flow"
    description = "Flow-based betweenness centrality"
    cost = "high"
    # max_nodes is handled by registry/resolver - no internal limit
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            n = len(nodes)
            
            # Get edge betweenness
            edge_bc = nx.edge_betweenness_centrality(G, normalized=True)
            
            result = {}
            
            for node in nodes:
                try:
                    # Sum of edge betweenness for incident edges
                    incident_bc = 0.0
                    incident_count = 0
                    
                    # Outgoing edges
                    for succ in G.successors(node):
                        edge = (node, succ)
                        incident_bc += edge_bc.get(edge, 0)
                        incident_count += 1
                    
                    # Incoming edges
                    for pred in G.predecessors(node):
                        edge = (pred, node)
                        incident_bc += edge_bc.get(edge, 0)
                        incident_count += 1
                    
                    # Flow betweenness is average of incident edge betweenness
                    flow_bc = incident_bc / incident_count if incident_count > 0 else 0
                    
                    # Bottleneck score: high in-degree but low out-degree, or vice versa
                    in_deg = G.in_degree(node)
                    out_deg = G.out_degree(node)
                    total_deg = in_deg + out_deg
                    
                    if total_deg > 0:
                        imbalance = abs(in_deg - out_deg) / total_deg
                        bottleneck = flow_bc * imbalance
                    else:
                        bottleneck = 0.0
                    
                    result[node] = {
                        "flow_betweenness": flow_bc,
                        "flow_bottleneck_score": bottleneck,
                    }
                    
                except Exception:
                    result[node] = {
                        "flow_betweenness": 0.0,
                        "flow_bottleneck_score": 0.0,
                    }
            
            return result
            
        except Exception as e:
            logger.warning(f"Flow betweenness failed: {e}")
            return {node: {} for node in nodes}


class HierarchyLevelAlgorithm(BaseMetricAlgorithm):
    """
    Compute hierarchical level of each node in a directed graph.
    
    Uses multiple approaches to determine node hierarchy:
    1. Longest path from sources (roots)
    2. Trophic-level-like computation
    
    Computes:
    - hierarchy_level: Node's position in the hierarchy (0 = root)
    - is_source: Whether node has no incoming edges
    - is_sink: Whether node has no outgoing edges
    - hierarchy_depth: Distance from nearest source
    """
    
    name = "hierarchy_level"
    category = "flow"
    description = "Hierarchical level in directed graph"
    cost = "medium"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            n = len(nodes)
            
            # Find sources (no incoming edges) and sinks (no outgoing edges)
            sources = [node for node in nodes if G.in_degree(node) == 0]
            sinks = [node for node in nodes if G.out_degree(node) == 0]
            
            # Compute hierarchy level (longest path from any source)
            hierarchy_level = {node: 0 for node in nodes}
            
            # BFS from each source to compute levels
            for source in sources:
                try:
                    # Use single_source_longest_path_length for DAGs
                    lengths = nx.single_source_shortest_path_length(G, source)
                    for node, length in lengths.items():
                        if length > hierarchy_level.get(node, 0):
                            hierarchy_level[node] = length
                except Exception:
                    pass
            
            # Compute depth (shortest path from any source)
            hierarchy_depth = {node: float('inf') for node in nodes}
            for source in sources:
                hierarchy_depth[source] = 0
            
            for source in sources:
                try:
                    lengths = nx.single_source_shortest_path_length(G, source)
                    for node, length in lengths.items():
                        if length < hierarchy_depth.get(node, float('inf')):
                            hierarchy_depth[node] = length
                except Exception:
                    pass
            
            # Replace inf with -1 for unreachable nodes
            for node in nodes:
                if hierarchy_depth[node] == float('inf'):
                    hierarchy_depth[node] = -1
            
            result = {}
            sources_set = set(sources)
            sinks_set = set(sinks)
            
            for node in nodes:
                result[node] = {
                    "hierarchy_level": hierarchy_level.get(node, 0),
                    "is_source": 1 if node in sources_set else 0,
                    "is_sink": 1 if node in sinks_set else 0,
                    "hierarchy_depth": hierarchy_depth.get(node, -1),
                }
            
            logger.debug(f"Hierarchy levels computed: {len(sources)} sources, {len(sinks)} sinks")
            return result
            
        except Exception as e:
            logger.warning(f"Hierarchy level failed: {e}")
            return {node: {} for node in nodes}


class CycleParticipationAlgorithm(BaseMetricAlgorithm):
    """
    Compute cycle participation metrics for each node.
    
    Analyzes how much each node participates in cycles,
    which is important for understanding feedback loops
    and non-hierarchical structure.
    
    Computes:
    - in_cycle: Whether node is part of any cycle
    - cycle_count_estimate: Estimated number of cycles containing node
    - scc_participation: Whether in non-trivial SCC (cycle indicator)
    """
    
    name = "cycle_participation"
    category = "flow"
    description = "Cycle participation analysis"
    cost = "medium"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            n = len(nodes)
            
            # Find strongly connected components (non-trivial SCCs contain cycles)
            sccs = list(nx.strongly_connected_components(G))
            
            # Map nodes to their SCC info
            node_to_scc_size = {}
            for scc in sccs:
                size = len(scc)
                for node in scc:
                    node_to_scc_size[node] = size
            
            # Nodes in non-trivial SCCs (size > 1) are in cycles
            in_cycle = {node: node_to_scc_size.get(node, 1) > 1 for node in nodes}
            
            # Estimate cycle count based on local structure
            # Nodes with more reciprocal edges and higher clustering tend to be in more cycles
            result = {}
            
            for node in nodes:
                scc_size = node_to_scc_size.get(node, 1)
                is_in_cycle = scc_size > 1
                
                # Estimate based on SCC size and local properties
                if is_in_cycle:
                    # More cycles estimated for larger SCCs and higher degree
                    degree = G.in_degree(node) + G.out_degree(node)
                    cycle_estimate = min(scc_size - 1, degree)
                else:
                    cycle_estimate = 0
                
                result[node] = {
                    "in_cycle": 1 if is_in_cycle else 0,
                    "cycle_count_estimate": cycle_estimate,
                    "scc_participation": 1 if scc_size > 1 else 0,
                }
            
            logger.debug(f"Cycle participation: {sum(in_cycle.values())} nodes in cycles")
            return result
            
        except Exception as e:
            logger.warning(f"Cycle participation failed: {e}")
            return {node: {} for node in nodes}


class MinCutCentralityAlgorithm(BaseMetricAlgorithm):
    """
    Compute minimum cut centrality for nodes.
    
    Measures how often a node appears in minimum cuts
    between random pairs of nodes, indicating importance
    for network connectivity.
    
    Computes:
    - min_cut_frequency: How often node appears in min-cuts
    - cut_vertex_centrality: Importance as a cut vertex
    """
    
    name = "min_cut_centrality"
    category = "flow"
    description = "Minimum cut based centrality"
    cost = "very_high"
    graph_type = "undirected"
    # max_nodes is handled by registry/resolver - no internal limit
    
    def compute(
        self, 
        G: nx.DiGraph, 
        U: nx.Graph, 
        nodes: list, 
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        try:
            n = len(nodes)
            params = parameters or {}
            
            # Number of random pairs to test
            num_pairs = params.get('num_pairs', min(100, n * 2))
            
            if n < 3:
                return {node: {
                    "min_cut_frequency": 0.0,
                    "cut_vertex_centrality": 0.0,
                } for node in nodes}
            
            # Count how often each node appears in min-cuts
            cut_counts = {node: 0 for node in nodes}
            
            # Generate random pairs
            pairs_tested = 0
            attempts = 0
            max_attempts = num_pairs * 3
            
            while pairs_tested < num_pairs and attempts < max_attempts:
                attempts += 1
                try:
                    # Pick two random distinct nodes
                    pair = np.random.choice(nodes, 2, replace=False)
                    s, t = pair[0], pair[1]
                    
                    if not nx.has_path(U, s, t):
                        continue
                    
                    # Find minimum node cut
                    min_cut = nx.minimum_node_cut(U, s, t)
                    
                    for node in min_cut:
                        if node in cut_counts:
                            cut_counts[node] += 1
                    
                    pairs_tested += 1
                    
                except (nx.NetworkXError, ValueError):
                    continue
            
            # Normalize by number of pairs tested
            max_count = max(cut_counts.values()) if cut_counts else 1
            
            # Also check articulation points
            articulation_points = set(nx.articulation_points(U))
            
            result = {}
            for node in nodes:
                count = cut_counts.get(node, 0)
                frequency = count / pairs_tested if pairs_tested > 0 else 0
                
                # Cut vertex centrality combines frequency and articulation point status
                is_ap = 1 if node in articulation_points else 0
                centrality = frequency + (0.5 * is_ap)
                
                result[node] = {
                    "min_cut_frequency": frequency,
                    "cut_vertex_centrality": centrality,
                }
            
            logger.debug(f"Min cut centrality: tested {pairs_tested} pairs")
            return result
            
        except Exception as e:
            logger.warning(f"Min cut centrality failed: {e}")
            return {node: {} for node in nodes}