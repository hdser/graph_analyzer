"""
Component Metric Algorithms

Component membership metrics including weak/strong components,
condensation graph analysis, attracting components, and biconnected components.
"""

from typing import Dict, Any
import logging

import networkx as nx

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


class ComponentsAlgorithm(BaseMetricAlgorithm):
    """Compute weak and strong component membership."""
    
    name = "components"
    category = "components"
    description = "Weak and strong component membership"
    cost = "low"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            weak_comps = list(nx.weakly_connected_components(G))
            strong_comps = list(nx.strongly_connected_components(G))
            
            logger.debug(f"Found {len(weak_comps)} weak components, {len(strong_comps)} strong components")
            
            # Map nodes to component IDs and sizes
            weak_map = {}
            weak_sizes = {}
            for idx, comp in enumerate(weak_comps):
                weak_sizes[idx] = len(comp)
                for node in comp:
                    weak_map[node] = idx
            
            strong_map = {}
            strong_sizes = {}
            for idx, comp in enumerate(strong_comps):
                strong_sizes[idx] = len(comp)
                for node in comp:
                    strong_map[node] = idx
            
            largest_weak = max(weak_sizes.values()) if weak_sizes else 0
            
            result = {}
            for node in nodes:
                wid = weak_map.get(node, -1)
                sid = strong_map.get(node, -1)
                result[node] = {
                    "weak_component_size": weak_sizes.get(wid, 0),
                    "strong_component_size": strong_sizes.get(sid, 0),
                    "in_largest_component": 1 if weak_sizes.get(wid, 0) == largest_weak else 0,
                }
            return result
        except Exception as e:
            logger.warning(f"Component metrics failed: {e}")
            return {node: {} for node in nodes}


class StronglyConnectedComponentsAlgorithm(BaseMetricAlgorithm):
    """
    Detailed strongly connected component analysis.
    
    Provides SCC membership, sizes, and identifies trivial (single-node) SCCs.
    Also computes condensation graph node mapping.
    """
    
    name = "strongly_connected_components"
    category = "components"
    description = "Detailed strongly connected component analysis"
    cost = "low"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            # Get strongly connected components
            sccs = list(nx.strongly_connected_components(G))
            
            # Find largest SCC
            largest_scc_size = max(len(scc) for scc in sccs) if sccs else 0
            largest_scc = None
            for scc in sccs:
                if len(scc) == largest_scc_size:
                    largest_scc = scc
                    break
            
            # Map nodes to SCC info
            node_to_scc = {}
            scc_sizes = {}
            for idx, scc in enumerate(sccs):
                scc_sizes[idx] = len(scc)
                for node in scc:
                    node_to_scc[node] = idx
            
            # Build condensation graph for condensation_id mapping
            try:
                condensation = nx.condensation(G, scc=sccs)
                # condensation.graph['mapping'] maps original nodes to condensation nodes
                node_to_condensation = condensation.graph.get('mapping', {})
            except Exception:
                node_to_condensation = {}
            
            result = {}
            for node in nodes:
                scc_id = node_to_scc.get(node, -1)
                scc_size = scc_sizes.get(scc_id, 0)
                
                result[node] = {
                    "strong_component_id": scc_id,
                    "scc_size": scc_size,
                    "in_largest_scc": 1 if (largest_scc and node in largest_scc) else 0,
                    "scc_is_trivial": 1 if scc_size == 1 else 0,
                    "scc_condensation_id": node_to_condensation.get(node, -1),
                }
            
            return result
            
        except Exception as e:
            logger.warning(f"Strongly connected components failed: {e}")
            return {node: {} for node in nodes}


class CondensationGraphAlgorithm(BaseMetricAlgorithm):
    """
    Condensation graph metrics.
    
    The condensation of a directed graph is a DAG where each node represents
    a strongly connected component. This algorithm computes metrics based on
    each node's position in this DAG hierarchy.
    """
    
    name = "condensation_graph"
    category = "components"
    description = "Condensation DAG hierarchy metrics"
    cost = "medium"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            # Build condensation graph
            sccs = list(nx.strongly_connected_components(G))
            condensation = nx.condensation(G, scc=sccs)
            
            # Get mapping from original nodes to condensation nodes
            node_to_cond = condensation.graph.get('mapping', {})
            
            # Compute condensation graph properties
            cond_in_degree = dict(condensation.in_degree())
            cond_out_degree = dict(condensation.out_degree())
            
            # Find roots (in_degree=0) and leaves (out_degree=0)
            roots = {n for n in condensation.nodes() if cond_in_degree[n] == 0}
            leaves = {n for n in condensation.nodes() if cond_out_degree[n] == 0}
            
            # Compute depth from roots using BFS
            cond_depth = {}
            for root in roots:
                for node, depth in nx.single_source_shortest_path_length(condensation, root).items():
                    if node not in cond_depth or depth < cond_depth[node]:
                        cond_depth[node] = depth
            
            # For nodes not reachable from any root
            for node in condensation.nodes():
                if node not in cond_depth:
                    cond_depth[node] = -1
            
            result = {}
            for node in nodes:
                cond_id = node_to_cond.get(node, -1)
                
                if cond_id >= 0:
                    result[node] = {
                        "condensation_in_degree": cond_in_degree.get(cond_id, 0),
                        "condensation_out_degree": cond_out_degree.get(cond_id, 0),
                        "condensation_is_root": 1 if cond_id in roots else 0,
                        "condensation_is_leaf": 1 if cond_id in leaves else 0,
                        "condensation_depth": cond_depth.get(cond_id, -1),
                    }
                else:
                    result[node] = {
                        "condensation_in_degree": 0,
                        "condensation_out_degree": 0,
                        "condensation_is_root": 0,
                        "condensation_is_leaf": 0,
                        "condensation_depth": -1,
                    }
            
            return result
            
        except Exception as e:
            logger.warning(f"Condensation graph metrics failed: {e}")
            return {node: {} for node in nodes}


class AttractingComponentsAlgorithm(BaseMetricAlgorithm):
    """
    Attracting component analysis.
    
    An attracting component is a strongly connected component with no edges
    leaving it (a sink in the condensation DAG). These are the 'final destinations'
    of random walks on the graph.
    """
    
    name = "attracting_components"
    category = "components"
    description = "Attracting (sink) component membership"
    cost = "low"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            # Get attracting components
            attracting = list(nx.attracting_components(G))
            
            # Map nodes to attracting component info
            node_to_attracting = {}
            attracting_sizes = {}
            
            for idx, comp in enumerate(attracting):
                attracting_sizes[idx] = len(comp)
                for node in comp:
                    node_to_attracting[node] = idx
            
            # Set of all nodes in any attracting component
            all_attracting_nodes = set()
            for comp in attracting:
                all_attracting_nodes.update(comp)
            
            result = {}
            for node in nodes:
                in_attracting = node in all_attracting_nodes
                attr_id = node_to_attracting.get(node, -1)
                
                result[node] = {
                    "in_attracting_component": 1 if in_attracting else 0,
                    "attracting_component_id": attr_id,
                    "attracting_component_size": attracting_sizes.get(attr_id, 0),
                }
            
            return result
            
        except Exception as e:
            logger.warning(f"Attracting components failed: {e}")
            return {node: {} for node in nodes}


class BiconnectedComponentsAlgorithm(BaseMetricAlgorithm):
    """
    Biconnected component analysis (undirected).
    
    A biconnected component is a maximal subgraph that remains connected
    after removing any single vertex. Nodes can belong to multiple
    biconnected components (articulation points).
    """
    
    name = "biconnected_components"
    category = "components"
    description = "Biconnected component membership (undirected)"
    cost = "medium"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            # Get biconnected components from undirected graph
            bicomps = list(nx.biconnected_components(U))
            
            # Count how many biconnected components each node belongs to
            node_bicomp_count = {node: 0 for node in nodes}
            node_max_bicomp_size = {node: 0 for node in nodes}
            
            for comp in bicomps:
                comp_size = len(comp)
                for node in comp:
                    if node in node_bicomp_count:
                        node_bicomp_count[node] += 1
                        node_max_bicomp_size[node] = max(
                            node_max_bicomp_size[node], 
                            comp_size
                        )
            
            # Get articulation points
            try:
                articulation_points = set(nx.articulation_points(U))
            except Exception:
                articulation_points = set()
            
            result = {}
            for node in nodes:
                result[node] = {
                    "biconnected_component_count": node_bicomp_count.get(node, 0),
                    "max_biconnected_size": node_max_bicomp_size.get(node, 0),
                    "is_articulation_point": 1 if node in articulation_points else 0,
                }
            
            return result
            
        except Exception as e:
            logger.warning(f"Biconnected components failed: {e}")
            return {node: {} for node in nodes}