"""
Graph Algorithms Router

API endpoints for interactive graph algorithms:
- Path finding between nodes
- Subgraph extraction
- Flow analysis
- Selection analysis

Location: web_viewer/backend/routers/graph_algorithms.py
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..services.network_service import network_service
from engines.graph_algorithms import (
    PathFinder,
    SubgraphExtractor,
    FlowAnalyzer,
    SelectionAnalyzer,
)


router = APIRouter(prefix="/api/algorithms", tags=["algorithms"])


# =============================================================================
# REQUEST MODELS
# =============================================================================

class PathRequest(BaseModel):
    """Request model for path finding."""
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    algorithm: str = Field("shortest_path", description="Algorithm to use")
    directed: bool = Field(True, description="Use directed graph")
    
    # Algorithm-specific parameters
    k: Optional[int] = Field(5, description="Number of paths for k-shortest")
    cutoff: Optional[int] = Field(10, description="Max path length for all_simple_paths")
    weight: Optional[str] = Field(None, description="Edge weight attribute")
    max_paths: Optional[int] = Field(1000, description="Maximum paths to return")


class SubgraphRequest(BaseModel):
    """Request model for subgraph extraction."""
    nodes: List[str] = Field(..., description="Node IDs")
    mode: str = Field("neighborhood", description="Extraction mode: neighborhood, induced, ego, component")
    hops: int = Field(1, description="Number of hops for neighborhood mode")
    directed: bool = Field(False, description="Use directed graph for neighborhood")


class FlowRequest(BaseModel):
    """Request model for flow analysis."""
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target (sink) node ID")
    capacity: str = Field("", description="Edge capacity attribute (empty = unit capacity)")


class SelectionAnalysisRequest(BaseModel):
    """Request model for selection analysis."""
    node_ids: List[str] = Field(..., description="Selected node IDs")
    metrics: List[str] = Field(default_factory=list, description="Metrics to compute")
    compare_to_full: bool = Field(True, description="Compare to full graph stats")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_graph(graph_name: Optional[str] = None):
    """Get graph from network service, using first available if not specified."""
    if graph_name:
        if graph_name in network_service.graphs:
            return network_service.graphs[graph_name]
        raise HTTPException(404, f"Graph '{graph_name}' not found")
    
    # Use first available graph
    if network_service.graphs:
        first_id = next(iter(network_service.graphs.keys()))
        return network_service.graphs[first_id]
    
    raise HTTPException(400, "No graphs loaded. Load a network first.")


# =============================================================================
# GRAPH INFO ENDPOINTS
# =============================================================================

@router.get("/edge-attributes")
async def get_edge_attributes(
    graph_name: Optional[str] = Query(None, description="Graph name")
):
    """Get available edge attributes (for capacity selection)."""
    G = get_graph(graph_name)
    
    # Collect all unique edge attributes
    attributes = set()
    numeric_attrs = set()
    
    for u, v, data in G.edges(data=True):
        for key, value in data.items():
            attributes.add(key)
            if isinstance(value, (int, float)):
                numeric_attrs.add(key)
    
    return {
        "all_attributes": sorted(list(attributes)),
        "numeric_attributes": sorted(list(numeric_attrs)),
        "edge_count": G.number_of_edges(),
    }


# =============================================================================
# PATH FINDING ENDPOINTS
# =============================================================================

@router.post("/paths")
async def find_paths(
    request: PathRequest, 
    graph_name: Optional[str] = Query(None, description="Graph name")
):
    """
    Find paths between two nodes.
    
    Supported algorithms:
    - shortest_path: Single shortest path (unweighted)
    - all_shortest_paths: All shortest paths
    - k_shortest_paths: K shortest paths (Yen's algorithm)
    - all_simple_paths: All simple paths up to cutoff
    - dijkstra: Weighted shortest path
    - node_disjoint_paths: Node-disjoint paths
    - edge_disjoint_paths: Edge-disjoint paths
    """
    # Debug: print which graph is being requested vs what's available
    print(f"[PathFinder] Requested graph: {graph_name}")
    print(f"[PathFinder] Available graphs: {list(network_service.graphs.keys())}")
    
    G = get_graph(graph_name)
    print(f"[PathFinder] Using graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    finder = PathFinder(G)
    
    algorithm = request.algorithm.lower()
    
    if algorithm == "shortest_path":
        result = finder.shortest_path(
            request.source, 
            request.target,
            directed=request.directed
        )
    
    elif algorithm == "all_shortest_paths":
        result = finder.all_shortest_paths(
            request.source,
            request.target,
            directed=request.directed,
            max_paths=request.max_paths
        )
    
    elif algorithm == "k_shortest_paths":
        result = finder.k_shortest_paths(
            request.source,
            request.target,
            k=request.k or 5,
            directed=request.directed,
            weight=request.weight
        )
    
    elif algorithm == "all_simple_paths":
        result = finder.all_simple_paths(
            request.source,
            request.target,
            cutoff=request.cutoff or 10,
            directed=request.directed,
            max_paths=request.max_paths
        )
    
    elif algorithm == "dijkstra":
        result = finder.dijkstra(
            request.source,
            request.target,
            directed=request.directed,
            weight=request.weight or "weight"
        )
    
    elif algorithm == "node_disjoint_paths":
        result = finder.node_disjoint_paths(
            request.source,
            request.target,
            directed=request.directed
        )
    
    elif algorithm == "edge_disjoint_paths":
        result = finder.edge_disjoint_paths(
            request.source,
            request.target,
            directed=request.directed
        )
    
    else:
        raise HTTPException(400, f"Unknown algorithm: {algorithm}")
    
    return result


@router.get("/paths/check")
async def check_path_exists(
    source: str,
    target: str,
    directed: bool = True,
    graph_name: Optional[str] = Query(None, description="Graph name")
):
    """Check if a path exists between two nodes."""
    G = get_graph(graph_name)
    finder = PathFinder(G)
    
    result = finder.path_exists(source, target, directed=directed)
    return result


@router.get("/paths/length")
async def get_path_length(
    source: str,
    target: str,
    directed: bool = True,
    weight: Optional[str] = None,
    graph_name: Optional[str] = Query(None, description="Graph name")
):
    """Get shortest path length between two nodes."""
    G = get_graph(graph_name)
    finder = PathFinder(G)
    
    result = finder.shortest_path_length(source, target, directed=directed, weight=weight)
    return result


# =============================================================================
# SUBGRAPH EXTRACTION ENDPOINTS
# =============================================================================

@router.post("/subgraph")
async def extract_subgraph(
    request: SubgraphRequest, 
    graph_name: Optional[str] = Query(None, description="Graph name")
):
    """
    Extract a subgraph based on node selection.
    
    Modes:
    - neighborhood: N-hop neighborhood of specified nodes
    - induced: Induced subgraph from node selection
    - ego: Ego graph centered on first node
    - component: Connected component containing first node
    - k_hop: K-hop subgraph from multiple seed nodes
    """
    G = get_graph(graph_name)
    extractor = SubgraphExtractor(G)
    
    mode = request.mode.lower()
    
    if mode == "neighborhood":
        if not request.nodes:
            raise HTTPException(400, "At least one node required for neighborhood")
        # For single node, use neighborhood; for multiple, use k_hop_subgraph
        if len(request.nodes) == 1:
            return extractor.neighborhood(
                request.nodes[0],
                hops=request.hops,
                directed=request.directed
            )
        else:
            return extractor.k_hop_subgraph(
                request.nodes,
                hops=request.hops,
                directed=request.directed
            )
    
    elif mode == "k_hop":
        if not request.nodes:
            raise HTTPException(400, "At least one node required for k-hop subgraph")
        return extractor.k_hop_subgraph(
            request.nodes,
            hops=request.hops,
            directed=request.directed
        )
    
    elif mode == "induced":
        return extractor.induced_subgraph(request.nodes)
    
    elif mode == "ego":
        if not request.nodes:
            raise HTTPException(400, "At least one node required for ego graph")
        return extractor.ego_graph(request.nodes[0], radius=request.hops)
    
    elif mode == "component":
        if not request.nodes:
            raise HTTPException(400, "At least one node required for component")
        return extractor.connected_component(request.nodes[0])
    
    else:
        raise HTTPException(400, f"Unknown mode: {mode}")


@router.get("/neighborhood/{node_id}")
async def get_neighborhood(
    node_id: str,
    hops: int = Query(1, ge=1, le=5),
    directed: bool = False,
    graph_name: Optional[str] = Query(None, description="Graph name")
):
    """Get N-hop neighborhood of a node."""
    G = get_graph(graph_name)
    extractor = SubgraphExtractor(G)
    return extractor.neighborhood(node_id, hops=hops, directed=directed)


@router.get("/ego/{node_id}")
async def get_ego_graph(
    node_id: str,
    radius: int = Query(1, ge=1, le=5),
    graph_name: Optional[str] = Query(None, description="Graph name")
):
    """Get ego graph centered on a node."""
    G = get_graph(graph_name)
    extractor = SubgraphExtractor(G)
    return extractor.ego_graph(node_id, radius=radius)


@router.get("/component/{node_id}")
async def get_component(
    node_id: str, 
    graph_name: Optional[str] = Query(None, description="Graph name")
):
    """Get connected component containing a node."""
    G = get_graph(graph_name)
    extractor = SubgraphExtractor(G)
    return extractor.connected_component(node_id)


# =============================================================================
# FLOW ANALYSIS ENDPOINTS
# =============================================================================

@router.post("/max-flow")
async def compute_max_flow(
    request: FlowRequest, 
    graph_name: Optional[str] = Query(None, description="Graph name")
):
    """Compute maximum flow between source and sink."""
    G = get_graph(graph_name)
    analyzer = FlowAnalyzer(G)
    return analyzer.maximum_flow(request.source, request.target, capacity=request.capacity)


@router.post("/min-cut")
async def compute_min_cut(
    request: FlowRequest, 
    graph_name: Optional[str] = Query(None, description="Graph name")
):
    """Compute minimum cut between source and sink."""
    G = get_graph(graph_name)
    analyzer = FlowAnalyzer(G)
    return analyzer.minimum_cut(request.source, request.target, capacity=request.capacity)


@router.get("/edge-connectivity")
async def compute_edge_connectivity(
    source: Optional[str] = Query(None, description="Source node ID"),
    target: Optional[str] = Query(None, description="Target node ID"),
    graph_name: Optional[str] = Query(None, description="Graph name")
):
    """Compute edge connectivity between nodes or for entire graph."""
    G = get_graph(graph_name)
    analyzer = FlowAnalyzer(G)
    return analyzer.edge_connectivity(source, target)


@router.get("/node-connectivity")
async def compute_node_connectivity(
    source: Optional[str] = Query(None, description="Source node ID"),
    target: Optional[str] = Query(None, description="Target node ID"),
    graph_name: Optional[str] = Query(None, description="Graph name")
):
    """Compute node connectivity between nodes or for entire graph."""
    G = get_graph(graph_name)
    analyzer = FlowAnalyzer(G)
    return analyzer.node_connectivity(source, target)


# =============================================================================
# SELECTION ANALYSIS ENDPOINTS
# =============================================================================

@router.post("/analyze-selection")
async def analyze_selection(
    request: SelectionAnalysisRequest, 
    graph_name: Optional[str] = Query(None, description="Graph name")
):
    """
    Analyze a selection of nodes.
    
    Returns statistics about the selected nodes and optionally
    compares them to the full graph.
    """
    G = get_graph(graph_name)
    analyzer = SelectionAnalyzer(G)
    
    result = analyzer.analyze_selection(
        request.node_ids,
        metrics=request.metrics if request.metrics else None
    )
    
    if request.compare_to_full:
        # Get full graph stats for comparison
        all_nodes = list(G.nodes())
        if len(all_nodes) > 10000:
            # Sample for large graphs
            import random
            sample = random.sample(all_nodes, 10000)
            full_stats = analyzer.analyze_selection(sample)
        else:
            full_stats = analyzer.analyze_selection(all_nodes)
        
        result["full_graph_stats"] = full_stats
    
    return result


@router.post("/boundary-nodes")
async def get_boundary_nodes(
    request: SelectionAnalysisRequest, 
    graph_name: Optional[str] = Query(None, description="Graph name")
):
    """Get boundary nodes of a selection (nodes with external connections)."""
    G = get_graph(graph_name)
    analyzer = SelectionAnalyzer(G)
    return analyzer.get_boundary_nodes(request.node_ids)


@router.post("/connecting-edges")
async def get_connecting_edges(
    request: SelectionAnalysisRequest, 
    graph_name: Optional[str] = Query(None, description="Graph name")
):
    """Get edges connecting selection to rest of graph."""
    G = get_graph(graph_name)
    analyzer = SelectionAnalyzer(G)
    return analyzer.get_connecting_edges(request.node_ids)


# =============================================================================
# INFO ENDPOINT
# =============================================================================

@router.get("/available-algorithms")
async def list_algorithms():
    """List available algorithms and their parameters."""
    return {
        "path_algorithms": {
            "shortest_path": {
                "description": "Find single shortest path (unweighted BFS)",
                "params": ["source", "target", "directed"]
            },
            "all_shortest_paths": {
                "description": "Find all shortest paths of equal length",
                "params": ["source", "target", "directed", "max_paths"]
            },
            "k_shortest_paths": {
                "description": "Find K shortest paths (Yen's algorithm)",
                "params": ["source", "target", "k", "directed", "weight"]
            },
            "all_simple_paths": {
                "description": "Find all simple paths up to cutoff length",
                "params": ["source", "target", "cutoff", "directed", "max_paths"]
            },
            "dijkstra": {
                "description": "Find weighted shortest path",
                "params": ["source", "target", "weight", "directed"]
            },
            "node_disjoint_paths": {
                "description": "Find paths sharing no intermediate nodes",
                "params": ["source", "target", "directed"]
            },
            "edge_disjoint_paths": {
                "description": "Find paths sharing no edges",
                "params": ["source", "target", "directed"]
            },
        },
        "subgraph_modes": [
            "neighborhood",
            "induced", 
            "ego",
            "component"
        ],
        "flow_algorithms": [
            "max_flow",
            "min_cut"
        ]
    }