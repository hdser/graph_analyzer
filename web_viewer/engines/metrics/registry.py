"""
Metric Registry

Central registry of all available metrics with their definitions,
categories, presets, and configurable parameters.

Enhanced to support 150+ metrics across 25+ categories including:
- New centrality metrics (Laplacian, Leverage, Semi-local, Decay)
- New clustering metrics (Clique count, Average neighbor clustering)
- New community metrics (Label propagation, Greedy modularity, Participation coefficient)
- Link prediction metrics (Common neighbors, Preferential attachment)
- Robustness metrics (Node/Edge connectivity, Resilience score)
- Spectral metrics (Fiedler vector, Spectral centrality)
- Influence metrics (Collective influence, Spreading activation)
- Bipartite metrics (Projection degree, Redundancy)
- Graph coloring metrics
- igraph-specific algorithms (Leiden, Infomap, Walktrap, Motifs)
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any


@dataclass
class MetricParameter:
    """Definition of a configurable metric parameter."""
    
    name: str                          # Parameter name
    type: str                          # Type: "int", "float", "bool", "str", "choice"
    default: Any                       # Default value
    description: str                   # Human-readable description
    min_value: Optional[float] = None  # Minimum value (for numeric types)
    max_value: Optional[float] = None  # Maximum value (for numeric types)
    choices: Optional[List[Any]] = None  # Available choices (for choice type)
    step: Optional[float] = None       # Step size for UI sliders


@dataclass
class MetricDefinition:
    """Definition of a single computable metric."""
    
    name: str                          # Unique identifier: "pagerank"
    category: str                      # Category: "centrality"
    description: str                   # Human-readable description
    
    # Computation
    algorithm_class: str               # Class name in algorithms module
    graph_type: str = "directed"       # "directed", "undirected", "both"
    
    # Dependencies & Cost
    dependencies: List[str] = field(default_factory=list)
    cost: str = "low"                  # "low", "medium", "high", "very_high"
    max_nodes: Optional[int] = None    # Skip if graph exceeds this size
    requires_connected: bool = False   # Requires connected graph
    
    # Output
    output_columns: List[str] = field(default_factory=list)
    
    # Metadata
    source: str = "networkx"           # "networkx", "igraph", "custom", "scipy"
    citation: Optional[str] = None
    requires_igraph: bool = False      # Requires igraph library
    
    # Parameters
    parameters: List[MetricParameter] = field(default_factory=list)


# =============================================================================
# METRIC REGISTRY - All available metrics
# =============================================================================

METRIC_REGISTRY: Dict[str, MetricDefinition] = {}


def register_metric(definition: MetricDefinition) -> MetricDefinition:
    """Register a metric definition."""
    METRIC_REGISTRY[definition.name] = definition
    return definition


# =============================================================================
# TOPOLOGY METRICS
# =============================================================================

register_metric(MetricDefinition(
    name="in_degree",
    category="topology",
    description="Number of incoming edges",
    algorithm_class="InDegreeAlgorithm",
    cost="low",
    output_columns=["in_degree"],
))

register_metric(MetricDefinition(
    name="out_degree",
    category="topology",
    description="Number of outgoing edges",
    algorithm_class="OutDegreeAlgorithm",
    cost="low",
    output_columns=["out_degree"],
))

register_metric(MetricDefinition(
    name="total_degree",
    category="topology",
    description="Sum of in and out degree",
    algorithm_class="TotalDegreeAlgorithm",
    cost="low",
    output_columns=["total_degree"],
    dependencies=["in_degree", "out_degree"],
))

register_metric(MetricDefinition(
    name="degree_imbalance",
    category="topology",
    description="Normalized difference between in and out degree",
    algorithm_class="DegreeImbalanceAlgorithm",
    cost="low",
    output_columns=["degree_imbalance"],
    dependencies=["in_degree", "out_degree"],
))

# =============================================================================
# CENTRALITY METRICS
# =============================================================================

register_metric(MetricDefinition(
    name="degree_centrality",
    category="centrality",
    description="Normalized degree centrality (in, out, undirected)",
    algorithm_class="DegreeCentralityAlgorithm",
    cost="low",
    output_columns=["in_degree_centrality", "out_degree_centrality", "degree_centrality_undirected"],
))

register_metric(MetricDefinition(
    name="closeness_centrality",
    category="centrality",
    description="Closeness centrality (directed and undirected)",
    algorithm_class="ClosenessCentralityAlgorithm",
    cost="medium",
    output_columns=["closeness_centrality", "closeness_centrality_in", "closeness_centrality_undirected"],
))

register_metric(MetricDefinition(
    name="betweenness_centrality",
    category="centrality",
    description="Betweenness centrality (directed and undirected)",
    algorithm_class="BetweennessCentralityAlgorithm",
    cost="high",
    output_columns=["betweenness_centrality", "betweenness_centrality_undirected"],
    parameters=[
        MetricParameter(
            name="normalized",
            type="bool",
            default=True,
            description="If True, normalize betweenness values"
        ),
        MetricParameter(
            name="endpoints",
            type="bool",
            default=False,
            description="If True, include endpoints in shortest path counts"
        ),
    ]
))

register_metric(MetricDefinition(
    name="eigenvector_centrality",
    category="centrality",
    description="Eigenvector centrality (directed and undirected)",
    algorithm_class="EigenvectorCentralityAlgorithm",
    cost="medium",
    output_columns=["eigenvector_centrality", "eigenvector_centrality_undirected"],
    parameters=[
        MetricParameter(
            name="max_iter",
            type="int",
            default=100,
            description="Maximum number of iterations for power method",
            min_value=10,
            max_value=1000,
            step=10
        ),
        MetricParameter(
            name="tol",
            type="float",
            default=1.0e-6,
            description="Convergence tolerance",
            min_value=1.0e-10,
            max_value=1.0e-3,
            step=1.0e-7
        ),
    ]
))

register_metric(MetricDefinition(
    name="katz_centrality",
    category="centrality",
    description="Katz centrality with safe alpha computation",
    algorithm_class="KatzCentralityAlgorithm",
    cost="medium",
    output_columns=["katz_centrality", "katz_centrality_undirected"],
    parameters=[
        MetricParameter(
            name="alpha",
            type="float",
            default=0.1,
            description="Attenuation factor (auto-calculated if None)",
            min_value=0.001,
            max_value=0.5,
            step=0.01
        ),
        MetricParameter(
            name="beta",
            type="float",
            default=1.0,
            description="Weight attributed to immediate neighbors",
            min_value=0.1,
            max_value=10.0,
            step=0.1
        ),
        MetricParameter(
            name="max_iter",
            type="int",
            default=1000,
            description="Maximum number of iterations",
            min_value=100,
            max_value=10000,
            step=100
        ),
        MetricParameter(
            name="tol",
            type="float",
            default=1.0e-6,
            description="Convergence tolerance",
            min_value=1.0e-10,
            max_value=1.0e-3,
            step=1.0e-7
        ),
    ]
))

register_metric(MetricDefinition(
    name="pagerank",
    category="centrality",
    description="Google PageRank score",
    algorithm_class="PageRankAlgorithm",
    cost="low",
    output_columns=["pagerank", "pagerank_undirected"],
    citation="Page et al., 1999",
    parameters=[
        MetricParameter(
            name="alpha",
            type="float",
            default=0.85,
            description="Damping parameter (probability of following a link)",
            min_value=0.0,
            max_value=1.0,
            step=0.05
        ),
        MetricParameter(
            name="max_iter",
            type="int",
            default=100,
            description="Maximum number of iterations",
            min_value=10,
            max_value=1000,
            step=10
        ),
        MetricParameter(
            name="tol",
            type="float",
            default=1.0e-6,
            description="Convergence tolerance",
            min_value=1.0e-10,
            max_value=1.0e-3,
            step=1.0e-7
        ),
    ]
))

register_metric(MetricDefinition(
    name="hits",
    category="centrality",
    description="HITS hub and authority scores",
    algorithm_class="HITSAlgorithm",
    cost="medium",
    output_columns=["hub_score", "authority_score"],
    citation="Kleinberg, 1999",
))

register_metric(MetricDefinition(
    name="harmonic_centrality",
    category="centrality",
    description="Harmonic centrality (sum of inverse distances)",
    algorithm_class="HarmonicCentralityAlgorithm",
    cost="medium",
    output_columns=["harmonic_centrality", "harmonic_centrality_undirected"],
))

register_metric(MetricDefinition(
    name="load_centrality",
    category="centrality",
    description="Load centrality (traffic flow through node)",
    algorithm_class="LoadCentralityAlgorithm",
    cost="high",
    output_columns=["load_centrality", "load_centrality_undirected"],
))

register_metric(MetricDefinition(
    name="subgraph_centrality",
    category="centrality",
    description="Subgraph centrality based on closed walks",
    algorithm_class="SubgraphCentralityAlgorithm",
    graph_type="undirected",
    cost="medium",
    output_columns=["subgraph_centrality"],
))

register_metric(MetricDefinition(
    name="second_order_centrality",
    category="centrality",
    description="Second order centrality (random walk variance)",
    algorithm_class="SecondOrderCentralityAlgorithm",
    graph_type="undirected",
    cost="high",
    max_nodes=100000,
    output_columns=["second_order_centrality"],
))

register_metric(MetricDefinition(
    name="percolation_centrality",
    category="centrality",
    description="Percolation centrality with random states",
    algorithm_class="PercolationCentralityAlgorithm",
    cost="medium",
    output_columns=["percolation_centrality"],
))

register_metric(MetricDefinition(
    name="trophic_level",
    category="centrality",
    description="Trophic level in directed graph hierarchy",
    algorithm_class="TrophicLevelAlgorithm",
    cost="medium",
    output_columns=["trophic_level"],
))

register_metric(MetricDefinition(
    name="current_flow_centrality",
    category="centrality",
    description="Current flow betweenness and closeness centrality",
    algorithm_class="CurrentFlowCentralityAlgorithm",
    graph_type="undirected",
    cost="very_high",
    max_nodes=100000,
    requires_connected=True,
    output_columns=["current_flow_betweenness", "current_flow_closeness"],
))

register_metric(MetricDefinition(
    name="information_centrality",
    category="centrality",
    description="Information centrality based on information flow",
    algorithm_class="InformationCentralityAlgorithm",
    graph_type="undirected",
    cost="very_high",
    max_nodes=100000,
    requires_connected=True,
    output_columns=["information_centrality"],
))

register_metric(MetricDefinition(
    name="communicability_betweenness",
    category="centrality",
    description="Communicability betweenness centrality",
    algorithm_class="CommunicabilityBetweennessAlgorithm",
    graph_type="undirected",
    cost="very_high",
    max_nodes=50000,
    output_columns=["communicability_betweenness"],
))

register_metric(MetricDefinition(
    name="voterank",
    category="centrality",
    description="VoteRank influence measure",
    algorithm_class="VoteRankAlgorithm",
    graph_type="undirected",
    cost="medium",
    output_columns=["voterank"],
))

register_metric(MetricDefinition(
    name="edge_betweenness_sum",
    category="centrality",
    description="Sum of edge betweenness for incident edges",
    algorithm_class="EdgeBetweennessSumAlgorithm",
    cost="high",
    output_columns=["edge_betweenness_sum"],
))

# NEW: Laplacian Centrality
register_metric(MetricDefinition(
    name="laplacian_centrality",
    category="centrality",
    description="Centrality based on Laplacian energy drop when node removed",
    algorithm_class="LaplacianCentralityAlgorithm",
    graph_type="undirected",
    cost="medium",
    output_columns=["laplacian_centrality"],
    source="networkx",
))

# NEW: Leverage Centrality
register_metric(MetricDefinition(
    name="leverage_centrality",
    category="centrality",
    description="Measures degree advantage over neighbors",
    algorithm_class="LeverageCentralityAlgorithm",
    cost="low",
    output_columns=["leverage_centrality"],
    source="custom",
))

# NEW: Semi-Local Centrality
register_metric(MetricDefinition(
    name="semi_local_centrality",
    category="centrality",
    description="2-hop neighborhood influence measure",
    algorithm_class="SemiLocalCentralityAlgorithm",
    cost="medium",
    output_columns=["semi_local_centrality"],
    source="custom",
))

# NEW: Decay Centrality
register_metric(MetricDefinition(
    name="decay_centrality",
    category="centrality",
    description="Distance-weighted influence with decay parameter",
    algorithm_class="DecayCentralityAlgorithm",
    cost="high",
    output_columns=["decay_centrality"],
    source="custom",
    parameters=[
        MetricParameter(
            name="delta",
            type="float",
            default=0.5,
            description="Decay factor (0-1, lower = faster decay)",
            min_value=0.1,
            max_value=0.9,
            step=0.1
        ),
    ]
))

# =============================================================================
# CLUSTERING METRICS
# =============================================================================

register_metric(MetricDefinition(
    name="clustering_coefficient",
    category="clustering",
    description="Local clustering coefficient",
    algorithm_class="ClusteringCoefficientAlgorithm",
    cost="medium",
    output_columns=["clustering_coefficient", "clustering_coefficient_directed"],
))

register_metric(MetricDefinition(
    name="triangles",
    category="clustering",
    description="Number of triangles involving node",
    algorithm_class="TrianglesAlgorithm",
    cost="medium",
    output_columns=["triangle_count", "triangle_count_directed"],
))

register_metric(MetricDefinition(
    name="square_clustering",
    category="clustering",
    description="Square clustering coefficient",
    algorithm_class="SquareClusteringAlgorithm",
    graph_type="undirected",
    cost="medium",
    output_columns=["square_clustering"],
))

register_metric(MetricDefinition(
    name="local_transitivity",
    category="clustering",
    description="Local transitivity (same as clustering coefficient)",
    algorithm_class="LocalTransitivityAlgorithm",
    cost="medium",
    output_columns=["local_transitivity"],
))

# NEW: Clique Count
register_metric(MetricDefinition(
    name="clique_count",
    category="clustering",
    description="Number of maximal cliques containing each node",
    algorithm_class="CliqueCountAlgorithm",
    graph_type="undirected",
    cost="high",
    max_nodes=50000,
    output_columns=["clique_count", "max_clique_size"],
    source="networkx",
))

# NEW: Average Neighbor Clustering
register_metric(MetricDefinition(
    name="average_neighbor_clustering",
    category="clustering",
    description="Average clustering coefficient of neighbors",
    algorithm_class="AverageNeighborClusteringAlgorithm",
    graph_type="undirected",
    cost="medium",
    output_columns=["average_neighbor_clustering"],
    source="networkx",
))

# =============================================================================
# COMMUNITY METRICS
# =============================================================================

register_metric(MetricDefinition(
    name="louvain_community",
    category="community",
    description="Louvain community detection",
    algorithm_class="LouvainCommunityAlgorithm",
    graph_type="undirected",
    cost="medium",
    output_columns=["community_id", "community_size"],
    parameters=[
        MetricParameter(
            name="resolution",
            type="float",
            default=1.0,
            description="Resolution parameter for modularity optimization",
            min_value=0.1,
            max_value=2.0,
            step=0.1
        ),
        MetricParameter(
            name="seed",
            type="int",
            default=42,
            description="Random seed for reproducibility",
            min_value=0,
            max_value=10000,
            step=1
        ),
    ]
))

register_metric(MetricDefinition(
    name="core_number",
    category="community",
    description="K-core membership number",
    algorithm_class="CoreNumberAlgorithm",
    graph_type="undirected",
    cost="low",
    output_columns=["core_number"],
))

register_metric(MetricDefinition(
    name="onion_layer",
    category="community",
    description="Onion decomposition layer",
    algorithm_class="OnionLayerAlgorithm",
    graph_type="undirected",
    cost="medium",
    output_columns=["onion_layer"],
))

register_metric(MetricDefinition(
    name="local_reaching_centrality",
    category="community",
    description="Proportion of network reachable from node",
    algorithm_class="LocalReachingCentralityAlgorithm",
    cost="high",
    max_nodes=50000,
    output_columns=["local_reaching_centrality"],
))

# NEW: Label Propagation
register_metric(MetricDefinition(
    name="label_propagation",
    category="community",
    description="Semi-synchronous label propagation community detection",
    algorithm_class="LabelPropagationAlgorithm",
    graph_type="undirected",
    cost="low",
    output_columns=["lp_community_id", "lp_community_size"],
    source="networkx",
))

# NEW: Async Label Propagation
register_metric(MetricDefinition(
    name="async_label_propagation",
    category="community",
    description="Asynchronous label propagation community detection",
    algorithm_class="AsyncLabelPropagationAlgorithm",
    graph_type="undirected",
    cost="low",
    output_columns=["async_lp_community_id", "async_lp_community_size"],
    source="networkx",
    parameters=[
        MetricParameter(
            name="seed",
            type="int",
            default=42,
            description="Random seed for reproducibility",
            min_value=0,
            max_value=10000,
            step=1
        ),
    ]
))

# NEW: Greedy Modularity Community
register_metric(MetricDefinition(
    name="greedy_modularity_community",
    category="community",
    description="Clauset-Newman-Moore greedy modularity optimization",
    algorithm_class="GreedyModularityCommunityAlgorithm",
    graph_type="undirected",
    cost="medium",
    output_columns=["gm_community_id", "gm_community_size", "graph_modularity"],
    source="networkx",
    citation="Clauset, Newman, Moore, 2004",
))

# NEW: Participation Coefficient
register_metric(MetricDefinition(
    name="participation_coefficient",
    category="community",
    description="Inter-community connectivity measure",
    algorithm_class="ParticipationCoefficientAlgorithm",
    graph_type="undirected",
    cost="medium",
    output_columns=["participation_coefficient"],
    source="custom",
    dependencies=["louvain_community"],
))

# NEW: Within Module Degree
register_metric(MetricDefinition(
    name="within_module_degree",
    category="community",
    description="Z-score of within-community degree",
    algorithm_class="WithinModuleDegreeAlgorithm",
    graph_type="undirected",
    cost="medium",
    output_columns=["within_module_degree_z"],
    source="custom",
    dependencies=["louvain_community"],
))

# =============================================================================
# PATH METRICS
# =============================================================================

register_metric(MetricDefinition(
    name="shortest_paths",
    category="paths",
    description="Shortest path statistics (avg, median, max)",
    algorithm_class="ShortestPathsAlgorithm",
    cost="high",
    output_columns=[
        "avg_shortest_path", "median_shortest_path", "max_shortest_path",
        "path_variance", "path_sum", "reachable_nodes"
    ],
))

register_metric(MetricDefinition(
    name="hop_paths",
    category="paths",
    description="Direct and 2-hop path counts",
    algorithm_class="HopPathsAlgorithm",
    cost="medium",
    output_columns=["paths_length_1", "paths_length_2_targets"],
))

register_metric(MetricDefinition(
    name="eccentricity",
    category="paths",
    description="Maximum distance to any other node",
    algorithm_class="EccentricityAlgorithm",
    graph_type="undirected",
    cost="high",
    output_columns=["eccentricity"],
))

register_metric(MetricDefinition(
    name="wiener_contribution",
    category="paths",
    description="Node's contribution to Wiener index",
    algorithm_class="WienerContributionAlgorithm",
    graph_type="undirected",
    cost="high",
    max_nodes=50000,
    output_columns=["wiener_contribution"],
))

# =============================================================================
# DISTANCE METRICS
# =============================================================================

register_metric(MetricDefinition(
    name="graph_distances",
    category="distances",
    description="Graph radius, diameter, center and periphery membership",
    algorithm_class="GraphDistancesAlgorithm",
    graph_type="undirected",
    cost="high",
    output_columns=["graph_radius", "graph_diameter", "is_center", "is_periphery"],
))

# =============================================================================
# STRUCTURAL METRICS
# =============================================================================

register_metric(MetricDefinition(
    name="structural_holes",
    category="structural",
    description="Burt's structural holes (constraint, effective size)",
    algorithm_class="StructuralHolesAlgorithm",
    graph_type="undirected",
    cost="medium",
    output_columns=["constraint", "effective_size", "redundancy"],
))

register_metric(MetricDefinition(
    name="articulation_points",
    category="structural",
    description="Whether node is an articulation point",
    algorithm_class="ArticulationPointsAlgorithm",
    graph_type="undirected",
    cost="medium",
    output_columns=["is_articulation_point"],
))

register_metric(MetricDefinition(
    name="bridges",
    category="structural",
    description="Number of bridge edges incident to node",
    algorithm_class="BridgesAlgorithm",
    graph_type="undirected",
    cost="medium",
    output_columns=["bridge_count"],
))

register_metric(MetricDefinition(
    name="neighbor_degree_stats",
    category="structural",
    description="Statistics of neighbor degrees",
    algorithm_class="NeighborDegreeStatsAlgorithm",
    cost="medium",
    output_columns=[
        "avg_neighbor_degree", "min_neighbor_degree", "max_neighbor_degree",
        "std_neighbor_degree", "avg_neighbor_degree_undirected", "avg_neighbor_degree_directed"
    ],
))

register_metric(MetricDefinition(
    name="biconnected_component",
    category="structural",
    description="Biconnected component membership",
    algorithm_class="BiconnectedComponentAlgorithm",
    graph_type="undirected",
    cost="medium",
    output_columns=["biconnected_component_id", "biconnected_component_size"],
))

# =============================================================================
# RECIPROCITY METRICS
# =============================================================================

register_metric(MetricDefinition(
    name="reciprocity",
    category="reciprocity",
    description="Mutual connection statistics",
    algorithm_class="ReciprocityAlgorithm",
    cost="low",
    output_columns=[
        "mutual_count", "mutual_ratio", "mutual_received_ratio",
        "one_way_out", "one_way_in"
    ],
))

# =============================================================================
# REACH METRICS
# =============================================================================

register_metric(MetricDefinition(
    name="reach",
    category="reach",
    description="N-hop reachability and network penetration",
    algorithm_class="ReachAlgorithm",
    cost="high",
    output_columns=[
        "reach_hop_1", "reach_hop_2", "reach_hop_3",
        "reach_hop_4", "reach_hop_5", "reach_hop_6",
        "total_reach", "network_penetration"
    ],
))

# =============================================================================
# COMPONENT METRICS
# =============================================================================

register_metric(MetricDefinition(
    name="components",
    category="components",
    description="Weak and strong component membership",
    algorithm_class="ComponentsAlgorithm",
    cost="low",
    output_columns=[
        "weak_component_size", "strong_component_size", "in_largest_component"
    ],
))

# NEW: Strongly Connected Components Analysis
register_metric(MetricDefinition(
    name="strongly_connected_components",
    category="components",
    description="Detailed strongly connected component analysis",
    algorithm_class="StronglyConnectedComponentsAlgorithm",
    cost="low",
    output_columns=[
        "strong_component_id", "scc_size", "in_largest_scc",
        "scc_is_trivial", "scc_condensation_id"
    ],
    source="networkx",
))

# NEW: Condensation Graph Analysis
register_metric(MetricDefinition(
    name="condensation_graph",
    category="components",
    description="Condensation DAG analysis for hierarchy detection",
    algorithm_class="CondensationGraphAlgorithm",
    cost="medium",
    output_columns=[
        "condensation_in_degree", "condensation_out_degree",
        "condensation_is_root", "condensation_is_leaf", "condensation_depth"
    ],
    source="networkx",
))

# NEW: Attracting Components
register_metric(MetricDefinition(
    name="attracting_components",
    category="components",
    description="Attracting (sink) components analysis",
    algorithm_class="AttractingComponentsAlgorithm",
    cost="low",
    output_columns=[
        "in_attracting_component", "attracting_component_id", "attracting_component_size"
    ],
    source="networkx",
))

# NEW: Biconnected Components Analysis
register_metric(MetricDefinition(
    name="biconnected_components",
    category="components",
    description="Biconnected components analysis",
    algorithm_class="BiconnectedComponentsAlgorithm",
    graph_type="undirected",
    cost="medium",
    output_columns=["biconnected_component_count", "max_biconnected_size"],
    source="networkx",
))

# =============================================================================
# VITALITY METRICS
# =============================================================================

register_metric(MetricDefinition(
    name="closeness_vitality",
    category="vitality",
    description="Change in Wiener index when node removed",
    algorithm_class="ClosenessVitalityAlgorithm",
    graph_type="undirected",
    cost="very_high",
    max_nodes=50000,
    requires_connected=True,
    output_columns=["closeness_vitality"],
))

# =============================================================================
# DISPERSION METRICS
# =============================================================================

register_metric(MetricDefinition(
    name="dispersion",
    category="dispersion",
    description="Dispersion of node's neighborhood",
    algorithm_class="DispersionAlgorithm",
    graph_type="undirected",
    cost="very_high",
    max_nodes=10000,
    output_columns=["avg_dispersion", "max_dispersion"],
))

# =============================================================================
# EFFICIENCY METRICS
# =============================================================================

register_metric(MetricDefinition(
    name="local_efficiency",
    category="efficiency",
    description="Local efficiency of node's neighborhood",
    algorithm_class="LocalEfficiencyAlgorithm",
    graph_type="undirected",
    cost="high",
    output_columns=["local_efficiency"],
))

# NEW: Global Efficiency Contribution
register_metric(MetricDefinition(
    name="global_efficiency_contribution",
    category="efficiency",
    description="Node's contribution to global network efficiency",
    algorithm_class="GlobalEfficiencyContributionAlgorithm",
    graph_type="undirected",
    cost="very_high",
    max_nodes=200000,
    output_columns=["global_efficiency_contribution", "global_efficiency_ratio"],
    source="networkx",
))

# NEW: Node Efficiency
register_metric(MetricDefinition(
    name="node_efficiency",
    category="efficiency",
    description="Per-node communication efficiency",
    algorithm_class="NodeEfficiencyAlgorithm",
    cost="high",
    max_nodes=100000,
    output_columns=["node_efficiency", "node_efficiency_in"],
    source="networkx",
))

# NEW: Robustness Efficiency
register_metric(MetricDefinition(
    name="robustness_efficiency",
    category="efficiency",
    description="Efficiency-based robustness analysis",
    algorithm_class="RobustnessEfficiencyAlgorithm",
    graph_type="undirected",
    cost="high",
    max_nodes=30000,
    output_columns=["efficiency_criticality", "efficiency_redundancy"],
    source="networkx",
))

# NEW: Routing Efficiency
register_metric(MetricDefinition(
    name="routing_efficiency",
    category="efficiency",
    description="Routing efficiency through each node",
    algorithm_class="RoutingEfficiencyAlgorithm",
    cost="high",
    max_nodes=50000,
    output_columns=["routing_efficiency", "path_diversity"],
    source="networkx",
))

# =============================================================================
# FLOW METRICS
# =============================================================================

register_metric(MetricDefinition(
    name="flow_hierarchy",
    category="flow",
    description="Flow hierarchy of directed graph",
    algorithm_class="FlowHierarchyAlgorithm",
    cost="medium",
    output_columns=["flow_hierarchy"],
))

# NEW: Max Flow Centrality
register_metric(MetricDefinition(
    name="max_flow_centrality",
    category="flow",
    description="Maximum flow centrality based on flow capacity",
    algorithm_class="MaxFlowCentralityAlgorithm",
    cost="very_high",
    max_nodes=100000,
    output_columns=["max_flow_in", "max_flow_out", "max_flow_centrality"],
    source="networkx",
    parameters=[
        MetricParameter(
            name="sample_size",
            type="int",
            default=20,
            description="Number of source/sink samples for flow computation",
            min_value=5,
            max_value=50,
            step=5
        ),
    ]
))

# NEW: Flow Betweenness
register_metric(MetricDefinition(
    name="flow_betweenness",
    category="flow",
    description="Flow-based betweenness centrality",
    algorithm_class="FlowBetweennessAlgorithm",
    cost="high",
    max_nodes=50000,
    output_columns=["flow_betweenness", "flow_bottleneck_score"],
    source="networkx",
))

# NEW: Hierarchy Level
register_metric(MetricDefinition(
    name="hierarchy_level",
    category="flow",
    description="Hierarchical level in directed graph",
    algorithm_class="HierarchyLevelAlgorithm",
    cost="medium",
    output_columns=["hierarchy_level", "is_source", "is_sink", "hierarchy_depth"],
    source="networkx",
))

# NEW: Cycle Participation
register_metric(MetricDefinition(
    name="cycle_participation",
    category="flow",
    description="Cycle participation analysis",
    algorithm_class="CycleParticipationAlgorithm",
    cost="medium",
    output_columns=["in_cycle", "cycle_count_estimate", "scc_participation"],
    source="networkx",
))

# NEW: Min Cut Centrality
register_metric(MetricDefinition(
    name="min_cut_centrality",
    category="flow",
    description="Minimum cut based centrality",
    algorithm_class="MinCutCentralityAlgorithm",
    graph_type="undirected",
    cost="very_high",
    max_nodes=50000,
    output_columns=["min_cut_frequency", "cut_vertex_centrality"],
    source="networkx",
    parameters=[
        MetricParameter(
            name="num_pairs",
            type="int",
            default=100,
            description="Number of random pairs to test",
            min_value=20,
            max_value=500,
            step=20
        ),
    ]
))

# =============================================================================
# DOMINANCE METRICS
# =============================================================================

register_metric(MetricDefinition(
    name="dominance",
    category="dominance",
    description="Dominated nodes count and ratio",
    algorithm_class="DominanceAlgorithm",
    cost="high",
    output_columns=["dominated_nodes_count", "dominance_ratio"],
))

# =============================================================================
# TRUST METRICS
# =============================================================================

register_metric(MetricDefinition(
    name="eigentrust",
    category="trust",
    description="EigenTrust score for social trust networks",
    algorithm_class="EigenTrustAlgorithm",
    source="custom",
    cost="medium",
    output_columns=["eigentrust"],
    citation="Kamvar et al., 2003",
    parameters=[
        MetricParameter(
            name="epsilon",
            type="float",
            default=0.01,
            description="Convergence threshold",
            min_value=0.0001,
            max_value=0.1,
            step=0.001
        ),
        MetricParameter(
            name="max_iter",
            type="int",
            default=100,
            description="Maximum number of iterations",
            min_value=10,
            max_value=1000,
            step=10
        ),
    ]
))

register_metric(MetricDefinition(
    name="appleseed",
    category="trust",
    description="Appleseed trust propagation with energy decay",
    algorithm_class="AppleseedAlgorithm",
    source="custom",
    cost="medium",
    output_columns=["appleseed"],
    citation="Ziegler and Lausen, 2005",
))

# =============================================================================
# SIMILARITY METRICS
# =============================================================================

register_metric(MetricDefinition(
    name="jaccard_similarity",
    category="similarity",
    description="Average Jaccard similarity with neighbors",
    algorithm_class="JaccardSimilarityAlgorithm",
    graph_type="undirected",
    cost="high",
    output_columns=["jaccard_similarity_avg", "jaccard_similarity_max"],
))

register_metric(MetricDefinition(
    name="cosine_similarity",
    category="similarity",
    description="Average cosine similarity with neighbors",
    algorithm_class="CosineSimilarityAlgorithm",
    graph_type="undirected",
    cost="high",
    output_columns=["cosine_similarity_avg", "cosine_similarity_max"],
))

register_metric(MetricDefinition(
    name="adamic_adar",
    category="similarity",
    description="Adamic-Adar similarity scores",
    algorithm_class="AdamicAdarAlgorithm",
    graph_type="undirected",
    cost="high",
    output_columns=["adamic_adar_sum", "adamic_adar_avg"],
))

register_metric(MetricDefinition(
    name="resource_allocation",
    category="similarity",
    description="Resource allocation index",
    algorithm_class="ResourceAllocationAlgorithm",
    graph_type="undirected",
    cost="high",
    output_columns=["resource_allocation_sum", "resource_allocation_avg"],
))

# =============================================================================
# LINK PREDICTION METRICS (NEW CATEGORY)
# =============================================================================

register_metric(MetricDefinition(
    name="common_neighbors",
    category="link_prediction",
    description="Common neighbors score for link prediction",
    algorithm_class="CommonNeighborsAlgorithm",
    graph_type="undirected",
    cost="high",
    output_columns=["common_neighbors_sum", "common_neighbors_max"],
    source="networkx",
))

register_metric(MetricDefinition(
    name="preferential_attachment",
    category="link_prediction",
    description="Preferential attachment score for link prediction",
    algorithm_class="PreferentialAttachmentAlgorithm",
    graph_type="undirected",
    cost="medium",
    output_columns=["preferential_attachment_score"],
    source="networkx",
))

register_metric(MetricDefinition(
    name="link_prediction_scores",
    category="link_prediction",
    description="Aggregated link prediction scores (AA, RA, Jaccard)",
    algorithm_class="LinkPredictionScoresAlgorithm",
    graph_type="undirected",
    cost="high",
    max_nodes=50000,
    output_columns=["link_pred_adamic_adar", "link_pred_resource_alloc", "link_pred_jaccard"],
    source="networkx",
    parameters=[
        MetricParameter(
            name="top_k",
            type="int",
            default=10,
            description="Number of top predictions to consider per node",
            min_value=5,
            max_value=50,
            step=5
        ),
    ]
))

# =============================================================================
# ROBUSTNESS METRICS (NEW CATEGORY)
# =============================================================================

register_metric(MetricDefinition(
    name="node_connectivity",
    category="robustness",
    description="Minimum nodes to disconnect from sampled targets",
    algorithm_class="NodeConnectivityAlgorithm",
    graph_type="undirected",
    cost="high",
    max_nodes=10000,
    output_columns=["node_connectivity"],
    source="networkx",
))

register_metric(MetricDefinition(
    name="edge_connectivity",
    category="robustness",
    description="Minimum edges to disconnect from sampled targets",
    algorithm_class="EdgeConnectivityAlgorithm",
    graph_type="undirected",
    cost="high",
    max_nodes=10000,
    output_columns=["edge_connectivity"],
    source="networkx",
))

register_metric(MetricDefinition(
    name="resilience_score",
    category="robustness",
    description="Impact on largest connected component when node removed",
    algorithm_class="ResilienceScoreAlgorithm",
    graph_type="undirected",
    cost="high",
    max_nodes=20000,
    output_columns=["resilience_score", "lcc_reduction"],
    source="networkx",
))

# =============================================================================
# SPECTRAL METRICS (NEW CATEGORY)
# =============================================================================

register_metric(MetricDefinition(
    name="fiedler_vector",
    category="spectral",
    description="Second eigenvector of Laplacian (algebraic connectivity)",
    algorithm_class="FiedlerVectorAlgorithm",
    graph_type="undirected",
    cost="high",
    max_nodes=50000,
    requires_connected=True,
    output_columns=["fiedler_component", "fiedler_partition"],
    source="scipy",
))

register_metric(MetricDefinition(
    name="spectral_centrality",
    category="spectral",
    description="Top eigenvector of adjacency matrix",
    algorithm_class="SpectralCentralityAlgorithm",
    graph_type="undirected",
    cost="high",
    max_nodes=30000,
    output_columns=["spectral_centrality"],
    source="scipy",
))

# =============================================================================
# INFLUENCE METRICS (NEW CATEGORY)
# =============================================================================

register_metric(MetricDefinition(
    name="collective_influence",
    category="influence",
    description="Morone & Makse collective influence for network dismantling",
    algorithm_class="CollectiveInfluenceAlgorithm",
    cost="medium",
    output_columns=["collective_influence"],
    source="custom",
    citation="Morone & Makse, 2015",
    parameters=[
        MetricParameter(
            name="ball_radius",
            type="int",
            default=2,
            description="Radius for ball neighborhood",
            min_value=1,
            max_value=4,
            step=1
        ),
    ]
))

register_metric(MetricDefinition(
    name="spreading_activation",
    category="influence",
    description="Simulated information spreading influence",
    algorithm_class="SpreadingActivationAlgorithm",
    cost="medium",
    output_columns=["spreading_activation", "influence_reach"],
    source="custom",
    parameters=[
        MetricParameter(
            name="steps",
            type="int",
            default=3,
            description="Number of spreading steps",
            min_value=1,
            max_value=10,
            step=1
        ),
        MetricParameter(
            name="decay",
            type="float",
            default=0.5,
            description="Decay factor per step",
            min_value=0.1,
            max_value=0.9,
            step=0.1
        ),
    ]
))

# =============================================================================
# BIPARTITE METRICS (NEW CATEGORY)
# =============================================================================

register_metric(MetricDefinition(
    name="bipartite_projection_degree",
    category="bipartite",
    description="Bipartite projection metrics (or 2-hop proxy for non-bipartite)",
    algorithm_class="BipartiteProjectionDegreeAlgorithm",
    graph_type="undirected",
    cost="medium",
    output_columns=["bipartite_projection_degree", "bipartite_redundancy"],
    source="networkx",
))

# =============================================================================
# GRAPH COLORING METRICS (NEW CATEGORY)
# =============================================================================

register_metric(MetricDefinition(
    name="greedy_color",
    category="graph_coloring",
    description="Greedy graph coloring analysis",
    algorithm_class="GreedyColorAlgorithm",
    graph_type="undirected",
    cost="low",
    output_columns=["greedy_color", "color_class_size", "chromatic_estimate"],
    source="networkx",
    parameters=[
        MetricParameter(
            name="strategy",
            type="choice",
            default="largest_first",
            description="Coloring strategy",
            choices=["largest_first", "smallest_last", "independent_set", "connected_sequential"]
        ),
    ]
))

# =============================================================================
# IGRAPH-SPECIFIC METRICS (NEW CATEGORY)
# =============================================================================

register_metric(MetricDefinition(
    name="leiden_community",
    category="igraph_community",
    description="Leiden community detection (improved Louvain)",
    algorithm_class="LeidenCommunityAlgorithm",
    graph_type="undirected",
    cost="medium",
    output_columns=["leiden_community_id", "leiden_community_size"],
    source="igraph",
    requires_igraph=True,
    parameters=[
        MetricParameter(
            name="resolution",
            type="float",
            default=1.0,
            description="Resolution parameter",
            min_value=0.1,
            max_value=2.0,
            step=0.1
        ),
    ]
))

register_metric(MetricDefinition(
    name="infomap_community",
    category="igraph_community",
    description="Infomap community detection based on information flow",
    algorithm_class="InfomapCommunityAlgorithm",
    cost="medium",
    output_columns=["infomap_community_id", "infomap_community_size", "infomap_codelength"],
    source="igraph",
    requires_igraph=True,
))

register_metric(MetricDefinition(
    name="walktrap_community",
    category="igraph_community",
    description="Walktrap community detection based on random walks",
    algorithm_class="WalktrapCommunityAlgorithm",
    graph_type="undirected",
    cost="medium",
    output_columns=["walktrap_community_id", "walktrap_community_size"],
    source="igraph",
    requires_igraph=True,
    parameters=[
        MetricParameter(
            name="steps",
            type="int",
            default=4,
            description="Length of random walks",
            min_value=2,
            max_value=10,
            step=1
        ),
    ]
))

register_metric(MetricDefinition(
    name="fast_greedy_community",
    category="igraph_community",
    description="Fast greedy modularity optimization",
    algorithm_class="FastGreedyCommunityAlgorithm",
    graph_type="undirected",
    cost="medium",
    output_columns=["fast_greedy_community_id", "fast_greedy_community_size"],
    source="igraph",
    requires_igraph=True,
))

register_metric(MetricDefinition(
    name="spinglass_community",
    category="igraph_community",
    description="Spinglass community detection (statistical physics approach)",
    algorithm_class="SpinglassCommunityAlgorithm",
    graph_type="undirected",
    cost="high",
    max_nodes=200000,
    requires_connected=True,
    output_columns=["spinglass_community_id", "spinglass_community_size"],
    source="igraph",
    requires_igraph=True,
))

register_metric(MetricDefinition(
    name="alpha_centrality",
    category="igraph_centrality",
    description="Alpha centrality (generalized eigenvector centrality)",
    algorithm_class="AlphaCentralityAlgorithm",
    cost="medium",
    output_columns=["alpha_centrality"],
    source="igraph",
    requires_igraph=True,
))

register_metric(MetricDefinition(
    name="motif_count",
    category="motifs",
    description="3-node and 4-node motif counting",
    algorithm_class="MotifCountAlgorithm",
    cost="high",
    max_nodes=50000,
    output_columns=["motif3_count", "motif4_count"],
    source="igraph",
    requires_igraph=True,
))


# =============================================================================
# METRIC CATEGORIES
# =============================================================================

METRIC_CATEGORIES: Dict[str, Dict[str, Any]] = {
    'topology': {
        'description': 'Basic degree and structure metrics',
        'metrics': ['in_degree', 'out_degree', 'total_degree', 'degree_imbalance'],
    },
    'centrality': {
        'description': 'Node importance and influence measures',
        'metrics': [
            'degree_centrality', 'closeness_centrality', 'betweenness_centrality',
            'eigenvector_centrality', 'katz_centrality', 'pagerank', 'hits',
            'harmonic_centrality', 'load_centrality', 'subgraph_centrality',
            'second_order_centrality', 'percolation_centrality', 'trophic_level',
            'current_flow_centrality', 'information_centrality',
            'communicability_betweenness', 'voterank', 'edge_betweenness_sum',
            'laplacian_centrality', 'leverage_centrality', 'semi_local_centrality',
            'decay_centrality'
        ],
    },
    'clustering': {
        'description': 'Local connectivity and triangle formation',
        'metrics': [
            'clustering_coefficient', 'triangles', 'square_clustering', 
            'local_transitivity', 'clique_count', 'average_neighbor_clustering'
        ],
    },
    'community': {
        'description': 'Community and core structure detection',
        'metrics': [
            'louvain_community', 'core_number', 'onion_layer', 
            'local_reaching_centrality', 'label_propagation',
            'async_label_propagation', 'greedy_modularity_community',
            'participation_coefficient', 'within_module_degree'
        ],
    },
    'paths': {
        'description': 'Shortest path and reachability analysis',
        'metrics': ['shortest_paths', 'hop_paths', 'eccentricity', 'wiener_contribution'],
    },
    'distances': {
        'description': 'Network distance measures',
        'metrics': ['graph_distances'],
    },
    'structural': {
        'description': 'Structural holes and robustness',
        'metrics': [
            'structural_holes', 'articulation_points', 'bridges', 
            'neighbor_degree_stats', 'biconnected_component'
        ],
    },
    'reciprocity': {
        'description': 'Mutual connection patterns',
        'metrics': ['reciprocity'],
    },
    'reach': {
        'description': 'N-hop reachability metrics',
        'metrics': ['reach'],
    },
    'components': {
        'description': 'Component membership and hierarchy',
        'metrics': [
            'components', 'strongly_connected_components', 
            'condensation_graph', 'attracting_components', 'biconnected_components'
        ],
    },
    'vitality': {
        'description': 'Node removal impact',
        'metrics': ['closeness_vitality'],
    },
    'dispersion': {
        'description': 'Neighborhood spread patterns',
        'metrics': ['dispersion'],
    },
    'efficiency': {
        'description': 'Communication efficiency',
        'metrics': [
            'local_efficiency', 'global_efficiency_contribution',
            'node_efficiency', 'robustness_efficiency', 'routing_efficiency'
        ],
    },
    'flow': {
        'description': 'Flow, hierarchy, and cycles',
        'metrics': [
            'flow_hierarchy', 'max_flow_centrality', 'flow_betweenness',
            'hierarchy_level', 'cycle_participation', 'min_cut_centrality'
        ],
    },
    'dominance': {
        'description': 'Dominance relationships',
        'metrics': ['dominance'],
    },
    'trust': {
        'description': 'Trust network algorithms',
        'metrics': ['eigentrust', 'appleseed'],
    },
    'similarity': {
        'description': 'Node similarity measures',
        'metrics': ['jaccard_similarity', 'cosine_similarity', 'adamic_adar', 'resource_allocation'],
    },
    'link_prediction': {
        'description': 'Link prediction metrics',
        'metrics': ['common_neighbors', 'preferential_attachment', 'link_prediction_scores'],
    },
    'robustness': {
        'description': 'Network robustness and resilience',
        'metrics': ['node_connectivity', 'edge_connectivity', 'resilience_score'],
    },
    'spectral': {
        'description': 'Spectral graph theory metrics',
        'metrics': ['fiedler_vector', 'spectral_centrality'],
    },
    'influence': {
        'description': 'Influence maximization metrics',
        'metrics': ['collective_influence', 'spreading_activation'],
    },
    'bipartite': {
        'description': 'Bipartite graph metrics',
        'metrics': ['bipartite_projection_degree'],
    },
    'graph_coloring': {
        'description': 'Graph coloring analysis',
        'metrics': ['greedy_color'],
    },
    'igraph_community': {
        'description': 'igraph community detection algorithms',
        'metrics': [
            'leiden_community', 'infomap_community', 'walktrap_community',
            'fast_greedy_community', 'spinglass_community'
        ],
    },
    'igraph_centrality': {
        'description': 'igraph centrality algorithms',
        'metrics': ['alpha_centrality'],
    },
    'motifs': {
        'description': 'Network motif analysis',
        'metrics': ['motif_count'],
    },
}


# =============================================================================
# METRIC PRESETS
# =============================================================================

METRIC_PRESETS: Dict[str, Dict[str, Any]] = {
    'basic': {
        'description': 'Quick overview metrics',
        'categories': ['topology'],
        'metrics': ['louvain_community', 'core_number'],
    },
    'essential': {
        'description': 'Standard analysis metrics',
        'categories': ['topology', 'clustering'],
        'metrics': [
            'pagerank', 'betweenness_centrality', 'eigenvector_centrality',
            'louvain_community', 'core_number', 'onion_layer'
        ],
    },
    'moderate': {
        'description': 'Detailed analysis',
        'categories': ['topology', 'clustering', 'reciprocity', 'components'],
        'metrics': [
            'degree_centrality', 'closeness_centrality', 'betweenness_centrality',
            'eigenvector_centrality', 'pagerank', 'hits', 'harmonic_centrality',
            'louvain_community', 'core_number', 'onion_layer',
            'structural_holes', 'articulation_points', 'bridges'
        ],
    },
    'comprehensive': {
        'description': 'Full analysis (excluding very expensive)',
        'categories': ['topology', 'clustering', 'community', 'reciprocity', 'components', 'structural'],
        'metrics': [
            'degree_centrality', 'closeness_centrality', 'betweenness_centrality',
            'eigenvector_centrality', 'katz_centrality', 'pagerank', 'hits',
            'harmonic_centrality', 'load_centrality', 'subgraph_centrality',
            'shortest_paths', 'hop_paths', 'reach', 'graph_distances',
            'laplacian_centrality', 'leverage_centrality', 'semi_local_centrality'
        ],
    },
    'all': {
        'description': 'All available metrics',
        'categories': list(METRIC_CATEGORIES.keys()),
        'metrics': [],  # Empty means all from categories
    },
    'trust_analysis': {
        'description': 'Trust network focused analysis',
        'categories': ['topology'],
        'metrics': [
            'pagerank', 'eigenvector_centrality', 'eigentrust', 'appleseed',
            'louvain_community', 'core_number', 'reciprocity'
        ],
    },
    'influence': {
        'description': 'Node influence and importance',
        'categories': ['topology'],
        'metrics': [
            'pagerank', 'betweenness_centrality', 'eigenvector_centrality',
            'katz_centrality', 'hits', 'voterank', 'collective_influence',
            'spreading_activation'
        ],
    },
    'structure': {
        'description': 'Network structure analysis',
        'categories': ['topology', 'structural', 'components'],
        'metrics': [
            'louvain_community', 'core_number', 'biconnected_component',
            'graph_distances', 'eccentricity', 'strongly_connected_components',
            'condensation_graph'
        ],
    },
    'community_detection': {
        'description': 'Compare multiple community detection methods',
        'categories': ['topology'],
        'metrics': [
            'louvain_community', 'label_propagation', 'greedy_modularity_community',
            'core_number', 'onion_layer', 'participation_coefficient',
            'within_module_degree'
        ],
    },
    'robustness_analysis': {
        'description': 'Network robustness and resilience analysis',
        'categories': ['topology', 'structural'],
        'metrics': [
            'node_connectivity', 'edge_connectivity', 'resilience_score',
            'articulation_points', 'bridges', 'efficiency_criticality',
            'global_efficiency_contribution'
        ],
    },
    'link_prediction': {
        'description': 'Link prediction focused analysis',
        'categories': ['topology', 'similarity'],
        'metrics': [
            'common_neighbors', 'preferential_attachment', 'link_prediction_scores',
            'jaccard_similarity', 'adamic_adar', 'resource_allocation'
        ],
    },
    'hierarchy': {
        'description': 'Network hierarchy analysis',
        'categories': ['topology'],
        'metrics': [
            'flow_hierarchy', 'hierarchy_level', 'trophic_level',
            'condensation_graph', 'cycle_participation', 'strongly_connected_components'
        ],
    },
    'igraph_advanced': {
        'description': 'Advanced igraph algorithms (requires igraph)',
        'categories': ['topology'],
        'metrics': [
            'leiden_community', 'infomap_community', 'walktrap_community',
            'alpha_centrality', 'motif_count'
        ],
    },
    'spectral': {
        'description': 'Spectral graph analysis',
        'categories': ['topology'],
        'metrics': [
            'fiedler_vector', 'spectral_centrality', 'eigenvector_centrality',
            'laplacian_centrality'
        ],
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_metric(name: str) -> Optional[MetricDefinition]:
    """Get metric definition by name."""
    return METRIC_REGISTRY.get(name)


def get_category_metrics(category: str) -> List[str]:
    """Get list of metric names in a category."""
    if category not in METRIC_CATEGORIES:
        return []
    return METRIC_CATEGORIES[category]['metrics']


def get_preset_metrics(preset: str) -> List[str]:
    """Get list of metric names in a preset."""
    if preset not in METRIC_PRESETS:
        return []
    
    preset_def = METRIC_PRESETS[preset]
    metrics = set()
    
    # Add metrics from categories
    for category in preset_def.get('categories', []):
        metrics.update(get_category_metrics(category))
    
    # Add explicit metrics
    metrics.update(preset_def.get('metrics', []))
    
    return list(metrics)


def list_all_metrics() -> List[Dict[str, Any]]:
    """List all available metrics with metadata."""
    return [
        {
            'name': m.name,
            'category': m.category,
            'description': m.description,
            'cost': m.cost,
            'max_nodes': m.max_nodes,
            'output_columns': m.output_columns,
            'requires_igraph': m.requires_igraph,
            'source': m.source,
        }
        for m in METRIC_REGISTRY.values()
    ]


def list_categories() -> List[Dict[str, Any]]:
    """List all categories with descriptions."""
    return [
        {
            'name': name,
            'description': cat['description'],
            'metric_count': len(cat['metrics']),
            'metrics': cat['metrics'],
        }
        for name, cat in METRIC_CATEGORIES.items()
    ]


def list_presets() -> List[Dict[str, Any]]:
    """List all presets with descriptions."""
    result = []
    for name, preset in METRIC_PRESETS.items():
        metrics = get_preset_metrics(name)
        result.append({
            'name': name,
            'description': preset['description'],
            'metric_count': len(metrics),
            'categories': preset.get('categories', []),
        })
    return result


def get_igraph_metrics() -> List[str]:
    """Get list of metrics that require igraph."""
    return [m.name for m in METRIC_REGISTRY.values() if m.requires_igraph]


def get_metrics_by_cost(cost: str) -> List[str]:
    """Get list of metrics with a specific cost level."""
    return [m.name for m in METRIC_REGISTRY.values() if m.cost == cost]


def get_metrics_by_source(source: str) -> List[str]:
    """Get list of metrics from a specific source library."""
    return [m.name for m in METRIC_REGISTRY.values() if m.source == source]