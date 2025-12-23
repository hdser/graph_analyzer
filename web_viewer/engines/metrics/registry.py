"""
Metric Registry

Central registry of all available metrics with their definitions,
categories, presets, and configurable parameters.
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
    source: str = "networkx"           # "networkx", "custom", "scipy"
    citation: Optional[str] = None
    
    # Parameters (NEW)
    parameters: List[MetricParameter] = field(default_factory=list)


# =============================================================================
# METRIC REGISTRY - All available metrics
# =============================================================================

METRIC_REGISTRY: Dict[str, MetricDefinition] = {}


def register_metric(definition: MetricDefinition) -> MetricDefinition:
    """Register a metric definition."""
    METRIC_REGISTRY[definition.name] = definition
    return definition


# -----------------------------------------------------------------------------
# TOPOLOGY METRICS
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# CENTRALITY METRICS
# -----------------------------------------------------------------------------

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
    max_nodes=1000,
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
    max_nodes=1000,
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
    max_nodes=1000,
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
    max_nodes=500,
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

# -----------------------------------------------------------------------------
# CLUSTERING METRICS
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# COMMUNITY METRICS
# -----------------------------------------------------------------------------

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
    max_nodes=500,
    output_columns=["local_reaching_centrality"],
))

# -----------------------------------------------------------------------------
# PATH METRICS
# -----------------------------------------------------------------------------

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
    max_nodes=500,
    output_columns=["wiener_contribution"],
))

# -----------------------------------------------------------------------------
# DISTANCE METRICS
# -----------------------------------------------------------------------------

register_metric(MetricDefinition(
    name="graph_distances",
    category="distances",
    description="Graph radius, diameter, center and periphery membership",
    algorithm_class="GraphDistancesAlgorithm",
    graph_type="undirected",
    cost="high",
    output_columns=["graph_radius", "graph_diameter", "is_center", "is_periphery"],
))

# -----------------------------------------------------------------------------
# STRUCTURAL METRICS
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# RECIPROCITY METRICS
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# REACH METRICS
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# COMPONENT METRICS
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# VITALITY METRICS
# -----------------------------------------------------------------------------

register_metric(MetricDefinition(
    name="closeness_vitality",
    category="vitality",
    description="Change in Wiener index when node removed",
    algorithm_class="ClosenessVitalityAlgorithm",
    graph_type="undirected",
    cost="very_high",
    max_nodes=500,
    requires_connected=True,
    output_columns=["closeness_vitality"],
))

# -----------------------------------------------------------------------------
# DISPERSION METRICS
# -----------------------------------------------------------------------------

register_metric(MetricDefinition(
    name="dispersion",
    category="dispersion",
    description="Dispersion of node's neighborhood",
    algorithm_class="DispersionAlgorithm",
    graph_type="undirected",
    cost="very_high",
    max_nodes=100,
    output_columns=["avg_dispersion", "max_dispersion"],
))

# -----------------------------------------------------------------------------
# EFFICIENCY METRICS
# -----------------------------------------------------------------------------

register_metric(MetricDefinition(
    name="local_efficiency",
    category="efficiency",
    description="Local efficiency of node's neighborhood",
    algorithm_class="LocalEfficiencyAlgorithm",
    graph_type="undirected",
    cost="high",
    output_columns=["local_efficiency"],
))

# -----------------------------------------------------------------------------
# FLOW METRICS
# -----------------------------------------------------------------------------

register_metric(MetricDefinition(
    name="flow_hierarchy",
    category="flow",
    description="Flow hierarchy of directed graph",
    algorithm_class="FlowHierarchyAlgorithm",
    cost="medium",
    output_columns=["flow_hierarchy"],
))

# -----------------------------------------------------------------------------
# DOMINANCE METRICS
# -----------------------------------------------------------------------------

register_metric(MetricDefinition(
    name="dominance",
    category="dominance",
    description="Dominated nodes count and ratio",
    algorithm_class="DominanceAlgorithm",
    cost="high",
    output_columns=["dominated_nodes_count", "dominance_ratio"],
))

# -----------------------------------------------------------------------------
# TRUST METRICS (NEW)
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# SIMILARITY METRICS (NEW)
# -----------------------------------------------------------------------------

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
            'communicability_betweenness', 'voterank', 'edge_betweenness_sum'
        ],
    },
    'clustering': {
        'description': 'Local connectivity and triangle formation',
        'metrics': ['clustering_coefficient', 'triangles', 'square_clustering', 'local_transitivity'],
    },
    'community': {
        'description': 'Community and core structure detection',
        'metrics': ['louvain_community', 'core_number', 'onion_layer', 'local_reaching_centrality'],
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
        'metrics': ['structural_holes', 'articulation_points', 'bridges', 'neighbor_degree_stats', 'biconnected_component'],
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
        'description': 'Component membership',
        'metrics': ['components'],
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
        'metrics': ['local_efficiency'],
    },
    'flow': {
        'description': 'Flow and hierarchy',
        'metrics': ['flow_hierarchy'],
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
            'shortest_paths', 'hop_paths', 'reach', 'graph_distances'
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
            'katz_centrality', 'hits', 'voterank'
        ],
    },
    'structure': {
        'description': 'Network structure analysis',
        'categories': ['topology', 'structural', 'components'],
        'metrics': [
            'louvain_community', 'core_number', 'biconnected_component',
            'graph_distances', 'eccentricity'
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