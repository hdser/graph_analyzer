"""
Metric Algorithms Package

Contains implementations of all graph metric algorithms.
Each algorithm is a class that inherits from BaseMetricAlgorithm.
"""

from .base import BaseMetricAlgorithm
from .topology import (
    InDegreeAlgorithm,
    OutDegreeAlgorithm,
    TotalDegreeAlgorithm,
    DegreeImbalanceAlgorithm,
)
from .centrality import (
    DegreeCentralityAlgorithm,
    ClosenessCentralityAlgorithm,
    BetweennessCentralityAlgorithm,
    EigenvectorCentralityAlgorithm,
    KatzCentralityAlgorithm,
    PageRankAlgorithm,
    HITSAlgorithm,
    HarmonicCentralityAlgorithm,
    LoadCentralityAlgorithm,
    SubgraphCentralityAlgorithm,
    SecondOrderCentralityAlgorithm,
    PercolationCentralityAlgorithm,
    TrophicLevelAlgorithm,
    CurrentFlowCentralityAlgorithm,
    InformationCentralityAlgorithm,
    CommunicabilityBetweennessAlgorithm,
    VoteRankAlgorithm,
    EdgeBetweennessSumAlgorithm,
)
from .clustering import (
    ClusteringCoefficientAlgorithm,
    TrianglesAlgorithm,
    SquareClusteringAlgorithm,
    LocalTransitivityAlgorithm,
)
from .community import (
    LouvainCommunityAlgorithm,
    CoreNumberAlgorithm,
    OnionLayerAlgorithm,
    LocalReachingCentralityAlgorithm,
)
from .paths import (
    ShortestPathsAlgorithm,
    HopPathsAlgorithm,
    EccentricityAlgorithm,
    WienerContributionAlgorithm,
)
from .distances import GraphDistancesAlgorithm
from .structural import (
    StructuralHolesAlgorithm,
    ArticulationPointsAlgorithm,
    BridgesAlgorithm,
    NeighborDegreeStatsAlgorithm,
    BiconnectedComponentAlgorithm,
)
from .reciprocity import ReciprocityAlgorithm
from .reach import ReachAlgorithm
from .components import ComponentsAlgorithm
from .vitality import ClosenessVitalityAlgorithm
from .dispersion import DispersionAlgorithm
from .efficiency import LocalEfficiencyAlgorithm
from .flow import FlowHierarchyAlgorithm
from .dominance import DominanceAlgorithm
from .trust import EigenTrustAlgorithm, AppleseedAlgorithm
from .similarity import (
    JaccardSimilarityAlgorithm,
    CosineSimilarityAlgorithm,
    AdamicAdarAlgorithm,
    ResourceAllocationAlgorithm,
)


# Algorithm class registry - maps class names to classes
ALGORITHM_CLASSES = {
    # Topology
    "InDegreeAlgorithm": InDegreeAlgorithm,
    "OutDegreeAlgorithm": OutDegreeAlgorithm,
    "TotalDegreeAlgorithm": TotalDegreeAlgorithm,
    "DegreeImbalanceAlgorithm": DegreeImbalanceAlgorithm,
    # Centrality
    "DegreeCentralityAlgorithm": DegreeCentralityAlgorithm,
    "ClosenessCentralityAlgorithm": ClosenessCentralityAlgorithm,
    "BetweennessCentralityAlgorithm": BetweennessCentralityAlgorithm,
    "EigenvectorCentralityAlgorithm": EigenvectorCentralityAlgorithm,
    "KatzCentralityAlgorithm": KatzCentralityAlgorithm,
    "PageRankAlgorithm": PageRankAlgorithm,
    "HITSAlgorithm": HITSAlgorithm,
    "HarmonicCentralityAlgorithm": HarmonicCentralityAlgorithm,
    "LoadCentralityAlgorithm": LoadCentralityAlgorithm,
    "SubgraphCentralityAlgorithm": SubgraphCentralityAlgorithm,
    "SecondOrderCentralityAlgorithm": SecondOrderCentralityAlgorithm,
    "PercolationCentralityAlgorithm": PercolationCentralityAlgorithm,
    "TrophicLevelAlgorithm": TrophicLevelAlgorithm,
    "CurrentFlowCentralityAlgorithm": CurrentFlowCentralityAlgorithm,
    "InformationCentralityAlgorithm": InformationCentralityAlgorithm,
    "CommunicabilityBetweennessAlgorithm": CommunicabilityBetweennessAlgorithm,
    "VoteRankAlgorithm": VoteRankAlgorithm,
    "EdgeBetweennessSumAlgorithm": EdgeBetweennessSumAlgorithm,
    # Clustering
    "ClusteringCoefficientAlgorithm": ClusteringCoefficientAlgorithm,
    "TrianglesAlgorithm": TrianglesAlgorithm,
    "SquareClusteringAlgorithm": SquareClusteringAlgorithm,
    "LocalTransitivityAlgorithm": LocalTransitivityAlgorithm,
    # Community
    "LouvainCommunityAlgorithm": LouvainCommunityAlgorithm,
    "CoreNumberAlgorithm": CoreNumberAlgorithm,
    "OnionLayerAlgorithm": OnionLayerAlgorithm,
    "LocalReachingCentralityAlgorithm": LocalReachingCentralityAlgorithm,
    # Paths
    "ShortestPathsAlgorithm": ShortestPathsAlgorithm,
    "HopPathsAlgorithm": HopPathsAlgorithm,
    "EccentricityAlgorithm": EccentricityAlgorithm,
    "WienerContributionAlgorithm": WienerContributionAlgorithm,
    # Distances
    "GraphDistancesAlgorithm": GraphDistancesAlgorithm,
    # Structural
    "StructuralHolesAlgorithm": StructuralHolesAlgorithm,
    "ArticulationPointsAlgorithm": ArticulationPointsAlgorithm,
    "BridgesAlgorithm": BridgesAlgorithm,
    "NeighborDegreeStatsAlgorithm": NeighborDegreeStatsAlgorithm,
    "BiconnectedComponentAlgorithm": BiconnectedComponentAlgorithm,
    # Reciprocity
    "ReciprocityAlgorithm": ReciprocityAlgorithm,
    # Reach
    "ReachAlgorithm": ReachAlgorithm,
    # Components
    "ComponentsAlgorithm": ComponentsAlgorithm,
    # Vitality
    "ClosenessVitalityAlgorithm": ClosenessVitalityAlgorithm,
    # Dispersion
    "DispersionAlgorithm": DispersionAlgorithm,
    # Efficiency
    "LocalEfficiencyAlgorithm": LocalEfficiencyAlgorithm,
    # Flow
    "FlowHierarchyAlgorithm": FlowHierarchyAlgorithm,
    # Dominance
    "DominanceAlgorithm": DominanceAlgorithm,
    # Trust
    "EigenTrustAlgorithm": EigenTrustAlgorithm,
    "AppleseedAlgorithm": AppleseedAlgorithm,
    # Similarity
    "JaccardSimilarityAlgorithm": JaccardSimilarityAlgorithm,
    "CosineSimilarityAlgorithm": CosineSimilarityAlgorithm,
    "AdamicAdarAlgorithm": AdamicAdarAlgorithm,
    "ResourceAllocationAlgorithm": ResourceAllocationAlgorithm,
}


def get_algorithm_class(class_name: str):
    """Get algorithm class by name."""
    return ALGORITHM_CLASSES.get(class_name)


__all__ = [
    "BaseMetricAlgorithm",
    "ALGORITHM_CLASSES",
    "get_algorithm_class",
]