"""
Metric Algorithms Package

Central registry of all algorithm classes.
Imports from individual algorithm modules and provides unified access.
"""

from typing import Dict, Type, Optional
import logging

logger = logging.getLogger(__name__)

# Import base class
from .base import BaseMetricAlgorithm

# Import from topology module
from .topology import (
    InDegreeAlgorithm,
    OutDegreeAlgorithm,
    TotalDegreeAlgorithm,
    DegreeImbalanceAlgorithm,
)

# Import from centrality module
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
    LaplacianCentralityAlgorithm,
    LeverageCentralityAlgorithm,
    SemiLocalCentralityAlgorithm,
    DecayCentralityAlgorithm,
)

# Import from clustering module
from .clustering import (
    ClusteringCoefficientAlgorithm,
    TrianglesAlgorithm,
    SquareClusteringAlgorithm,
    LocalTransitivityAlgorithm,
    CliqueCountAlgorithm,
    AverageNeighborClusteringAlgorithm,
)

# Import from community module
from .community import (
    LouvainCommunityAlgorithm,
    CoreNumberAlgorithm,
    OnionLayerAlgorithm,
    LocalReachingCentralityAlgorithm,
    LabelPropagationAlgorithm,
    AsyncLabelPropagationAlgorithm,
    GreedyModularityCommunityAlgorithm,
    ParticipationCoefficientAlgorithm,
    WithinModuleDegreeAlgorithm,
)

# Import from paths module
from .paths import (
    ShortestPathsAlgorithm,
    HopPathsAlgorithm,
    EccentricityAlgorithm,
    WienerContributionAlgorithm,
)

# Import from distances module
from .distances import (
    GraphDistancesAlgorithm,
)

# Import from structural module
from .structural import (
    StructuralHolesAlgorithm,
    ArticulationPointsAlgorithm,
    BridgesAlgorithm,
    NeighborDegreeStatsAlgorithm,
    BiconnectedComponentAlgorithm,
)

# Import from reciprocity module
from .reciprocity import (
    ReciprocityAlgorithm,
)

# Import from reach module
from .reach import (
    ReachAlgorithm,
)

# Import from components module
from .components import (
    ComponentsAlgorithm,
    StronglyConnectedComponentsAlgorithm,
    CondensationGraphAlgorithm,
    AttractingComponentsAlgorithm,
    BiconnectedComponentsAlgorithm,
)

# Import from vitality module
from .vitality import (
    ClosenessVitalityAlgorithm,
)

# Import from dispersion module
from .dispersion import (
    DispersionAlgorithm,
)

# Import from efficiency module
from .efficiency import (
    LocalEfficiencyAlgorithm,
    GlobalEfficiencyContributionAlgorithm,
    NodeEfficiencyAlgorithm,
    RobustnessEfficiencyAlgorithm,
    RoutingEfficiencyAlgorithm,
)

# Import from flow module
from .flow import (
    FlowHierarchyAlgorithm,
    MaxFlowCentralityAlgorithm,
    FlowBetweennessAlgorithm,
    HierarchyLevelAlgorithm,
    CycleParticipationAlgorithm,
    MinCutCentralityAlgorithm,
)

# Import from dominance module
from .dominance import (
    DominanceAlgorithm,
)

# Import from trust module
from .trust import (
    EigenTrustAlgorithm,
    AppleseedAlgorithm,
)

# Import from similarity module
from .similarity import (
    JaccardSimilarityAlgorithm,
    CosineSimilarityAlgorithm,
    AdamicAdarAlgorithm,
    ResourceAllocationAlgorithm,
)

# Import from spectral module (NEW)
from .spectral import (
    FiedlerVectorAlgorithm,
    SpectralCentralityAlgorithm,
)

# Import from influence module (NEW)
from .influence import (
    CollectiveInfluenceAlgorithm,
    SpreadingActivationAlgorithm,
)

# Import from link_prediction module (NEW)
from .link_prediction import (
    CommonNeighborsAlgorithm,
    PreferentialAttachmentAlgorithm,
    LinkPredictionScoresAlgorithm,
)

# Import from robustness module (NEW)
from .robustness import (
    NodeConnectivityAlgorithm,
    EdgeConnectivityAlgorithm,
    ResilienceScoreAlgorithm,
)

# Import from bipartite module (NEW)
from .bipartite import (
    BipartiteProjectionDegreeAlgorithm,
)

# Import from graph_coloring module (NEW)
from .graph_coloring import (
    GreedyColorAlgorithm,
)

# Import from igraph_algorithms module (NEW) - conditional import
try:
    from .igraph_algorithms import (
        LeidenCommunityAlgorithm,
        InfomapCommunityAlgorithm,
        WalktrapCommunityAlgorithm,
        FastGreedyCommunityAlgorithm,
        SpinglassCommunityAlgorithm,
        AlphaCentralityAlgorithm,
        MotifCountAlgorithm,
    )
    HAS_IGRAPH_ALGORITHMS = True
except ImportError as e:
    logger.warning(f"igraph algorithms not available: {e}")
    HAS_IGRAPH_ALGORITHMS = False
    # Create placeholder classes for graceful degradation
    LeidenCommunityAlgorithm = None
    InfomapCommunityAlgorithm = None
    WalktrapCommunityAlgorithm = None
    FastGreedyCommunityAlgorithm = None
    SpinglassCommunityAlgorithm = None
    AlphaCentralityAlgorithm = None
    MotifCountAlgorithm = None


# =============================================================================
# ALGORITHM REGISTRY
# =============================================================================

ALGORITHM_CLASSES: Dict[str, Type[BaseMetricAlgorithm]] = {
    # Topology
    "InDegreeAlgorithm": InDegreeAlgorithm,
    "OutDegreeAlgorithm": OutDegreeAlgorithm,
    "TotalDegreeAlgorithm": TotalDegreeAlgorithm,
    "DegreeImbalanceAlgorithm": DegreeImbalanceAlgorithm,
    
    # Centrality (18 original + 4 new)
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
    "LaplacianCentralityAlgorithm": LaplacianCentralityAlgorithm,
    "LeverageCentralityAlgorithm": LeverageCentralityAlgorithm,
    "SemiLocalCentralityAlgorithm": SemiLocalCentralityAlgorithm,
    "DecayCentralityAlgorithm": DecayCentralityAlgorithm,
    
    # Clustering (4 original + 2 new)
    "ClusteringCoefficientAlgorithm": ClusteringCoefficientAlgorithm,
    "TrianglesAlgorithm": TrianglesAlgorithm,
    "SquareClusteringAlgorithm": SquareClusteringAlgorithm,
    "LocalTransitivityAlgorithm": LocalTransitivityAlgorithm,
    "CliqueCountAlgorithm": CliqueCountAlgorithm,
    "AverageNeighborClusteringAlgorithm": AverageNeighborClusteringAlgorithm,
    
    # Community (4 original + 5 new)
    "LouvainCommunityAlgorithm": LouvainCommunityAlgorithm,
    "CoreNumberAlgorithm": CoreNumberAlgorithm,
    "OnionLayerAlgorithm": OnionLayerAlgorithm,
    "LocalReachingCentralityAlgorithm": LocalReachingCentralityAlgorithm,
    "LabelPropagationAlgorithm": LabelPropagationAlgorithm,
    "AsyncLabelPropagationAlgorithm": AsyncLabelPropagationAlgorithm,
    "GreedyModularityCommunityAlgorithm": GreedyModularityCommunityAlgorithm,
    "ParticipationCoefficientAlgorithm": ParticipationCoefficientAlgorithm,
    "WithinModuleDegreeAlgorithm": WithinModuleDegreeAlgorithm,
    
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
    
    # Components (1 original + 4 new)
    "ComponentsAlgorithm": ComponentsAlgorithm,
    "StronglyConnectedComponentsAlgorithm": StronglyConnectedComponentsAlgorithm,
    "CondensationGraphAlgorithm": CondensationGraphAlgorithm,
    "AttractingComponentsAlgorithm": AttractingComponentsAlgorithm,
    "BiconnectedComponentsAlgorithm": BiconnectedComponentsAlgorithm,
    
    # Vitality
    "ClosenessVitalityAlgorithm": ClosenessVitalityAlgorithm,
    
    # Dispersion
    "DispersionAlgorithm": DispersionAlgorithm,
    
    # Efficiency (1 original + 4 new)
    "LocalEfficiencyAlgorithm": LocalEfficiencyAlgorithm,
    "GlobalEfficiencyContributionAlgorithm": GlobalEfficiencyContributionAlgorithm,
    "NodeEfficiencyAlgorithm": NodeEfficiencyAlgorithm,
    "RobustnessEfficiencyAlgorithm": RobustnessEfficiencyAlgorithm,
    "RoutingEfficiencyAlgorithm": RoutingEfficiencyAlgorithm,
    
    # Flow (1 original + 5 new)
    "FlowHierarchyAlgorithm": FlowHierarchyAlgorithm,
    "MaxFlowCentralityAlgorithm": MaxFlowCentralityAlgorithm,
    "FlowBetweennessAlgorithm": FlowBetweennessAlgorithm,
    "HierarchyLevelAlgorithm": HierarchyLevelAlgorithm,
    "CycleParticipationAlgorithm": CycleParticipationAlgorithm,
    "MinCutCentralityAlgorithm": MinCutCentralityAlgorithm,
    
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
    
    # Spectral (NEW)
    "FiedlerVectorAlgorithm": FiedlerVectorAlgorithm,
    "SpectralCentralityAlgorithm": SpectralCentralityAlgorithm,
    
    # Influence (NEW)
    "CollectiveInfluenceAlgorithm": CollectiveInfluenceAlgorithm,
    "SpreadingActivationAlgorithm": SpreadingActivationAlgorithm,
    
    # Link Prediction (NEW)
    "CommonNeighborsAlgorithm": CommonNeighborsAlgorithm,
    "PreferentialAttachmentAlgorithm": PreferentialAttachmentAlgorithm,
    "LinkPredictionScoresAlgorithm": LinkPredictionScoresAlgorithm,
    
    # Robustness (NEW)
    "NodeConnectivityAlgorithm": NodeConnectivityAlgorithm,
    "EdgeConnectivityAlgorithm": EdgeConnectivityAlgorithm,
    "ResilienceScoreAlgorithm": ResilienceScoreAlgorithm,
    
    # Bipartite (NEW)
    "BipartiteProjectionDegreeAlgorithm": BipartiteProjectionDegreeAlgorithm,
    
    # Graph Coloring (NEW)
    "GreedyColorAlgorithm": GreedyColorAlgorithm,
}

# Add igraph algorithms if available
if HAS_IGRAPH_ALGORITHMS:
    ALGORITHM_CLASSES.update({
        "LeidenCommunityAlgorithm": LeidenCommunityAlgorithm,
        "InfomapCommunityAlgorithm": InfomapCommunityAlgorithm,
        "WalktrapCommunityAlgorithm": WalktrapCommunityAlgorithm,
        "FastGreedyCommunityAlgorithm": FastGreedyCommunityAlgorithm,
        "SpinglassCommunityAlgorithm": SpinglassCommunityAlgorithm,
        "AlphaCentralityAlgorithm": AlphaCentralityAlgorithm,
        "MotifCountAlgorithm": MotifCountAlgorithm,
    })


def get_algorithm_class(name: str) -> Optional[Type[BaseMetricAlgorithm]]:
    """
    Get an algorithm class by name.
    
    Args:
        name: Algorithm class name
        
    Returns:
        Algorithm class or None if not found
    """
    return ALGORITHM_CLASSES.get(name)


def list_algorithm_classes() -> Dict[str, Type[BaseMetricAlgorithm]]:
    """
    Get all available algorithm classes.
    
    Returns:
        Dictionary mapping names to classes
    """
    return ALGORITHM_CLASSES.copy()


def get_algorithms_by_category() -> Dict[str, list]:
    """
    Get algorithms organized by category.
    
    Returns:
        Dictionary mapping category names to lists of algorithm names
    """
    by_category = {}
    for name, cls in ALGORITHM_CLASSES.items():
        if cls is None:
            continue
        cat = getattr(cls, 'category', 'unknown')
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(name)
    return by_category


def igraph_available() -> bool:
    """Check if igraph algorithms are available."""
    return HAS_IGRAPH_ALGORITHMS


# Export all
__all__ = [
    # Base
    "BaseMetricAlgorithm",
    
    # Registry
    "ALGORITHM_CLASSES",
    "get_algorithm_class",
    "list_algorithm_classes",
    "get_algorithms_by_category",
    "igraph_available",
    "HAS_IGRAPH_ALGORITHMS",
    
    # Topology
    "InDegreeAlgorithm",
    "OutDegreeAlgorithm",
    "TotalDegreeAlgorithm",
    "DegreeImbalanceAlgorithm",
    
    # Centrality
    "DegreeCentralityAlgorithm",
    "ClosenessCentralityAlgorithm",
    "BetweennessCentralityAlgorithm",
    "EigenvectorCentralityAlgorithm",
    "KatzCentralityAlgorithm",
    "PageRankAlgorithm",
    "HITSAlgorithm",
    "HarmonicCentralityAlgorithm",
    "LoadCentralityAlgorithm",
    "SubgraphCentralityAlgorithm",
    "SecondOrderCentralityAlgorithm",
    "PercolationCentralityAlgorithm",
    "TrophicLevelAlgorithm",
    "CurrentFlowCentralityAlgorithm",
    "InformationCentralityAlgorithm",
    "CommunicabilityBetweennessAlgorithm",
    "VoteRankAlgorithm",
    "EdgeBetweennessSumAlgorithm",
    "LaplacianCentralityAlgorithm",
    "LeverageCentralityAlgorithm",
    "SemiLocalCentralityAlgorithm",
    "DecayCentralityAlgorithm",
    
    # Clustering
    "ClusteringCoefficientAlgorithm",
    "TrianglesAlgorithm",
    "SquareClusteringAlgorithm",
    "LocalTransitivityAlgorithm",
    "CliqueCountAlgorithm",
    "AverageNeighborClusteringAlgorithm",
    
    # Community
    "LouvainCommunityAlgorithm",
    "CoreNumberAlgorithm",
    "OnionLayerAlgorithm",
    "LocalReachingCentralityAlgorithm",
    "LabelPropagationAlgorithm",
    "AsyncLabelPropagationAlgorithm",
    "GreedyModularityCommunityAlgorithm",
    "ParticipationCoefficientAlgorithm",
    "WithinModuleDegreeAlgorithm",
    
    # Paths
    "ShortestPathsAlgorithm",
    "HopPathsAlgorithm",
    "EccentricityAlgorithm",
    "WienerContributionAlgorithm",
    
    # Distances
    "GraphDistancesAlgorithm",
    
    # Structural
    "StructuralHolesAlgorithm",
    "ArticulationPointsAlgorithm",
    "BridgesAlgorithm",
    "NeighborDegreeStatsAlgorithm",
    "BiconnectedComponentAlgorithm",
    
    # Reciprocity
    "ReciprocityAlgorithm",
    
    # Reach
    "ReachAlgorithm",
    
    # Components
    "ComponentsAlgorithm",
    "StronglyConnectedComponentsAlgorithm",
    "CondensationGraphAlgorithm",
    "AttractingComponentsAlgorithm",
    "BiconnectedComponentsAlgorithm",
    
    # Vitality
    "ClosenessVitalityAlgorithm",
    
    # Dispersion
    "DispersionAlgorithm",
    
    # Efficiency
    "LocalEfficiencyAlgorithm",
    "GlobalEfficiencyContributionAlgorithm",
    "NodeEfficiencyAlgorithm",
    "RobustnessEfficiencyAlgorithm",
    "RoutingEfficiencyAlgorithm",
    
    # Flow
    "FlowHierarchyAlgorithm",
    "MaxFlowCentralityAlgorithm",
    "FlowBetweennessAlgorithm",
    "HierarchyLevelAlgorithm",
    "CycleParticipationAlgorithm",
    "MinCutCentralityAlgorithm",
    
    # Dominance
    "DominanceAlgorithm",
    
    # Trust
    "EigenTrustAlgorithm",
    "AppleseedAlgorithm",
    
    # Similarity
    "JaccardSimilarityAlgorithm",
    "CosineSimilarityAlgorithm",
    "AdamicAdarAlgorithm",
    "ResourceAllocationAlgorithm",
    
    # Spectral
    "FiedlerVectorAlgorithm",
    "SpectralCentralityAlgorithm",
    
    # Influence
    "CollectiveInfluenceAlgorithm",
    "SpreadingActivationAlgorithm",
    
    # Link Prediction
    "CommonNeighborsAlgorithm",
    "PreferentialAttachmentAlgorithm",
    "LinkPredictionScoresAlgorithm",
    
    # Robustness
    "NodeConnectivityAlgorithm",
    "EdgeConnectivityAlgorithm",
    "ResilienceScoreAlgorithm",
    
    # Bipartite
    "BipartiteProjectionDegreeAlgorithm",
    
    # Graph Coloring
    "GreedyColorAlgorithm",
    
    # igraph (conditional)
    "LeidenCommunityAlgorithm",
    "InfomapCommunityAlgorithm",
    "WalktrapCommunityAlgorithm",
    "FastGreedyCommunityAlgorithm",
    "SpinglassCommunityAlgorithm",
    "AlphaCentralityAlgorithm",
    "MotifCountAlgorithm",
]