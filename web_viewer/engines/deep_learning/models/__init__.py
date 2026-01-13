"""
Deep Learning Models

Provides neural network architectures for graph learning:
- GNN modules (SAGEConv, HeteroConv)
- Transformer modules (Dynamic HIN Attention)
- Clustering modules (Deep Clustering Head)
- Full GIT-CD model
"""

from .gnn import HomogeneousSAGE, HeterogeneousSAGE
from .transformer import DynamicHINAttention, HINTransformerBlock
from .clustering_head import DeepClusteringHead
from .gitcd import GITCD

__all__ = [
    "HomogeneousSAGE",
    "HeterogeneousSAGE",
    "DynamicHINAttention",
    "HINTransformerBlock",
    "DeepClusteringHead",
    "GITCD",
]
