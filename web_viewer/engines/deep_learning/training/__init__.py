"""
Training Module for GIT-CD

Provides training infrastructure:
- GITCDTrainer: Full training loop with multi-objective optimization
- Loss functions: KL divergence, silhouette, contrastive
- Mini-batch samplers for large graphs
"""

from .losses import (
    kl_divergence_loss,
    soft_silhouette_loss,
    cluster_contrastive_loss,
    combined_clustering_loss,
)
from .trainer import GITCDTrainer, TrainingCallback, EarlyStoppingCallback

__all__ = [
    # Losses
    "kl_divergence_loss",
    "soft_silhouette_loss",
    "cluster_contrastive_loss",
    "combined_clustering_loss",
    # Trainer
    "GITCDTrainer",
    "TrainingCallback",
    "EarlyStoppingCallback",
]
