"""
Self-Optimizing Clustering Head for GIT-CD

Implements the deep clustering module from the paper:
- KMeans initialization of cluster centers
- Soft assignments using Student-t distribution
- Target distribution (DEC-style sharpening)
- KL divergence loss for alignment
- Silhouette loss for geometric quality

This is a differentiable clustering module that refines
communities during training.
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

logger = logging.getLogger(__name__)


class DeepClusteringHead(nn.Module):
    """
    Self-optimizing clustering module.
    
    Implements the clustering approach from GIT-CD:
    1. Initialize centers with KMeans
    2. Compute soft assignments Q using Student-t distribution
    3. Compute target distribution P (sharpened Q)
    4. Minimize KL(P||Q) to refine clusters
    5. Use silhouette loss for geometric quality
    
    Args:
        hidden_dim: Input embedding dimension
        num_clusters: Number of clusters (communities)
        init_temperature: Initial temperature for soft assignments (trainable)
        alpha: Student-t degrees of freedom (default: 1.0)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_clusters: int,
        init_temperature: float = 1.0,
        alpha: float = 1.0,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        self.alpha = alpha
        
        # Cluster centers (updated via KMeans, but can also be learned)
        self.register_buffer(
            'centers',
            torch.zeros(num_clusters, hidden_dim, dtype=torch.float32)
        )
        
        # Trainable temperature parameter (log-space for positivity)
        # Use math.log instead of np.log to avoid float64
        log_temp = math.log(max(init_temperature, 1e-6))
        self.log_temperature = nn.Parameter(
            torch.tensor(log_temp, dtype=torch.float32)
        )
        
        # Track initialization state
        self._centers_initialized = False
    
    @property
    def temperature(self) -> Tensor:
        """Get temperature (always positive via exp)."""
        return torch.exp(self.log_temperature).clamp(min=1e-6)
    
    @property
    def centers_initialized(self) -> bool:
        """Check if centers have been initialized."""
        return self._centers_initialized
    
    def initialize_centers(
        self,
        embeddings: Tensor,
        method: str = 'kmeans',
        n_init: int = 10,
    ) -> None:
        """
        Initialize cluster centers from embeddings.
        
        Args:
            embeddings: Node embeddings [N, D]
            method: Initialization method ('kmeans', 'random', 'kmeans++')
            n_init: Number of KMeans initializations
        """
        device = embeddings.device
        
        with torch.no_grad():
            if method == 'kmeans':
                centers = self._kmeans_init(embeddings, n_init)
            elif method == 'kmeans++':
                centers = self._kmeanspp_init(embeddings)
            elif method == 'random':
                centers = self._random_init(embeddings)
            else:
                raise ValueError(f"Unknown initialization method: {method}")
            
            # Ensure float32 for MPS compatibility
            centers = centers.to(dtype=torch.float32, device=device)
            self.centers.copy_(centers)
            self._centers_initialized = True
            
            logger.info(f"Initialized {self.num_clusters} cluster centers using {method}")
    
    def _kmeans_init(self, embeddings: Tensor, n_init: int = 10) -> Tensor:
        """Initialize centers using sklearn KMeans."""
        try:
            from sklearn.cluster import KMeans
            
            # Convert to float32 numpy array
            X = embeddings.detach().cpu().to(torch.float32).numpy()
            
            # Handle case where we have fewer samples than clusters
            if X.shape[0] < self.num_clusters:
                logger.warning(
                    f"Fewer samples ({X.shape[0]}) than clusters ({self.num_clusters}), "
                    "using random initialization"
                )
                return self._random_init(embeddings)
            
            kmeans = KMeans(
                n_clusters=self.num_clusters,
                n_init=n_init,
                random_state=42,
                max_iter=300,
            )
            kmeans.fit(X)
            
            # Explicitly convert to float32
            centers = torch.tensor(
                kmeans.cluster_centers_,
                dtype=torch.float32,
            )
            return centers
            
        except ImportError:
            logger.warning("sklearn not available, using random initialization")
            return self._random_init(embeddings)
    
    def _kmeanspp_init(self, embeddings: Tensor) -> Tensor:
        """Initialize centers using KMeans++ algorithm."""
        N = embeddings.size(0)
        K = self.num_clusters
        
        if N < K:
            return self._random_init(embeddings)
        
        # Ensure float32
        embeddings = embeddings.to(torch.float32)
        
        # First center: random
        indices = [torch.randint(0, N, (1,)).item()]
        centers = [embeddings[indices[0]]]
        
        for _ in range(1, K):
            # Compute distances to nearest center
            dists = torch.cdist(embeddings, torch.stack(centers))
            min_dists = dists.min(dim=1)[0]  # [N]
            
            # Sample proportional to squared distance
            probs = min_dists ** 2
            probs = probs / probs.sum()
            
            idx = torch.multinomial(probs, 1).item()
            indices.append(idx)
            centers.append(embeddings[idx])
        
        return torch.stack(centers).to(torch.float32)
    
    def _random_init(self, embeddings: Tensor) -> Tensor:
        """Random center initialization."""
        N = embeddings.size(0)
        K = min(self.num_clusters, N)
        
        indices = torch.randperm(N)[:K]
        centers = embeddings[indices].clone().to(torch.float32)
        
        # Pad with zeros if needed
        if K < self.num_clusters:
            padding = torch.zeros(
                self.num_clusters - K,
                self.hidden_dim,
                dtype=torch.float32,
                device=embeddings.device
            )
            centers = torch.cat([centers, padding], dim=0)
        
        return centers
    
    def compute_soft_assignments(self, embeddings: Tensor) -> Tensor:
        """
        Compute soft cluster assignments using Student-t distribution.
        
        Q_ij = (1 + ||z_i - c_j||^2 / (alpha * t))^(-(alpha+1)/2)
        
        Then normalize: Q_ij = Q_ij / sum_j(Q_ij)
        
        Args:
            embeddings: Node embeddings [N, D]
            
        Returns:
            Soft assignments Q [N, K]
        """
        # Ensure float32
        embeddings = embeddings.to(torch.float32)
        
        # Squared Euclidean distances: ||z - c||^2
        # [N, 1, D] - [1, K, D] -> [N, K, D] -> sum -> [N, K]
        z = embeddings.unsqueeze(1)  # [N, 1, D]
        c = self.centers.unsqueeze(0)  # [1, K, D]
        dist_sq = ((z - c) ** 2).sum(dim=2)  # [N, K]
        
        # Student-t kernel with trainable temperature
        # Q = (1 + dist^2 / (alpha * t))^(-(alpha+1)/2)
        t = self.temperature
        alpha = self.alpha
        
        numerator = 1.0 + dist_sq / (alpha * t)
        Q = numerator ** (-(alpha + 1) / 2)
        
        # Normalize to get probability distribution
        Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-12)
        
        return Q
    
    def compute_target_distribution(self, Q: Tensor) -> Tensor:
        """
        Compute target distribution by sharpening soft assignments.
        
        This is the DEC-style target distribution:
        P_ij = (Q_ij^2 / f_j) / sum_j(Q_ij^2 / f_j)
        
        where f_j = sum_i Q_ij is the cluster frequency.
        
        Args:
            Q: Soft assignments [N, K]
            
        Returns:
            Target distribution P [N, K]
        """
        # Cluster frequencies
        f = Q.sum(dim=0, keepdim=True) + 1e-12  # [1, K]
        
        # Square and normalize by frequency
        P = (Q ** 2) / f
        
        # Normalize rows to get probability distribution
        P = P / (P.sum(dim=1, keepdim=True) + 1e-12)
        
        return P
    
    def forward(
        self,
        embeddings: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass.
        
        Args:
            embeddings: Node embeddings [N, D]
            
        Returns:
            Tuple of:
            - Q: Soft assignments [N, K]
            - P: Target distribution [N, K]
            - hard_labels: Hard cluster assignments [N]
        """
        Q = self.compute_soft_assignments(embeddings)
        P = self.compute_target_distribution(Q)
        hard_labels = Q.argmax(dim=1)
        
        return Q, P, hard_labels
    
    def get_cluster_sizes(self, Q: Tensor) -> Tensor:
        """Get soft cluster sizes."""
        return Q.sum(dim=0)
    
    def get_cluster_centers(self) -> Tensor:
        """Get current cluster centers."""
        return self.centers.clone()


def kl_divergence_loss(P: Tensor, Q: Tensor, reduction: str = 'mean') -> Tensor:
    """
    Compute KL divergence loss KL(P || Q).
    
    KL(P||Q) = sum_i sum_j P_ij * log(P_ij / Q_ij)
    
    Args:
        P: Target distribution [N, K]
        Q: Soft assignments [N, K]
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        KL divergence loss
    """
    # Avoid log(0) with small epsilon
    eps = 1e-12
    log_ratio = torch.log(P + eps) - torch.log(Q + eps)
    kl = (P * log_ratio).sum(dim=1)  # [N]
    
    if reduction == 'mean':
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    else:
        return kl


def soft_silhouette_loss(
    embeddings: Tensor,
    Q: Tensor,
    temperature: float = 0.1,
    reduction: str = 'mean',
) -> Tensor:
    """
    Compute differentiable silhouette loss using soft assignments.
    
    Silhouette coefficient: s(i) = (b(i) - a(i)) / max(a(i), b(i))
    - a(i): Mean distance to points in same cluster
    - b(i): Mean distance to points in nearest other cluster
    
    Loss = -mean(s(i))  (minimize to maximize silhouette)
    
    This version uses soft assignments for differentiability.
    
    Args:
        embeddings: Node embeddings [N, D]
        Q: Soft assignments [N, K]
        temperature: Temperature for softmin over other clusters
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Silhouette loss (negative mean silhouette)
    """
    N, K = Q.shape
    device = embeddings.device
    
    # Ensure float32
    embeddings = embeddings.to(torch.float32)
    Q = Q.to(torch.float32)
    
    # Pairwise distances [N, N]
    dist = torch.cdist(embeddings, embeddings)
    
    # Soft cluster sizes [K]
    cluster_sizes = Q.sum(dim=0) + 1e-12
    
    # Compute mean distance to each cluster for each node
    # For node i, mean dist to cluster k = sum_j(Q_jk * dist_ij) / sum_j(Q_jk)
    # = (dist @ Q) / cluster_sizes
    mean_dist_to_cluster = torch.matmul(dist, Q) / cluster_sizes.unsqueeze(0)  # [N, K]
    
    # Intra-cluster distance a(i)
    # Weight by own cluster membership
    own_cluster = Q.argmax(dim=1)  # [N]
    a = mean_dist_to_cluster[torch.arange(N, device=device), own_cluster]  # [N]
    
    # Inter-cluster distance b(i): nearest other cluster
    # Mask own cluster with large value
    mask = F.one_hot(own_cluster, K).bool()
    other_dist = mean_dist_to_cluster.masked_fill(mask, float('inf'))
    
    # Soft minimum using temperature (more differentiable than hard min)
    # b = softmin(other_dist) = sum_k softmax(-d/tau) * d
    weights = F.softmax(-other_dist / (temperature + 1e-8), dim=1)
    
    # Handle inf values
    weights = weights.masked_fill(torch.isinf(other_dist), 0)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)
    
    b = (weights * other_dist.masked_fill(torch.isinf(other_dist), 0)).sum(dim=1)  # [N]
    
    # Silhouette coefficient
    # s = (b - a) / max(a, b)
    max_ab = torch.maximum(a, b) + 1e-12
    s = (b - a) / max_ab
    
    # Clamp to valid range [-1, 1]
    s = s.clamp(-1, 1)
    
    # Loss = negative silhouette (minimize to maximize quality)
    if reduction == 'mean':
        return -s.mean()
    elif reduction == 'sum':
        return -s.sum()
    else:
        return -s


def cluster_contrastive_loss(
    embeddings: Tensor,
    Q: Tensor,
    temperature: float = 0.5,
) -> Tensor:
    """
    Contrastive loss to separate clusters.
    
    Pulls together nodes in the same cluster and pushes apart
    nodes in different clusters.
    
    Args:
        embeddings: Node embeddings [N, D]
        Q: Soft assignments [N, K]
        temperature: Temperature for contrastive loss
        
    Returns:
        Contrastive loss
    """
    N = embeddings.size(0)
    
    # Normalize embeddings
    z = F.normalize(embeddings.to(torch.float32), p=2, dim=1)
    
    # Similarity matrix
    sim = torch.matmul(z, z.t()) / temperature  # [N, N]
    
    # Cluster assignment similarity (how similar are assignments)
    cluster_sim = torch.matmul(Q, Q.t())  # [N, N] in [0, 1]
    
    # Use cluster similarity as soft labels
    # High cluster_sim -> should be similar
    # Low cluster_sim -> should be different
    
    # InfoNCE-style loss
    # For each node, other nodes in same cluster are positives
    exp_sim = torch.exp(sim)
    
    # Mask diagonal
    mask = torch.eye(N, device=embeddings.device).bool()
    exp_sim = exp_sim.masked_fill(mask, 0)
    
    # Weighted by cluster membership
    pos_weight = cluster_sim.masked_fill(mask, 0)
    neg_weight = 1 - cluster_sim
    neg_weight = neg_weight.masked_fill(mask, 0)
    
    # Loss
    pos_term = (exp_sim * pos_weight).sum(dim=1)
    neg_term = (exp_sim * neg_weight).sum(dim=1)
    
    loss = -torch.log(pos_term / (pos_term + neg_term + 1e-12) + 1e-12)
    
    return loss.mean()


def compute_clustering_metrics(
    embeddings: Tensor,
    labels: Tensor,
) -> Dict[str, float]:
    """
    Compute clustering quality metrics.
    
    Args:
        embeddings: Node embeddings [N, D]
        labels: Cluster labels [N]
        
    Returns:
        Dict with silhouette, davies_bouldin, calinski_harabasz scores
    """
    try:
        from sklearn.metrics import (
            silhouette_score,
            davies_bouldin_score,
            calinski_harabasz_score,
        )
        
        # Convert to float32 numpy
        X = embeddings.detach().cpu().to(torch.float32).numpy()
        y = labels.detach().cpu().numpy()
        
        # Need at least 2 clusters with samples
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            return {
                'silhouette': 0.0,
                'davies_bouldin': float('inf'),
                'calinski_harabasz': 0.0,
            }
        
        return {
            'silhouette': float(silhouette_score(X, y)),
            'davies_bouldin': float(davies_bouldin_score(X, y)),
            'calinski_harabasz': float(calinski_harabasz_score(X, y)),
        }
        
    except ImportError:
        logger.warning("sklearn not available for clustering metrics")
        return {}