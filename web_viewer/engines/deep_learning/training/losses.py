"""
Loss Functions for GIT-CD Training

Implements various loss functions for deep graph clustering:
- KL divergence loss (DEC-style)
- Soft silhouette loss (differentiable)
- Cluster contrastive loss
- Combined losses with configurable weights
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def kl_divergence_loss(
    P: Tensor,
    Q: Tensor,
    reduction: str = 'mean',
) -> Tensor:
    """
    Compute KL divergence loss KL(P || Q).
    
    This is the main clustering loss from DEC and GIT-CD papers.
    Minimizing KL(P||Q) aligns soft assignments Q with sharpened targets P.
    
    KL(P||Q) = sum_i sum_j P_ij * log(P_ij / Q_ij)
    
    Args:
        P: Target distribution [N, K] (sharpened assignments)
        Q: Soft assignments [N, K]
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        KL divergence loss
    """
    eps = 1e-12
    
    # KL divergence per sample
    log_ratio = torch.log(P + eps) - torch.log(Q + eps)
    kl = (P * log_ratio).sum(dim=1)  # [N]
    
    if reduction == 'mean':
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    return kl


def soft_silhouette_loss(
    embeddings: Tensor,
    Q: Tensor,
    temperature: float = 0.1,
    reduction: str = 'mean',
) -> Tensor:
    """
    Compute differentiable silhouette loss using soft assignments.
    
    Silhouette coefficient measures clustering quality:
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    
    where:
    - a(i) = mean distance to same-cluster points
    - b(i) = mean distance to nearest other cluster
    
    Loss = -mean(s)  (minimize to maximize silhouette)
    
    Uses soft assignments Q for differentiability.
    
    Args:
        embeddings: Node embeddings [N, D]
        Q: Soft cluster assignments [N, K]
        temperature: Temperature for softmin over other clusters
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Silhouette loss (negative silhouette score)
    """
    N, K = Q.shape
    device = embeddings.device
    
    # Pairwise Euclidean distances
    dist = torch.cdist(embeddings, embeddings)  # [N, N]
    
    # Soft cluster sizes
    cluster_sizes = Q.sum(dim=0) + 1e-12  # [K]
    
    # Mean distance from each node to each cluster
    # mean_dist[i, k] = sum_j(Q[j,k] * dist[i,j]) / sum_j(Q[j,k])
    mean_dist_to_cluster = torch.matmul(dist, Q) / cluster_sizes  # [N, K]
    
    # Hard cluster assignment for each node
    own_cluster = Q.argmax(dim=1)  # [N]
    
    # Intra-cluster distance a(i)
    a = mean_dist_to_cluster[torch.arange(N, device=device), own_cluster]
    
    # Inter-cluster distance b(i): nearest other cluster
    # Mask own cluster with inf
    mask = F.one_hot(own_cluster, K).bool()
    other_dist = mean_dist_to_cluster.clone()
    other_dist[mask] = float('inf')
    
    # Softmin for differentiability (approximates min)
    weights = F.softmax(-other_dist / temperature, dim=1)
    
    # Handle inf values
    valid_mask = ~torch.isinf(other_dist)
    weights = weights * valid_mask.float()
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)
    
    b = (weights * other_dist.masked_fill(~valid_mask, 0)).sum(dim=1)
    
    # Silhouette coefficient
    max_ab = torch.maximum(a, b) + 1e-12
    s = (b - a) / max_ab
    s = s.clamp(-1, 1)
    
    if reduction == 'mean':
        return -s.mean()
    elif reduction == 'sum':
        return -s.sum()
    return -s


def cluster_contrastive_loss(
    embeddings: Tensor,
    Q: Tensor,
    temperature: float = 0.5,
    reduction: str = 'mean',
) -> Tensor:
    """
    Contrastive loss for cluster separation.
    
    Encourages nodes in same cluster to be similar and
    nodes in different clusters to be dissimilar.
    
    Args:
        embeddings: Node embeddings [N, D]
        Q: Soft cluster assignments [N, K]
        temperature: Temperature scaling
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Contrastive loss
    """
    N = embeddings.size(0)
    device = embeddings.device
    
    # L2 normalize embeddings
    z = F.normalize(embeddings, p=2, dim=1)
    
    # Cosine similarity matrix
    sim = torch.matmul(z, z.t()) / temperature  # [N, N]
    
    # Cluster membership similarity
    cluster_sim = torch.matmul(Q, Q.t())  # [N, N]
    
    # Self-similarity mask (diagonal)
    self_mask = torch.eye(N, device=device).bool()
    
    # Exponential similarities
    exp_sim = torch.exp(sim)
    exp_sim = exp_sim.masked_fill(self_mask, 0)
    
    # Positive pairs: same cluster (weighted by membership similarity)
    pos_weight = cluster_sim.masked_fill(self_mask, 0)
    
    # Negative pairs: different clusters
    neg_weight = (1 - cluster_sim).clamp(min=0)
    neg_weight = neg_weight.masked_fill(self_mask, 0)
    
    # Contrastive loss: -log(pos / (pos + neg))
    pos_term = (exp_sim * pos_weight).sum(dim=1) + 1e-12
    neg_term = (exp_sim * neg_weight).sum(dim=1) + 1e-12
    
    loss = -torch.log(pos_term / (pos_term + neg_term))
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def cluster_entropy_loss(Q: Tensor, reduction: str = 'mean') -> Tensor:
    """
    Entropy regularization to encourage confident assignments.
    
    H(q_i) = -sum_k q_ik * log(q_ik)
    
    Minimizing entropy pushes assignments to be more confident.
    
    Args:
        Q: Soft cluster assignments [N, K]
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Entropy loss
    """
    eps = 1e-12
    entropy = -(Q * torch.log(Q + eps)).sum(dim=1)  # [N]
    
    if reduction == 'mean':
        return entropy.mean()
    elif reduction == 'sum':
        return entropy.sum()
    return entropy


def cluster_balance_loss(Q: Tensor) -> Tensor:
    """
    Loss to encourage balanced cluster sizes.
    
    Penalizes deviation from uniform cluster distribution.
    
    Args:
        Q: Soft cluster assignments [N, K]
        
    Returns:
        Balance loss
    """
    N, K = Q.shape
    
    # Target: uniform distribution
    target_size = N / K
    
    # Actual soft cluster sizes
    cluster_sizes = Q.sum(dim=0)  # [K]
    
    # MSE from uniform
    loss = ((cluster_sizes - target_size) ** 2).mean()
    
    return loss / (N ** 2)  # Normalize


def orthogonality_loss(embeddings: Tensor) -> Tensor:
    """
    Loss to encourage orthogonal/diverse embeddings.
    
    Args:
        embeddings: Node embeddings [N, D]
        
    Returns:
        Orthogonality loss
    """
    # Normalize
    z = F.normalize(embeddings, p=2, dim=1)
    
    # Gram matrix
    gram = torch.matmul(z, z.t())
    
    # Target: identity (orthogonal)
    N = z.size(0)
    identity = torch.eye(N, device=z.device)
    
    # Frobenius norm of difference
    loss = ((gram - identity) ** 2).mean()
    
    return loss


def combined_clustering_loss(
    embeddings: Tensor,
    Q: Tensor,
    P: Tensor,
    kl_weight: float = 1.0,
    silhouette_weight: float = 0.1,
    contrastive_weight: float = 0.0,
    entropy_weight: float = 0.0,
    balance_weight: float = 0.0,
    silhouette_temperature: float = 0.1,
    contrastive_temperature: float = 0.5,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    Compute combined clustering loss with configurable weights.
    
    Total = w_kl * L_kl + w_sil * L_sil + w_con * L_con + w_ent * L_ent + w_bal * L_bal
    
    Args:
        embeddings: Node embeddings [N, D]
        Q: Soft cluster assignments [N, K]
        P: Target distribution [N, K]
        kl_weight: Weight for KL divergence loss
        silhouette_weight: Weight for silhouette loss
        contrastive_weight: Weight for contrastive loss
        entropy_weight: Weight for entropy loss
        balance_weight: Weight for balance loss
        silhouette_temperature: Temperature for silhouette softmin
        contrastive_temperature: Temperature for contrastive loss
        
    Returns:
        Tuple of (total_loss, individual_losses_dict)
    """
    losses = {}
    total = torch.tensor(0.0, device=embeddings.device)
    
    # KL divergence
    if kl_weight > 0:
        loss_kl = kl_divergence_loss(P, Q)
        losses['kl'] = loss_kl
        total = total + kl_weight * loss_kl
    
    # Silhouette
    if silhouette_weight > 0:
        loss_sil = soft_silhouette_loss(
            embeddings, Q, temperature=silhouette_temperature
        )
        losses['silhouette'] = loss_sil
        total = total + silhouette_weight * loss_sil
    
    # Contrastive
    if contrastive_weight > 0:
        loss_con = cluster_contrastive_loss(
            embeddings, Q, temperature=contrastive_temperature
        )
        losses['contrastive'] = loss_con
        total = total + contrastive_weight * loss_con
    
    # Entropy
    if entropy_weight > 0:
        loss_ent = cluster_entropy_loss(Q)
        losses['entropy'] = loss_ent
        total = total + entropy_weight * loss_ent
    
    # Balance
    if balance_weight > 0:
        loss_bal = cluster_balance_loss(Q)
        losses['balance'] = loss_bal
        total = total + balance_weight * loss_bal
    
    losses['total'] = total
    
    return total, losses


class ClusteringLoss(nn.Module):
    """
    Modular clustering loss as nn.Module.
    
    Combines multiple clustering objectives with configurable weights.
    
    Args:
        kl_weight: Weight for KL divergence loss
        silhouette_weight: Weight for silhouette loss
        contrastive_weight: Weight for contrastive loss
        entropy_weight: Weight for entropy regularization
        balance_weight: Weight for cluster balance loss
        silhouette_temperature: Temperature for silhouette computation
        contrastive_temperature: Temperature for contrastive loss
    """
    
    def __init__(
        self,
        kl_weight: float = 1.0,
        silhouette_weight: float = 0.1,
        contrastive_weight: float = 0.0,
        entropy_weight: float = 0.0,
        balance_weight: float = 0.0,
        silhouette_temperature: float = 0.1,
        contrastive_temperature: float = 0.5,
    ):
        super().__init__()
        
        self.kl_weight = kl_weight
        self.silhouette_weight = silhouette_weight
        self.contrastive_weight = contrastive_weight
        self.entropy_weight = entropy_weight
        self.balance_weight = balance_weight
        self.silhouette_temperature = silhouette_temperature
        self.contrastive_temperature = contrastive_temperature
    
    def forward(
        self,
        embeddings: Tensor,
        Q: Tensor,
        P: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute loss.
        
        Args:
            embeddings: Node embeddings [N, D]
            Q: Soft assignments [N, K]
            P: Target distribution [N, K]
            
        Returns:
            Tuple of (total_loss, losses_dict)
        """
        return combined_clustering_loss(
            embeddings=embeddings,
            Q=Q,
            P=P,
            kl_weight=self.kl_weight,
            silhouette_weight=self.silhouette_weight,
            contrastive_weight=self.contrastive_weight,
            entropy_weight=self.entropy_weight,
            balance_weight=self.balance_weight,
            silhouette_temperature=self.silhouette_temperature,
            contrastive_temperature=self.contrastive_temperature,
        )