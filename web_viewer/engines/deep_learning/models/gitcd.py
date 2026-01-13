"""
GIT-CD: Graph Integrated Transformer for Community Detection

Complete model implementation combining:
1. GNN module (SAGEConv) for local structure
2. Transformer encoder (Dynamic HIN Attention) for global context
3. Deep clustering head for community detection
4. Optional classification head for semi-supervised learning

Based on paper: "GIT-CD: Graph Integrated Transformer for Community Detection"
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .gnn import HomogeneousSAGE, HeterogeneousSAGE
from .transformer import HINTransformerBlock, TransformerEncoder
from .clustering_head import (
    DeepClusteringHead,
    kl_divergence_loss,
    soft_silhouette_loss,
)
from ..config import GITCDConfig

logger = logging.getLogger(__name__)


class GITCD(nn.Module):
    """
    Graph Integrated Transformer for Community Detection.
    
    Architecture:
    1. GNN layers (SAGEConv) for local neighborhood structure
    2. Transformer blocks (Dynamic HIN Attention) for global dependencies
    3. Clustering head for community detection
    4. Optional classification head for node labels
    
    The model can operate in two modes:
    - Homogeneous: All nodes same type (simpler, faster)
    - Heterogeneous: Multiple node types with type-aware attention
    
    Args:
        config: GITCDConfig with all model parameters
    """
    
    def __init__(self, config: GITCDConfig):
        super().__init__()
        
        self.config = config
        self.is_heterogeneous = len(config.node_types) > 1 and len(config.edge_types) > 0
        
        # Input projection (if input_dim != hidden_dim)
        if config.input_dim != config.hidden_dim:
            self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        else:
            self.input_proj = nn.Identity()
        
        # GNN Module
        if self.is_heterogeneous:
            self.gnn = HeterogeneousSAGE(
                node_types=config.node_types,
                edge_types=config.edge_types,
                in_channels_dict={t: config.hidden_dim for t in config.node_types},
                hidden_channels=config.hidden_dim,
                out_channels=config.hidden_dim,
                num_layers=config.num_gnn_layers,
                dropout=config.dropout,
                aggregation=config.gnn_aggregation,
            )
        else:
            self.gnn = HomogeneousSAGE(
                in_channels=config.hidden_dim,
                hidden_channels=config.hidden_dim,
                out_channels=config.hidden_dim,
                num_layers=config.num_gnn_layers,
                dropout=config.dropout,
                aggregation=config.gnn_aggregation,
            )
        
        # Transformer Encoder
        self.transformer = TransformerEncoder(
            node_types=config.node_types,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_transformer_layers,
            num_heads=config.num_attention_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
        )
        
        # Clustering Head
        self.clustering_head = DeepClusteringHead(
            hidden_dim=config.hidden_dim,
            num_clusters=config.num_clusters,
            init_temperature=config.clustering_temperature,
        )
        
        # Optional Classification Head
        self.classifier = None
        if config.num_classes is not None and config.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, config.num_classes),
            )
        
        # Output projection (if embedding_dim != hidden_dim)
        if config.embedding_dim != config.hidden_dim:
            self.output_proj = nn.Linear(config.hidden_dim, config.embedding_dim)
        else:
            self.output_proj = nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        if isinstance(self.input_proj, nn.Linear):
            nn.init.xavier_uniform_(self.input_proj.weight)
            if self.input_proj.bias is not None:
                nn.init.zeros_(self.input_proj.bias)
        
        if isinstance(self.output_proj, nn.Linear):
            nn.init.xavier_uniform_(self.output_proj.weight)
            if self.output_proj.bias is not None:
                nn.init.zeros_(self.output_proj.bias)
    
    def add_classifier(self, num_classes: int):
        """
        Add or replace classification head.
        
        Args:
            num_classes: Number of classes
        """
        self.config.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim // 2, num_classes),
        )
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Forward pass for homogeneous graphs.
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            edge_weight: Optional edge weights [E]
            return_attention: Return transformer attention weights
            
        Returns:
            Dict with:
            - embeddings: Node embeddings [N, embedding_dim]
            - Q: Soft cluster assignments [N, K]
            - P: Target distribution [N, K]
            - hard_labels: Hard cluster assignments [N]
            - logits: Classification logits [N, C] (if classifier exists)
            - attention_weights: List of attention matrices (if return_attention)
        """
        # Input projection
        h = self.input_proj(x)  # [N, hidden_dim]
        
        # GNN: local structure
        h = self.gnn(h, edge_index, edge_weight)  # [N, hidden_dim]
        
        # Prepare for transformer (single node type)
        h_by_type = {self.config.node_types[0]: h}
        
        # Transformer: global context
        h, attn_weights = self.transformer(
            h_by_type,
            return_all_attentions=return_attention,
        )  # [N, hidden_dim]
        
        # Output projection
        embeddings = self.output_proj(h)  # [N, embedding_dim]
        
        # Clustering
        Q, P, hard_labels = self.clustering_head(h)
        
        output = {
            'embeddings': embeddings,
            'hidden': h,
            'Q': Q,
            'P': P,
            'hard_labels': hard_labels,
        }
        
        # Classification if available
        if self.classifier is not None:
            output['logits'] = self.classifier(h)
        
        if return_attention and attn_weights is not None:
            output['attention_weights'] = attn_weights
        
        return output
    
    def forward_hetero(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        target_type: str = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass for heterogeneous graphs.
        
        Args:
            x_dict: Dict mapping node type to features [N_t, D]
            edge_index_dict: Dict mapping edge type to indices [2, E]
            target_type: Node type to cluster/classify (default: first type)
            
        Returns:
            Dict with embeddings, Q, P, hard_labels for target type
        """
        target_type = target_type or self.config.node_types[0]
        
        # Input projection per type
        h_dict = {
            ntype: self.input_proj(x)
            for ntype, x in x_dict.items()
        }
        
        # GNN: local structure (heterogeneous)
        h_dict = self.gnn(h_dict, edge_index_dict)
        
        # Transformer: global context
        h, _ = self.transformer(h_dict)
        
        # Split back to get target type embeddings
        # This is simplified - in practice track offsets
        offset = 0
        target_h = None
        for ntype in self.config.node_types:
            if ntype in x_dict:
                n = x_dict[ntype].size(0)
                if ntype == target_type:
                    target_h = h[offset:offset + n]
                offset += n
        
        if target_h is None:
            raise ValueError(f"Target type {target_type} not found in input")
        
        # Output projection
        embeddings = self.output_proj(target_h)
        
        # Clustering
        Q, P, hard_labels = self.clustering_head(target_h)
        
        output = {
            'embeddings': embeddings,
            'hidden': target_h,
            'Q': Q,
            'P': P,
            'hard_labels': hard_labels,
        }
        
        if self.classifier is not None:
            output['logits'] = self.classifier(target_h)
        
        return output
    
    def compute_loss(
        self,
        output: Dict[str, Tensor],
        labels: Optional[Tensor] = None,
        label_mask: Optional[Tensor] = None,
        classification_weight: Optional[float] = None,
        clustering_weight: Optional[float] = None,
        silhouette_weight: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute training losses.
        
        Multi-objective loss:
        L = w_cls * L_classification + w_kl * L_clustering + w_sil * L_silhouette
        
        Args:
            output: Forward pass output dict
            labels: Optional node labels [N]
            label_mask: Mask for labeled nodes [N] (for semi-supervised)
            classification_weight: Override config weight
            clustering_weight: Override config weight
            silhouette_weight: Override config weight
            
        Returns:
            Dict with individual losses and total loss
        """
        losses = {}
        
        # Use config weights as defaults
        cls_weight = classification_weight or self.config.classification_weight if hasattr(self.config, 'classification_weight') else 1.0
        kl_weight = clustering_weight or self.config.clustering_weight if hasattr(self.config, 'clustering_weight') else 1.0
        sil_weight = silhouette_weight or self.config.silhouette_weight if hasattr(self.config, 'silhouette_weight') else 0.1
        
        # Clustering loss (KL divergence)
        if self.clustering_head.centers_initialized:
            loss_kl = kl_divergence_loss(output['P'], output['Q'])
            losses['loss_kl'] = loss_kl
            
            # Silhouette loss
            loss_sil = soft_silhouette_loss(output['hidden'], output['Q'])
            losses['loss_silhouette'] = loss_sil
        else:
            # Before initialization, use zero placeholders
            losses['loss_kl'] = torch.tensor(0.0, device=output['embeddings'].device)
            losses['loss_silhouette'] = torch.tensor(0.0, device=output['embeddings'].device)
        
        # Classification loss
        if labels is not None and 'logits' in output:
            if label_mask is not None:
                # Semi-supervised: only compute on labeled nodes
                if label_mask.any():
                    loss_cls = F.cross_entropy(
                        output['logits'][label_mask],
                        labels[label_mask]
                    )
                else:
                    loss_cls = torch.tensor(0.0, device=output['embeddings'].device)
            else:
                loss_cls = F.cross_entropy(output['logits'], labels)
            losses['loss_classification'] = loss_cls
        else:
            losses['loss_classification'] = torch.tensor(0.0, device=output['embeddings'].device)
        
        # Total weighted loss
        total = (
            cls_weight * losses['loss_classification'] +
            kl_weight * losses['loss_kl'] +
            sil_weight * losses['loss_silhouette']
        )
        losses['total'] = total
        
        return losses
    
    def get_embeddings(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Get node embeddings (inference mode).
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            edge_weight: Optional edge weights
            
        Returns:
            Node embeddings [N, embedding_dim]
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x, edge_index, edge_weight)
        return output['embeddings']
    
    def get_communities(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get community assignments and confidences.
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            edge_weight: Optional edge weights
            
        Returns:
            Tuple of:
            - hard_labels: Community assignments [N]
            - confidences: Assignment confidence scores [N]
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x, edge_index, edge_weight)
        
        hard_labels = output['hard_labels']
        confidences = output['Q'].max(dim=1)[0]
        
        return hard_labels, confidences
    
    def predict_classes(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Predict node classes (requires classifier).
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_weight: Optional edge weights
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.classifier is None:
            raise ValueError("No classifier head - add one with add_classifier()")
        
        self.eval()
        with torch.no_grad():
            output = self.forward(x, edge_index, edge_weight)
        
        probs = F.softmax(output['logits'], dim=1)
        predictions = probs.argmax(dim=1)
        
        return predictions, probs
    
    def save(self, path: Union[str, Path]):
        """
        Save model checkpoint.
        
        Args:
            path: Save path (directory or .pt file)
        """
        path = Path(path)
        
        if path.suffix != '.pt':
            path.mkdir(parents=True, exist_ok=True)
            model_path = path / 'model.pt'
            config_path = path / 'config.json'
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            model_path = path
            config_path = path.with_suffix('.json')
        
        # Save model state
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
            'centers_initialized': self.clustering_head.centers_initialized,
        }, model_path)
        
        # Save config as JSON for readability
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info(f"Saved model to {model_path}")
    
    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[str] = None) -> 'GITCD':
        """
        Load model from checkpoint.
        
        Args:
            path: Checkpoint path
            device: Device to load to. If None, uses CUDA if available, else CPU.
            
        Returns:
            Loaded GITCD model
        """
        path = Path(path)
        
        if path.is_dir():
            model_path = path / 'model.pt'
        else:
            model_path = path
        
        # Determine device: CUDA if available, else CPU
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Loading model from {model_path} to device '{device}'")
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        config = GITCDConfig.from_dict(checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if checkpoint.get('centers_initialized', False):
            model.clustering_head._centers_initialized = True
        
        model.to(device)
        logger.info(f"Loaded model from {model_path} on {device}")
        
        return model
    
    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def __repr__(self) -> str:
        return (
            f"GITCD(\n"
            f"  input_dim={self.config.input_dim},\n"
            f"  hidden_dim={self.config.hidden_dim},\n"
            f"  embedding_dim={self.config.embedding_dim},\n"
            f"  num_gnn_layers={self.config.num_gnn_layers},\n"
            f"  num_transformer_layers={self.config.num_transformer_layers},\n"
            f"  num_clusters={self.config.num_clusters},\n"
            f"  num_classes={self.config.num_classes},\n"
            f"  parameters={self.num_parameters():,}\n"
            f")"
        )


def create_gitcd_model(
    input_dim: int,
    num_clusters: int,
    hidden_dim: int = 128,
    num_classes: Optional[int] = None,
    **kwargs,
) -> GITCD:
    """
    Factory function to create GIT-CD model.
    
    Args:
        input_dim: Input feature dimension
        num_clusters: Number of communities
        hidden_dim: Hidden dimension
        num_classes: Optional number of classes for classification
        **kwargs: Additional config parameters
        
    Returns:
        GITCD model
    """
    config = GITCDConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embedding_dim=hidden_dim,
        num_clusters=num_clusters,
        num_classes=num_classes,
        **kwargs,
    )
    return GITCD(config)