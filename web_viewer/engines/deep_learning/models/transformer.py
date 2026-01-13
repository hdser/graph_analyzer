"""
Dynamic Heterogeneous Transformer for GIT-CD

Implements type-aware multi-head self-attention for heterogeneous
information networks - the key innovation from the GIT-CD paper.

Key features:
- Separate Q/K/V projections per node type
- Computes both intra-type and inter-type attention
- Full transformer block with FFN and residual connections
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class DynamicHINAttention(nn.Module):
    """
    Dynamic Heterogeneous Information Network Attention.
    
    This is the key innovation from GIT-CD paper:
    - Uses separate Q/K/V projections per node type
    - Computes attention scores considering node types
    - Enables learning type-specific interaction patterns
    
    From paper equations:
    - Q_t, K_t, V_t = type-specific projections
    - Score(v,v) = Q_v · K_v^T (intra-type)
    - Score(v,u) = Q_v · K_u^T (inter-type)
    
    Args:
        node_types: List of node type names
        hidden_dim: Hidden dimension (must be divisible by num_heads)
        num_heads: Number of attention heads
        dropout: Attention dropout rate
        use_bias: Whether to use bias in projections
    """
    
    def __init__(
        self,
        node_types: List[str],
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_bias: bool = False,
    ):
        super().__init__()
        
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        self.node_types = list(node_types)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1/sqrt(d_k)
        
        # Type-specific Q/K/V projections (the paper's key innovation)
        self.W_q = nn.ModuleDict({
            t: nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
            for t in node_types
        })
        self.W_k = nn.ModuleDict({
            t: nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
            for t in node_types
        })
        self.W_v = nn.ModuleDict({
            t: nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
            for t in node_types
        })
        
        # Output projection
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Layer normalization (pre-norm style)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform."""
        for module_dict in [self.W_q, self.W_k, self.W_v]:
            for module in module_dict.values():
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        nn.init.xavier_uniform_(self.W_o.weight)
        if self.W_o.bias is not None:
            nn.init.zeros_(self.W_o.bias)
    
    def forward(
        self,
        h_by_type: Dict[str, Tensor],
        attention_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass with type-aware attention.
        
        Args:
            h_by_type: Dict mapping node type to embeddings [n_t, hidden_dim]
            attention_mask: Optional mask [N, N] where True = attend, False = mask
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of:
            - Updated embeddings [N, hidden_dim] (concatenated all types)
            - Optional attention weights [num_heads, N, N] if return_attention=True
        """
        # Track node types and concatenate
        type_offsets = {}  # type -> (start, end)
        feat_list = []
        offset = 0
        
        for node_type in self.node_types:
            if node_type in h_by_type:
                h = h_by_type[node_type]
                type_offsets[node_type] = (offset, offset + h.size(0))
                offset += h.size(0)
                feat_list.append(h)
        
        if not feat_list:
            empty = torch.tensor([], device=next(iter(self.W_q.values())).weight.device)
            return empty, None if return_attention else empty
        
        x = torch.cat(feat_list, dim=0)  # [N, D]
        N, D = x.shape
        
        # Apply layer norm (pre-norm)
        x_norm = self.layer_norm(x)
        
        # Build type-specific Q/K/V
        Q = torch.zeros(N, self.hidden_dim, device=x.device, dtype=x.dtype)
        K = torch.zeros(N, self.hidden_dim, device=x.device, dtype=x.dtype)
        V = torch.zeros(N, self.hidden_dim, device=x.device, dtype=x.dtype)
        
        for node_type, (start, end) in type_offsets.items():
            h_t = x_norm[start:end]
            Q[start:end] = self.W_q[node_type](h_t)
            K[start:end] = self.W_k[node_type](h_t)
            V[start:end] = self.W_v[node_type](h_t)
        
        # Reshape for multi-head attention: [N, D] -> [num_heads, N, head_dim]
        Q = Q.view(N, self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view(N, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(N, self.num_heads, self.head_dim).transpose(0, 1)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [H, N, N]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: [N, N] with True for valid pairs
            scores = scores.masked_fill(~attention_mask.unsqueeze(0), float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # [H, N, head_dim]
        
        # Merge heads: [H, N, head_dim] -> [N, D]
        out = out.transpose(0, 1).contiguous().view(N, self.hidden_dim)
        
        # Output projection
        out = self.W_o(out)
        out = self.resid_dropout(out)
        
        # Residual connection
        out = x + out
        
        if return_attention:
            return out, attn_weights
        return out, None


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = Linear(GELU(Linear(x)))
    
    Args:
        hidden_dim: Input/output dimension
        ffn_dim: Hidden dimension (typically 4x hidden_dim)
        dropout: Dropout rate
        activation: Activation function (gelu, relu, silu)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        
        ffn_dim = ffn_dim or hidden_dim * 4
        
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Activation
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class HINTransformerBlock(nn.Module):
    """
    Full Transformer block with Dynamic HIN Attention.
    
    Architecture (pre-norm style):
    1. LayerNorm + Dynamic HIN Multi-Head Attention + Residual
    2. LayerNorm + FFN + Residual
    
    Args:
        node_types: List of node type names
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        ffn_dim: FFN hidden dimension (default: 4 * hidden_dim)
        dropout: Dropout rate
        attention_dropout: Attention-specific dropout
        activation: FFN activation function
    """
    
    def __init__(
        self,
        node_types: List[str],
        hidden_dim: int,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        attention_dropout: Optional[float] = None,
        activation: str = "gelu",
    ):
        super().__init__()
        
        attention_dropout = attention_dropout or dropout
        ffn_dim = ffn_dim or hidden_dim * 4
        
        self.node_types = node_types
        self.hidden_dim = hidden_dim
        
        # Dynamic HIN Attention
        self.attention = DynamicHINAttention(
            node_types=node_types,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        
        # FFN with pre-norm
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForwardNetwork(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
            activation=activation,
        )
    
    def forward(
        self,
        h_by_type: Dict[str, Tensor],
        attention_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass.
        
        Args:
            h_by_type: Dict mapping node type to embeddings
            attention_mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of:
            - Updated embeddings (concatenated) [N, hidden_dim]
            - Optional attention weights
        """
        # Attention sublayer (includes residual in DynamicHINAttention)
        x, attn_weights = self.attention(h_by_type, attention_mask, return_attention)
        
        # FFN sublayer with pre-norm and residual
        x = x + self.ffn(self.ffn_norm(x))
        
        if return_attention:
            return x, attn_weights
        return x, None
    
    def forward_dict(
        self,
        h_by_type: Dict[str, Tensor],
        attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass returning dict by type (for stacking blocks).
        
        Args:
            h_by_type: Dict mapping node type to embeddings
            attention_mask: Optional attention mask
            
        Returns:
            Dict mapping node type to updated embeddings
        """
        # Get concatenated output
        x, _ = self.forward(h_by_type, attention_mask)
        
        # Split back by type
        result = {}
        offset = 0
        for node_type in self.node_types:
            if node_type in h_by_type:
                n = h_by_type[node_type].size(0)
                result[node_type] = x[offset:offset + n]
                offset += n
        
        return result


class TransformerEncoder(nn.Module):
    """
    Stack of Transformer blocks.
    
    Args:
        node_types: List of node type names
        hidden_dim: Hidden dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        ffn_dim: FFN hidden dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        node_types: List[str],
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            HINTransformerBlock(
                node_types=node_types,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        h_by_type: Dict[str, Tensor],
        attention_mask: Optional[Tensor] = None,
        return_all_attentions: bool = False,
    ) -> Tuple[Tensor, Optional[List[Tensor]]]:
        """
        Forward pass through all transformer layers.
        
        Args:
            h_by_type: Dict mapping node type to embeddings
            attention_mask: Optional attention mask
            return_all_attentions: Return attention weights from all layers
            
        Returns:
            Tuple of:
            - Final embeddings (concatenated) [N, hidden_dim]
            - Optional list of attention weights per layer
        """
        all_attentions = [] if return_all_attentions else None
        
        for layer in self.layers:
            if return_all_attentions:
                h_by_type = layer.forward_dict(h_by_type, attention_mask)
                # Re-run to get attention (not efficient but correct)
                _, attn = layer(h_by_type, attention_mask, return_attention=True)
                all_attentions.append(attn)
            else:
                h_by_type = layer.forward_dict(h_by_type, attention_mask)
        
        # Concatenate final output
        feat_list = [h_by_type[t] for t in self.layers[0].node_types if t in h_by_type]
        x = torch.cat(feat_list, dim=0) if feat_list else torch.tensor([])
        
        # Final normalization
        x = self.final_norm(x)
        
        return x, all_attentions


class PositionalEncoding(nn.Module):
    """
    Optional positional encoding for graph nodes.
    
    Uses Laplacian eigenvector-based positional encoding.
    
    Args:
        hidden_dim: Embedding dimension
        max_nodes: Maximum number of nodes
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        hidden_dim: int,
        max_nodes: int = 10000,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Learnable positional embeddings
        self.pe = nn.Embedding(max_nodes, hidden_dim)
        
        nn.init.normal_(self.pe.weight, std=0.02)
    
    def forward(
        self,
        x: Tensor,
        positions: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Add positional encoding.
        
        Args:
            x: Node embeddings [N, D]
            positions: Optional position indices [N]
            
        Returns:
            Embeddings with positional encoding
        """
        N = x.size(0)
        
        if positions is None:
            positions = torch.arange(N, device=x.device)
        
        # Clamp to max positions
        positions = positions.clamp(0, self.pe.num_embeddings - 1)
        
        pe = self.pe(positions)
        x = x + pe
        x = self.dropout(x)
        
        return x