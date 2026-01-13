"""
GNN Module for GIT-CD

Implements Graph Neural Network layers for local neighborhood aggregation.

Supports:
- Homogeneous graphs (single node/edge type)
- Heterogeneous graphs (multiple node/edge types)
- Configurable number of layers
- Multiple aggregation methods
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# Conditional imports for PyTorch Geometric
try:
    from torch_geometric.nn import SAGEConv, HeteroConv, Linear, GATConv, GCNConv
    from torch_geometric.typing import EdgeType
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    logger.warning("PyTorch Geometric not available - GNN module disabled")


class HomogeneousSAGE(nn.Module):
    """
    GraphSAGE for homogeneous graphs (single node type).
    
    Applies multiple layers of SAGEConv with configurable aggregation.
    Based on GIT-CD paper which uses 1 SAGEConv layer.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output embedding dimension
        num_layers: Number of SAGEConv layers (paper: 1)
        dropout: Dropout rate (paper: 0.8, but 0.5 more stable)
        aggregation: Aggregation method (mean, sum, max)
        normalize: Whether to L2 normalize output embeddings
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 1,
        dropout: float = 0.5,
        aggregation: str = "mean",
        normalize: bool = False,
    ):
        super().__init__()
        
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric required for GNN module")
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.normalize = normalize
        
        # Build convolution layers
        self.convs = nn.ModuleList()
        
        if num_layers == 1:
            # Single layer: in -> out
            self.convs.append(SAGEConv(in_channels, out_channels, aggr=aggregation))
        else:
            # First layer: in -> hidden
            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregation))
            
            # Middle layers: hidden -> hidden
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregation))
            
            # Last layer: hidden -> out
            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggregation))
        
        # Optional batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels if i < num_layers - 1 else out_channels)
            for i in range(num_layers)
        ])
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
            
        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        for i, conv in enumerate(self.convs):
            # Message passing
            x = conv(x, edge_index)
            
            # Apply batch norm, activation, dropout for all but last layer
            if i < self.num_layers - 1:
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Optional L2 normalization
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        
        return x


class HeterogeneousSAGE(nn.Module):
    """
    GraphSAGE for heterogeneous graphs (multiple node/edge types).
    
    Uses HeteroConv to apply different convolutions per edge type,
    then aggregates results per node type.
    
    Args:
        node_types: List of node type names
        edge_types: List of edge type tuples (src_type, relation, dst_type)
        in_channels_dict: Dict mapping node type to input dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output embedding dimension
        num_layers: Number of convolution layers
        dropout: Dropout rate
        aggregation: Aggregation for HeteroConv (sum, mean, max)
    """
    
    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        in_channels_dict: Dict[str, int],
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 1,
        dropout: float = 0.5,
        aggregation: str = "sum",
    ):
        super().__init__()
        
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric required for GNN module")
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection layers to common dimension
        self.input_linears = nn.ModuleDict()
        for node_type in node_types:
            in_dim = in_channels_dict.get(node_type, hidden_channels)
            self.input_linears[node_type] = Linear(in_dim, hidden_channels)
        
        # Build heterogeneous convolution layers
        self.convs = nn.ModuleList()
        
        for layer_idx in range(num_layers):
            # Output dimension: hidden for all but last, out for last
            out_dim = out_channels if layer_idx == num_layers - 1 else hidden_channels
            
            # Create SAGEConv for each edge type
            conv_dict = {}
            for edge_type in edge_types:
                src_type, rel, dst_type = edge_type
                conv_dict[edge_type] = SAGEConv(
                    hidden_channels,
                    out_dim,
                    aggr="mean"
                )
            
            self.convs.append(HeteroConv(conv_dict, aggr=aggregation))
        
        # Batch normalization per node type
        self.batch_norms = nn.ModuleList()
        for layer_idx in range(num_layers):
            out_dim = out_channels if layer_idx == num_layers - 1 else hidden_channels
            bn_dict = nn.ModuleDict({
                nt: nn.BatchNorm1d(out_dim) for nt in node_types
            })
            self.batch_norms.append(bn_dict)
    
    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> Dict[str, Tensor]:
        """
        Forward pass for heterogeneous graph.
        
        Args:
            x_dict: Dict mapping node type to feature tensor [num_nodes_type, in_dim]
            edge_index_dict: Dict mapping edge type to edge indices [2, num_edges]
            
        Returns:
            Dict mapping node type to embedding tensor [num_nodes_type, out_channels]
        """
        # Project inputs to common dimension
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.input_linears:
                h_dict[node_type] = self.input_linears[node_type](x)
            else:
                h_dict[node_type] = x
        
        # Apply convolutions
        for layer_idx, conv in enumerate(self.convs):
            # Message passing
            h_dict = conv(h_dict, edge_index_dict)
            
            # Apply batch norm, activation, dropout
            if layer_idx < self.num_layers - 1:
                bn_dict = self.batch_norms[layer_idx]
                h_dict = {
                    nt: F.relu(bn_dict[nt](h)) if nt in bn_dict else F.relu(h)
                    for nt, h in h_dict.items()
                }
                h_dict = {
                    nt: F.dropout(h, p=self.dropout, training=self.training)
                    for nt, h in h_dict.items()
                }
        
        return h_dict


class GATLayer(nn.Module):
    """
    Graph Attention Network layer (alternative to SAGE).
    
    Uses attention mechanism for neighbor aggregation.
    
    Args:
        in_channels: Input dimension
        out_channels: Output dimension
        heads: Number of attention heads
        dropout: Dropout rate
        concat: Concatenate heads (True) or average (False)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.5,
        concat: bool = True,
    ):
        super().__init__()
        
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric required")
        
        self.conv = GATConv(
            in_channels,
            out_channels,
            heads=heads,
            dropout=dropout,
            concat=concat,
        )
        
        actual_out = out_channels * heads if concat else out_channels
        self.batch_norm = nn.BatchNorm1d(actual_out)
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Forward pass."""
        x = self.conv(x, edge_index)
        x = self.batch_norm(x)
        x = F.elu(x)
        return x


class MultiLayerGNN(nn.Module):
    """
    Flexible multi-layer GNN with configurable layer types.
    
    Supports mixing different GNN architectures.
    
    Args:
        in_channels: Input dimension
        hidden_channels: Hidden dimension
        out_channels: Output dimension
        num_layers: Number of layers
        layer_type: Type of GNN layer (sage, gat, gcn)
        dropout: Dropout rate
        **kwargs: Additional layer-specific arguments
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        layer_type: str = "sage",
        dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric required")
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Layer factory
        def make_layer(in_dim: int, out_dim: int) -> nn.Module:
            if layer_type == "sage":
                return SAGEConv(in_dim, out_dim, **kwargs)
            elif layer_type == "gat":
                heads = kwargs.get("heads", 4)
                return GATConv(in_dim, out_dim // heads, heads=heads, concat=True)
            elif layer_type == "gcn":
                return GCNConv(in_dim, out_dim, **kwargs)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
        
        # Build layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        if num_layers == 1:
            self.layers.append(make_layer(in_channels, out_channels))
            self.batch_norms.append(nn.BatchNorm1d(out_channels))
        else:
            # First
            self.layers.append(make_layer(in_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
            # Middle
            for _ in range(num_layers - 2):
                self.layers.append(make_layer(hidden_channels, hidden_channels))
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
            # Last
            self.layers.append(make_layer(hidden_channels, out_channels))
            self.batch_norms.append(nn.BatchNorm1d(out_channels))
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Forward pass."""
        for i, (layer, bn) in enumerate(zip(self.layers, self.batch_norms)):
            x = layer(x, edge_index)
            
            if i < self.num_layers - 1:
                x = bn(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


def create_gnn(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int = 1,
    gnn_type: str = "sage",
    dropout: float = 0.5,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create GNN module.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension
        out_channels: Output dimension
        num_layers: Number of layers
        gnn_type: Type of GNN (sage, gat, gcn, multi)
        dropout: Dropout rate
        **kwargs: Additional arguments
        
    Returns:
        GNN module
    """
    if gnn_type == "sage":
        return HomogeneousSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            aggregation=kwargs.get("aggregation", "mean"),
        )
    elif gnn_type == "multi":
        return MultiLayerGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            layer_type=kwargs.get("layer_type", "sage"),
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown GNN type: {gnn_type}")