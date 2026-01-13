"""
Graph Converter for Deep Learning

Converts NetworkX graphs to PyTorch Geometric Data/HeteroData objects.

Supports:
- Homogeneous graphs (single node/edge type)
- Heterogeneous graphs (multiple node/edge types)
- Edge weights and attributes
- Node labels and masks
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

# Conditional imports for PyTorch Geometric
try:
    import torch
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Tensor = Any

try:
    from torch_geometric.data import Data, HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    Data = Any
    HeteroData = Any


@dataclass
class ConversionConfig:
    """Configuration for graph conversion."""
    
    # Node type inference
    node_type_attr: str = "node_type"     # Node attribute for type
    default_node_type: str = "node"       # Default type if not specified
    
    # Edge type inference  
    edge_type_attr: str = "edge_type"     # Edge attribute for type
    default_edge_type: str = "connects"   # Default relation name
    
    # Edge weights
    weight_attr: Optional[str] = "weight"  # Edge weight attribute
    default_weight: float = 1.0            # Default if not present
    
    # Labels
    label_attr: Optional[str] = None       # Node attribute for labels
    
    # Masks
    train_mask_attr: Optional[str] = None  # Node attribute for train mask
    val_mask_attr: Optional[str] = None    # Node attribute for val mask
    test_mask_attr: Optional[str] = None   # Node attribute for test mask


class GraphConverter:
    """
    Converts NetworkX graphs to PyTorch Geometric format.
    
    Usage:
        converter = GraphConverter()
        
        # Homogeneous graph
        data = converter.to_pyg_data(G, node_features)
        
        # Heterogeneous graph
        hetero_data = converter.to_pyg_hetero_data(G, node_features_dict)
    """
    
    def __init__(self, config: Optional[ConversionConfig] = None):
        """
        Initialize converter.
        
        Args:
            config: Conversion configuration
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for GraphConverter")
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric required for GraphConverter")
        
        self.config = config or ConversionConfig()
    
    def to_pyg_data(
        self,
        G: nx.Graph,
        node_features: Optional[np.ndarray] = None,
        node_to_idx: Optional[Dict[str, int]] = None,
        labels: Optional[np.ndarray] = None,
        train_mask: Optional[np.ndarray] = None,
        val_mask: Optional[np.ndarray] = None,
        test_mask: Optional[np.ndarray] = None,
    ) -> Data:
        """
        Convert NetworkX graph to PyG Data object.
        
        Args:
            G: NetworkX graph (directed or undirected)
            node_features: Node feature matrix [N, D]
            node_to_idx: Optional pre-computed node to index mapping
            labels: Optional node labels [N]
            train_mask: Optional training mask [N]
            val_mask: Optional validation mask [N]
            test_mask: Optional test mask [N]
            
        Returns:
            PyG Data object
        """
        # Build node mapping
        nodes = list(G.nodes())
        if node_to_idx is None:
            node_to_idx = {str(n): i for i, n in enumerate(nodes)}
        
        num_nodes = len(nodes)
        
        # Build edge index
        edge_index = self._build_edge_index(G, node_to_idx)
        
        # Build edge weights if available
        edge_weight = self._extract_edge_weights(G, node_to_idx)
        
        # Create features tensor
        if node_features is not None:
            x = torch.tensor(node_features, dtype=torch.float32)
        else:
            # Identity features
            x = torch.eye(num_nodes, dtype=torch.float32)
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            num_nodes=num_nodes,
        )
        
        # Add edge weight if available
        if edge_weight is not None:
            data.edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
        
        # Add labels
        if labels is not None:
            data.y = torch.tensor(labels, dtype=torch.long)
        
        # Add masks
        if train_mask is not None:
            data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        if val_mask is not None:
            data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
        if test_mask is not None:
            data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
        
        # Store node mapping for reference
        data.node_to_idx = node_to_idx
        data.idx_to_node = {i: n for n, i in node_to_idx.items()}
        
        logger.info(f"Created PyG Data: {num_nodes} nodes, {edge_index.shape[1]} edges")
        
        return data
    
    def to_pyg_hetero_data(
        self,
        G: nx.Graph,
        node_features_dict: Dict[str, np.ndarray],
        node_to_idx_dict: Optional[Dict[str, Dict[str, int]]] = None,
        labels_dict: Optional[Dict[str, np.ndarray]] = None,
        train_mask_dict: Optional[Dict[str, np.ndarray]] = None,
    ) -> HeteroData:
        """
        Convert NetworkX graph to PyG HeteroData object.
        
        Requires node type and edge type information in graph attributes.
        
        Args:
            G: NetworkX graph with node_type and edge_type attributes
            node_features_dict: Dict mapping node type to feature matrix
            node_to_idx_dict: Optional dict mapping node type to node-index mapping
            labels_dict: Optional dict mapping node type to labels
            train_mask_dict: Optional dict mapping node type to training mask
            
        Returns:
            PyG HeteroData object
        """
        # Extract node types
        node_types = self._extract_node_types(G)
        
        # Build node mappings per type
        if node_to_idx_dict is None:
            node_to_idx_dict = {}
            for ntype, nodes in node_types.items():
                node_to_idx_dict[ntype] = {str(n): i for i, n in enumerate(nodes)}
        
        # Extract edge types
        edge_types = self._extract_edge_types(G, node_types)
        
        # Create HeteroData
        data = HeteroData()
        
        # Add node features
        for ntype, features in node_features_dict.items():
            data[ntype].x = torch.tensor(features, dtype=torch.float32)
            data[ntype].num_nodes = features.shape[0]
        
        # Add edge indices
        for (src_type, rel, dst_type), edges in edge_types.items():
            src_mapping = node_to_idx_dict.get(src_type, {})
            dst_mapping = node_to_idx_dict.get(dst_type, {})
            
            src_indices = []
            dst_indices = []
            
            for src, dst in edges:
                src_str = str(src)
                dst_str = str(dst)
                if src_str in src_mapping and dst_str in dst_mapping:
                    src_indices.append(src_mapping[src_str])
                    dst_indices.append(dst_mapping[dst_str])
            
            if src_indices:
                edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
                data[src_type, rel, dst_type].edge_index = edge_index
        
        # Add labels
        if labels_dict:
            for ntype, labels in labels_dict.items():
                data[ntype].y = torch.tensor(labels, dtype=torch.long)
        
        # Add training masks
        if train_mask_dict:
            for ntype, mask in train_mask_dict.items():
                data[ntype].train_mask = torch.tensor(mask, dtype=torch.bool)
        
        # Store mappings
        data.node_to_idx_dict = node_to_idx_dict
        
        return data
    
    def _build_edge_index(
        self,
        G: nx.Graph,
        node_to_idx: Dict[str, int]
    ) -> np.ndarray:
        """Build edge index from graph."""
        edges = list(G.edges())
        
        if not edges:
            return np.zeros((2, 0), dtype=np.int64)
        
        src_indices = []
        dst_indices = []
        
        for u, v in edges:
            u_str = str(u)
            v_str = str(v)
            if u_str in node_to_idx and v_str in node_to_idx:
                src_indices.append(node_to_idx[u_str])
                dst_indices.append(node_to_idx[v_str])
        
        return np.array([src_indices, dst_indices], dtype=np.int64)
    
    def _extract_edge_weights(
        self,
        G: nx.Graph,
        node_to_idx: Dict[str, int]
    ) -> Optional[np.ndarray]:
        """Extract edge weights from graph."""
        weight_attr = self.config.weight_attr
        if weight_attr is None:
            return None
        
        weights = []
        for u, v, data in G.edges(data=True):
            u_str = str(u)
            v_str = str(v)
            if u_str in node_to_idx and v_str in node_to_idx:
                w = data.get(weight_attr, self.config.default_weight)
                weights.append(float(w))
        
        if not weights:
            return None
        
        return np.array(weights, dtype=np.float32)
    
    def _extract_node_types(
        self,
        G: nx.Graph
    ) -> Dict[str, List[Any]]:
        """Extract nodes grouped by type."""
        type_attr = self.config.node_type_attr
        default_type = self.config.default_node_type
        
        node_types: Dict[str, List] = {}
        
        for node, data in G.nodes(data=True):
            ntype = data.get(type_attr, default_type)
            if ntype not in node_types:
                node_types[ntype] = []
            node_types[ntype].append(node)
        
        return node_types
    
    def _extract_edge_types(
        self,
        G: nx.Graph,
        node_types: Dict[str, List]
    ) -> Dict[Tuple[str, str, str], List[Tuple]]:
        """Extract edges grouped by type tuple (src_type, rel, dst_type)."""
        type_attr = self.config.node_type_attr
        edge_type_attr = self.config.edge_type_attr
        default_node_type = self.config.default_node_type
        default_edge_type = self.config.default_edge_type
        
        # Build node -> type mapping
        node_to_type = {}
        for ntype, nodes in node_types.items():
            for node in nodes:
                node_to_type[node] = ntype
        
        edge_types: Dict[Tuple[str, str, str], List] = {}
        
        for u, v, data in G.edges(data=True):
            src_type = node_to_type.get(u, default_node_type)
            dst_type = node_to_type.get(v, default_node_type)
            rel = data.get(edge_type_attr, default_edge_type)
            
            key = (src_type, rel, dst_type)
            if key not in edge_types:
                edge_types[key] = []
            edge_types[key].append((u, v))
        
        return edge_types
    
    @staticmethod
    def from_edge_list(
        edges: List[Tuple[Any, Any]],
        node_features: Optional[np.ndarray] = None,
        edge_weights: Optional[List[float]] = None,
    ) -> Data:
        """
        Create PyG Data from edge list.
        
        Args:
            edges: List of (source, target) tuples
            node_features: Optional node features
            edge_weights: Optional edge weights
            
        Returns:
            PyG Data object
        """
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric required")
        
        # Build node mapping
        nodes = set()
        for u, v in edges:
            nodes.add(u)
            nodes.add(v)
        nodes = sorted(list(nodes))
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        
        # Build edge index
        src = [node_to_idx[e[0]] for e in edges]
        dst = [node_to_idx[e[1]] for e in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        
        # Features
        num_nodes = len(nodes)
        if node_features is not None:
            x = torch.tensor(node_features, dtype=torch.float32)
        else:
            x = torch.eye(num_nodes, dtype=torch.float32)
        
        data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
        
        if edge_weights is not None:
            data.edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
        
        data.node_to_idx = node_to_idx
        data.idx_to_node = {i: n for n, i in node_to_idx.items()}
        
        return data
    
    @staticmethod
    def add_self_loops(data: Data) -> Data:
        """Add self-loops to graph."""
        from torch_geometric.utils import add_self_loops as pyg_add_self_loops
        
        edge_index, edge_weight = pyg_add_self_loops(
            data.edge_index,
            data.edge_weight if hasattr(data, 'edge_weight') else None,
            num_nodes=data.num_nodes
        )
        
        data.edge_index = edge_index
        if edge_weight is not None:
            data.edge_weight = edge_weight
        
        return data
    
    @staticmethod
    def to_undirected(data: Data) -> Data:
        """Convert directed graph to undirected by adding reverse edges."""
        from torch_geometric.utils import to_undirected as pyg_to_undirected
        
        edge_index = pyg_to_undirected(data.edge_index, num_nodes=data.num_nodes)
        data.edge_index = edge_index
        
        # Edge weight handling for undirected
        if hasattr(data, 'edge_weight') and data.edge_weight is not None:
            # Duplicate weights for reverse edges
            data.edge_weight = torch.cat([data.edge_weight, data.edge_weight])
        
        return data


def create_train_val_test_masks(
    num_nodes: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create random train/val/test masks.
    
    Args:
        num_nodes: Number of nodes
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        seed: Random seed
        
    Returns:
        Tuple of (train_mask, val_mask, test_mask)
    """
    np.random.seed(seed)
    indices = np.random.permutation(num_nodes)
    
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    return train_mask, val_mask, test_mask


def create_label_masks(
    labels: np.ndarray,
    labeled_ratio: float = 0.1,
    seed: int = 42
) -> np.ndarray:
    """
    Create mask for labeled nodes (semi-supervised setting).
    
    Args:
        labels: Node labels (-1 for unlabeled)
        labeled_ratio: Ratio of labeled nodes to use
        seed: Random seed
        
    Returns:
        Boolean mask for labeled nodes
    """
    np.random.seed(seed)
    
    # Find actually labeled nodes
    labeled_indices = np.where(labels >= 0)[0]
    
    if labeled_ratio >= 1.0:
        mask = np.zeros(len(labels), dtype=bool)
        mask[labeled_indices] = True
        return mask
    
    # Sample subset
    num_to_sample = int(len(labeled_indices) * labeled_ratio)
    sampled = np.random.choice(labeled_indices, size=num_to_sample, replace=False)
    
    mask = np.zeros(len(labels), dtype=bool)
    mask[sampled] = True
    
    return mask