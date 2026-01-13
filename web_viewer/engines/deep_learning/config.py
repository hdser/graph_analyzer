"""
Deep Learning Configuration

All configurable parameters for the GIT-CD model and training pipeline.
No hardcoded values - everything is configurable via these dataclasses.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import os


class DeviceType(Enum):
    """Device types for computation."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


class NormalizationType(Enum):
    """Feature normalization types."""
    NONE = "none"
    STANDARD = "standard"  # z-score
    MINMAX = "minmax"      # [0, 1]
    ROBUST = "robust"      # IQR-based


class NaNStrategy(Enum):
    """NaN handling strategies."""
    ZERO = "zero"
    MEAN = "mean"
    MEDIAN = "median"
    DROP = "drop"


class AggregationType(Enum):
    """Aggregation types for heterogeneous convolutions."""
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"


@dataclass
class DeepLearningConfig:
    """
    Global configuration for deep learning module.
    
    Can be loaded from environment variables or config file.
    """
    
    # Device configuration - None means auto-detect
    device: Optional[str] = field(default_factory=lambda: os.getenv("DL_DEVICE", None))
    num_workers: int = field(default_factory=lambda: int(os.getenv("DL_NUM_WORKERS", "4")))
    pin_memory: bool = field(default_factory=lambda: os.getenv("DL_PIN_MEMORY", "true").lower() == "true")
    
    # Model storage
    model_dir: str = field(default_factory=lambda: os.getenv("DL_MODEL_DIR", "cache/models"))
    embedding_cache_dir: str = field(default_factory=lambda: os.getenv("DL_EMBEDDING_CACHE_DIR", "cache/embeddings"))
    
    # Default model parameters (can be overridden per-model)
    default_hidden_dim: int = field(default_factory=lambda: int(os.getenv("DL_HIDDEN_DIM", "128")))
    default_num_clusters: int = field(default_factory=lambda: int(os.getenv("DL_NUM_CLUSTERS", "10")))
    
    # Memory limits
    max_nodes_gpu: int = field(default_factory=lambda: int(os.getenv("DL_MAX_NODES_GPU", "500000")))
    batch_size_inference: int = field(default_factory=lambda: int(os.getenv("DL_BATCH_SIZE_INFERENCE", "10000")))
    
    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("DL_LOG_LEVEL", "INFO"))
    
    def get_device(self) -> str:
        """Get actual device string. None means auto-detect best available."""
        if self.device is None:
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                return "cpu"
            except ImportError:
                return "cpu"
        return self.device


@dataclass
class FeatureConfig:
    """
    Configuration for feature extraction from graph metrics.
    
    Defines which metrics to use as features and how to preprocess them.
    """
    
    # Metrics to include as node features
    # These should match column names in metrics DataFrame
    metric_columns: List[str] = field(default_factory=lambda: [
        "in_degree",
        "out_degree", 
        "pagerank",
        "betweenness_centrality",
        "clustering_coefficient",
        "eigenvector_centrality",
        "core_number",
    ])
    
    # Node properties to include (from external APIs or SQL)
    property_columns: List[str] = field(default_factory=list)
    
    # Columns to apply log1p transform (good for power-law distributed metrics)
    log_transform_columns: List[str] = field(default_factory=lambda: [
        "in_degree",
        "out_degree",
        "pagerank",
        "betweenness_centrality",
    ])
    
    # Preprocessing
    normalization: str = "standard"  # none, standard, minmax, robust
    nan_strategy: str = "zero"       # zero, mean, median, drop
    
    # For heterogeneous graphs
    node_type_column: Optional[str] = None  # Column indicating node type
    default_node_type: str = "node"         # Default type if not specified
    
    # Edge features
    edge_weight_column: Optional[str] = None
    edge_type_column: Optional[str] = None
    
    # Dimensionality
    max_feature_dim: Optional[int] = None  # Truncate/pad to this dim if set
    use_identity_features: bool = True      # Use identity matrix if no features


@dataclass
class GITCDConfig:
    """
    Configuration for GIT-CD model architecture.
    
    Based on paper defaults but fully configurable.
    """
    
    # Input/Output dimensions
    input_dim: int = 64              # Input feature dimension
    hidden_dim: int = 128            # Hidden layer dimension (paper: 128)
    embedding_dim: int = 128         # Output embedding dimension
    
    # GNN configuration
    num_gnn_layers: int = 1          # Number of SAGEConv layers (paper: 1)
    gnn_aggregation: str = "mean"    # Aggregation: mean, sum, max
    
    # Transformer configuration
    num_transformer_layers: int = 2  # Number of transformer blocks (paper: 2)
    num_attention_heads: int = 8     # Attention heads
    attention_dropout: float = 0.1   # Dropout in attention
    ffn_dim_multiplier: int = 4      # FFN hidden dim = hidden_dim * multiplier
    
    # Clustering configuration
    num_clusters: int = 10           # Number of communities to detect
    clustering_temperature: float = 1.0  # Initial temperature (trainable)
    
    # Regularization
    dropout: float = 0.5             # Global dropout (paper: 0.8, but 0.5 more stable)
    
    # Node types (for heterogeneous graphs)
    node_types: List[str] = field(default_factory=lambda: ["node"])
    edge_types: List[Tuple[str, str, str]] = field(default_factory=list)
    
    # Classification head (optional)
    num_classes: Optional[int] = None  # Set to enable classification
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.hidden_dim % self.num_attention_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_attention_heads ({self.num_attention_heads})"
        assert self.num_gnn_layers >= 1, "Need at least 1 GNN layer"
        assert self.num_transformer_layers >= 1, "Need at least 1 transformer layer"
        assert self.num_clusters >= 2, "Need at least 2 clusters"
    
    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_dim // self.num_attention_heads
    
    @property  
    def ffn_dim(self) -> int:
        """Feed-forward network hidden dimension."""
        return self.hidden_dim * self.ffn_dim_multiplier
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "embedding_dim": self.embedding_dim,
            "num_gnn_layers": self.num_gnn_layers,
            "gnn_aggregation": self.gnn_aggregation,
            "num_transformer_layers": self.num_transformer_layers,
            "num_attention_heads": self.num_attention_heads,
            "attention_dropout": self.attention_dropout,
            "ffn_dim_multiplier": self.ffn_dim_multiplier,
            "num_clusters": self.num_clusters,
            "clustering_temperature": self.clustering_temperature,
            "dropout": self.dropout,
            "node_types": self.node_types,
            "edge_types": self.edge_types,
            "num_classes": self.num_classes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GITCDConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TrainingConfig:
    """
    Configuration for GIT-CD training pipeline.
    
    Based on paper: Adam lr=3e-4, wd=5e-4, patience=5, max_epochs=200
    """
    
    # Optimization
    learning_rate: float = 3e-4      # Paper: 3e-4
    weight_decay: float = 5e-4       # Paper: 5e-4
    max_epochs: int = 200            # Paper: 200
    
    # Learning rate scheduling
    lr_scheduler: str = "none"       # none, cosine, step, plateau
    lr_warmup_epochs: int = 0        # Warmup epochs
    lr_min: float = 1e-6             # Minimum LR for schedulers
    
    # Gradient clipping
    gradient_clip_norm: Optional[float] = 1.0  # Max gradient norm
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 5                # Paper: 5
    min_delta: float = 1e-4          # Minimum improvement
    monitor_metric: str = "loss"     # Metric to monitor
    
    # KMeans clustering updates
    kmeans_warmup_epochs: int = 10   # Epochs before enabling clustering loss
    kmeans_update_interval: int = 5  # Epochs between center updates
    kmeans_n_init: int = 10          # KMeans initializations
    
    # Loss weights
    classification_weight: float = 1.0
    clustering_weight: float = 1.0
    silhouette_weight: float = 0.1
    
    # Batch configuration
    batch_size: int = 512            # Mini-batch size
    num_neighbors: List[int] = field(default_factory=lambda: [25, 10])  # Neighbors per layer
    
    # Validation
    val_split: float = 0.1           # Validation split ratio
    val_interval: int = 1            # Validate every N epochs
    
    # Device - None means auto-detect (CUDA if available, else CPU)
    device: Optional[str] = None
    mixed_precision: bool = False    # Use AMP for faster training
    
    # Checkpointing
    save_best_model: bool = True
    checkpoint_dir: str = "cache/checkpoints"
    checkpoint_interval: int = 10    # Save every N epochs
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False      # Slower but reproducible
    
    # Logging
    log_interval: int = 10           # Log every N epochs
    verbose: bool = True
    
    def get_device(self) -> str:
        """Get actual device string."""
        if self.device is None:
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                return "cpu"
            except ImportError:
                return "cpu"
        return self.device
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "lr_scheduler": self.lr_scheduler,
            "lr_warmup_epochs": self.lr_warmup_epochs,
            "lr_min": self.lr_min,
            "gradient_clip_norm": self.gradient_clip_norm,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "monitor_metric": self.monitor_metric,
            "kmeans_warmup_epochs": self.kmeans_warmup_epochs,
            "kmeans_update_interval": self.kmeans_update_interval,
            "kmeans_n_init": self.kmeans_n_init,
            "classification_weight": self.classification_weight,
            "clustering_weight": self.clustering_weight,
            "silhouette_weight": self.silhouette_weight,
            "batch_size": self.batch_size,
            "num_neighbors": self.num_neighbors,
            "val_split": self.val_split,
            "val_interval": self.val_interval,
            "device": self.device,
            "mixed_precision": self.mixed_precision,
            "seed": self.seed,
            "deterministic": self.deterministic,
        }


@dataclass
class InferenceConfig:
    """Configuration for embedding inference."""
    
    # Batching
    batch_size: int = 10000
    
    # Device - None means auto-detect (CUDA if available, else CPU)
    device: Optional[str] = None
    
    # Caching
    cache_embeddings: bool = True
    cache_dir: str = "cache/embeddings"
    cache_format: str = "parquet"  # parquet, npy, pt
    
    # Similarity search
    similarity_metric: str = "cosine"  # cosine, euclidean, dot
    top_k_default: int = 10
    
    # Visualization
    reduction_method: str = "umap"  # umap, tsne, pca
    reduction_dim: int = 2
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    
    def get_device(self) -> str:
        """Get actual device string."""
        if self.device is None:
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                return "cpu"
            except ImportError:
                return "cpu"
        return self.device


# Default configurations
DEFAULT_FEATURE_CONFIG = FeatureConfig()
DEFAULT_GITCD_CONFIG = GITCDConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_INFERENCE_CONFIG = InferenceConfig()


def create_circles_config(
    num_clusters: int = 20,
    hidden_dim: int = 128,
) -> GITCDConfig:
    """
    Create a GIT-CD config optimized for Circles trust networks.
    
    Args:
        num_clusters: Number of communities to detect
        hidden_dim: Hidden dimension
        
    Returns:
        GITCDConfig for Circles network
    """
    return GITCDConfig(
        hidden_dim=hidden_dim,
        embedding_dim=hidden_dim,
        num_clusters=num_clusters,
        node_types=["avatar", "group", "token_pool"],
        edge_types=[
            ("avatar", "trusts", "avatar"),
            ("avatar", "holds", "token_pool"),
            ("token_pool", "trusted_by", "avatar"),
            ("group", "mints_to", "avatar"),
        ],
        dropout=0.5,
    )


def create_feature_config_for_circles() -> FeatureConfig:
    """
    Create feature config for Circles trust networks.
    
    Returns:
        FeatureConfig optimized for Circles metrics
    """
    return FeatureConfig(
        metric_columns=[
            "in_degree",
            "out_degree",
            "pagerank",
            "eigentrust",
            "betweenness_centrality",
            "clustering_coefficient",
            "core_number",
            "community_id",
        ],
        property_columns=[
            "isBlacklisted",
            "balance",
        ],
        log_transform_columns=[
            "in_degree",
            "out_degree",
            "pagerank",
            "balance",
        ],
        normalization="standard",
        nan_strategy="zero",
    )