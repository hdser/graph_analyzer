"""
Pydantic Schemas for Embedding API

Request and response models for the embedding endpoints.
All parameters are configurable through these schemas.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


class DeviceType(str, Enum):
    """Computation device types."""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class ReductionMethod(str, Enum):
    """Dimensionality reduction methods."""
    UMAP = "umap"
    TSNE = "tsne"
    PCA = "pca"


class SimilarityMetric(str, Enum):
    """Similarity metrics."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT = "dot"


# ============== Training ==============

class TrainEmbeddingRequest(BaseModel):
    """Request to train a GIT-CD model."""
    
    # Graph to train on
    graph_name: Optional[str] = Field(
        default=None,
        description="Name of graph to train on. If None, uses the currently loaded graph."
    )
    
    # Model name (auto-generated if not provided)
    model_name: Optional[str] = Field(
        default=None,
        description="Name for the trained model. If None, auto-generates based on graph name."
    )
    
    # Model architecture
    num_clusters: int = Field(
        default=20,
        ge=2,
        le=1000,
        description="Number of communities to detect"
    )
    hidden_dim: int = Field(
        default=128,
        ge=32,
        le=512,
        description="Hidden layer dimension"
    )
    num_gnn_layers: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Number of GNN layers"
    )
    num_transformer_layers: int = Field(
        default=2,
        ge=1,
        le=6,
        description="Number of transformer layers"
    )
    num_attention_heads: int = Field(
        default=8,
        ge=1,
        le=16,
        description="Number of attention heads"
    )
    dropout: float = Field(
        default=0.5,
        ge=0.0,
        le=0.9,
        description="Dropout rate"
    )
    
    # Training parameters
    max_epochs: int = Field(
        default=200,
        ge=10,
        le=1000,
        description="Maximum training epochs"
    )
    learning_rate: float = Field(
        default=3e-4,
        ge=1e-6,
        le=1e-1,
        description="Learning rate"
    )
    weight_decay: float = Field(
        default=5e-4,
        ge=0.0,
        le=1e-1,
        description="Weight decay for regularization"
    )
    patience: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Early stopping patience"
    )
    
    # Feature configuration
    metric_columns: Optional[List[str]] = Field(
        default=None,
        description="Metric columns to use as features (None = auto)"
    )
    
    # Device
    device: DeviceType = Field(
        default=DeviceType.AUTO,
        description="Computation device"
    )


class TrainEmbeddingResponse(BaseModel):
    """Response from training."""
    
    success: bool
    model_name: str
    num_nodes: int
    num_features: int
    num_clusters: int
    final_loss: Optional[float]
    epochs_trained: int
    training_time_seconds: float
    silhouette_score: Optional[float]
    message: Optional[str] = None


# ============== Inference ==============

class ComputeEmbeddingsRequest(BaseModel):
    """Request to compute embeddings."""
    
    model_name: Optional[str] = Field(
        default=None,
        description="Model to use (None = current model)"
    )
    cache_key: Optional[str] = Field(
        default=None,
        description="Cache key for results"
    )
    include_communities: bool = Field(
        default=True,
        description="Include community assignments"
    )
    include_confidences: bool = Field(
        default=True,
        description="Include confidence scores"
    )


class EmbeddingNode(BaseModel):
    """Single node embedding."""
    
    node_id: str
    embedding: List[float]
    community: Optional[int] = None
    confidence: Optional[float] = None


class ComputeEmbeddingsResponse(BaseModel):
    """Response with embeddings."""
    
    success: bool
    num_nodes: int
    embedding_dim: int
    num_communities: int
    nodes: Optional[List[EmbeddingNode]] = None  # Optional for large graphs
    communities_summary: Optional[Dict[int, int]] = None  # community -> count
    message: Optional[str] = None


# ============== Communities ==============

class GetCommunitiesRequest(BaseModel):
    """Request community assignments."""
    
    model_name: Optional[str] = Field(
        default=None,
        description="Model to use"
    )
    include_confidence: bool = Field(
        default=True,
        description="Include confidence scores"
    )


class CommunityAssignment(BaseModel):
    """Community assignment for a node."""
    
    node_id: str
    community: int
    confidence: Optional[float] = None


class GetCommunitiesResponse(BaseModel):
    """Response with community assignments."""
    
    success: bool
    num_nodes: int
    num_communities: int
    assignments: List[CommunityAssignment]
    community_sizes: Dict[int, int]
    message: Optional[str] = None


# ============== Similarity Search ==============

class SimilarNodeRequest(BaseModel):
    """Request similar nodes."""
    
    query_node: str = Field(
        description="Node ID to find similar nodes for"
    )
    k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of similar nodes to return"
    )
    metric: SimilarityMetric = Field(
        default=SimilarityMetric.COSINE,
        description="Similarity metric"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Model to use"
    )


class SimilarNode(BaseModel):
    """Similar node result."""
    
    node_id: str
    similarity: float
    community: Optional[int] = None


class SimilarNodeResponse(BaseModel):
    """Response with similar nodes."""
    
    success: bool
    query_node: str
    similar_nodes: List[SimilarNode]
    query_community: Optional[int] = None
    message: Optional[str] = None


# ============== Visualization ==============

class VisualizationRequest(BaseModel):
    """Request embedding visualization data."""
    
    method: ReductionMethod = Field(
        default=ReductionMethod.UMAP,
        description="Dimensionality reduction method"
    )
    n_components: int = Field(
        default=2,
        ge=2,
        le=3,
        description="Output dimensions (2D or 3D)"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Model to use"
    )
    
    # UMAP-specific parameters
    umap_n_neighbors: int = Field(
        default=15,
        ge=2,
        le=200,
        description="UMAP n_neighbors parameter"
    )
    umap_min_dist: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="UMAP min_dist parameter"
    )
    
    # t-SNE specific parameters
    tsne_perplexity: float = Field(
        default=30.0,
        ge=5.0,
        le=100.0,
        description="t-SNE perplexity parameter"
    )


class VisualizationNode(BaseModel):
    """Node with visualization coordinates."""
    
    id: str
    x: float
    y: float
    z: Optional[float] = None
    community: int
    confidence: float


class VisualizationResponse(BaseModel):
    """Response with visualization data."""
    
    success: bool
    method: str
    dimensions: int
    num_nodes: int
    nodes: List[VisualizationNode]
    bounds: Optional[Dict[str, Dict[str, float]]] = None  # x/y/z min/max
    message: Optional[str] = None


# ============== Model Management ==============

class ModelInfo(BaseModel):
    """Information about a saved model."""
    
    name: str
    graph_name: Optional[str] = None
    num_clusters: Optional[int] = None
    hidden_dim: Optional[int] = None
    created_at: Optional[str] = None
    num_parameters: Optional[int] = None


class ListModelsResponse(BaseModel):
    """Response with list of models."""
    
    success: bool
    models: List[ModelInfo]
    current_model: Optional[str] = None


class LoadModelRequest(BaseModel):
    """Request to load a model."""
    
    model_name: str = Field(description="Name of model to load")


class LoadModelResponse(BaseModel):
    """Response from loading model."""
    
    success: bool
    model_name: Optional[str] = None
    message: Optional[str] = None


# ============== Deep Learning Info ==============

class DeepLearningInfo(BaseModel):
    """Information about deep learning availability."""
    
    available: bool
    torch_available: bool
    torch_version: Optional[str] = None
    cuda_available: bool
    cuda_device_count: int = 0
    pyg_available: bool
    pyg_version: Optional[str] = None
    umap_available: bool
    features: Dict[str, bool]


class GetInfoResponse(BaseModel):
    """Response with service information."""
    
    deep_learning: DeepLearningInfo
    has_model: bool
    cached_embeddings: List[str]
    model_dir: str