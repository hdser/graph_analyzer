"""
Embedding Service

Service layer for GIT-CD embeddings integration.
Handles:
- Model management (loading, training, inference)
- Feature extraction from graph metrics
- Community detection using learned embeddings
- Similarity search
- Caching and persistence
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import time
import json
import hashlib

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

# Check deep learning availability
try:
    from engines.deep_learning import (
        HAS_DEEP_LEARNING,
        get_deep_learning_info,
    )
    if HAS_DEEP_LEARNING:
        from engines.deep_learning import (
            GITCD,
            GITCDConfig,
            GITCDTrainer,
            TrainingConfig,
            TrainingCallback,
            FeatureBuilder,
            FeatureConfig,
            GraphConverter,
            EmbeddingPredictor,
            SimilaritySearch,
            EmbeddingVisualizer,
        )
        import torch
except ImportError:
    HAS_DEEP_LEARNING = False
    logger.warning("Deep learning module not available")


@dataclass
class EmbeddingServiceConfig:
    """Configuration for embedding service."""
    
    # Model settings
    model_dir: str = "cache/models"
    embedding_cache_dir: str = "cache/embeddings"
    
    # Default model parameters
    hidden_dim: int = 128
    num_clusters: int = 20
    num_gnn_layers: int = 1
    num_transformer_layers: int = 2
    dropout: float = 0.5
    
    # Training defaults
    max_epochs: int = 200
    learning_rate: float = 3e-4
    patience: int = 5
    
    # Feature extraction defaults
    metric_columns: List[str] = field(default_factory=lambda: [
        "in_degree",
        "out_degree",
        "pagerank",
        "betweenness_centrality",
        "clustering_coefficient",
        "eigenvector_centrality",
    ])
    normalization: str = "standard"
    
    # Inference
    batch_size: int = 10000
    device: Optional[str] = None  # None = auto-detect (CUDA if available, else CPU)
    
    # Visualization
    reduction_method: str = "umap"
    reduction_dim: int = 2


class EmbeddingService:
    """
    Service for managing node embeddings and GIT-CD models.
    
    Provides high-level API for:
    - Training GIT-CD models on graph data
    - Computing embeddings for nodes
    - Detecting communities using learned representations
    - Finding similar nodes
    - Visualizing embeddings
    """
    
    def __init__(self, config: Optional[EmbeddingServiceConfig] = None):
        """
        Initialize embedding service.
        
        Args:
            config: Service configuration
        """
        self.config = config or EmbeddingServiceConfig()
        
        # Check availability
        if not HAS_DEEP_LEARNING:
            logger.warning("Deep learning not available. Embedding features disabled.")
            self._available = False
            return
        
        self._available = True
        
        # Model cache
        self._models: Dict[str, GITCD] = {}
        self._predictors: Dict[str, EmbeddingPredictor] = {}
        
        # Feature builder
        self._feature_builder: Optional[FeatureBuilder] = None
        
        # Similarity searcher
        self._similarity_searcher: Optional[SimilaritySearch] = None
        
        # Visualizer
        self._visualizer: Optional[EmbeddingVisualizer] = None
        
        # Create directories
        Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.embedding_cache_dir).mkdir(parents=True, exist_ok=True)
    
    @property
    def available(self) -> bool:
        """Check if service is available."""
        return self._available
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get service status and capabilities.
        
        Returns:
            Dict with availability and feature info
        """
        if not self._available:
            return {
                "available": False,
                "reason": "Deep learning dependencies not installed",
                "features": {},
            }
        
        dl_info = get_deep_learning_info()
        
        return {
            "available": True,
            "features": dl_info["features"],
            "torch_version": dl_info["torch"]["version"],
            "cuda_available": dl_info["torch"]["cuda_available"],
            "loaded_models": list(self._models.keys()),
        }
    
    def train_model(
        self,
        G: nx.Graph,
        metrics_df: Any,  # pd.DataFrame
        model_id: str = "default",
        num_clusters: Optional[int] = None,
        num_epochs: Optional[int] = None,
        properties_df: Any = None,  # pd.DataFrame
        labels: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train a GIT-CD model on graph data.
        
        Args:
            G: NetworkX graph
            metrics_df: DataFrame with computed node metrics
            model_id: Identifier for this model
            num_clusters: Number of communities to detect
            num_epochs: Training epochs
            properties_df: Optional node properties DataFrame
            labels: Optional node labels for semi-supervised learning
            **kwargs: Additional model/training config overrides
            
        Returns:
            Dict with training results and metrics
        """
        if not self._available:
            return {"error": "Deep learning not available"}
        
        # Extract epoch_callback from kwargs
        epoch_callback = kwargs.pop('epoch_callback', None)
        
        start_time = time.time()
        
        # Build features
        feature_config = FeatureConfig(
            metric_columns=kwargs.get('metric_columns', self.config.metric_columns),
            normalization=kwargs.get('normalization', self.config.normalization),
        )
        
        self._feature_builder = FeatureBuilder(feature_config)
        
        X, nodes, node_to_idx, feature_stats = self._feature_builder.build_features(
            G, metrics_df, properties_df
        )
        
        # Build edge index
        edge_index = self._feature_builder.build_edge_index(G, node_to_idx)
        
        logger.info(f"Built features: {X.shape}, edges: {edge_index.shape[1]}")
        
        # Create model config
        model_config = GITCDConfig(
            input_dim=X.shape[1],
            hidden_dim=kwargs.get('hidden_dim', self.config.hidden_dim),
            embedding_dim=kwargs.get('hidden_dim', self.config.hidden_dim),
            num_clusters=num_clusters or self.config.num_clusters,
            num_gnn_layers=kwargs.get('num_gnn_layers', self.config.num_gnn_layers),
            num_transformer_layers=kwargs.get('num_transformer_layers', self.config.num_transformer_layers),
            dropout=kwargs.get('dropout', self.config.dropout),
            num_classes=len(np.unique(labels)) if labels is not None else None,
        )
        
        # Create model
        model = GITCD(model_config)
        logger.info(f"Created model: {model}")
        
        # Create training config
        training_config = TrainingConfig(
            max_epochs=num_epochs or self.config.max_epochs,
            learning_rate=kwargs.get('learning_rate', self.config.learning_rate),
            patience=kwargs.get('patience', self.config.patience),
            device=self.config.device,
        )
        
        # Train - create callback wrapper if provided
        callbacks = []
        if epoch_callback:
            class EpochProgressCallback(TrainingCallback):
                def on_epoch_end(self, epoch: int, metrics: dict, trainer):
                    epoch_callback(epoch, metrics)
            callbacks.append(EpochProgressCallback())
        
        trainer = GITCDTrainer(model, training_config, callbacks=callbacks)
        
        x_tensor = torch.tensor(X, dtype=torch.float32)
        edge_tensor = torch.tensor(edge_index, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        
        history = trainer.train(x_tensor, edge_tensor, labels=labels_tensor)
        
        # Save model
        model_path = Path(self.config.model_dir) / f"{model_id}.pt"
        model.save(model_path)
        
        # Cache model
        self._models[model_id] = model
        self._predictors[model_id] = EmbeddingPredictor(model)
        
        training_time = time.time() - start_time
        
        return {
            "model_id": model_id,
            "model_path": str(model_path),
            "training_time": training_time,
            "num_nodes": len(nodes),
            "num_edges": edge_index.shape[1],
            "input_dim": X.shape[1],
            "num_clusters": model_config.num_clusters,
            "best_epoch": history.best_epoch,
            "best_loss": history.best_loss,
            "final_metrics": history.metrics[-1] if history.metrics else {},
            "feature_stats": {
                "num_features": feature_stats.num_features,
                "feature_names": feature_stats.feature_names,
            },
        }
    
    def compute_embeddings(
        self,
        G: nx.Graph,
        metrics_df: Any,  # pd.DataFrame
        model_id: str = "default",
        properties_df: Any = None,
    ) -> Dict[str, Any]:
        """
        Compute embeddings for graph nodes.
        
        Args:
            G: NetworkX graph
            metrics_df: DataFrame with node metrics
            model_id: Model to use
            properties_df: Optional node properties
            
        Returns:
            Dict with embeddings and communities
        """
        if not self._available:
            return {"error": "Deep learning not available"}
        
        # Load model if needed
        model = self._get_model(model_id)
        if model is None:
            return {"error": f"Model {model_id} not found"}
        
        # Get or create predictor
        predictor = self._get_predictor(model_id)
        
        # Build features using same config as training
        if self._feature_builder is None:
            feature_config = FeatureConfig(
                metric_columns=self.config.metric_columns,
                normalization=self.config.normalization,
            )
            self._feature_builder = FeatureBuilder(feature_config)
        
        X, nodes, node_to_idx, _ = self._feature_builder.build_features(
            G, metrics_df, properties_df, fit=False
        )
        edge_index = self._feature_builder.build_edge_index(G, node_to_idx)
        
        # Predict
        result = predictor.predict(X, edge_index, node_ids=nodes)
        
        # Build output with node mapping - result is an EmbeddingResult object
        return {
            "model_id": model_id,
            "num_nodes": len(nodes),
            "embedding_dim": result.embeddings.shape[1],
            "nodes": result.node_ids,
            "embeddings": result.embeddings.tolist(),
            "communities": result.communities.tolist(),
            "confidences": result.confidences.tolist() if result.confidences is not None else [],
            "num_communities": len(np.unique(result.communities)),
        }
    
    def get_communities(
        self,
        G: nx.Graph,
        metrics_df: Any,
        model_id: str = "default",
    ) -> Dict[str, int]:
        """
        Get community assignments for nodes.
        
        Args:
            G: NetworkX graph
            metrics_df: Node metrics
            model_id: Model to use
            
        Returns:
            Dict mapping node ID to community ID
        """
        result = self.compute_embeddings(G, metrics_df, model_id)
        
        if "error" in result:
            return {}
        
        return dict(zip(result['nodes'], result['communities']))
    
    def find_similar_nodes(
        self,
        node_id: str,
        G: nx.Graph,
        metrics_df: Any,
        model_id: str = "default",
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Find nodes similar to a given node.
        
        Args:
            node_id: Query node
            G: NetworkX graph
            metrics_df: Node metrics
            model_id: Model to use
            top_k: Number of results
            
        Returns:
            List of (node_id, similarity) tuples
        """
        if not self._available:
            return []
        
        # Compute embeddings
        result = self.compute_embeddings(G, metrics_df, model_id)
        
        if "error" in result:
            return []
        
        # Create similarity searcher with data
        embeddings = np.array(result['embeddings'])
        nodes = result['nodes']
        
        searcher = SimilaritySearch(embeddings, nodes, metric='cosine')
        
        return searcher.search(node_id, k=top_k)
    
    def get_embedding_visualization(
        self,
        G: nx.Graph,
        metrics_df: Any,
        model_id: str = "default",
        method: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get 2D visualization coordinates for embeddings.
        
        Args:
            G: NetworkX graph
            metrics_df: Node metrics
            model_id: Model to use
            method: Reduction method (umap, tsne, pca)
            
        Returns:
            List of points with x, y, community, node_id
        """
        if not self._available:
            return []
        
        # Compute embeddings
        result = self.compute_embeddings(G, metrics_df, model_id)
        
        if "error" in result:
            return []
        
        # Create visualizer
        method = method or self.config.reduction_method
        
        if self._visualizer is None or self._visualizer.method != method:
            self._visualizer = EmbeddingVisualizer(
                method=method,
                n_components=self.config.reduction_dim,
            )
        
        embeddings = np.array(result['embeddings'])
        communities = np.array(result['communities'])
        nodes = result['nodes']
        
        return self._visualizer.to_visualization_data(
            embeddings, nodes, communities
        )
    
    def save_embeddings(
        self,
        result: Dict[str, Any],
        path: Union[str, Path],
        format: str = "parquet",
    ):
        """Save computed embeddings to file."""
        if not self._available:
            return
        
        predictor = EmbeddingPredictor(self._models.get("default"), None)
        embeddings = np.array(result['embeddings'])
        nodes = result['nodes']
        
        predictor.save_embeddings(embeddings, nodes, path, format)
    
    def load_model(self, model_id: str, path: Optional[Union[str, Path]] = None) -> bool:
        """
        Load a saved model.
        
        Args:
            model_id: Model identifier
            path: Model path (defaults to model_dir/model_id.pt)
            
        Returns:
            True if loaded successfully
        """
        if not self._available:
            return False
        
        if path is None:
            path = Path(self.config.model_dir) / f"{model_id}.pt"
        
        path = Path(path)
        if not path.exists():
            logger.error(f"Model file not found: {path}")
            return False
        
        try:
            # Resolve device - CUDA if available, else CPU
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            model = GITCD.load(path, device=device)
            self._models[model_id] = model
            self._predictors[model_id] = EmbeddingPredictor(model)
            logger.info(f"Loaded model {model_id} from {path} on {device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        models = []
        
        # In-memory models
        for model_id, model in self._models.items():
            models.append({
                "model_id": model_id,
                "loaded": True,
                "num_clusters": model.config.num_clusters,
                "hidden_dim": model.config.hidden_dim,
                "parameters": model.num_parameters(),
            })
        
        # Saved models not yet loaded
        model_dir = Path(self.config.model_dir)
        if model_dir.exists():
            for path in model_dir.glob("*.pt"):
                model_id = path.stem
                if model_id not in self._models:
                    models.append({
                        "model_id": model_id,
                        "loaded": False,
                        "path": str(path),
                    })
        
        return models
    
    def _get_model(self, model_id: str) -> Optional[Any]:
        """Get model, loading if necessary."""
        if model_id in self._models:
            return self._models[model_id]
        
        # Try to load
        if self.load_model(model_id):
            return self._models.get(model_id)
        
        return None
    
    def _get_predictor(self, model_id: str) -> Optional[EmbeddingPredictor]:
        """Get predictor for model."""
        if model_id in self._predictors:
            return self._predictors[model_id]
        
        model = self._get_model(model_id)
        if model is None:
            return None
        
        predictor = EmbeddingPredictor(model)
        self._predictors[model_id] = predictor
        return predictor


# Global service instance
_embedding_service: Optional[EmbeddingService] = None


def is_deep_learning_available() -> bool:
    """
    Check if deep learning dependencies are available.
    
    Returns:
        True if PyTorch and PyTorch Geometric are installed
    """
    return HAS_DEEP_LEARNING


def get_embedding_service(config: Optional[EmbeddingServiceConfig] = None) -> EmbeddingService:
    """
    Get or create the global embedding service instance.
    
    Args:
        config: Optional configuration (used on first call)
        
    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = EmbeddingService(config)
    
    return _embedding_service