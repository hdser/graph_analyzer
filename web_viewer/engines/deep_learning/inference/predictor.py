"""
Embedding Predictor for GIT-CD Inference

Provides inference utilities:
- Batch embedding computation
- Similarity search (KNN)
- Embedding visualization (UMAP/t-SNE/PCA)
- Caching and persistence
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from ..config import InferenceConfig
from ..models.gitcd import GITCD

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Container for embedding inference results."""
    
    embeddings: np.ndarray          # [N, D] embedding matrix
    node_ids: List[str]             # Node IDs in order
    communities: Optional[np.ndarray] = None     # [N] community labels
    confidences: Optional[np.ndarray] = None     # [N] confidence scores
    predictions: Optional[np.ndarray] = None     # [N] class predictions
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        df = pd.DataFrame(
            self.embeddings,
            index=self.node_ids,
            columns=[f'emb_{i}' for i in range(self.embeddings.shape[1])]
        )
        
        if self.communities is not None:
            df['community'] = self.communities
        if self.confidences is not None:
            df['confidence'] = self.confidences
        if self.predictions is not None:
            df['prediction'] = self.predictions
        
        return df
    
    def save(self, path: Union[str, Path], format: str = 'parquet'):
        """Save embeddings to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        df = self.to_dataframe()
        
        if format == 'parquet':
            df.to_parquet(path.with_suffix('.parquet'))
        elif format == 'csv':
            df.to_csv(path.with_suffix('.csv'))
        elif format == 'npy':
            np.save(path.with_suffix('.npy'), self.embeddings)
            # Save metadata
            with open(path.with_suffix('.json'), 'w') as f:
                json.dump({
                    'node_ids': self.node_ids,
                    'communities': self.communities.tolist() if self.communities is not None else None,
                }, f)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'EmbeddingResult':
        """Load embeddings from file."""
        path = Path(path)
        
        if path.suffix == '.parquet' or path.with_suffix('.parquet').exists():
            df = pd.read_parquet(path.with_suffix('.parquet'))
            emb_cols = [c for c in df.columns if c.startswith('emb_')]
            return cls(
                embeddings=df[emb_cols].values,
                node_ids=df.index.tolist(),
                communities=df['community'].values if 'community' in df else None,
                confidences=df['confidence'].values if 'confidence' in df else None,
            )
        elif path.suffix == '.npy' or path.with_suffix('.npy').exists():
            embeddings = np.load(path.with_suffix('.npy'))
            with open(path.with_suffix('.json')) as f:
                meta = json.load(f)
            return cls(
                embeddings=embeddings,
                node_ids=meta['node_ids'],
                communities=np.array(meta['communities']) if meta.get('communities') else None,
            )
        else:
            raise ValueError(f"Cannot load from {path}")


class EmbeddingPredictor:
    """
    Batch embedding predictor for GIT-CD.
    
    Handles inference with:
    - Automatic batching for large graphs
    - GPU/CPU device management
    - Caching of results
    
    Args:
        model: Trained GITCD model
        config: Inference configuration
    """
    
    def __init__(
        self,
        model: GITCD,
        config: Optional[InferenceConfig] = None,
    ):
        self.model = model
        self.config = config or InferenceConfig()
        
        self.device = torch.device(self.config.get_device())
        self.model.to(self.device)
        self.model.eval()
        
        # Cache
        self._cache: Dict[str, EmbeddingResult] = {}
    
    def predict(
        self,
        x: Union[Tensor, np.ndarray],
        edge_index: Union[Tensor, np.ndarray],
        node_ids: Optional[List[str]] = None,
        edge_weight: Optional[Union[Tensor, np.ndarray]] = None,
        cache_key: Optional[str] = None,
    ) -> EmbeddingResult:
        """
        Compute embeddings for all nodes.
        
        Args:
            x: Node features [N, D]
            edge_index: Edge indices [2, E]
            node_ids: Optional list of node IDs
            edge_weight: Optional edge weights
            cache_key: Optional cache key
            
        Returns:
            EmbeddingResult with embeddings and communities
        """
        # Check cache
        if cache_key and cache_key in self._cache:
            logger.info(f"Using cached embeddings for {cache_key}")
            return self._cache[cache_key]
        
        # Convert inputs
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if isinstance(edge_index, np.ndarray):
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        if edge_weight is not None and isinstance(edge_weight, np.ndarray):
            edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
        
        # Move to device
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)
        
        # Generate node IDs if not provided
        if node_ids is None:
            node_ids = [str(i) for i in range(x.size(0))]
        
        # Inference
        with torch.no_grad():
            output = self.model(x, edge_index, edge_weight)
        
        # Extract results
        embeddings = output['embeddings'].cpu().numpy()
        communities = output['hard_labels'].cpu().numpy()
        confidences = output['Q'].max(dim=1)[0].cpu().numpy()
        
        # Classifications if available
        predictions = None
        if 'logits' in output:
            predictions = output['logits'].argmax(dim=1).cpu().numpy()
        
        result = EmbeddingResult(
            embeddings=embeddings,
            node_ids=node_ids,
            communities=communities,
            confidences=confidences,
            predictions=predictions,
        )
        
        # Cache if requested
        if cache_key and self.config.cache_embeddings:
            self._cache[cache_key] = result
        
        return result
    
    def predict_batch(
        self,
        x: Union[Tensor, np.ndarray],
        edge_index: Union[Tensor, np.ndarray],
        node_ids: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
    ) -> EmbeddingResult:
        """
        Compute embeddings in batches (for very large graphs).
        
        Note: This is a simplified version. For true mini-batch inference
        on large graphs, use PyG's NeighborLoader.
        
        Args:
            x: Node features [N, D]
            edge_index: Edge indices [2, E]
            node_ids: Optional node IDs
            batch_size: Batch size (default from config)
            
        Returns:
            EmbeddingResult
        """
        batch_size = batch_size or self.config.batch_size
        
        # For small graphs, use regular prediction
        if x.shape[0] <= batch_size:
            return self.predict(x, edge_index, node_ids)
        
        # For larger graphs, we need to use the full graph anyway
        # because GNN aggregation needs neighbors
        logger.info(f"Graph has {x.shape[0]} nodes, using full inference")
        return self.predict(x, edge_index, node_ids)
    
    def get_similar_nodes(
        self,
        query_node: str,
        result: EmbeddingResult,
        k: int = 10,
        metric: str = 'cosine',
    ) -> List[Tuple[str, float]]:
        """
        Find nodes most similar to query.
        
        Args:
            query_node: Query node ID
            result: EmbeddingResult containing embeddings
            k: Number of similar nodes to return
            metric: Distance metric (cosine, euclidean, dot)
            
        Returns:
            List of (node_id, similarity_score) tuples
        """
        if query_node not in result.node_ids:
            raise ValueError(f"Node {query_node} not found")
        
        query_idx = result.node_ids.index(query_node)
        query_emb = result.embeddings[query_idx]
        
        # Compute similarities
        if metric == 'cosine':
            # Normalize embeddings
            norms = np.linalg.norm(result.embeddings, axis=1, keepdims=True) + 1e-8
            normalized = result.embeddings / norms
            query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
            similarities = np.dot(normalized, query_norm)
        elif metric == 'dot':
            similarities = np.dot(result.embeddings, query_emb)
        elif metric == 'euclidean':
            distances = np.linalg.norm(result.embeddings - query_emb, axis=1)
            similarities = -distances  # Negative for ranking
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Get top-k (excluding query itself)
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices:
            if idx != query_idx:
                results.append((result.node_ids[idx], float(similarities[idx])))
                if len(results) >= k:
                    break
        
        return results
    
    def clear_cache(self):
        """Clear embedding cache."""
        self._cache.clear()


class SimilaritySearch:
    """
    Efficient similarity search over embeddings.
    
    Uses approximate nearest neighbor search for large datasets.
    
    Args:
        embeddings: Embedding matrix [N, D]
        node_ids: Node IDs
        metric: Distance metric
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        node_ids: List[str],
        metric: str = 'cosine',
    ):
        self.embeddings = embeddings
        self.node_ids = node_ids
        self.metric = metric
        self.node_to_idx = {n: i for i, n in enumerate(node_ids)}
        
        # Normalize for cosine similarity
        if metric == 'cosine':
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
            self.normalized = embeddings / norms
        else:
            self.normalized = embeddings
    
    def search(
        self,
        query: Union[str, np.ndarray],
        k: int = 10,
        exclude_self: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Search for similar nodes.
        
        Args:
            query: Node ID or embedding vector
            k: Number of results
            exclude_self: Exclude query node from results
            
        Returns:
            List of (node_id, score) tuples
        """
        # Get query embedding
        if isinstance(query, str):
            if query not in self.node_to_idx:
                raise ValueError(f"Node {query} not found")
            query_idx = self.node_to_idx[query]
            query_emb = self.normalized[query_idx]
        else:
            query_emb = query
            if self.metric == 'cosine':
                query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
            query_idx = None
        
        # Compute scores
        if self.metric in ['cosine', 'dot']:
            scores = np.dot(self.normalized, query_emb)
        else:  # euclidean
            scores = -np.linalg.norm(self.embeddings - query_emb, axis=1)
        
        # Get top-k
        top_k = k + (1 if exclude_self and query_idx is not None else 0)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if exclude_self and idx == query_idx:
                continue
            results.append((self.node_ids[idx], float(scores[idx])))
            if len(results) >= k:
                break
        
        return results
    
    def batch_search(
        self,
        queries: List[str],
        k: int = 10,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Batch similarity search.
        
        Args:
            queries: List of query node IDs
            k: Number of results per query
            
        Returns:
            Dict mapping query to results
        """
        return {q: self.search(q, k) for q in queries}


class EmbeddingVisualizer:
    """
    Embedding visualization with dimensionality reduction.
    
    Supports UMAP, t-SNE, and PCA for 2D/3D projections.
    
    Args:
        config: Inference configuration
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
    
    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        method: Optional[str] = None,
        n_components: int = 2,
        **kwargs,
    ) -> np.ndarray:
        """
        Reduce embedding dimensions for visualization.
        
        Args:
            embeddings: High-dimensional embeddings [N, D]
            method: Reduction method (umap, tsne, pca)
            n_components: Target dimensions (2 or 3)
            **kwargs: Method-specific arguments
            
        Returns:
            Reduced embeddings [N, n_components]
        """
        method = method or self.config.reduction_method
        
        if method == 'umap':
            return self._umap_reduce(embeddings, n_components, **kwargs)
        elif method == 'tsne':
            return self._tsne_reduce(embeddings, n_components, **kwargs)
        elif method == 'pca':
            return self._pca_reduce(embeddings, n_components, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _umap_reduce(
        self,
        embeddings: np.ndarray,
        n_components: int,
        n_neighbors: Optional[int] = None,
        min_dist: Optional[float] = None,
        metric: str = 'cosine',
        **kwargs,
    ) -> np.ndarray:
        """UMAP dimensionality reduction."""
        try:
            import umap
        except ImportError:
            logger.warning("UMAP not available, falling back to PCA")
            return self._pca_reduce(embeddings, n_components)
        
        n_neighbors = n_neighbors or self.config.umap_n_neighbors
        min_dist = min_dist or self.config.umap_min_dist
        
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42,
            **kwargs,
        )
        
        return reducer.fit_transform(embeddings)
    
    def _tsne_reduce(
        self,
        embeddings: np.ndarray,
        n_components: int,
        perplexity: float = 30.0,
        **kwargs,
    ) -> np.ndarray:
        """t-SNE dimensionality reduction."""
        from sklearn.manifold import TSNE
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=min(perplexity, embeddings.shape[0] - 1),
            random_state=42,
            **kwargs,
        )
        
        return tsne.fit_transform(embeddings)
    
    def _pca_reduce(
        self,
        embeddings: np.ndarray,
        n_components: int,
        **kwargs,
    ) -> np.ndarray:
        """PCA dimensionality reduction."""
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=n_components, random_state=42)
        return pca.fit_transform(embeddings)
    
    def create_visualization_data(
        self,
        result: EmbeddingResult,
        method: Optional[str] = None,
        n_components: int = 2,
    ) -> Dict[str, Any]:
        """
        Create data for embedding visualization.
        
        Args:
            result: EmbeddingResult
            method: Reduction method
            n_components: Dimensions
            
        Returns:
            Dict with visualization data
        """
        coords = self.reduce_dimensions(
            result.embeddings,
            method=method,
            n_components=n_components,
        )
        
        data = {
            'nodes': [
                {
                    'id': result.node_ids[i],
                    'x': float(coords[i, 0]),
                    'y': float(coords[i, 1]),
                    'z': float(coords[i, 2]) if n_components == 3 else 0,
                    'community': int(result.communities[i]) if result.communities is not None else 0,
                    'confidence': float(result.confidences[i]) if result.confidences is not None else 1.0,
                }
                for i in range(len(result.node_ids))
            ],
            'method': method or self.config.reduction_method,
            'dimensions': n_components,
        }
        
        return data


def compute_embeddings(
    model: GITCD,
    x: Union[Tensor, np.ndarray],
    edge_index: Union[Tensor, np.ndarray],
    node_ids: Optional[List[str]] = None,
    device: str = 'auto',
) -> EmbeddingResult:
    """
    Convenience function to compute embeddings.
    
    Args:
        model: Trained GITCD model
        x: Node features
        edge_index: Edge indices
        node_ids: Optional node IDs
        device: Computation device
        
    Returns:
        EmbeddingResult
    """
    config = InferenceConfig(device=device)
    predictor = EmbeddingPredictor(model, config)
    return predictor.predict(x, edge_index, node_ids)