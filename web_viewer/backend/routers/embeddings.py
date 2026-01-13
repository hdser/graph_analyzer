"""
Embedding API Router

FastAPI endpoints for GIT-CD embeddings with background training.
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
import threading
import uuid

from ..models.embedding_schemas import (
    DeviceType,
    ReductionMethod,
    SimilarityMetric,
    TrainEmbeddingRequest,
    TrainEmbeddingResponse,
    ComputeEmbeddingsRequest,
    ComputeEmbeddingsResponse,
    EmbeddingNode,
    GetCommunitiesRequest,
    GetCommunitiesResponse,
    CommunityAssignment,
    SimilarNodeRequest,
    SimilarNodeResponse,
    SimilarNode,
    VisualizationRequest,
    VisualizationResponse,
    VisualizationNode,
    ModelInfo,
    ListModelsResponse,
    LoadModelRequest,
    LoadModelResponse,
    DeepLearningInfo,
    GetInfoResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/embeddings", tags=["embeddings"])


# ============== Background Task Manager ==============

@dataclass
class TrainingTask:
    """Represents a background training task."""
    task_id: str
    model_name: str
    status: str = "pending"
    progress: float = 0.0
    current_epoch: int = 0
    max_epochs: int = 0
    message: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    graph_name: Optional[str] = None


class TrainingTaskManager:
    """Manages background training tasks."""
    
    def __init__(self, max_workers: int = 2):
        self._tasks: Dict[str, TrainingTask] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
    
    def create_task(self, model_name: str, max_epochs: int) -> str:
        task_id = str(uuid.uuid4())[:8]
        task = TrainingTask(
            task_id=task_id,
            model_name=model_name,
            max_epochs=max_epochs,
            started_at=datetime.now(),
        )
        with self._lock:
            self._tasks[task_id] = task
        return task_id
    
    def get_task(self, task_id: str) -> Optional[TrainingTask]:
        return self._tasks.get(task_id)
    
    def update_task(self, task_id: str, **kwargs):
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                for key, value in kwargs.items():
                    if hasattr(task, key):
                        setattr(task, key, value)
    
    def submit(self, task_id: str, fn, *args, **kwargs):
        self.update_task(task_id, status="running")
        future = self._executor.submit(self._run_task, task_id, fn, *args, **kwargs)
        return future
    
    def _run_task(self, task_id: str, fn, *args, **kwargs):
        try:
            result = fn(*args, **kwargs)
            self.update_task(
                task_id,
                status="completed",
                completed_at=datetime.now(),
                result=result,
                progress=100.0,
                message="Training completed successfully"
            )
            return result
        except Exception as e:
            logger.error(f"Training task {task_id} failed: {e}")
            self.update_task(
                task_id,
                status="failed",
                completed_at=datetime.now(),
                error=str(e),
                message=f"Training failed: {e}"
            )
            raise
    
    def list_tasks(self, limit: int = 10) -> List[TrainingTask]:
        tasks = list(self._tasks.values())
        tasks.sort(key=lambda t: t.started_at or datetime.min, reverse=True)
        return tasks[:limit]


_task_manager = TrainingTaskManager()


# ============== Dependencies ==============

def get_embedding_service():
    """Get embedding service instance."""
    try:
        from ..services.embedding_service import get_embedding_service as _get_service, is_deep_learning_available
        
        if not is_deep_learning_available():
            raise HTTPException(
                status_code=503,
                detail="Deep learning not available. Install PyTorch and PyTorch Geometric."
            )
        
        return _get_service()
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Deep learning dependencies not available: {e}"
        )


def get_network_service():
    """Get network service for graph access."""
    from ..services.network_service import network_service
    return network_service


def get_graph_and_metrics(network_service):
    """Get current graph and metrics from network service."""
    if not network_service.graphs:
        raise HTTPException(
            status_code=400,
            detail="No graph loaded. Please load a network first."
        )
    
    graph_name = list(network_service.graphs.keys())[0]
    G = network_service.graphs[graph_name]
    
    metrics_df = None
    if hasattr(network_service, 'node_metrics') and network_service.node_metrics is not None:
        metrics_df = network_service.node_metrics
    
    return G, metrics_df, graph_name


# ============== Info Endpoint ==============

@router.get("/info", response_model=GetInfoResponse)
async def get_info():
    """Get deep learning availability and service status."""
    try:
        from engines.deep_learning import (
            HAS_DEEP_LEARNING,
            HAS_TORCH,
            HAS_PYG,
            HAS_UMAP,
            get_deep_learning_info,
        )
        
        if HAS_DEEP_LEARNING:
            info = get_deep_learning_info()
            
            dl_info = DeepLearningInfo(
                available=True,
                torch_available=info['torch']['available'],
                torch_version=info['torch'].get('version'),
                cuda_available=info['torch'].get('cuda_available', False),
                cuda_device_count=info['torch'].get('cuda_device_count', 0),
                pyg_available=info['torch_geometric']['available'],
                pyg_version=info['torch_geometric'].get('version'),
                umap_available=info['umap']['available'],
                features={
                    'training': True,
                    'inference': True,
                    'similarity_search': True,
                    'visualization': info['umap']['available'],
                }
            )
            
            try:
                from ..services.embedding_service import get_embedding_service
                service = get_embedding_service()
                has_model = len(service._models) > 0
                model_dir = str(service.config.model_dir)
            except:
                has_model = False
                model_dir = ""
            
            return GetInfoResponse(
                deep_learning=dl_info,
                has_model=has_model,
                cached_embeddings=[],
                model_dir=model_dir
            )
        else:
            return GetInfoResponse(
                deep_learning=DeepLearningInfo(
                    available=False,
                    torch_available=HAS_TORCH,
                    torch_version=None,
                    cuda_available=False,
                    cuda_device_count=0,
                    pyg_available=HAS_PYG,
                    pyg_version=None,
                    umap_available=HAS_UMAP,
                    features={}
                ),
                has_model=False,
                cached_embeddings=[],
                model_dir=""
            )
            
    except ImportError:
        return GetInfoResponse(
            deep_learning=DeepLearningInfo(
                available=False,
                torch_available=False,
                torch_version=None,
                cuda_available=False,
                cuda_device_count=0,
                pyg_available=False,
                pyg_version=None,
                umap_available=False,
                features={}
            ),
            has_model=False,
            cached_embeddings=[],
            model_dir=""
        )


# ============== Training Endpoints ==============

def _run_training(request_dict: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """Run training in background thread."""
    import time
    
    from ..services.embedding_service import get_embedding_service
    from ..services.network_service import network_service
    
    service = get_embedding_service()
    
    if not network_service.graphs:
        raise ValueError("No graph loaded")
    
    # Debug: log available graphs
    available_graphs = list(network_service.graphs.keys())
    logger.info(f"[Task {task_id}] Available graphs: {available_graphs}")
    
    # Use specified graph or first available
    graph_name = request_dict.get('graph_name')
    logger.info(f"[Task {task_id}] Requested graph_name: {graph_name}")
    
    if graph_name and graph_name in network_service.graphs:
        G = network_service.graphs[graph_name]
        logger.info(f"[Task {task_id}] Using requested graph: {graph_name}")
    else:
        # Fall back to first graph
        graph_name = list(network_service.graphs.keys())[0]
        G = network_service.graphs[graph_name]
        logger.info(f"[Task {task_id}] Falling back to first graph: {graph_name}")
    
    # Generate model name from graph if not specified
    model_name = request_dict.get('model_name')
    if not model_name:
        model_name = f"gitcd_{graph_name}"
    
    metrics_df = None
    if hasattr(network_service, 'node_metrics') and network_service.node_metrics is not None:
        metrics_df = network_service.node_metrics
    
    logger.info(f"[Task {task_id}] Training on graph '{graph_name}': {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    logger.info(f"[Task {task_id}] Model will be saved as '{model_name}'")
    
    max_epochs = request_dict['max_epochs']
    
    # Progress callback for epoch updates
    def epoch_callback(epoch: int, metrics: dict):
        progress = ((epoch + 1) / max_epochs) * 100
        # Map trainer metric names to our expected names
        total_loss = metrics.get('train_loss', 0)
        cluster_loss = metrics.get('loss_kl', 0)  # KL divergence is the clustering loss
        silhouette_loss = metrics.get('loss_silhouette', 0)
        
        _task_manager.update_task(
            task_id,
            current_epoch=epoch + 1,
            max_epochs=max_epochs,
            progress=progress,
            message=f"Epoch {epoch + 1}/{max_epochs} - Loss: {total_loss:.4f}",
            metrics={
                'loss': total_loss,
                'cluster_loss': cluster_loss,
                'recon_loss': silhouette_loss,  # Using silhouette loss as "reconstruction"
            },
            model_name=model_name,
            graph_name=graph_name,
        )
        logger.debug(f"[Task {task_id}] Epoch {epoch + 1}: loss={total_loss:.4f}")
    
    _task_manager.update_task(
        task_id, 
        message=f"Training on {graph_name} ({G.number_of_nodes()} nodes)...",
        model_name=model_name,
        graph_name=graph_name,
        max_epochs=max_epochs,
        current_epoch=0,
    )
    
    start_time = time.time()
    
    result = service.train_model(
        G=G,
        metrics_df=metrics_df,
        model_id=model_name,
        num_clusters=request_dict['num_clusters'],
        num_epochs=max_epochs,
        hidden_dim=request_dict['hidden_dim'],
        num_gnn_layers=request_dict['num_gnn_layers'],
        num_transformer_layers=request_dict['num_transformer_layers'],
        dropout=request_dict['dropout'],
        learning_rate=request_dict['learning_rate'],
        weight_decay=request_dict['weight_decay'],
        patience=request_dict['patience'],
        epoch_callback=epoch_callback,
    )
    
    training_time = time.time() - start_time
    
    return {
        'model_name': model_name,
        'graph_name': graph_name,
        'num_nodes': result.get('num_nodes', G.number_of_nodes()),
        'num_features': result.get('input_dim', 0),
        'num_clusters': request_dict['num_clusters'],
        'final_loss': result.get('best_loss'),
        'epochs_trained': result.get('best_epoch', 0),
        'training_time': training_time,
        'silhouette_score': result.get('final_metrics', {}).get('silhouette'),
    }


@router.post("/train")
async def train_model(request: TrainEmbeddingRequest):
    """Start training a GIT-CD model in the background."""
    try:
        service = get_embedding_service()
        net_service = get_network_service()
        
        if not net_service.graphs:
            raise HTTPException(status_code=400, detail="No graph loaded. Please load a network first.")
        
        # Determine which graph to use
        graph_name = request.graph_name
        if graph_name and graph_name not in net_service.graphs:
            raise HTTPException(status_code=404, detail=f"Graph '{graph_name}' not found")
        
        if not graph_name:
            graph_name = list(net_service.graphs.keys())[0]
        
        # Generate model name from graph if not specified
        model_name = request.model_name
        if not model_name:
            model_name = f"gitcd_{graph_name}"
        
        # Build config dict for monitoring
        config = {
            'graph_name': graph_name,
            'model_name': model_name,
            'num_clusters': request.num_clusters,
            'hidden_dim': request.hidden_dim,
            'num_gnn_layers': request.num_gnn_layers,
            'num_transformer_layers': request.num_transformer_layers,
            'dropout': request.dropout,
            'max_epochs': request.max_epochs,
            'learning_rate': request.learning_rate,
            'patience': request.patience,
        }
        
        task_id = _task_manager.create_task(
            model_name=model_name,
            max_epochs=request.max_epochs,
        )
        
        # Store config in task
        task = _task_manager.get_task(task_id)
        if task:
            task.config = config
        
        request_dict = {
            'graph_name': graph_name,
            'model_name': model_name,
            'num_clusters': request.num_clusters,
            'hidden_dim': request.hidden_dim,
            'num_gnn_layers': request.num_gnn_layers,
            'num_transformer_layers': request.num_transformer_layers,
            'dropout': request.dropout,
            'max_epochs': request.max_epochs,
            'learning_rate': request.learning_rate,
            'weight_decay': request.weight_decay,
            'patience': request.patience,
        }
        
        _task_manager.submit(task_id, _run_training, request_dict, task_id)
        
        logger.info(f"Started training task {task_id} for model '{model_name}' on graph '{graph_name}'")
        
        return {
            "success": True,
            "task_id": task_id,
            "model_name": model_name,
            "graph_name": graph_name,
            "message": f"Training started on '{graph_name}'. Use GET /api/embeddings/train/status/{task_id} to check progress."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/train/status/{task_id}")
async def get_training_status(task_id: str):
    """Get status of a training task."""
    task = _task_manager.get_task(task_id)
    
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    
    response = {
        "task_id": task.task_id,
        "model_name": task.model_name,
        "graph_name": task.graph_name,
        "status": task.status,
        "progress": task.progress,
        "current_epoch": task.current_epoch,
        "max_epochs": task.max_epochs,
        "message": task.message,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
    }
    
    # Include training config if available
    if hasattr(task, 'config') and task.config:
        response["config"] = task.config
    
    # Include metrics if available
    if hasattr(task, 'metrics') and task.metrics:
        response["metrics"] = task.metrics
    
    if task.status == "completed" and task.result:
        response["result"] = task.result
    
    if task.status == "failed" and task.error:
        response["error"] = task.error
    
    return response


@router.get("/train/tasks")
async def list_training_tasks(limit: int = Query(default=10, ge=1, le=100)):
    """List recent training tasks."""
    tasks = _task_manager.list_tasks(limit)
    
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "model_name": t.model_name,
                "status": t.status,
                "progress": t.progress,
                "message": t.message,
                "started_at": t.started_at.isoformat() if t.started_at else None,
                "completed_at": t.completed_at.isoformat() if t.completed_at else None,
            }
            for t in tasks
        ]
    }


# ============== Communities Endpoint ==============

@router.get("/communities", response_model=GetCommunitiesResponse)
async def get_communities(include_confidence: bool = Query(default=True)):
    """Get community assignments for all nodes."""
    try:
        service = get_embedding_service()
        net_service = get_network_service()
        G, metrics_df, graph_name = get_graph_and_metrics(net_service)
        
        if len(service._models) == 0:
            raise HTTPException(status_code=400, detail="No model loaded. Train a model first.")
        
        # Get first loaded model
        model_id = list(service._models.keys())[0]
        
        loop = asyncio.get_event_loop()
        
        # Use compute_embeddings to get both communities and confidences
        result = await loop.run_in_executor(
            None,
            lambda: service.compute_embeddings(G, metrics_df, model_id=model_id)
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Build maps from result
        nodes = result.get('nodes', [])
        communities = result.get('communities', [])
        confidences = result.get('confidences', [])
        
        community_map = dict(zip(nodes, communities))
        confidence_map = dict(zip(nodes, confidences)) if confidences else {}
        
        assignments = []
        community_sizes = {}
        
        for node_id, community in community_map.items():
            confidence = confidence_map.get(node_id) if include_confidence else None
            
            assignments.append(CommunityAssignment(
                node_id=str(node_id),
                community=community,
                confidence=confidence
            ))
            
            community_sizes[community] = community_sizes.get(community, 0) + 1
        
        return GetCommunitiesResponse(
            success=True,
            num_nodes=len(assignments),
            num_communities=len(community_sizes),
            assignments=assignments,
            community_sizes=community_sizes,
            message=f"Found {len(community_sizes)} communities"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Community detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Similarity Endpoint ==============

@router.post("/similar", response_model=SimilarNodeResponse)
async def find_similar_nodes(request: SimilarNodeRequest):
    """Find nodes most similar to a query node."""
    try:
        service = get_embedding_service()
        net_service = get_network_service()
        G, metrics_df, graph_name = get_graph_and_metrics(net_service)
        
        if len(service._models) == 0:
            raise HTTPException(status_code=400, detail="No model loaded. Train a model first.")
        
        # Get first loaded model
        model_id = list(service._models.keys())[0]
        
        if request.query_node not in G.nodes():
            raise HTTPException(status_code=404, detail=f"Query node '{request.query_node}' not found")
        
        loop = asyncio.get_event_loop()
        similar = await loop.run_in_executor(
            None,
            lambda: service.find_similar_nodes(
                node_id=request.query_node,
                G=G,
                metrics_df=metrics_df,
                model_id=model_id,
                top_k=request.k
            )
        )
        
        # Get community assignments
        community_map = await loop.run_in_executor(
            None,
            lambda: service.get_communities(G, metrics_df, model_id=model_id)
        )
        query_community = community_map.get(request.query_node)
        
        similar_nodes = [
            SimilarNode(
                node_id=str(node_id),
                similarity=sim,
                community=community_map.get(node_id)
            )
            for node_id, sim in similar
        ]
        
        return SimilarNodeResponse(
            success=True,
            query_node=request.query_node,
            similar_nodes=similar_nodes,
            query_community=query_community,
            message=f"Found {len(similar_nodes)} similar nodes"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Model Management ==============

@router.get("/models", response_model=ListModelsResponse)
async def list_models():
    """List all saved models."""
    try:
        service = get_embedding_service()
        models = service.list_models()
        
        model_infos = [
            ModelInfo(
                name=m.get('model_id') or m.get('name', 'unknown'),
                num_clusters=m.get('num_clusters'),
                hidden_dim=m.get('hidden_dim'),
                created_at=m.get('created_at'),
                num_parameters=m.get('parameters') or m.get('num_parameters')
            )
            for m in models
        ]
        
        current = None
        # Get first loaded model as "current"
        loaded_models = list(service._models.keys())
        if loaded_models:
            current = loaded_models[0]
        
        return ListModelsResponse(success=True, models=model_infos, current_model=current)
        
    except Exception as e:
        logger.error(f"List models failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/load", response_model=LoadModelResponse)
async def load_model(request: LoadModelRequest):
    """Load a saved model."""
    try:
        from pathlib import Path
        import os
        
        service = get_embedding_service()
        
        # First check if model exists in list to get info
        models = service.list_models()
        
        model_info = None
        for m in models:
            model_id = m.get('model_id') or m.get('name')
            if model_id == request.model_name:
                model_info = m
                break
        
        if model_info is None:
            raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found")
        
        # If model is already loaded in memory, just return success
        if model_info.get('loaded', False):
            return LoadModelResponse(
                success=True,
                model_name=request.model_name,
                message=f"Model '{request.model_name}' already loaded"
            )
        
        # Get path - use stored path or construct from model_dir
        model_path = None
        if model_info.get('path'):
            model_path = Path(model_info['path'])
            
            # If relative, resolve from current working directory
            if not model_path.is_absolute():
                model_path = Path(os.getcwd()) / model_path
            
            model_path = model_path.resolve()
            
            if not model_path.exists():
                raise HTTPException(status_code=404, detail=f"Model file not found at {model_path}")
        
        logger.info(f"Loading model '{request.model_name}' from {model_path}")
        
        # Check service availability
        if not service._available:
            raise HTTPException(status_code=503, detail="Deep learning service not available")
        
        success = service.load_model(request.model_name, path=model_path)
        
        if success:
            return LoadModelResponse(
                success=True,
                model_name=request.model_name,
                message=f"Model '{request.model_name}' loaded"
            )
        else:
            raise HTTPException(status_code=500, detail=f"Failed to load model '{request.model_name}'")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Load model failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache")
async def clear_cache():
    """Clear embedding cache."""
    try:
        service = get_embedding_service()
        service.clear_cache()
        return {"success": True, "message": "Cache cleared"}
    except Exception as e:
        logger.error(f"Clear cache failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))