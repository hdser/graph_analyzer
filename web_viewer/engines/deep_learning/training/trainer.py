"""
GIT-CD Training Pipeline

Complete training loop with:
- Multi-objective optimization
- KMeans warmup and periodic updates
- Early stopping
- Learning rate scheduling
- Gradient clipping
- Checkpointing
- Logging and metrics

Note: MPS (Apple Silicon) requires float32 - this module automatically
converts tensors when MPS is detected.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
    LambdaLR,
)

from ..config import TrainingConfig
from ..models.gitcd import GITCD
from ..models.clustering_head import compute_clustering_metrics

logger = logging.getLogger(__name__)


def to_device_safe(tensor: Tensor, device: torch.device) -> Tensor:
    """
    Move tensor to device with MPS-safe dtype conversion.
    
    MPS (Apple Silicon) doesn't support float64, so we convert to float32.
    
    Args:
        tensor: Input tensor
        device: Target device
        
    Returns:
        Tensor on target device with compatible dtype
    """
    # Check if MPS and tensor is float64
    if device.type == 'mps' and tensor.dtype == torch.float64:
        tensor = tensor.to(torch.float32)
    
    return tensor.to(device)


@dataclass
class TrainingHistory:
    """Training history container."""
    
    epoch: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    
    # Individual losses
    loss_kl: List[float] = field(default_factory=list)
    loss_silhouette: List[float] = field(default_factory=list)
    loss_classification: List[float] = field(default_factory=list)
    
    # Metrics
    silhouette_score: List[float] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)
    
    # Timing
    epoch_time: List[float] = field(default_factory=list)
    
    # Best results
    best_epoch: int = 0
    best_loss: float = float('inf')
    metrics: List[Dict[str, float]] = field(default_factory=list)
    
    def add(self, **kwargs):
        """Add values to history."""
        for key, value in kwargs.items():
            if hasattr(self, key) and isinstance(getattr(self, key), list):
                getattr(self, key).append(value)
    
    def to_dict(self) -> Dict[str, List]:
        """Convert to dictionary."""
        return {
            'epoch': self.epoch,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'loss_kl': self.loss_kl,
            'loss_silhouette': self.loss_silhouette,
            'loss_classification': self.loss_classification,
            'silhouette_score': self.silhouette_score,
            'learning_rate': self.learning_rate,
            'epoch_time': self.epoch_time,
        }
    
    def save(self, path: Union[str, Path]):
        """Save history to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class TrainingCallback:
    """Base class for training callbacks."""
    
    def on_epoch_start(self, epoch: int, trainer: 'GITCDTrainer'):
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: 'GITCDTrainer'):
        pass
    
    def on_train_start(self, trainer: 'GITCDTrainer'):
        pass
    
    def on_train_end(self, trainer: 'GITCDTrainer'):
        pass


class EarlyStoppingCallback(TrainingCallback):
    """Early stopping based on validation loss."""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        monitor: str = 'val_loss',
        mode: str = 'min',
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.should_stop = False
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: 'GITCDTrainer'):
        current = metrics.get(self.monitor, metrics.get('train_loss', 0))
        
        if self.mode == 'min':
            improved = current < self.best_value - self.min_delta
        else:
            improved = current > self.best_value + self.min_delta
        
        if improved:
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered at epoch {epoch}")


class CheckpointCallback(TrainingCallback):
    """Save model checkpoints."""
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        save_best_only: bool = True,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_interval: Optional[int] = None,
    ):
        self.save_dir = Path(save_dir)
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.save_interval = save_interval
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: 'GITCDTrainer'):
        current = metrics.get(self.monitor, metrics.get('train_loss', 0))
        
        # Check if improved
        if self.mode == 'min':
            improved = current < self.best_value
        else:
            improved = current > self.best_value
        
        # Save best model
        if improved:
            self.best_value = current
            if self.save_best_only:
                path = self.save_dir / 'best_model.pt'
                trainer.model.save(path)
                logger.info(f"Saved best model (epoch {epoch}, {self.monitor}={current:.4f})")
        
        # Save periodic checkpoint
        if self.save_interval and epoch % self.save_interval == 0:
            path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
            trainer.model.save(path)


class GITCDTrainer:
    """
    Training pipeline for GIT-CD model.
    
    Implements the training procedure from the paper:
    - KMeans warmup (10 epochs)
    - Periodic cluster center updates
    - Multi-objective loss optimization
    - Early stopping
    
    Args:
        model: GITCD model to train
        config: TrainingConfig with hyperparameters
        callbacks: Optional list of callbacks
    """
    
    def __init__(
        self,
        model: GITCD,
        config: Optional[TrainingConfig] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
    ):
        self.model = model
        self.config = config or TrainingConfig()
        self.callbacks = callbacks or []
        
        # Setup device
        self.device = self._get_device()
        self._is_mps = self.device.type == 'mps'
        
        if self._is_mps:
            logger.info("MPS detected - using float32 for all operations")
        
        # Move model to device (with float32 conversion for MPS)
        if self._is_mps:
            self.model = self.model.float()  # Ensure float32
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.history = TrainingHistory()
        self.best_loss = float('inf')
        
        # Early stopping (add default if enabled)
        if self.config.early_stopping:
            has_early_stop = any(isinstance(c, EarlyStoppingCallback) for c in self.callbacks)
            if not has_early_stop:
                self.callbacks.append(EarlyStoppingCallback(
                    patience=self.config.patience,
                    min_delta=self.config.min_delta,
                    monitor=self.config.monitor_metric,
                ))
        
        logger.info(f"Initialized trainer on device: {self.device}")
    
    def _get_device(self) -> torch.device:
        """Get computation device."""
        device_str = self.config.get_device()
        return torch.device(device_str)
    
    def _to_device(self, tensor: Tensor) -> Tensor:
        """Move tensor to device with MPS-safe conversion."""
        if self._is_mps and tensor.dtype == torch.float64:
            tensor = tensor.to(torch.float32)
        return tensor.to(self.device)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        return optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
    
    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler."""
        if self.config.lr_scheduler == 'none':
            return None
        
        if self.config.lr_scheduler == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_epochs,
                eta_min=self.config.lr_min,
            )
        
        if self.config.lr_scheduler == 'step':
            return StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )
        
        if self.config.lr_scheduler == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                min_lr=self.config.lr_min,
            )
        
        return None
    
    def _warmup_lr(self, epoch: int) -> float:
        """Compute learning rate warmup factor."""
        if epoch < self.config.lr_warmup_epochs:
            return (epoch + 1) / self.config.lr_warmup_epochs
        return 1.0
    
    def train(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        label_mask: Optional[Tensor] = None,
        val_mask: Optional[Tensor] = None,
    ) -> TrainingHistory:
        """
        Train the model.
        
        Args:
            x: Node features [N, D]
            edge_index: Edge indices [2, E]
            edge_weight: Optional edge weights [E]
            labels: Optional node labels [N]
            label_mask: Optional mask for labeled nodes
            val_mask: Optional validation mask
            
        Returns:
            TrainingHistory with metrics
        """
        # Ensure float32 for MPS compatibility
        if self._is_mps:
            x = x.to(torch.float32)
            if edge_weight is not None:
                edge_weight = edge_weight.to(torch.float32)
        
        # Move data to device
        x = self._to_device(x)
        edge_index = self._to_device(edge_index)
        if edge_weight is not None:
            edge_weight = self._to_device(edge_weight)
        if labels is not None:
            labels = self._to_device(labels)
        if label_mask is not None:
            label_mask = self._to_device(label_mask)
        if val_mask is not None:
            val_mask = self._to_device(val_mask)
        
        # Set seed for reproducibility
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)
        
        # Callbacks: training start
        for callback in self.callbacks:
            callback.on_train_start(self)
        
        logger.info(f"Starting training for {self.config.max_epochs} epochs")
        
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Callbacks: epoch start
            for callback in self.callbacks:
                callback.on_epoch_start(epoch, self)
            
            # Initialize cluster centers after warmup
            if epoch == self.config.kmeans_warmup_epochs:
                self._initialize_clusters(x, edge_index, edge_weight)
            
            # Update cluster centers periodically
            elif (epoch > self.config.kmeans_warmup_epochs and 
                  epoch % self.config.kmeans_update_interval == 0):
                self._update_clusters(x, edge_index, edge_weight)
            
            # Training step
            train_metrics = self._train_epoch(
                x, edge_index, edge_weight, labels, label_mask, epoch
            )
            
            # Validation step
            val_metrics = {}
            if val_mask is not None and val_mask.any():
                val_metrics = self._validate(
                    x, edge_index, edge_weight, labels, val_mask
                )
            
            # Compute clustering metrics
            cluster_metrics = {}
            if self.model.clustering_head.centers_initialized:
                cluster_metrics = self._compute_cluster_metrics(x, edge_index, edge_weight)
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics, **cluster_metrics}
            
            # Track best
            if metrics['train_loss'] < self.history.best_loss:
                self.history.best_loss = metrics['train_loss']
                self.history.best_epoch = epoch
            
            # Store metrics
            self.history.metrics.append(metrics)
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(metrics.get('val_loss', metrics['train_loss']))
                else:
                    self.scheduler.step()
            
            # Apply warmup
            if epoch < self.config.lr_warmup_epochs:
                warmup_factor = self._warmup_lr(epoch)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config.learning_rate * warmup_factor
            
            # Record history
            epoch_time = time.time() - epoch_start
            self.history.add(
                epoch=epoch,
                train_loss=metrics['train_loss'],
                val_loss=metrics.get('val_loss', 0),
                loss_kl=metrics.get('loss_kl', 0),
                loss_silhouette=metrics.get('loss_silhouette', 0),
                loss_classification=metrics.get('loss_classification', 0),
                silhouette_score=cluster_metrics.get('silhouette', 0),
                learning_rate=current_lr,
                epoch_time=epoch_time,
            )
            
            # Logging
            if self.config.verbose and epoch % self.config.log_interval == 0:
                self._log_epoch(epoch, metrics, epoch_time)
            
            # Callbacks: epoch end
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, metrics, self)
            
            # Check early stopping
            should_stop = any(
                getattr(c, 'should_stop', False)
                for c in self.callbacks
                if isinstance(c, EarlyStoppingCallback)
            )
            if should_stop:
                break
        
        # Callbacks: training end
        for callback in self.callbacks:
            callback.on_train_end(self)
        
        logger.info(f"Training completed after {self.current_epoch + 1} epochs")
        
        return self.history
    
    def _train_epoch(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor],
        labels: Optional[Tensor],
        label_mask: Optional[Tensor],
        epoch: int,
    ) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        
        # Forward pass
        self.optimizer.zero_grad()
        output = self.model(x, edge_index, edge_weight)
        
        # Compute losses
        loss_dict = self._compute_losses(output, labels, label_mask, epoch)
        total_loss = loss_dict['total']
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm
            )
        
        # Update weights
        self.optimizer.step()
        
        return {
            'train_loss': total_loss.item(),
            'loss_kl': loss_dict.get('kl', torch.tensor(0)).item(),
            'loss_silhouette': loss_dict.get('silhouette', torch.tensor(0)).item(),
            'loss_classification': loss_dict.get('classification', torch.tensor(0)).item(),
        }
    
    def _compute_losses(
        self,
        output: Dict[str, Tensor],
        labels: Optional[Tensor],
        label_mask: Optional[Tensor],
        epoch: int,
    ) -> Dict[str, Tensor]:
        """Compute training losses."""
        from ..models.clustering_head import kl_divergence_loss, soft_silhouette_loss
        
        losses = {}
        total = torch.tensor(0.0, device=self.device)
        
        # KL divergence loss (after warmup)
        if epoch >= self.config.kmeans_warmup_epochs:
            Q = output.get('soft_assignments')
            P = output.get('target_distribution')
            
            if Q is not None and P is not None:
                kl_loss = kl_divergence_loss(P, Q)
                losses['kl'] = kl_loss
                total = total + self.config.clustering_weight * kl_loss
            
            # Silhouette loss (optional)
            embeddings = output.get('embeddings')
            if embeddings is not None and Q is not None and self.config.silhouette_weight > 0:
                sil_loss = soft_silhouette_loss(embeddings, Q)
                losses['silhouette'] = sil_loss
                total = total + self.config.silhouette_weight * sil_loss
        
        # Classification loss (if labels provided)
        if labels is not None and label_mask is not None:
            logits = output.get('logits')
            if logits is not None:
                class_loss = nn.functional.cross_entropy(
                    logits[label_mask],
                    labels[label_mask]
                )
                losses['classification'] = class_loss
                total = total + self.config.classification_weight * class_loss
        
        # Ensure we have some loss
        if total == 0:
            embeddings = output.get('embeddings')
            if embeddings is not None:
                # Regularization loss to prevent collapse
                total = 0.01 * (embeddings.norm(dim=1).mean() - 1).abs()
        
        losses['total'] = total
        return losses
    
    def _validate(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor],
        labels: Optional[Tensor],
        val_mask: Tensor,
    ) -> Dict[str, float]:
        """Run validation step."""
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(x, edge_index, edge_weight)
            
            if labels is not None:
                logits = output.get('logits')
                if logits is not None:
                    val_loss = nn.functional.cross_entropy(
                        logits[val_mask],
                        labels[val_mask]
                    )
                    
                    # Accuracy
                    preds = logits[val_mask].argmax(dim=1)
                    acc = (preds == labels[val_mask]).float().mean()
                    
                    return {
                        'val_loss': val_loss.item(),
                        'val_accuracy': acc.item(),
                    }
        
        return {}
    
    def _initialize_clusters(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor],
    ) -> None:
        """Initialize cluster centers using KMeans."""
        logger.info("Initializing cluster centers...")
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(x, edge_index, edge_weight)
            embeddings = output['embeddings']
            
            self.model.clustering_head.initialize_centers(
                embeddings,
                method='kmeans',
                n_init=self.config.kmeans_n_init,
            )
        
        self.model.train()
    
    def _update_clusters(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor],
    ) -> None:
        """Update cluster centers."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(x, edge_index, edge_weight)
            embeddings = output['embeddings']
            
            # Re-initialize with current embeddings
            self.model.clustering_head.initialize_centers(
                embeddings,
                method='kmeans',
                n_init=self.config.kmeans_n_init,
            )
        
        self.model.train()
    
    def _compute_cluster_metrics(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor],
    ) -> Dict[str, float]:
        """Compute clustering quality metrics."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(x, edge_index, edge_weight)
            embeddings = output['embeddings']
            labels = output['hard_labels']
            
            metrics = compute_clustering_metrics(embeddings, labels)
        
        self.model.train()
        return metrics
    
    def _log_epoch(
        self,
        epoch: int,
        metrics: Dict[str, float],
        epoch_time: float,
    ) -> None:
        """Log epoch metrics."""
        parts = [f"Epoch {epoch:4d}"]
        
        for key, value in metrics.items():
            if isinstance(value, float):
                parts.append(f"{key}={value:.4f}")
        
        parts.append(f"time={epoch_time:.2f}s")
        
        logger.info(" | ".join(parts))