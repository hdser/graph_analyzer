# Deep Learning & Graph Embeddings

Graph Analyzer includes **GIT-CD (Graph-Informed Transformer for Community Detection)**, a deep learning module for computing node embeddings, detecting communities, and finding similar nodes using Graph Neural Networks and Transformers.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Architecture](#architecture)
4. [Training Models](#training-models)
5. [Computing Embeddings](#computing-embeddings)
6. [Community Detection](#community-detection)
7. [Similarity Search](#similarity-search)
8. [Visualization](#visualization)
9. [Training Monitor](#training-monitor)
10. [Model Management](#model-management)
11. [API Reference](#api-reference)
12. [Best Practices](#best-practices)

---

## Overview

GIT-CD combines Graph Neural Networks (GNNs) with Transformer attention mechanisms to learn meaningful node representations that capture both local graph structure and global patterns.

### Key Features

| Feature | Description |
|---------|-------------|
| **Node Embeddings** | Dense vector representations capturing node roles and relationships |
| **Community Detection** | Soft clustering based on learned embeddings |
| **Similarity Search** | Find similar nodes using embedding space |
| **Visualization** | 2D/3D embedding projections via UMAP/t-SNE |
| **Background Training** | Non-blocking training with real-time progress updates |
| **Model Management** | Save, load, and manage multiple trained models |

### When to Use Deep Learning

| Use Case | Traditional Metrics | Deep Learning |
|----------|---------------------|---------------|
| Node importance | ✅ PageRank, Centrality | ⚠️ Overkill |
| Community detection | ✅ Louvain, Label Prop | ✅ More nuanced |
| Similar node search | ⚠️ Limited | ✅ Excellent |
| Role identification | ⚠️ Complex combinations | ✅ Automatic |
| Large-scale patterns | ⚠️ Many metrics needed | ✅ Unified |
| Anomaly detection | ✅ Statistical | ✅ Embedding-based |

---

## Requirements

### Dependencies

```bash
# Required
pip install torch>=2.0.0
pip install torch-geometric>=2.3.0

# Optional (for visualization)
pip install umap-learn>=0.5.0

# Full installation
pip install torch torch-geometric umap-learn scikit-learn
```

### Hardware

| Hardware | Training | Inference |
|----------|----------|-----------|
| CPU | Slow (10K nodes: ~10min) | Fast |
| CUDA GPU | Fast (10K nodes: ~1min) | Very Fast |
| MPS (Apple Silicon) | Medium | Fast |

### Checking Availability

```bash
# API endpoint
curl http://localhost:8000/api/embeddings/info
```

**Response**:
```json
{
  "deep_learning": {
    "available": true,
    "torch_available": true,
    "torch_version": "2.1.0",
    "cuda_available": true,
    "cuda_device_count": 1,
    "pyg_available": true,
    "pyg_version": "2.4.0",
    "umap_available": true,
    "features": {
      "training": true,
      "inference": true,
      "similarity_search": true,
      "visualization": true
    }
  },
  "has_model": false,
  "model_dir": "cache/models"
}
```

---

## Architecture

### GIT-CD Model Structure

```
Input Features (metrics)
        │
        ▼
┌───────────────────┐
│   GNN Layers      │  ← Graph structure learning
│   (GCN/GAT)       │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Transformer     │  ← Global attention patterns
│   Encoder         │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Embedding       │  ← Node representations
│   Layer           │
└───────────────────┘
        │
        ├──────────────────┐
        ▼                  ▼
┌───────────────┐  ┌───────────────┐
│   Clustering  │  │ Classification│
│   Head (KL)   │  │ Head (optional)│
└───────────────┘  └───────────────┘
```

### Components

| Component | Description | Output |
|-----------|-------------|--------|
| **GNN Layers** | Learn from graph topology | Structure-aware features |
| **Transformer** | Capture global dependencies | Attention-weighted features |
| **Embedding Layer** | Final node representations | Dense vectors (128-dim default) |
| **Clustering Head** | Soft community assignment | Community probabilities |
| **Classification Head** | Optional supervised labels | Label predictions |

### Loss Functions

| Loss | Purpose | Weight |
|------|---------|--------|
| **KL Divergence** | Community clustering quality | Primary |
| **Silhouette** | Cluster separation | Secondary |
| **Classification** | Label prediction (if labels provided) | Optional |

---

## Training Models

### Via UI

1. **Open Deep Learning Panel**: Click the neural network icon (🧠) in the sidebar
2. **Configure Parameters**:
   - Clusters: Number of communities to detect (2-1000)
   - Hidden Dim: Embedding dimension (32-512)
   - Max Epochs: Training iterations (10-1000)
   - Learning Rate: Optimization step size (1e-6 to 0.1)
   - Patience: Early stopping epochs (1-50)
   - Dropout: Regularization (0-0.9)
   - GNN Layers: Graph convolution depth (1-5)
   - Transformer Layers: Attention depth (1-6)
3. **Click "Train Model"**
4. **Monitor Progress**: Open Training Monitor for real-time updates

### Via API

```bash
POST /api/embeddings/train
Content-Type: application/json

{
  "graph_name": "crc_v2_trusts",
  "model_name": "gitcd_trusts_v1",
  "num_clusters": 20,
  "hidden_dim": 128,
  "num_gnn_layers": 1,
  "num_transformer_layers": 2,
  "num_attention_heads": 8,
  "dropout": 0.5,
  "max_epochs": 200,
  "learning_rate": 0.0003,
  "patience": 5,
  "device": "auto"
}
```

**Response** (immediate):
```json
{
  "task_id": "abc12345",
  "status": "started",
  "message": "Training started in background"
}
```

### Training Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `graph_name` | string | null | - | Graph to train on (null = current) |
| `model_name` | string | auto | - | Model identifier |
| `num_clusters` | int | 20 | 2-1000 | Number of communities |
| `hidden_dim` | int | 128 | 32-512 | Embedding dimension |
| `num_gnn_layers` | int | 1 | 1-5 | GNN depth |
| `num_transformer_layers` | int | 2 | 1-6 | Transformer depth |
| `num_attention_heads` | int | 8 | 1-16 | Attention heads |
| `dropout` | float | 0.5 | 0-0.9 | Dropout rate |
| `max_epochs` | int | 200 | 10-1000 | Maximum epochs |
| `learning_rate` | float | 3e-4 | 1e-6 to 0.1 | Learning rate |
| `patience` | int | 5 | 1-50 | Early stopping patience |
| `weight_decay` | float | 5e-4 | 0 to 0.1 | L2 regularization |
| `device` | string | "auto" | auto, cpu, cuda | Compute device |

### Feature Selection

By default, the following metrics are used as input features:
- `in_degree`
- `out_degree`
- `pagerank`
- `betweenness_centrality`
- `clustering_coefficient`
- `eigenvector_centrality`

Custom features can be specified:
```json
{
  "metric_columns": [
    "in_degree", "out_degree", "pagerank", 
    "eigentrust", "community_id"
  ]
}
```

---

## Computing Embeddings

After training, compute embeddings for all nodes:

### Via API

```bash
POST /api/embeddings/compute
Content-Type: application/json

{
  "model_name": "gitcd_trusts_v1",
  "include_communities": true,
  "include_confidences": true
}
```

**Response**:
```json
{
  "success": true,
  "num_nodes": 13428,
  "embedding_dim": 128,
  "num_communities": 20,
  "communities_summary": {
    "0": 1245,
    "1": 892,
    "2": 756,
    ...
  }
}
```

### Embedding Structure

Each node gets:
- **embedding**: 128-dimensional dense vector
- **community**: Assigned community ID (0 to num_clusters-1)
- **confidence**: Cluster assignment confidence (0-1)

---

## Community Detection

GIT-CD provides soft clustering where each node has probability scores for each community.

### Get Community Assignments

```bash
POST /api/embeddings/communities
Content-Type: application/json

{
  "model_name": "gitcd_trusts_v1",
  "include_confidence": true
}
```

**Response**:
```json
{
  "success": true,
  "num_nodes": 13428,
  "num_communities": 20,
  "assignments": [
    {"node_id": "0x123...", "community": 5, "confidence": 0.92},
    {"node_id": "0x456...", "community": 12, "confidence": 0.78},
    ...
  ],
  "community_sizes": {
    "0": 1245,
    "1": 892,
    ...
  }
}
```

### Community Interpretation

| Confidence | Interpretation |
|------------|----------------|
| > 0.9 | Strong community membership |
| 0.7 - 0.9 | Clear membership |
| 0.5 - 0.7 | Moderate membership |
| < 0.5 | Boundary node (between communities) |

---

## Similarity Search

Find nodes with similar embeddings (similar roles/positions in the network).

### Via API

```bash
POST /api/embeddings/similar
Content-Type: application/json

{
  "query_node": "0x123456789...",
  "k": 10,
  "metric": "cosine",
  "model_name": "gitcd_trusts_v1"
}
```

**Response**:
```json
{
  "success": true,
  "query_node": "0x123456789...",
  "query_community": 5,
  "similar_nodes": [
    {"node_id": "0xabc...", "similarity": 0.98, "community": 5},
    {"node_id": "0xdef...", "similarity": 0.95, "community": 5},
    {"node_id": "0x789...", "similarity": 0.91, "community": 12},
    ...
  ]
}
```

### Similarity Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `cosine` | Cosine similarity (direction) | [-1, 1] |
| `euclidean` | Euclidean distance (magnitude) | [0, ∞) |
| `dot` | Dot product | (-∞, ∞) |

---

## Visualization

Project high-dimensional embeddings to 2D/3D for visualization.

### Via API

```bash
POST /api/embeddings/visualize
Content-Type: application/json

{
  "model_name": "gitcd_trusts_v1",
  "method": "umap",
  "n_components": 2,
  "umap_n_neighbors": 15,
  "umap_min_dist": 0.1
}
```

**Response**:
```json
{
  "success": true,
  "method": "umap",
  "dimensions": 2,
  "num_nodes": 13428,
  "nodes": [
    {"id": "0x123...", "x": -5.23, "y": 12.45, "community": 5, "confidence": 0.92},
    {"id": "0x456...", "x": 8.91, "y": -3.22, "community": 12, "confidence": 0.78},
    ...
  ],
  "bounds": {
    "x": {"min": -15.2, "max": 18.7},
    "y": {"min": -12.1, "max": 14.3}
  }
}
```

### Reduction Methods

| Method | Description | Speed | Quality |
|--------|-------------|-------|---------|
| `umap` | Uniform Manifold Approximation | Fast | Best |
| `tsne` | t-SNE | Slow | Good |
| `pca` | Principal Component Analysis | Very Fast | Basic |

### UMAP Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `umap_n_neighbors` | 15 | 2-200 | Local vs global structure |
| `umap_min_dist` | 0.1 | 0-1 | Cluster tightness |

### t-SNE Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `tsne_perplexity` | 30 | 5-100 | Balance local/global |

---

## Training Monitor

The Training Monitor provides real-time visualization of training progress.

### Opening the Monitor

1. Click **"Open Training Monitor"** in the Deep Learning panel
2. A popup window opens with live metrics

### Monitor Features

| Section | Information |
|---------|-------------|
| **Progress** | Current epoch, progress bar, percentage |
| **Metrics** | Total loss, clustering loss, reconstruction loss |
| **Chart** | Loss curves over time |
| **Configuration** | Model name, graph, clusters, hidden dim, learning rate |
| **Training Log** | Timestamped event log |
| **Results** | Final metrics on completion |

### Loss Metrics

| Metric | Description | Good Values |
|--------|-------------|-------------|
| **Total Loss** | Combined training loss | Decreasing |
| **Clustering Loss** | KL divergence for clusters | Decreasing |
| **Reconstruction Loss** | Silhouette-based quality | Low |

### Interpreting Training

| Pattern | Interpretation | Action |
|---------|----------------|--------|
| Loss decreasing steadily | Good training | Continue |
| Loss plateaus early | Underfitting | Increase capacity |
| Loss oscillates | High learning rate | Reduce LR |
| Loss increases | Overfitting | Add dropout, reduce epochs |

---

## Model Management

### List Models

```bash
GET /api/embeddings/models
```

**Response**:
```json
{
  "success": true,
  "models": [
    {
      "name": "gitcd_trusts_v1",
      "graph_name": "crc_v2_trusts",
      "num_clusters": 20,
      "hidden_dim": 128,
      "created_at": "2024-01-15T10:30:00Z",
      "num_parameters": 156432
    }
  ],
  "current_model": "gitcd_trusts_v1"
}
```

### Load Model

```bash
POST /api/embeddings/load
Content-Type: application/json

{
  "model_name": "gitcd_trusts_v1"
}
```

### Delete Model

```bash
DELETE /api/embeddings/models/{model_name}
```

### Model Storage

Models are saved to `cache/models/` directory:
- `{model_name}.pt` - PyTorch model weights
- `{model_name}_config.json` - Model configuration

---

## API Reference

### Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/embeddings/info` | GET | Get service status |
| `/api/embeddings/train` | POST | Start training |
| `/api/embeddings/train/status/{task_id}` | GET | Get training status |
| `/api/embeddings/compute` | POST | Compute embeddings |
| `/api/embeddings/communities` | POST | Get community assignments |
| `/api/embeddings/similar` | POST | Find similar nodes |
| `/api/embeddings/visualize` | POST | Get visualization data |
| `/api/embeddings/models` | GET | List saved models |
| `/api/embeddings/load` | POST | Load a model |
| `/api/embeddings/models/{name}` | DELETE | Delete a model |

### Training Status Response

```json
{
  "task_id": "abc12345",
  "model_name": "gitcd_trusts_v1",
  "graph_name": "crc_v2_trusts",
  "status": "running",
  "progress": 45.0,
  "current_epoch": 45,
  "max_epochs": 100,
  "message": "Epoch 45/100 - Loss: 0.0892",
  "metrics": {
    "loss": 0.0892,
    "cluster_loss": 0.0234,
    "recon_loss": 0.0012
  },
  "config": {
    "num_clusters": 20,
    "hidden_dim": 128,
    "learning_rate": 0.0003
  },
  "started_at": "2024-01-15T10:30:00Z"
}
```

### Status Values

| Status | Description |
|--------|-------------|
| `pending` | Task queued |
| `running` | Training in progress |
| `completed` | Training finished successfully |
| `failed` | Training failed with error |

---

## Best Practices

### 1. Data Preparation

- **Compute metrics first**: Ensure metrics are computed before training
- **Use relevant metrics**: Select metrics that capture node roles
- **Handle missing values**: Metrics should not have NaN values

### 2. Model Configuration

| Graph Size | Clusters | Hidden Dim | Epochs |
|------------|----------|------------|--------|
| < 1K | 5-10 | 64 | 100 |
| 1K-10K | 10-30 | 128 | 200 |
| 10K-100K | 20-50 | 128-256 | 200-500 |
| > 100K | 30-100 | 256 | 300-500 |

### 3. Training Tips

- **Start with defaults**: Default parameters work well for most cases
- **Use GPU**: Training is much faster on GPU
- **Monitor loss**: Watch for convergence
- **Early stopping**: Patience=5 is usually sufficient

### 4. Evaluation

- **Check cluster sizes**: Very uneven clusters may indicate issues
- **Validate communities**: Compare with known labels if available
- **Test similarity**: Verify similar nodes make sense

### 5. Production Use

- **Save models**: Always save trained models
- **Version models**: Use descriptive names
- **Document settings**: Record training parameters
- **Test inference**: Verify embeddings before deployment

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Deep learning not available" | Missing PyTorch/PyG | Install dependencies |
| "No graph loaded" | Graph not loaded | Load network first |
| Training very slow | Using CPU | Install CUDA, use GPU |
| NaN loss | Learning rate too high | Reduce learning rate |
| No convergence | Underfitting | Increase model capacity |
| All same community | Too few clusters | Increase num_clusters |

### Memory Issues

For large graphs (>100K nodes):
```json
{
  "batch_size": 5000,
  "max_samples": 50000
}
```

### GPU Issues

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU
{
  "device": "cpu"
}
```