# API Reference

Graph Analyzer provides a REST API for programmatic access to all features. This document covers all available endpoints.

## Base URL

```
http://localhost:8000/api
```

## Response Format

All responses are JSON. Error responses follow this format:

```json
{
  "detail": "Error message describing the issue"
}
```

---

## Network Endpoints

### Get Configuration

Returns application configuration including available SQL files and features.

```
GET /api/config
```

**Response**:
```json
{
  "sql_files": [
    {"filename": "crc_v2_trusts.sql", "path": "/path/to/file"}
  ],
  "node_properties_files": [
    {"filename": "crc_v2_avatars.sql", "path": "/path/to/file"}
  ],
  "metric_modes": {
    "presets": {
      "basic": ["topology", "community"],
      "essential": ["topology", "centrality", "clustering", "community"]
    },
    "categories": {
      "topology": "Basic Topology (in/out degree, degree imbalance)"
    }
  },
  "cytoscape_desktop_available": true,
  "cached_layouts": [
    {"graph_id": "crc_v2_trusts", "node_count": 50000}
  ],
  "anomaly_available": true,
  "auto_reload_available": true,
  "hide_data_source_ui": false
}
```

---

### Load Network

Load network data from SQL files.

```
POST /api/load
Content-Type: application/json
```

**Request Body**:
```json
{
  "sql_files": ["crc_v2_trusts.sql", "crc_v2_flows.sql"],
  "node_properties_files": ["crc_v2_avatars.sql"],
  "use_cached_layout": true,
  "skip_sql": false,
  "metrics_mode": "essential"
}
```

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| sql_files | array | Yes | List of SQL file names |
| node_properties_files | array | No | List of properties SQL files |
| use_cached_layout | boolean | No | Use cached layout positions |
| skip_sql | boolean | No | Skip SQL, use cached edges |
| metrics_mode | string | No | Metrics preset (basic, essential, moderate, all) |

**Response**:
```json
{
  "loaded_graphs": ["crc_v2_trusts", "crc_v2_flows"],
  "node_count": 50000,
  "edge_count": 150000,
  "metrics_computed": ["in_degree", "out_degree", "pagerank"],
  "computation_time": 15.3,
  "layout_computation_time": 5.2,
  "layout_algorithm": "cached",
  "layout_cached": true,
  "data_source": "sql",
  "node_properties_loaded": ["name", "signup_timestamp"],
  "metrics_source": "computed"
}
```

---

### Get Graph Elements

Return graph elements for Cytoscape.js rendering.

```
GET /api/graphs/{graph_id}/elements?mode=full
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| graph_id | string | Graph identifier |
| mode | string | `full` or `nodes_only` |

**Response** (nodes_only mode):
```json
{
  "elements": [
    {
      "group": "nodes",
      "data": {
        "id": "0x1234...",
        "in_degree": 10,
        "out_degree": 5,
        "pagerank": 0.001
      },
      "position": {"x": 100.5, "y": 200.3}
    }
  ],
  "total_nodes": 50000,
  "total_edges": 150000,
  "edges_included": false
}
```

---

### Get Edge Chunk

Get paginated edges for progressive loading.

```
GET /api/graphs/{graph_id}/edges?offset=0&limit=50000
```

**Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| graph_id | string | - | Graph identifier |
| offset | int | 0 | Starting edge index |
| limit | int | 50000 | Maximum edges to return |

**Response**:
```json
{
  "edges": [
    {
      "group": "edges",
      "data": {
        "id": "0x1234-0x5678",
        "source": "0x1234...",
        "target": "0x5678..."
      }
    }
  ],
  "offset": 0,
  "limit": 50000,
  "returned": 50000,
  "total": 150000,
  "has_more": true
}
```

---

### Get Neighbors

Get neighbors of specified nodes.

```
POST /api/graphs/{graph_id}/neighbors
Content-Type: application/json
```

**Request Body**:
```json
{
  "node_ids": ["0x1234...", "0x5678..."],
  "direction": "both"
}
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| node_ids | array | Node IDs to query |
| direction | string | `in`, `out`, or `both` |

**Response**:
```json
{
  "neighbors": {
    "0x1234...": {
      "in": ["0xaaaa...", "0xbbbb..."],
      "out": ["0xcccc...", "0xdddd..."]
    }
  }
}
```

---

### Get All Nodes (Data Explorer)

Get paginated list of all nodes with attributes.

```
GET /api/nodes?offset=0&limit=100
```

**Response**:
```json
{
  "nodes": [
    {
      "avatar": "0x1234...",
      "in_degree": 10,
      "out_degree": 5,
      "pagerank": 0.001,
      "name": "User123"
    }
  ],
  "columns": [
    {"name": "avatar", "type": "string"},
    {"name": "in_degree", "type": "number"}
  ],
  "total": 50000,
  "offset": 0,
  "limit": 100
}
```

---

## Metrics Endpoints

### Run Metrics

Compute graph metrics.

```
POST /api/metrics/run
Content-Type: application/json
```

**Request Body**:
```json
{
  "categories": ["topology", "centrality"],
  "preset": null
}
```

Or use preset:
```json
{
  "preset": "essential"
}
```

**Response**:
```json
{
  "metrics_computed": ["in_degree", "out_degree", "pagerank"],
  "node_count": 50000,
  "computation_time": 15.3,
  "categories_run": ["topology", "centrality", "clustering", "community"]
}
```

---

### Get Metric Categories

List available metric categories.

```
GET /api/metrics/categories
```

**Response**:
```json
{
  "categories": {
    "topology": "Basic Topology (in/out degree, degree imbalance)",
    "centrality": "Centrality Measures (pagerank, betweenness, etc.)"
  },
  "presets": {
    "basic": ["topology", "community"],
    "essential": ["topology", "centrality", "clustering", "community"]
  }
}
```

---

## Anomaly Detection Endpoints

### Detect Anomalies

Run anomaly detection on specified metrics.

```
POST /api/anomaly/detect
Content-Type: application/json
```

**Request Body**:
```json
{
  "name": "anomaly_score",
  "metrics": ["pagerank", "betweenness_centrality", "clustering_coefficient"],
  "algorithm": "isolation_forest",
  "parameters": {
    "n_estimators": 100,
    "contamination": 0.1
  },
  "config": {
    "nan_strategy": "zero",
    "global_scaling": "standard"
  },
  "apply_to_graph": true,
  "node_ids": null
}
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| name | string | Name for anomaly score metric |
| metrics | array | Metrics to analyze |
| algorithm | string | Detection algorithm |
| parameters | object | Algorithm-specific parameters |
| config | object | Preprocessing configuration |
| apply_to_graph | boolean | Apply scores to graph nodes |
| node_ids | array | Filter to specific nodes (null = all) |

**Response**:
```json
{
  "algorithm": "isolation_forest",
  "n_total": 50000,
  "n_anomalies": 523,
  "anomaly_percentage": 1.05,
  "threshold": 0.65,
  "computation_time": 3.45,
  "top_anomalies": [
    {
      "id": "0x1234...",
      "score": 0.95,
      "is_anomaly": true,
      "rank": 1
    }
  ],
  "statistics": {
    "mean": 0.42,
    "std": 0.18,
    "min": 0.0,
    "max": 0.98
  },
  "visualization": {
    "histogram": {
      "bins": [0, 0.1, 0.2],
      "counts": [1000, 2000, 1500]
    }
  }
}
```

---

### Get Algorithms

List available anomaly detection algorithms.

```
GET /api/anomaly/algorithms
```

**Response**:
```json
{
  "zscore": {
    "name": "zscore",
    "display_name": "Z-Score",
    "description": "Statistical z-score based outlier detection",
    "complexity": "O(n × d)",
    "multivariate": true,
    "requires_sklearn": false,
    "parameters": {
      "threshold": {
        "name": "threshold",
        "type": "float",
        "default": 3.0,
        "min": 1.0,
        "max": 10.0,
        "description": "Z-score threshold"
      }
    }
  }
}
```

---

### Profile Metrics

Analyze metrics for preprocessing recommendations.

```
POST /api/anomaly/profile
Content-Type: application/json
```

**Request Body**:
```json
{
  "metrics": ["pagerank", "in_degree", "clustering_coefficient"]
}
```

**Response**:
```json
{
  "profiles": {
    "pagerank": {
      "name": "pagerank",
      "n_samples": 50000,
      "mean": 0.00002,
      "std": 0.0001,
      "skewness": 15.3,
      "warnings": ["High positive skewness, log transform suggested"],
      "suggested_transform": {
        "log": true,
        "clip_max": null
      }
    }
  }
}
```

---

### Run PCA

Perform PCA dimensionality reduction.

```
POST /api/anomaly/pca
Content-Type: application/json
```

**Request Body**:
```json
{
  "metrics": ["pagerank", "betweenness_centrality", "clustering_coefficient"],
  "n_components": "auto",
  "standardize": true,
  "node_ids": null
}
```

**Response**:
```json
{
  "n_components": 2,
  "explained_variance_ratio": [0.65, 0.25],
  "total_variance_explained": 0.90,
  "loadings": {
    "PC1": {"pagerank": 0.8, "betweenness": 0.5},
    "PC2": {"clustering": 0.9}
  },
  "projections": [
    {"id": "0x1234...", "PC1": 1.5, "PC2": -0.3}
  ]
}
```

---

## Composite Metrics Endpoints

### Get Operations

List available composite operations.

```
GET /api/metrics/composite/operations
```

**Response**:
```json
{
  "operations": [
    {
      "name": "multiply",
      "symbol": "×",
      "description": "Product of two metrics"
    },
    {
      "name": "add",
      "symbol": "+",
      "description": "Sum of two metrics"
    }
  ]
}
```

---

### Preview Composite

Preview composite without saving.

```
POST /api/metrics/composite/preview
Content-Type: application/json
```

**Request Body**:
```json
{
  "metrics": ["pagerank", "betweenness_centrality"],
  "operation": "multiply",
  "normalize": true,
  "node_ids": null
}
```

**Response**:
```json
{
  "formula": "norm(pagerank) × norm(betweenness_centrality)",
  "statistics": {
    "min": 0.0,
    "max": 0.85,
    "mean": 0.04,
    "std": 0.09,
    "median": 0.01
  },
  "correlations": {
    "input_correlation": 0.45,
    "m1_composite": 0.82,
    "m2_composite": 0.71
  },
  "histogram": {
    "bins": [0, 0.1, 0.2],
    "counts": [8000, 1500, 400]
  },
  "values": [
    {"id": "0x1234...", "metric1": 0.001, "metric2": 0.05, "composite": 0.012}
  ]
}
```

---

### Create Composite

Create and save composite metric.

```
POST /api/metrics/composite/create
Content-Type: application/json
```

**Request Body**:
```json
{
  "name": "influence_score",
  "metrics": ["pagerank", "betweenness_centrality"],
  "operation": "multiply",
  "weights": null,
  "normalize": true,
  "save": true
}
```

**Response**:
```json
{
  "metric_name": "influence_score",
  "formula": "norm(pagerank) × norm(betweenness_centrality)",
  "statistics": {"min": 0, "max": 0.85, "mean": 0.04},
  "saved": true,
  "composite_id": "abc123",
  "node_updates": [
    {"id": "0x1234...", "influence_score": 0.012}
  ]
}
```

---

### List Saved Composites

```
GET /api/metrics/composite/saved
```

**Response**:
```json
{
  "composites": [
    {
      "id": "abc123",
      "name": "influence_score",
      "formula": "norm(pagerank) × norm(betweenness)",
      "operation": "multiply",
      "source_metrics": ["pagerank", "betweenness_centrality"],
      "normalize": true,
      "created_at": "2024-01-15T10:30:00Z"
    }
  ]
}
```

---

### Delete Composite

```
DELETE /api/metrics/composite/saved/{name}
```

**Response**:
```json
{
  "deleted": true,
  "name": "influence_score"
}
```

---

## Auto-Reload Endpoints

### Get Status

```
GET /api/auto-reload/status
```

**Response**:
```json
{
  "enabled": true,
  "interval_seconds": 300,
  "last_reload_time": "2024-01-15T10:30:00Z",
  "next_reload_time": "2024-01-15T10:35:00Z",
  "reload_in_progress": false,
  "current_node_count": 50000,
  "last_reload_duration": 5.3,
  "last_reload_nodes_added": 50,
  "last_reload_nodes_removed": 10,
  "error": null
}
```

---

### Start Auto-Reload

```
POST /api/auto-reload/start
Content-Type: application/json
```

**Request Body**:
```json
{
  "enabled": true,
  "interval_seconds": 300,
  "sql_files": ["crc_v2_trusts.sql"],
  "node_properties_files": ["crc_v2_avatars.sql"],
  "compute_metrics": true,
  "metrics_mode": "basic"
}
```

---

### Stop Auto-Reload

```
POST /api/auto-reload/stop
```

---

### SSE Events Stream

```
GET /api/auto-reload/events
```

**Event Format**:
```
event: reload_started
data: {"timestamp": "2024-01-15T10:30:00Z"}

event: reload_complete
data: {"nodes_added": 50, "nodes_removed": 10, "duration": 5.3}

event: reload_error
data: {"error": "Database connection failed"}
```

---

## Utility Endpoints

### Health Check

```
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "mode": "admin",
  "data_status": "ready",
  "graphs_loaded": true,
  "node_count": 50000
}
```

---

### Startup Status

```
GET /api/startup-status
```

**Response**:
```json
{
  "status": "ready",
  "message": "Loaded 50000 nodes, 150000 edges",
  "node_count": 50000,
  "edge_count": 150000,
  "loaded_graphs": ["crc_v2_trusts"]
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Resource not found |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Feature not available |

**Example Error Response**:
```json
{
  "detail": "Metrics not found: ['invalid_metric']. Available: ['in_degree', 'out_degree']"
}
```

---

## Rate Limiting

Currently no rate limiting is implemented. For production deployments, consider adding rate limiting at the load balancer level.

---

## CORS

CORS is enabled for all origins by default. Configure in production as needed:

```python
# main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```