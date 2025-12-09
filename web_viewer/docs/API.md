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
  "hide_data_source_ui": false,
  "api_properties": {
    "enabled": true,
    "providers": [
      {
        "name": "blacklist",
        "display_name": "Blacklist (Bot Detection)",
        "columns": ["isBlacklisted", "blacklistReason"],
        "enabled": true
      }
    ],
    "base_url": "https://squid-app-3gxnl.ondigitalocean.app",
    "cache_ttl_seconds": 3600
  }
}
```

---

### Load Network

Load network data from SQL files with optional API properties.

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
  "metrics_mode": "essential",
  "load_api_properties": true,
  "api_properties_providers": null,
  "skip_api_cache": false
}
```

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| sql_files | array | Yes | - | List of SQL file names |
| node_properties_files | array | No | [] | List of properties SQL files |
| use_cached_layout | boolean | No | true | Use cached layout positions |
| skip_sql | boolean | No | false | Skip SQL, use cached edges |
| metrics_mode | string | No | "basic" | Metrics preset |
| load_api_properties | boolean | No | true | Load properties from external APIs |
| api_properties_providers | array | No | null | Specific providers to use (null = all enabled) |
| skip_api_cache | boolean | No | false | Skip API cache, always fetch fresh |

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
  "node_properties_loaded": ["name", "signup_timestamp", "isBlacklisted", "blacklistReason"],
  "node_properties_source": "sql",
  "metrics_source": "computed",
  "api_properties_loaded": {
    "blacklist": ["isBlacklisted", "blacklistReason"]
  },
  "api_properties_source": "api"
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
        "pagerank": 0.001,
        "isBlacklisted": true,
        "blacklistReason": "repeated_username"
      },
      "position": {"x": 100.5, "y": 200.3}
    }
  ],
  "count": 50000
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

### Get Node Updates

Get updated node data for incremental refresh.

```
GET /api/graphs/{graph_id}/node-updates?node_ids=0x1234,0x5678
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| graph_id | string | Graph identifier |
| node_ids | string | Comma-separated node IDs (optional) |

**Response**:
```json
{
  "updates": [
    {
      "id": "0x1234...",
      "in_degree": 10,
      "out_degree": 5,
      "isBlacklisted": true,
      "position": {"x": 100.5, "y": 200.3}
    }
  ],
  "count": 2
}
```

---

### Get Neighbors

Get neighbors of specified nodes.

```
POST /api/network/graphs/{graph_id}/neighbors
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
  "incoming": ["0xaaaa...", "0xbbbb..."],
  "outgoing": ["0xcccc...", "0xdddd..."],
  "incoming_count": 2,
  "outgoing_count": 2,
  "source_nodes": ["0x1234...", "0x5678..."]
}
```

---

### Get Application State

```
GET /api/state
```

**Response**:
```json
{
  "loaded": true,
  "loaded_graphs": ["crc_v2_trusts"],
  "node_count": 50000,
  "edge_count": 150000,
  "metrics_computed": ["in_degree", "out_degree"],
  "cytoscape_available": false,
  "anomaly_available": true,
  "auto_reload_available": true,
  "api_properties_loaded": {
    "blacklist": ["isBlacklisted", "blacklistReason"]
  },
  "api_properties_source": "cache"
}
```

---

### Get All Nodes (Data Explorer)

Get paginated list of all nodes with attributes.

```
GET /api/nodes/data?offset=0&limit=100
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
      "name": "User123",
      "isBlacklisted": false,
      "blacklistReason": null
    }
  ],
  "columns": [
    {"name": "avatar", "type": "string"},
    {"name": "in_degree", "type": "number"},
    {"name": "isBlacklisted", "type": "boolean"}
  ],
  "total": 50000,
  "offset": 0,
  "limit": 100
}
```

---

## External API Properties Endpoints

Endpoints for managing node properties loaded from external REST APIs.

### List API Properties Providers

Get available external API providers and their status.

```
GET /api/api-properties/providers
```

**Response**:
```json
{
  "providers": [
    {
      "name": "blacklist",
      "display_name": "Blacklist (Bot Detection)",
      "columns": ["isBlacklisted", "blacklistReason"],
      "enabled": true
    }
  ],
  "all_columns": ["isBlacklisted", "blacklistReason"],
  "cache_ttl_seconds": 3600,
  "base_url": "https://squid-app-3gxnl.ondigitalocean.app"
}
```

---

### List API Properties Cache

List cached API properties with metadata.

```
GET /api/api-properties/cache
```

**Response**:
```json
{
  "caches": [
    {
      "filename": "api_properties_blacklist_v2.parquet",
      "provider": "blacklist",
      "version": "v2",
      "row_count": 3404,
      "columns": ["avatar", "isBlacklisted", "blacklistReason"],
      "timestamp": 1704067200.0,
      "age_seconds": 1800.5,
      "size_mb": 0.15
    }
  ]
}
```

---

### Clear API Properties Cache

Clear API properties cache, optionally for specific provider/version.

```
DELETE /api/api-properties/cache?provider=blacklist&version=v2
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| provider | string | Provider name (optional, null = all) |
| version | string | Version (optional, null = all) |

**Response**:
```json
{
  "status": "cleared",
  "provider": "blacklist",
  "version": "v2"
}
```

---

### Refresh API Properties

Force refresh API properties by fetching fresh data from APIs.

```
POST /api/api-properties/refresh?version=v2&providers=blacklist
```

**Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| version | string | "v2" | Version to refresh |
| providers | string | null | Comma-separated provider names (null = all) |

**Response**:
```json
{
  "status": "success",
  "rows_fetched": 3404,
  "columns": ["avatar", "isBlacklisted", "blacklistReason"],
  "provider_columns": {
    "blacklist": ["isBlacklisted", "blacklistReason"]
  },
  "source": "api"
}
```

**Error Response** (no data):
```json
{
  "status": "no_data",
  "message": "No data fetched from API providers",
  "providers_queried": ["blacklist"]
}
```

---

## Metrics Endpoints

### Run Metrics

Compute graph metrics.

```
POST /api/metrics
Content-Type: application/json
```

**Request Body**:
```json
{
  "metrics_mode": "essential",
  "metrics_graph_id": "crc_v2_trusts"
}
```

**Response**:
```json
{
  "metrics_computed": ["avatar", "in_degree", "out_degree", "pagerank"],
  "computation_time": 15.3,
  "node_data": [...]
}
```

---

## Anomaly Detection Endpoints

### Get Available Algorithms

```
GET /api/anomaly/algorithms
```

**Response**:
```json
{
  "zscore": {
    "name": "zscore",
    "display_name": "Z-Score",
    "description": "Statistical outlier detection using standard deviations",
    "multivariate": false,
    "parameters": {...}
  },
  "isolation_forest": {...},
  "lof": {...}
}
```

---

### Run Anomaly Detection

```
POST /api/anomaly/detect
Content-Type: application/json
```

**Request Body**:
```json
{
  "name": "anomaly_score",
  "metrics": ["in_degree", "out_degree", "pagerank"],
  "algorithm": "isolation_forest",
  "parameters": {
    "contamination": 0.05
  },
  "apply_to_graph": true
}
```

**Response**:
```json
{
  "metric_name": "anomaly_score",
  "algorithm": "isolation_forest",
  "n_anomalies": 250,
  "n_total": 50000,
  "anomaly_percentage": 0.5,
  "computation_time": 2.5,
  "top_anomalies": [...]
}
```

---

## Composite Metrics Endpoints

### Create Composite Metric

```
POST /api/metrics/composite
Content-Type: application/json
```

**Request Body**:
```json
{
  "name": "influence_score",
  "metrics": ["pagerank", "out_degree"],
  "operation": "multiply",
  "normalize": true,
  "save": true
}
```

---

## Cache Management Endpoints

### List Cached Layouts

```
GET /api/cached-layouts
```

**Response**:
```json
[
  {
    "filename": "crc_v2_trusts.parquet",
    "graph_id": "crc_v2_trusts",
    "node_count": 50000,
    "is_base": false,
    "size_mb": 2.5
  }
]
```

---

### Clear Cached Layouts

```
DELETE /api/cached-layouts?graph_id=crc_v2_trusts
```

---

## Auto-Reload Endpoints

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
  "load_api_properties": true,
  "preserve_layout": true
}
```

---

### Stop Auto-Reload

```
POST /api/auto-reload/stop
```

---

### Get Auto-Reload Status

```
GET /api/auto-reload/status
```

---

### SSE Stream (Auto-Reload Events)

```
GET /api/auto-reload/stream
Accept: text/event-stream
```

**Events**:
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