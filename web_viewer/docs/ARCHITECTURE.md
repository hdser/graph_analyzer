# System Architecture

This document describes the architecture of the Graph Analyzer application, including its components, data flow, and design decisions.

## Overview

Graph Analyzer follows a modular architecture with clear separation between:

- **Backend (FastAPI)**: REST API, business logic, computation engines
- **Frontend (JavaScript)**: Interactive visualization, user interface
- **External Services**: Layout computation, Cytoscape Desktop integration

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Client Browser                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │   Main UI   │  │Distributions│  │Data Explorer│  │   Info Panel    │ │
│  │  (index)    │  │   (popup)   │  │  (table)    │  │   (sidebar)     │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘ │
│         │                │                │                   │          │
│         └────────────────┴────────────────┴───────────────────┘          │
│                                   │                                       │
│                          Cytoscape.js (WebGL)                            │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │ HTTP/SSE
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                           FastAPI Backend                                  │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                           Routers (API)                              │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐  │  │
│  │  │ network  │ │ metrics  │ │ anomaly  │ │composite │ │auto_reload│  │  │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └─────┬─────┘  │  │
│  └───────┼────────────┼────────────┼────────────┼─────────────┼────────┘  │
│          │            │            │            │             │           │
│  ┌───────┴────────────┴────────────┴────────────┴─────────────┴────────┐  │
│  │                          Services                                    │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────────┐  │  │
│  │  │  network_   │ │   layout_   │ │   cache_    │ │ auto_reload_   │  │  │
│  │  │   service   │ │   service   │ │   service   │ │    service     │  │  │
│  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └───────┬────────┘  │  │
│  └─────────┼───────────────┼───────────────┼────────────────┼──────────┘  │
│            │               │               │                │             │
│  ┌─────────┴───────────────┴───────────────┴────────────────┴──────────┐  │
│  │                           Engines                                    │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────────┐  │  │
│  │  │   graph_    │ │  anomaly_   │ │ composite_  │ │    metric_     │  │  │
│  │  │   metrics   │ │   engine    │ │   engine    │ │    profiler    │  │  │
│  │  └─────────────┘ └──────┬──────┘ └─────────────┘ └────────────────┘  │  │
│  │                         │                                            │  │
│  │              ┌──────────┴──────────┐                                 │  │
│  │              │      Algorithms     │                                 │  │
│  │              │  zscore │ iqr │ if  │                                 │  │
│  │              │  lof │ dbscan │ mah │                                 │  │
│  │              └─────────────────────┘                                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
         │                    │                        │
         ▼                    ▼                        ▼
   ┌──────────┐        ┌──────────────┐         ┌──────────────┐
   │PostgreSQL│        │ Layout Svc   │         │  Cytoscape   │
   │ Database │        │  (Node.js)   │         │   Desktop    │
   └──────────┘        └──────────────┘         └──────────────┘
```

## Backend Architecture

### Directory Structure

```
backend/
├── __init__.py          # Package initialization
├── config.py            # Configuration settings
├── main.py              # FastAPI application entry point
├── models/
│   ├── requests.py      # Pydantic request models
│   └── responses.py     # Pydantic response models
├── routers/
│   ├── network.py       # Network/graph endpoints
│   ├── metrics.py       # Metrics endpoints
│   ├── anomaly.py       # Anomaly detection endpoints
│   ├── composite.py     # Composite metrics endpoints
│   └── auto_reload.py   # Auto-reload SSE endpoints
├── services/
│   ├── network_service.py    # Main network management
│   ├── layout_service.py     # Layout computation
│   ├── cache_service.py      # Caching logic
│   └── auto_reload_service.py# Background reload
└── utils/
    └── helpers.py       # Utility functions
```

### Key Components

#### NetworkService

Central service managing all network data:

```python
class NetworkService:
    # Data storage
    edge_layers: Dict[str, pd.DataFrame]      # Edge data by layer
    metrics_dfs: Dict[str, pd.DataFrame]      # Computed metrics
    node_properties_dfs: Dict[str, pd.DataFrame]  # External properties
    layouts: Dict[str, Dict[str, Dict[str, float]]]  # Position data
    graphs: Dict[str, nx.DiGraph]             # NetworkX graphs
    
    # Services
    cache_service: CacheService
    layout_service: LayoutService
    anomaly_engine: AnomalyEngine
    composite_engine: CompositeMetricEngine
```

#### LayoutService

Manages layout computation with multiple backends:

```
Priority Order:
1. Cached layout (Parquet files)
2. Cytoscape Desktop (via py4cytoscape)
3. External Node.js service
4. Local spring layout (NumPy)
5. Circular layout (fallback)
```

#### AnomalyEngine

Orchestrates anomaly detection:

```
┌─────────────────────────────────────────────────────────────┐
│                      AnomalyEngine                          │
│                                                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐│
│  │Preprocessor│→ │ Algorithm  │→ │    Result Builder      ││
│  │            │  │  Registry  │  │                        ││
│  │ - scaling  │  │            │  │ - normalize scores     ││
│  │ - NaN fill │  │ - zscore   │  │ - compute threshold    ││
│  │ - clipping │  │ - iqr      │  │ - rank anomalies       ││
│  │ - log xform│  │ - iforest  │  │ - generate stats       ││
│  │            │  │ - lof      │  │                        ││
│  └────────────┘  │ - dbscan   │  └────────────────────────┘│
│                  │ - mahal    │                            │
│                  │ - pca      │                            │
│                  │ - ocsvm    │                            │
│                  └────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

## Frontend Architecture

### JavaScript Modules

```
static/js/
├── api.js              # API client
├── app.js              # Main entry point
├── state.js            # Global state management
├── cytoscape-manager.js # Cytoscape.js wrapper
├── graph-loader.js     # Graph loading logic
├── metrics.js          # Metrics computation UI
├── search.js           # Node search
├── info-panel.js       # Node info sidebar
├── auto-reload.js      # SSE handling
├── composite-metrics.js # Composite UI
├── distributions.js    # Distribution popup comm
├── distributions-popup.js # Full distributions UI
├── export.js           # Data export
├── icons.js            # SVG icon system
├── toast.js            # Notifications
└── utils.js            # Utilities
```

### State Management

Global state is managed in `state.js`:

```javascript
const State = {
    cy: null,                    // Cytoscape instance
    availableConfig: {},         // Server configuration
    currentElements: [],         // Current graph elements
    graphs: {},                  // Cached graph data
    styleCache: {                // Style computation cache
        sizeRange: { min: 0, max: 1 },
        colorRange: { min: 0, max: 1 }
    },
    performanceMode: false,      // Render mode flag
    neighborHighlight: false     // Neighbor highlighting
};
```

### Module Pattern

Each module follows a consistent pattern:

```javascript
const ModuleName = {
    // Public methods
    setup() { /* initialization */ },
    action() { /* user action */ },
    
    // Internal methods
    _helper() { /* private helper */ }
};
```

## Data Flow

### Loading Network Data

```
1. User selects SQL files → POST /api/load
2. Backend loads SQL from PostgreSQL
3. NetworkX graphs created from DataFrames
4. Metrics computed (GraphMetrics)
5. Layout computed (LayoutService)
6. Elements returned to frontend
7. Cytoscape.js renders visualization
```

### Anomaly Detection Flow

```
1. User selects metrics and algorithm
2. POST /api/anomaly/detect with configuration
3. AnomalyEngine:
   a. Preprocess data (scaling, NaN handling)
   b. Run algorithm (e.g., Isolation Forest)
   c. Build results (scores, labels, stats)
4. Scores applied to graph nodes
5. Frontend updates visualization
6. Results shown in distributions popup
```

### Auto-Reload Flow

```
1. User enables auto-reload with interval
2. Backend starts background task
3. At each interval:
   a. Re-query database
   b. Compute diff (added/removed nodes)
   c. Update graph data
   d. Broadcast SSE event
4. Frontend receives event
5. Graph updates with animation
```

## Caching Strategy

### Layout Cache

```
cache/layouts/
├── {graph_id}.parquet        # Current working layout
└── {graph_id}_base.parquet   # Base layout (from Cytoscape Desktop)
```

- Parquet format for type preservation and efficiency
- Base layouts are protected from overwrites
- Incremental updates for new nodes

### Data Cache

```
cache/data/
├── {graph_id}_edges.parquet        # Edge data
├── node_metrics_{version}.parquet  # Computed metrics
└── node_properties_{version}.parquet # External properties
```

## Performance Optimizations

### Backend

- **Vectorized operations**: NumPy/Pandas throughout
- **Parallel processing**: ThreadPoolExecutor for metrics
- **Chunked processing**: Large datasets processed in chunks
- **Lazy loading**: Edges loaded incrementally
- **Efficient serialization**: Parquet for cache

### Frontend

- **WebGL rendering**: Hardware-accelerated visualization
- **Batch updates**: Cytoscape.js batch() for bulk operations
- **Progressive loading**: Nodes first, edges incrementally
- **Performance mode**: Simplified styling for large graphs
- **Viewport culling**: Hide elements outside view

## Configuration

### Environment Variables

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=circles
DB_USER=readonly_user
DB_PASSWORD=secret

# Layout
LAYOUT_SERVICE_URL=http://localhost:3000/layout
MAX_EDGES_FOR_CYTOSCAPE_DESKTOP=5000000

# Performance
N_JOBS=-1  # Parallel workers
EDGE_CHUNK_SIZE=50000

# UI Mode
HIDE_DATA_SOURCE_UI=false  # true for production
DEFAULT_SQL_FILES=file1.sql,file2.sql
```

## Extension Points

### Adding New Algorithms

1. Create algorithm class extending `AnomalyAlgorithmBase`
2. Register in `algorithms/__init__.py`
3. Algorithm automatically appears in UI

### Adding New Metrics

1. Add computation in `graph_metrics.py`
2. Add to appropriate category in `METRIC_CATEGORIES`
3. Metrics automatically included in computation

### Adding New Composite Operations

1. Add operation in `CompositeMetricEngine`
2. Update `get_available_operations()`
3. Operation available in UI dropdowns