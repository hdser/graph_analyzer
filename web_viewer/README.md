# Graph Analyzer

A powerful web dashboard for large-scale graph visualization and analysis, built with FastAPI and Cytoscape.js.

## Features

- **Interactive Visualization**: WebGL-accelerated graph rendering with Cytoscape.js
- **120+ Graph Metrics**: Comprehensive network analysis (centrality, clustering, community detection)
- **8 Anomaly Detection Algorithms**: Z-Score, IQR, Isolation Forest, LOF, DBSCAN, Mahalanobis, PCA, One-Class SVM
- **Composite Metrics**: Create custom metrics by combining existing ones
- **External API Properties**: Enrich nodes with data from external REST APIs (e.g., bot detection)
- **Distribution Analysis**: Statistical visualizations and histograms
- **Data Explorer**: Searchable, sortable table view of all node data
- **Auto-Reload**: Real-time graph updates via Server-Sent Events
- **Multiple Layout Algorithms**: Force-directed, hierarchical, circular
- **Cytoscape Desktop Integration**: Professional layouts via py4cytoscape
- **Export Options**: PNG, JSON, CSV
- **DuckDB Data Engine**: Parquet I/O, SQL Explorer, postgres_scanner for direct DB queries
- **Arrow IPC Streaming**: Binary transport ~10× smaller than JSON, zero-copy on frontend
- **Multi-Backend Compute**: Auto-selects NetworkX (<50K), igraph (50K–5M), or cuGraph (>5M nodes)
- **Historical Snapshots**: Block-based graph snapshots with layout persistence and diff comparison
- **WebGL Rendering**: GPU-accelerated visualization via cosmos.gl for 100K+ node graphs

## Quick Start

### Prerequisites

- Python 3.9+
- PostgreSQL database with network data
- Node.js (optional, for layout service)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-repo/graph-analyzer.git
cd graph-analyzer
```

2. **Install Python dependencies**:
```bash
cd web_viewer
pip install -r requirements.txt
```

3. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your database credentials
```

4. **Start the application**:
```bash
python run.py
```

5. **Open in browser**:
```
http://localhost:8000
```

### Docker Deployment

```bash
cd web_viewer
docker-compose up -d
```

## Configuration

### Essential Environment Variables

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=circles
DB_USER=readonly_user
DB_PASSWORD=your_password

# Metrics
DEFAULT_METRICS_MODE=essential

# External API Properties (optional)
EXTERNAL_API_PROVIDERS=blacklist
EXTERNAL_API_BASE_URL=https://your-api-server.com
EXTERNAL_API_CACHE_TTL=3600
```

See [Configuration Guide](docs/CONFIGURATION.md) for all options.

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | System architecture and components |
| [Algorithms](docs/ALGORITHMS.md) | Anomaly detection algorithms |
| [Graph Metrics](docs/METRICS.md) | 120+ computed graph metrics |
| [Features Guide](docs/FEATURES.md) | Complete feature documentation |
| [Filters & Search](docs/FILTERS.md) | Filtering with regex examples |
| [Composite Metrics](docs/COMPOSITE_METRICS.md) | Creating custom metrics |
| [API Reference](docs/API.md) | REST API documentation |
| [Configuration](docs/CONFIGURATION.md) | Environment variables |
| [Deployment](docs/DEPLOYMENT.md) | Production deployment guide |

## Project Structure

```
graph-analyzer/
├── web_viewer/                # Main web application
│   ├── backend/               # FastAPI backend
│   │   ├── routers/           # API endpoints
│   │   │   ├── network.py     # Graph data & Arrow IPC endpoints
│   │   │   └── snapshots.py   # Historical snapshot API
│   │   ├── services/          # Business logic
│   │   │   ├── network_service.py
│   │   │   ├── cache_service.py
│   │   │   ├── layout_service.py
│   │   │   ├── auto_reload_service.py
│   │   │   ├── api_properties_service.py  # External API integration
│   │   │   ├── duckdb_service.py          # DuckDB data engine
│   │   │   ├── arrow_service.py           # Arrow IPC serialization
│   │   │   └── snapshot_service.py        # Snapshot management
│   │   └── models/            # Pydantic models
│   ├── engines/               # Computation engines
│   │   ├── algorithms/        # Anomaly detection algorithms
│   │   ├── anomaly_engine.py  # Main anomaly orchestrator
│   │   ├── composite_engine.py# Composite metrics
│   │   ├── graph_metrics.py   # NetworkX metrics
│   │   └── metrics/backends/
│   │       └── dispatcher.py  # Auto-select NX/igraph/cuGraph
│   ├── static/                # Frontend assets
│   │   ├── js/                # JavaScript modules
│   │   │   ├── arrow-reader.js    # Arrow IPC client reader
│   │   │   ├── cosmos-adapter.js  # cosmos.gl WebGL renderer
│   │   │   └── ...
│   │   └── css/               # Stylesheets
│   ├── layout_service/        # Node.js layout service
│   └── cache/                 # Cached data and layouts
├── sql/                       # SQL query files
│   └── properties/            # Node properties SQL
└── docs/                      # Documentation
```

## Key Features

### Graph Loading

Load networks from SQL files with automatic metric computation:

```python
POST /api/load
{
  "sql_files": ["crc_v2_trusts.sql"],
  "node_properties_files": ["crc_v2_avatars.sql"],
  "metrics_mode": "essential",
  "load_api_properties": true
}
```

### External API Properties

Enrich node data with information from external REST APIs:

```bash
# Configure in .env
EXTERNAL_API_PROVIDERS=blacklist
EXTERNAL_API_BLACKLIST_ENABLED=true
```

This adds properties like `isBlacklisted` and `blacklistReason` to nodes, with intelligent caching and fallback mechanisms.

### Anomaly Detection

Detect outliers using multiple algorithms:

```python
POST /api/anomaly/detect
{
  "metrics": ["in_degree", "out_degree", "pagerank"],
  "algorithm": "isolation_forest",
  "parameters": {"contamination": 0.05}
}
```

### Auto-Reload

Enable real-time updates with SSE:

```python
POST /api/auto-reload/start
{
  "interval_seconds": 300,
  "preserve_layout": true,
  "load_api_properties": true
}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/config` | GET | Application configuration |
| `/api/load` | POST | Load network from SQL |
| `/api/state` | GET | Current application state |
| `/api/graphs/{id}/elements` | GET | Graph elements for rendering |
| `/api/metrics` | POST | Compute graph metrics |
| `/api/anomaly/detect` | POST | Run anomaly detection |
| `/api/api-properties/providers` | GET | List API property providers |
| `/api/api-properties/refresh` | POST | Refresh API properties |
| `/api/auto-reload/start` | POST | Start auto-reload |
| `/api/graphs/{id}/elements/arrow` | GET | Nodes + positions as Arrow IPC |
| `/api/graphs/{id}/edges/arrow` | GET | Paginated edges as Arrow IPC |
| `/api/graphs/{id}/metrics/arrow` | GET | Metrics as Arrow IPC |
| `/api/sql/query` | GET | SQL Explorer (DuckDB-sandboxed) |
| `/api/snapshots/create` | POST | Create historical snapshot |
| `/api/snapshots/batch` | POST | Batch snapshot creation |
| `/api/snapshots/{id}/data` | GET | Full snapshot data |

See [API Reference](docs/API.md) for complete documentation.

## Scaling Architecture

The application uses a 4-layer scaling stack for graphs from 1K to 5M+ nodes:

```
Client (Arrow JS) ←── Arrow IPC ──→ FastAPI ←── DuckDB ──→ Parquet / PostgreSQL
       │                                │
   cosmos.gl                    Compute Dispatcher
   (WebGL)                   NX │ igraph │ cuGraph
```

### Data Engine — DuckDB

- Replaces Pandas for all Parquet I/O and joins
- SQL Explorer: run sandboxed SELECT queries against in-memory graph data
- `postgres_scanner` extension for direct PostgreSQL queries without ETL
- Stateless `:memory:` connections — no file locking under concurrency

### Compute Dispatcher

- Auto-selects graph analysis backend by node count:
  - **< 50K nodes** → NetworkX (pure Python, always available)
  - **50K – 5M** → igraph (C core, ~10–100× faster)
  - **> 5M** → cuGraph (GPU, ~100–1000× faster, requires CUDA)
- Graceful fallback if preferred backend is unavailable

### Arrow IPC Transport

- Binary serialization replacing JSON for graph data transfer
- ~10× smaller payloads for numeric data (positions, metrics)
- Zero-copy deserialization via Apache Arrow JS
- Automatic fallback to JSON if Arrow JS library is unavailable
- Edge pagination with pre-computed cosmos.gl link indices

### Historical Snapshots

- Block-based snapshots of graph state at specific blockchain blocks
- Layout persistence across snapshots (master layout + incremental derivation)
- Configurable metrics computation (NONE / BASIC / STANDARD / FULL)
- Snapshot comparison (diff added/removed nodes and edges)
- Animation data endpoint for time-lapse visualization

## Screenshots

### Main Dashboard
![Dashboard](img/cytoscape_webapp.png)

### Cytoscape Desktop Integration
![Cytoscape Desktop](img/cytoscape_desktop.png)

## Performance

| Graph Size | Load Time | Transport | Renderer |
|------------|-----------|-----------|----------|
| < 10K nodes | < 3s | Arrow IPC | Cytoscape.js |
| 10K – 50K | 3–15s | Arrow IPC | cosmos.gl (WebGL) |
| 50K – 200K | 10–30s | Arrow IPC | cosmos.gl (WebGL) |
| > 200K | 30s+ | Arrow IPC (batched edges) | cosmos.gl (WebGL) |

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## License

MIT License - see [LICENSE](LICENSE) for details.
