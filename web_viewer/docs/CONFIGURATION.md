# Configuration Guide

Graph Analyzer is configured through environment variables. This document describes all available settings.

## Environment Variables

### Database Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | `circles` | Database name |
| `DB_USER` | `readonly_user` | Database username |
| `DB_PASSWORD` | (empty) | Database password |

**Example**:
```bash
DB_HOST=db.example.com
DB_PORT=5432
DB_NAME=network_db
DB_USER=analyzer
DB_PASSWORD=secret123
```

---

### Layout Service

| Variable | Default | Description |
|----------|---------|-------------|
| `LAYOUT_SERVICE_URL` | `http://localhost:3000/layout` | Node.js layout service URL |
| `MAX_EDGES_FOR_CYTOSCAPE_DESKTOP` | `5000000` | Max edges for Cytoscape Desktop |

**Layout Service Priority**:
1. Cached layout
2. Cytoscape Desktop (if available and within edge limit)
3. External layout service
4. Local spring layout
5. Circular layout (fallback)

---

### Metrics Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_METRICS_MODE` | `essential` | Default metrics mode |
| `N_JOBS` | `-1` | Parallel workers (-1 = all CPUs) |

**Metrics Modes**:
- `basic`: topology, community
- `essential`: + centrality, clustering
- `moderate`: + paths, structural
- `all`: all categories

---

### Layout Algorithm Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `SPRING_STRENGTH` | `0.0008` | Spring attraction strength |
| `SPRING_LENGTH` | `200` | Natural spring length |
| `REPULSION_STRENGTH` | `5000` | Node repulsion strength |
| `DAMPING` | `0.9` | Velocity damping factor |
| `MAX_VELOCITY` | `50` | Maximum node velocity |
| `CONVERGENCE_THRESHOLD` | `0.5` | Convergence detection threshold |
| `MAX_ITERATIONS` | `100` | Maximum layout iterations |

**Tuning Guide**:
- **Tighter layout**: Increase `SPRING_STRENGTH`, decrease `SPRING_LENGTH`
- **Looser layout**: Decrease `SPRING_STRENGTH`, increase `REPULSION_STRENGTH`
- **Faster convergence**: Lower `CONVERGENCE_THRESHOLD`, higher `DAMPING`

---

### Performance Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_NODES_FOR_LOF` | `50000` | LOF algorithm node limit |
| `EDGE_CHUNK_SIZE` | `50000` | Edges per loading chunk |

**LOF Warning**: LOF has O(n²) complexity. For datasets larger than `MAX_NODES_FOR_LOF`, consider using Isolation Forest or sampling.

---

### Auto-Reload Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTO_RELOAD_DEFAULT_INTERVAL` | `300` | Default interval (seconds) |

**Constraints**:
- Minimum: 60 seconds
- Maximum: 3600 seconds

---

### UI Mode Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HIDE_DATA_SOURCE_UI` | `false` | Hide data source controls |
| `DEFAULT_SQL_FILES` | (empty) | Comma-separated SQL files |
| `DEFAULT_PROPERTIES_FILES` | (empty) | Comma-separated properties files |

**Admin Mode** (`HIDE_DATA_SOURCE_UI=false`):
- Full UI with all controls
- Manual data source selection
- Development and exploration

**Production Mode** (`HIDE_DATA_SOURCE_UI=true`):
- Hidden data source controls
- Auto-load from `DEFAULT_SQL_FILES`
- End-user focused

**Example Production Config**:
```bash
HIDE_DATA_SOURCE_UI=true
DEFAULT_SQL_FILES=crc_v2_trusts.sql,crc_v2_flows.sql
DEFAULT_PROPERTIES_FILES=crc_v2_avatars.sql
DEFAULT_METRICS_MODE=essential
AUTO_RELOAD_DEFAULT_INTERVAL=300
```

---

## Configuration File (.env)

Create a `.env` file in the `web_viewer` directory:

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=circles
DB_USER=readonly_user
DB_PASSWORD=mysecretpassword

# Layout
LAYOUT_SERVICE_URL=http://localhost:3000/layout
MAX_EDGES_FOR_CYTOSCAPE_DESKTOP=5000000

# Metrics
DEFAULT_METRICS_MODE=essential
N_JOBS=-1

# Spring Layout
SPRING_STRENGTH=0.0008
SPRING_LENGTH=200
REPULSION_STRENGTH=5000
DAMPING=0.9
MAX_VELOCITY=50
CONVERGENCE_THRESHOLD=0.5
MAX_ITERATIONS=100

# Performance
MAX_NODES_FOR_LOF=50000
EDGE_CHUNK_SIZE=50000

# Auto-Reload
AUTO_RELOAD_DEFAULT_INTERVAL=300

# UI Mode (Production)
# HIDE_DATA_SOURCE_UI=true
# DEFAULT_SQL_FILES=crc_v2_trusts.sql,crc_v2_flows.sql
# DEFAULT_PROPERTIES_FILES=crc_v2_avatars.sql
```

---

## Docker Configuration

### Environment Variables in Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DB_HOST=db
      - DB_PORT=5432
      - DB_NAME=circles
      - DB_USER=readonly_user
      - DB_PASSWORD=${DB_PASSWORD}
      - LAYOUT_SERVICE_URL=http://layout:3000/layout
      - HIDE_DATA_SOURCE_UI=true
      - DEFAULT_SQL_FILES=crc_v2_trusts.sql
    depends_on:
      - db
      - layout

  layout:
    build: ./layout_service
    ports:
      - "3000:3000"

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=circles
      - POSTGRES_USER=readonly_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
```

### Using .env with Docker

```bash
# .env file
DB_PASSWORD=supersecret
```

```bash
docker-compose --env-file .env up
```

---

## Directory Structure

### Required Directories

```
web_viewer/
├── sql/                    # SQL query files
│   └── properties/         # Node properties SQL
├── cache/                  # Auto-created cache
│   ├── layouts/            # Layout cache
│   └── data/               # Data cache
└── static/                 # Frontend files
```

### Cache Structure

```
cache/
├── layouts/
│   ├── {graph_id}.parquet        # Current layout
│   └── {graph_id}_base.parquet   # Base layout (protected)
└── data/
    ├── {graph_id}_edges.parquet  # Edge data
    ├── node_metrics_{version}.parquet
    └── node_properties_{version}.parquet
```

---

## Feature Flags

Certain features are enabled/disabled based on installed packages:

### Anomaly Detection

Requires `scikit-learn`:
```bash
pip install scikit-learn
```

Without scikit-learn:
- Z-Score and IQR still available
- ML algorithms (IF, LOF, etc.) disabled

### Cytoscape Desktop Integration

Requires `py4cytoscape` and running Cytoscape Desktop:
```bash
pip install py4cytoscape
```

### Server-Sent Events

Requires `sse-starlette`:
```bash
pip install sse-starlette
```

Without SSE:
- Auto-reload disabled
- No real-time updates

---

## Logging

### Log Level

Set via environment:
```bash
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

### Log Format

```
2024-01-15 10:30:00 - INFO - [load_network] Loading 3 SQL files
2024-01-15 10:30:05 - INFO - [compute_layout] Using local spring layout
```

---

## Health Check

The `/health` endpoint provides status information:

```bash
curl http://localhost:8000/health
```

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

## Troubleshooting

### Database Connection

```
Error: could not connect to server
```

**Check**:
1. PostgreSQL is running
2. `DB_HOST` is correct
3. `DB_PORT` is accessible
4. Credentials are valid

### Layout Service

```
[LAYOUT] External service failed
```

**Check**:
1. Layout service is running: `http://localhost:3000/health`
2. `LAYOUT_SERVICE_URL` is correct
3. No firewall blocking

### Memory Issues

```
MemoryError during metrics computation
```

**Solutions**:
1. Reduce `N_JOBS` to limit parallel workers
2. Use `basic` or `essential` metrics mode
3. Increase system swap space
4. Filter to smaller subgraph

### Slow Performance

**For large graphs**:
1. Enable performance mode in UI
2. Use cached layouts
3. Load edges incrementally
4. Reduce metrics categories