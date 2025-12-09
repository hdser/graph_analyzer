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

## External API Properties

Graph Analyzer can fetch additional node properties from external REST APIs. This allows enriching node data with information not stored in the PostgreSQL database, such as blacklist status, reputation scores, or labels from external services.

### Base Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EXTERNAL_API_BASE_URL` | `https://squid-app-3gxnl.ondigitalocean.app` | Base URL for external APIs |
| `EXTERNAL_API_TIMEOUT` | `30` | HTTP request timeout (seconds) |
| `EXTERNAL_API_RETRIES` | `3` | Number of retry attempts on failure |
| `EXTERNAL_API_CACHE_TTL` | `3600` | Cache time-to-live (seconds, 0 = no expiry) |
| `EXTERNAL_API_PROVIDERS` | `blacklist` | Comma-separated list of enabled providers |

### Blacklist Provider

The blacklist provider fetches bot detection data, adding `isBlacklisted` and `blacklistReason` columns to nodes.

| Variable | Default | Description |
|----------|---------|-------------|
| `EXTERNAL_API_BLACKLIST_ENABLED` | `true` | Enable blacklist provider |
| `EXTERNAL_API_BLACKLIST_ENDPOINT` | `/aboutcircles-advanced-analytics2/bot-analytics/blacklist` | API endpoint path |
| `EXTERNAL_API_BLACKLIST_V2_ONLY` | `true` | Filter for v2 addresses only |

**Properties Added**:
| Column | Type | Description |
|--------|------|-------------|
| `isBlacklisted` | boolean | `true` if address is blacklisted |
| `blacklistReason` | string | Reason (e.g., `repeated_username`, `duplicate_avatar`) |

### Example Configuration

```bash
# External API Properties
EXTERNAL_API_BASE_URL=https://squid-app-3gxnl.ondigitalocean.app
EXTERNAL_API_TIMEOUT=30
EXTERNAL_API_RETRIES=3
EXTERNAL_API_CACHE_TTL=3600
EXTERNAL_API_PROVIDERS=blacklist

# Blacklist Provider
EXTERNAL_API_BLACKLIST_ENABLED=true
EXTERNAL_API_BLACKLIST_ENDPOINT=/aboutcircles-advanced-analytics2/bot-analytics/blacklist
EXTERNAL_API_BLACKLIST_V2_ONLY=true
```

### Caching Behavior

- API properties are cached in `cache/data/api_properties_{provider}_{version}.parquet`
- Cache TTL is configurable via `EXTERNAL_API_CACHE_TTL`
- Setting TTL to `0` disables cache expiry
- On API failure, the system falls back to cached data (ignoring TTL)
- Cache metadata is stored in `.meta.json` files for TTL tracking

### Adding New Providers

To add a new external API provider:

1. **Add environment variables** following the naming pattern:
   ```bash
   EXTERNAL_API_{PROVIDER}_ENABLED=true
   EXTERNAL_API_{PROVIDER}_ENDPOINT=/path/to/api
   ```

2. **Implement provider class** in `api_properties_service.py`:
   ```python
   class NewProvider(ExternalPropertyProvider):
       name = "newprovider"
       display_name = "New Provider"
       
       @property
       def columns_provided(self) -> List[str]:
           return ['column1', 'column2']
       
       def transform_to_df(self, response_data):
           # Transform API response to DataFrame with 'avatar' column
           ...
   ```

3. **Register provider** in `PROVIDER_CLASSES` dictionary

4. **Add to `EXTERNAL_API_PROVIDERS`** environment variable

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

# External API Properties
EXTERNAL_API_BASE_URL=https://squid-app-3gxnl.ondigitalocean.app
EXTERNAL_API_TIMEOUT=30
EXTERNAL_API_RETRIES=3
EXTERNAL_API_CACHE_TTL=3600
EXTERNAL_API_PROVIDERS=blacklist
EXTERNAL_API_BLACKLIST_ENABLED=true
EXTERNAL_API_BLACKLIST_ENDPOINT=/aboutcircles-advanced-analytics2/bot-analytics/blacklist
EXTERNAL_API_BLACKLIST_V2_ONLY=true

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
      - DEFAULT_METRICS_MODE=essential
      # External API
      - EXTERNAL_API_BASE_URL=https://squid-app-3gxnl.ondigitalocean.app
      - EXTERNAL_API_PROVIDERS=blacklist
      - EXTERNAL_API_BLACKLIST_ENABLED=true
    volumes:
      - ./sql:/app/sql:ro
      - cache_data:/app/cache
    depends_on:
      - db
      - layout
```

---

## Environment Variable Precedence

1. Environment variables set in shell
2. Variables in `.env` file
3. Default values in `config.py`

---

## Validating Configuration

Check your configuration at startup:

```bash
python run.py
```

The startup banner displays current settings:

```
============================================================
  Graph Analyzer Web Viewer v2.0.0
============================================================
  Database: localhost:5432/circles
  SQL Dir:  /path/to/sql
  Properties Dir: /path/to/sql/properties
  Cache:    /path/to/cache
------------------------------------------------------------
  SSE Support:        Y
  Anomaly Detection:  Y
  Cytoscape Desktop:  N
------------------------------------------------------------
  External API Properties:
    Base URL: https://squid-app-3gxnl.ondigitalocean.app
    Providers: blacklist
    Blacklist: Y
------------------------------------------------------------
  Mode: Admin (manual control)
============================================================
```

Or via API:

```bash
curl http://localhost:8000/api/config | jq
```