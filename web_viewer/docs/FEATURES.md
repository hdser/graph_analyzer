# Features Guide

This guide provides comprehensive documentation of all features in Graph Analyzer.

## Table of Contents

1. [Graph Loading](#graph-loading)
2. [Node Properties](#node-properties)
3. [External API Properties](#external-api-properties)
4. [Visualization](#visualization)
5. [Node Information](#node-information)
6. [Metrics Computation](#metrics-computation)
7. [Anomaly Detection](#anomaly-detection)
8. [Distribution Analysis](#distribution-analysis)
9. [Data Explorer](#data-explorer)
10. [Auto-Reload](#auto-reload)
11. [Export Functions](#export-functions)

---

## Graph Loading

### Loading Networks

1. **Select SQL Files**: Check the SQL files to load from the dropdown
2. **Select Properties**: Optionally select node properties SQL files
3. **Configure Options**:
   - **Use Cached Layout**: Load pre-computed positions
   - **Skip SQL**: Use cached edge data
   - **Metrics Mode**: basic, essential, moderate, or all
4. **Click "Load Graphs"**

### SQL File Structure

SQL files should return edge data with source/target columns:

```sql
-- Example: crc_v2_trusts.sql
SELECT 
    truster AS source,
    trustee AS target
FROM circles_v2.trusts
WHERE ...
```

Supported column names:
- `source` / `target`
- `truster` / `trustee`
- `sender` / `receiver`

### Multi-Graph Support

Load multiple SQL files simultaneously:
- Each file creates a separate graph layer
- Switch between graphs using the dropdown
- Shared node properties across layers

---

## Node Properties

### SQL-Based Properties

Properties SQL files add attributes to nodes from the PostgreSQL database:

```sql
-- Example: crc_v2_avatars.sql
SELECT 
    avatar,
    name,
    signup_timestamp,
    verified
FROM circles_v2.avatars
```

**Requirements**:
- Must include `avatar` column (used as join key)
- Column names become property names
- Supports all data types (string, number, boolean, timestamp, arrays)

### Property Sources

Node properties can come from multiple sources:

| Source | Description | Example Properties |
|--------|-------------|-------------------|
| SQL Properties | From PostgreSQL via properties SQL files | name, signup_timestamp, verified |
| Computed Metrics | Calculated from graph structure | in_degree, pagerank, clustering |
| External APIs | Fetched from REST APIs | isBlacklisted, blacklistReason |
| Anomaly Scores | From anomaly detection | anomaly_score, is_anomaly |
| Composite Metrics | User-defined combinations | influence_score, activity_ratio |

---

## External API Properties

Graph Analyzer can enrich node data with properties from external REST APIs. This is useful for incorporating data that isn't stored in your PostgreSQL database, such as:

- Bot detection / blacklist status
- Reputation scores
- External labels or classifications
- Real-time data from other services

### Blacklist Provider

The built-in blacklist provider fetches bot detection data:

**Properties Added**:
| Property | Type | Description |
|----------|------|-------------|
| `isBlacklisted` | boolean | `true` if address is flagged as a bot |
| `blacklistReason` | string | Reason for blacklisting |

**Blacklist Reasons**:
- `repeated_username` - Multiple accounts with same username pattern
- `duplicate_avatar` - Multiple accounts with same avatar image
- `suspicious_activity` - Unusual transaction patterns
- Other custom reasons from the detection system

### How It Works

1. **During Load**: When you load a graph, the system:
   - Fetches data from configured API endpoints
   - Caches results for performance (configurable TTL)
   - Merges API properties with SQL properties
   - Applies all properties to graph nodes

2. **Caching**: API results are cached to avoid repeated requests:
   - Default cache TTL: 1 hour (3600 seconds)
   - Cache stored in `cache/data/api_properties_{provider}_{version}.parquet`
   - On API failure, falls back to cached data

3. **Load Options**:
   - `load_api_properties`: Enable/disable API fetching (default: true)
   - `skip_api_cache`: Force fresh fetch from API
   - `api_properties_providers`: Select specific providers

### Using API Properties

**In Visualization**:
- Color nodes by `isBlacklisted` to highlight bots
- Filter to show only blacklisted nodes
- Use in anomaly detection as a feature

**In Data Explorer**:
- Filter table by `isBlacklisted = true`
- Export blacklisted nodes for review
- Cross-reference with other properties

**In Anomaly Detection**:
- Include `isBlacklisted` as a feature
- Compare ML-detected anomalies with known bots
- Validate detection algorithms

### API Endpoints

Manage external API properties via REST API:

```bash
# List available providers
curl http://localhost:8000/api/api-properties/providers

# View cached data
curl http://localhost:8000/api/api-properties/cache

# Clear cache
curl -X DELETE "http://localhost:8000/api/api-properties/cache?provider=blacklist"

# Force refresh
curl -X POST "http://localhost:8000/api/api-properties/refresh?version=v2"
```

### Configuration

Configure in `.env`:

```bash
# Enable external API properties
EXTERNAL_API_PROVIDERS=blacklist

# Base URL for APIs
EXTERNAL_API_BASE_URL=https://your-api-server.com

# Cache TTL (seconds)
EXTERNAL_API_CACHE_TTL=3600

# Blacklist provider settings
EXTERNAL_API_BLACKLIST_ENABLED=true
EXTERNAL_API_BLACKLIST_ENDPOINT=/bot-analytics/blacklist
EXTERNAL_API_BLACKLIST_V2_ONLY=true
```

See [Configuration Guide](CONFIGURATION.md#external-api-properties) for full details.

---

## Visualization

### Cytoscape.js Renderer

The graph uses WebGL-accelerated rendering for performance.

### Navigation Controls

| Action | Mouse | Keyboard |
|--------|-------|----------|
| Pan | Click and drag | Arrow keys |
| Zoom | Scroll wheel | +/- keys |
| Select node | Click | - |
| Multi-select | Shift+Click | - |
| Box select | Click+Drag on empty | - |
| Fit to view | - | Button or F |
| Center | - | Button or C |

### Performance Mode

For large graphs, enable performance mode:
- Simplified node styling
- No dynamic colors/sizes
- Faster rendering

Toggle in the toolbar: "⚡ Performance Mode"

### Visual Styling

#### Node Size

Map metric to node size:

1. Select metric from "Node Size" dropdown
2. Set min/max size range (default: 8-25 pixels)
3. Click "Apply Style"

#### Node Color

Map metric to color gradient:

1. Select metric from "Node Color" dropdown
2. Choose gradient:
   - **Spectral**: Blue → Yellow → Red
   - **Viridis**: Purple → Blue → Green → Yellow
   - **Blues**: Light blue → Dark blue
   - **Reds**: Light red → Dark red
   - **Purples**: Light purple → Dark purple
3. Click "Apply Style"

#### Boolean Properties

For boolean properties like `isBlacklisted`:

1. Select the boolean property for Node Color
2. Nodes with `true` values appear in red
3. Nodes with `false`/null appear in default color

---

## Node Information

### Info Panel

Click any node to see its details:

- **ID**: Node identifier (address)
- **Metrics**: All computed metrics with values
- **Properties**: Loaded properties from SQL and APIs
- **Neighbors**: In/out degree links

### Copy Functions

- Copy node ID to clipboard
- Copy all data as JSON
- Link to external explorer (Etherscan, etc.)

---

## Metrics Computation

### Available Metrics

Graph Analyzer computes 120+ metrics in categories:

| Category | Example Metrics |
|----------|-----------------|
| Topology | in_degree, out_degree, total_degree |
| Centrality | pagerank, betweenness, eigenvector |
| Clustering | clustering_coefficient, triangles |
| Community | component_id, component_size |
| Paths | eccentricity, avg_path_length |
| Structural | core_number, authority, hub |

### Metrics Modes

| Mode | Categories | Use Case |
|------|------------|----------|
| basic | topology, community | Quick overview |
| essential | + centrality, clustering | Standard analysis |
| moderate | + paths, structural | Detailed analysis |
| all | all categories | Complete analysis |

---

## Anomaly Detection

### Algorithms

| Algorithm | Type | Best For |
|-----------|------|----------|
| Z-Score | Statistical | Single metrics, normal distributions |
| IQR | Statistical | Robust to outliers |
| Isolation Forest | ML | Multivariate, unknown patterns |
| LOF | Density | Local anomalies, clusters |
| DBSCAN | Clustering | Density-based outliers |
| Mahalanobis | Distance | Correlated features |
| PCA Reconstruction | Dimensionality | High-dimensional data |
| One-Class SVM | ML | Complex boundaries |

### Running Detection

1. Select metrics to analyze
2. Choose algorithm
3. Configure parameters
4. Set threshold method
5. Click "Run Detection"

Results are applied as node attributes:
- `anomaly_score`: Normalized score (0-1)
- `is_anomaly`: Boolean flag

---

## Distribution Analysis

### Viewing Distributions

1. Click "Distributions" in toolbar
2. Select metrics to analyze
3. View histograms and statistics

### Statistics Shown

- Count, unique values
- Min, max, mean, median
- Standard deviation
- Percentiles (25th, 75th, 95th, 99th)
- Skewness, kurtosis

---

## Data Explorer

### Features

- Sortable columns
- Search/filter
- Column visibility toggle
- Pagination
- Export to CSV

### Accessing

Click "Data Explorer" in toolbar or navigate to `/data-explorer.html`

---

## Auto-Reload

### Purpose

Automatically refresh graph data at intervals:
- Detect new nodes/edges
- Update properties
- Maintain real-time view

### Configuration

1. Enable auto-reload toggle
2. Set interval (60-3600 seconds)
3. Configure options:
   - Preserve layout positions
   - Recompute metrics
   - Refresh API properties

### Events

Monitor via SSE stream:
- `reload_started`: Reload beginning
- `reload_complete`: Success with stats
- `reload_error`: Failure with message

---

## Export Functions

### Export Options

| Format | Contents |
|--------|----------|
| PNG | Graph visualization |
| JSON | Node/edge data |
| CSV | Tabular node data |

### Node Selection Export

Export selected nodes:
1. Select nodes (click or box select)
2. Click "Export Selected"
3. Choose format

### Full Export

Export entire dataset from Data Explorer.