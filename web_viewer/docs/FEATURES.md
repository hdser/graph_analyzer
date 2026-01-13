# Features Guide

This guide provides comprehensive documentation of all features in Graph Analyzer, a web-based dashboard for large-scale graph visualization and analysis.

## Table of Contents

1. [Graph Loading](#graph-loading)
2. [Node Properties](#node-properties)
3. [External API Properties](#external-api-properties)
4. [Visualization](#visualization)
5. [Node Information](#node-information)
6. [Metrics Computation](#metrics-computation)
7. [Anomaly Detection](#anomaly-detection)
8. [Deep Learning & Embeddings](#deep-learning--embeddings)
9. [Composite Metrics](#composite-metrics)
10. [Distribution Analysis](#distribution-analysis)
11. [Path Analysis](#path-analysis)
12. [Flow Analysis](#flow-analysis)
13. [Subgraph Tools](#subgraph-tools)
14. [Snapshots & Time Series](#snapshots--time-series)
15. [Data Explorer](#data-explorer)
16. [Auto-Reload](#auto-reload)
17. [Export Functions](#export-functions)
18. [Keyboard Shortcuts](#keyboard-shortcuts)

---

## Graph Loading

### Loading Networks

1. **Select SQL Files**: Check the SQL files to load from the dropdown
2. **Select Properties**: Optionally select node properties SQL files
3. **Configure Options**:
   - **Use Cached Layout**: Load pre-computed positions
   - **Skip SQL**: Use cached edge data
   - **Metrics Mode**: basic, essential, moderate, comprehensive, or all
   - **Load API Properties**: Fetch external API data
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
- `from` / `to`

### Weighted Edges

Include a `weight` column for weighted graphs:

```sql
SELECT 
    sender AS source,
    receiver AS target,
    amount AS weight
FROM transfers
```

### Multi-Graph Support

Load multiple SQL files simultaneously:
- Each file creates a separate graph layer
- Switch between graphs using the dropdown
- Shared node properties across layers
- Compare metrics across different relationship types

### Graph Info Display

After loading, view:
- Node count
- Edge count
- Connected components
- Average degree
- Density

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
    verified,
    balance
FROM circles_v2.avatars
```

**Requirements**:
- Must include `avatar` column (used as join key)
- Column names become property names
- Supports all data types

### Property Types

| Type | Example | Use Case |
|------|---------|----------|
| String | name, address | Labels, identification |
| Number | balance, age | Metrics, filtering |
| Boolean | verified, isBot | Binary classification |
| Timestamp | signup_date | Temporal analysis |
| Array | tags, categories | Multi-value attributes |

### Property Sources

Node properties can come from multiple sources:

| Source | Priority | Description |
|--------|----------|-------------|
| SQL Properties | 1 | From PostgreSQL via properties SQL files |
| External APIs | 2 | Fetched from REST APIs (blacklist, etc.) |
| Computed Metrics | 3 | Calculated from graph structure |
| Composite Metrics | 4 | User-defined combinations |
| Anomaly Scores | 5 | From anomaly detection |
| Deep Learning | 6 | Embeddings and communities |

---

## External API Properties

### Overview

Graph Analyzer can enrich node data with properties from external REST APIs:

- Bot detection / blacklist status
- Reputation scores
- External labels or classifications
- Real-time data from other services

### Blacklist Provider

The built-in blacklist provider fetches bot detection data:

**Properties Added**:

| Property | Type | Description |
|----------|------|-------------|
| `isBlacklisted` | boolean | `true` if flagged as bot |
| `blacklistReason` | string | Reason for blacklisting |

**Blacklist Reasons**:
- `repeated_username` - Multiple accounts with same username pattern
- `duplicate_avatar` - Multiple accounts with same avatar image
- `suspicious_activity` - Unusual transaction patterns

### Caching

API results are cached to avoid repeated requests:
- Default cache TTL: 1 hour (3600 seconds)
- Cache location: `cache/data/api_properties_{provider}_{version}.parquet`
- Fallback to cached data on API failure

### API Endpoints

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

---

## Visualization

### Cytoscape.js Renderer

The graph uses WebGL-accelerated Cytoscape.js for high-performance rendering:
- Supports 100K+ nodes
- Hardware-accelerated pan/zoom
- Efficient edge bundling

### Navigation Controls

| Action | Mouse | Keyboard |
|--------|-------|----------|
| Pan | Click and drag | Arrow keys |
| Zoom | Scroll wheel | +/- keys |
| Select node | Click | - |
| Multi-select | Shift+Click | - |
| Box select | Click+Drag on empty | - |
| Fit to view | - | F |
| Center | - | C |
| Reset zoom | - | R |

### Performance Mode

For large graphs (>50K nodes), enable performance mode:
- Simplified node styling
- No dynamic colors/sizes
- Reduced edge rendering
- Faster interactions

Toggle: "⚡ Performance Mode" in toolbar

### Visual Styling

#### Node Size

Map any numeric metric to node size:

1. Select metric from "Node Size" dropdown
2. Set min/max size range (default: 8-25 pixels)
3. Choose scaling: linear, log, sqrt
4. Click "Apply Style"

#### Node Color

Map any metric to color gradient:

1. Select metric from "Node Color" dropdown
2. Choose gradient:
   - **Spectral**: Blue → Yellow → Red
   - **Viridis**: Purple → Blue → Green → Yellow
   - **Plasma**: Purple → Pink → Orange → Yellow
   - **Blues**: Light blue → Dark blue
   - **Reds**: Light red → Dark red
   - **Greens**: Light green → Dark green
   - **Purples**: Light purple → Dark purple
   - **RdYlBu**: Red → Yellow → Blue (diverging)
3. Click "Apply Style"

#### Boolean Properties

For boolean properties like `isBlacklisted`:
- `true` values: Red
- `false`/null values: Default color

#### Community Coloring

For community IDs:
- Automatic categorical color assignment
- Up to 20 distinct colors
- Beyond 20: color cycling

### Layout Algorithms

| Layout | Description | Best For |
|--------|-------------|----------|
| Force-Directed | Physics simulation | General |
| Cytoscape Desktop | External layout server | High quality |
| Grid | Regular grid positions | Uniform display |
| Circle | Circular arrangement | Ring structures |
| Concentric | Concentric circles by metric | Hierarchical |
| Breadthfirst | Tree-like layout | DAGs |

---

## Node Information

### Info Panel

Click any node to see its details:

**Sections**:
- **Identity**: Node ID, name, address
- **Properties**: All loaded properties
- **Metrics**: Computed graph metrics
- **Neighbors**: In/out connections with counts
- **Community**: Community assignment and confidence
- **Embedding**: Deep learning embedding info

### Copy Functions

- Copy node ID to clipboard
- Copy address to clipboard
- Copy all data as JSON
- Link to external explorer (Etherscan, etc.)

### Multi-Node Selection

Select multiple nodes to see:
- Combined statistics
- Common properties
- Induced subgraph metrics

---

## Metrics Computation

### Overview

Graph Analyzer computes **150+ metrics** in **25 categories**.

See [METRICS.md](METRICS.md) for complete documentation.

### Quick Reference

| Category | Example Metrics | Cost |
|----------|-----------------|------|
| Topology | in_degree, out_degree | Low |
| Centrality | pagerank, betweenness | Medium-High |
| Clustering | clustering_coefficient | Medium |
| Community | louvain, leiden | Medium |
| Paths | shortest_paths, eccentricity | High |
| Trust | eigentrust, appleseed | Medium |

### Metrics Modes

| Mode | Categories | Speed |
|------|------------|-------|
| basic | topology, community basics | Very Fast |
| essential | + centrality, clustering | Fast |
| moderate | + paths, structural | Medium |
| comprehensive | most categories | Slow |
| all | all categories | Very Slow |

### Custom Metric Selection

Via UI:
1. Open Metrics panel
2. Select categories or individual metrics
3. Configure parameters
4. Click "Compute"

Via API:
```bash
curl -X POST "http://localhost:8000/api/metrics/run" \
  -H "Content-Type: application/json" \
  -d '{"metrics": ["pagerank", "betweenness_centrality"]}'
```

---

## Anomaly Detection

### Overview

Eight algorithms for detecting anomalous nodes.

See [ALGORITHMS.md](ALGORITHMS.md) for complete documentation.

### Available Algorithms

| Algorithm | Type | Best For |
|-----------|------|----------|
| Z-Score | Statistical | Normal distributions |
| IQR | Statistical | Robust to outliers |
| Isolation Forest | ML | High-dimensional |
| LOF | Density | Local anomalies |
| DBSCAN | Clustering | Clustered data |
| Mahalanobis | Distance | Correlated features |
| PCA | Manifold | Linear relationships |
| One-Class SVM | Boundary | Complex patterns |

### Running Detection

1. **Select Metrics**: Choose metrics to analyze
2. **Choose Algorithm**: Select from dropdown
3. **Configure Parameters**: Adjust algorithm settings
4. **Set Threshold**: Choose threshold method
5. **Run Detection**: Click "Detect Anomalies"

### Results

Results are applied as node attributes:
- `anomaly_score`: Normalized score (0-1)
- `is_anomaly`: Boolean flag
- `anomaly_rank`: Rank by score

### Visualization

- Color nodes by `anomaly_score`
- Filter to show only anomalies
- View distribution in Distributions panel

---

## Deep Learning & Embeddings

### Overview

GIT-CD (Graph-Informed Transformer for Community Detection) provides:
- Node embeddings
- Learned community detection
- Similarity search
- Visualization

See [DEEP_LEARNING.md](DEEP_LEARNING.md) for complete documentation.

### Quick Start

1. **Load Graph**: Ensure a graph is loaded with metrics computed
2. **Open Panel**: Click the 🧠 neural network icon
3. **Configure**: Set clusters, hidden dim, epochs
4. **Train**: Click "Train Model"
5. **Monitor**: Open Training Monitor for progress

### Key Features

| Feature | Description |
|---------|-------------|
| Training | Background training with progress updates |
| Communities | Soft clustering with confidence scores |
| Similarity | Find similar nodes by embedding |
| Visualization | UMAP/t-SNE projection |

### Requirements

```bash
pip install torch torch-geometric umap-learn
```

---

## Composite Metrics

### Overview

Create custom metrics by combining existing ones.

See [COMPOSITE_METRICS.md](COMPOSITE_METRICS.md) for complete documentation.

### Available Operations

| Operation | Formula | Use Case |
|-----------|---------|----------|
| Multiply | M1 × M2 | Joint importance |
| Add | M1 + M2 | Aggregation |
| Subtract | M1 - M2 | Difference/imbalance |
| Divide | M1 / M2 | Ratios |
| Average | (M1 + M2) / 2 | Balanced score |
| Maximum | max(M1, M2) | Upper bound |
| Minimum | min(M1, M2) | Lower bound |
| Weighted | w1×M1 + w2×M2 | Custom weighting |

### Creating Composites

Via UI:
1. Open Composite Metrics panel
2. Select two metrics
3. Choose operation
4. Enable normalization (optional)
5. Name and create

Via API:
```bash
curl -X POST "http://localhost:8000/api/metrics/composite/create" \
  -d '{"name": "influence", "metrics": ["pagerank", "betweenness"], "operation": "multiply"}'
```

---

## Distribution Analysis

### Opening Distributions

Click "📊 Distributions" in toolbar or press D.

### Features

- **Histograms**: Distribution visualization
- **Statistics**: min, max, mean, median, std
- **Percentiles**: 25th, 50th, 75th, 95th, 99th
- **Skewness/Kurtosis**: Distribution shape
- **Correlation Matrix**: Metric relationships
- **Scatter Plots**: Pairwise comparisons

### Composite Preview

Create and preview composite metrics:
1. Select metrics and operation
2. View resulting distribution
3. Check correlations
4. Create if satisfied

---

## Path Analysis

### Overview

Analyze shortest paths and reachability between nodes.

### Features

| Feature | Description |
|---------|-------------|
| Shortest Path | Find path between two nodes |
| All Paths | List all paths up to length k |
| Path Statistics | Length distribution |
| Reachability | Nodes reachable from source |

### Usage

1. Select source node (click or search)
2. Select target node
3. Click "Find Path"
4. View highlighted path

### API

```bash
curl "http://localhost:8000/api/paths/shortest?source=0x123&target=0x456"
```

---

## Flow Analysis

### Circles Capacity Flow

Analyze token flow capacity in trust networks using the Circles protocol model.

### Features

| Feature | Description |
|---------|-------------|
| Max Flow | Maximum flow between addresses |
| Flow Decomposition | Path-by-path flow breakdown |
| Capacity Graph | Build capacity-weighted graph |
| Bottleneck Analysis | Identify flow constraints |

### Usage

1. Open Flow Analysis panel
2. Enter source and target addresses
3. Set max flow amount
4. Click "Calculate Max Flow"
5. View decomposed flows

### Configuration

| Parameter | Description |
|-----------|-------------|
| Max Hops | Maximum path length (default: 10) |
| Simplify Paths | Remove redundant edges |
| Use Netted | Use netted balances |

---

## Subgraph Tools

### Overview

Extract and analyze subgraphs based on selection or criteria.

### Selection Methods

| Method | Description |
|--------|-------------|
| Manual | Click/box-select nodes |
| Ego | N-hop neighborhood of node |
| Community | All nodes in a community |
| Filter | Nodes matching criteria |
| Path | Nodes on path between two points |

### Operations

| Operation | Description |
|-----------|-------------|
| Extract | Create subgraph from selection |
| Analyze | Compute metrics on subgraph |
| Export | Save subgraph data |
| Highlight | Visual emphasis |

### Ego Networks

1. Select center node
2. Set radius (1-5 hops)
3. Choose direction (in, out, both)
4. Extract ego network

---

## Snapshots & Time Series

### Snapshots

Save graph state at points in time for comparison.

| Feature | Description |
|---------|-------------|
| Create | Save current state |
| Compare | Diff two snapshots |
| Load | Restore previous state |
| Delete | Remove saved snapshot |

### Time Series

Analyze metric changes over time when multiple snapshots exist.

| Analysis | Description |
|----------|-------------|
| Trend | Metric value over time |
| Velocity | Rate of change |
| Anomaly | Deviation from trend |

### Temporal Composite

Combine metrics across time periods:
- Rolling averages
- Time-weighted scores
- Change detection

---

## Data Explorer

### Features

Full-featured data table for node analysis:

| Feature | Description |
|---------|-------------|
| Sorting | Click column headers |
| Filtering | Search box and column filters |
| Column Toggle | Show/hide columns |
| Pagination | Navigate large datasets |
| Export | CSV, JSON download |
| Selection | Click to highlight on graph |

### Accessing

- Click "Data Explorer" in toolbar
- Or navigate to `/data-explorer.html`
- Or press E

### Column Categories

| Category | Columns |
|----------|---------|
| Identity | id, name, address |
| Properties | SQL-loaded properties |
| API Properties | isBlacklisted, etc. |
| Topology | degrees, ratios |
| Centrality | pagerank, betweenness, etc. |
| Community | community_id, core_number |
| Anomaly | anomaly_score, is_anomaly |
| Embedding | community, confidence |

---

## Auto-Reload

### Purpose

Automatically refresh graph data at intervals for:
- Detecting new nodes/edges
- Updating properties
- Maintaining real-time view

### Configuration

1. Enable auto-reload toggle
2. Set interval (60-3600 seconds)
3. Configure options:
   - Preserve layout positions
   - Recompute metrics
   - Refresh API properties
   - Detect changes only

### Events (SSE Stream)

```javascript
// Connect to event stream
const events = new EventSource('/api/auto-reload/events');

events.addEventListener('reload_started', (e) => {
  console.log('Reload beginning');
});

events.addEventListener('reload_complete', (e) => {
  const data = JSON.parse(e.data);
  console.log('New nodes:', data.new_nodes);
});

events.addEventListener('reload_error', (e) => {
  console.error('Reload failed:', e.data);
});
```

---

## Export Functions

### Export Options

| Format | Contents | Use Case |
|--------|----------|----------|
| PNG | Graph visualization | Reports |
| SVG | Vector visualization | High-quality |
| JSON | Complete node/edge data | Analysis |
| CSV | Tabular node data | Spreadsheets |
| GraphML | Standard graph format | Other tools |

### Node Selection Export

1. Select nodes (click or box select)
2. Click "Export Selected"
3. Choose format
4. Download file

### Full Export

Export entire dataset from Data Explorer:
- All nodes with all properties
- Filtered view export
- Custom column selection

### API Export

```bash
# Export nodes as JSON
curl "http://localhost:8000/api/export/nodes?format=json"

# Export edges as CSV
curl "http://localhost:8000/api/export/edges?format=csv"

# Export subgraph
curl "http://localhost:8000/api/export/subgraph?nodes=0x123,0x456"
```

---

## Keyboard Shortcuts

### Navigation

| Key | Action |
|-----|--------|
| Arrow keys | Pan graph |
| +/- | Zoom in/out |
| F | Fit to view |
| C | Center on selection |
| R | Reset zoom |
| Home | Reset view |

### Selection

| Key | Action |
|-----|--------|
| Escape | Clear selection |
| A | Select all |
| Delete | Remove selected (temporary) |

### Panels

| Key | Action |
|-----|--------|
| I | Toggle Info panel |
| M | Toggle Metrics panel |
| D | Open Distributions |
| E | Open Data Explorer |
| S | Toggle Search |

### Actions

| Key | Action |
|-----|--------|
| Ctrl+S | Save snapshot |
| Ctrl+E | Export selection |
| Ctrl+F | Search nodes |
| ? | Show help |

---

## Configuration

### Environment Variables

See [CONFIGURATION.md](CONFIGURATION.md) for complete settings.

Key variables:
```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=circles

# Server
HOST=0.0.0.0
PORT=8000

# Performance
N_JOBS=-1
CACHE_DIR=cache

# External APIs
EXTERNAL_API_PROVIDERS=blacklist
EXTERNAL_API_BASE_URL=https://api.example.com
```

### Cache Management

```bash
# Clear all caches
curl -X DELETE http://localhost:8000/api/cache/all

# Clear specific cache
curl -X DELETE http://localhost:8000/api/cache/metrics
curl -X DELETE http://localhost:8000/api/cache/layouts
curl -X DELETE http://localhost:8000/api/cache/api-properties
```