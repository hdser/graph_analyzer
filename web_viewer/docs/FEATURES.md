# Features Guide

This guide provides comprehensive documentation of all features in Graph Analyzer.

## Table of Contents

1. [Graph Loading](#graph-loading)
2. [Visualization](#visualization)
3. [Node Information](#node-information)
4. [Metrics Computation](#metrics-computation)
5. [Anomaly Detection](#anomaly-detection)
6. [Distribution Analysis](#distribution-analysis)
7. [Data Explorer](#data-explorer)
8. [Auto-Reload](#auto-reload)
9. [Export Functions](#export-functions)

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

### Node Properties

Properties SQL files add attributes to nodes:

```sql
-- Example: crc_v2_avatars.sql
SELECT 
    avatar,
    name,
    signup_timestamp,
    verified
FROM circles_v2.avatars
```

### Multi-Graph Support

Load multiple SQL files simultaneously:
- Each file creates a separate graph layer
- Switch between graphs using the dropdown
- Shared node properties across layers

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

#### Edge Styling

Configure edges:
- **Width Range**: Min/max edge thickness
- **Opacity**: Edge transparency (0-100%)
- **Color**: Edge color picker

### Node Highlighting

#### Selection Highlight

Selected nodes show:
- Blue border
- Enlarged size
- Connected edges highlighted

#### Neighbor Highlighting

Toggle "Highlight Neighbors" to show:
- Direct connections of selected nodes
- In-neighbors (incoming edges)
- Out-neighbors (outgoing edges)

#### Anomaly Highlighting

Anomalous nodes (from detection) get red styling.

#### Search Highlighting

Searched nodes get orange highlight.

---

## Node Information

### Info Panel

Click any node to see its info panel:

1. **Header**: Node ID with copy button
2. **Quick Stats**: Degree, community, core
3. **All Attributes**: Scrollable list of all metrics/properties
4. **Neighbors Section**: In/out neighbor lists

### Neighbor Navigation

Click neighbor IDs in the info panel to:
- Jump to that node
- Center view
- Show its info

### Copying Node Data

- **Copy ID**: Click the copy icon next to node ID
- **Copy Selection**: Copy all selected node IDs

---

## Metrics Computation

### Running Metrics

1. Select metrics mode (or individual categories)
2. Click "Run Metrics"
3. Wait for computation
4. Metrics added to all nodes

### Metrics Modes

| Mode | Categories | Speed | Use Case |
|------|------------|-------|----------|
| basic | topology, community | Fast | Quick overview |
| essential | + centrality, clustering | Medium | Standard analysis |
| moderate | + paths, structural | Slow | Detailed analysis |
| all | All categories | Very slow | Comprehensive |

### Metric Categories

See [METRICS.md](METRICS.md) for full documentation.

### Post-Computation

After metrics run:
- Dropdowns populated with new metrics
- Size/color mappings available
- Filter options expanded
- Distributions page updated

---

## Anomaly Detection

### Quick Start

1. Open Distributions page (📊 button)
2. Switch to "Anomaly" tab
3. Select metrics to analyze
4. Choose algorithm
5. Configure parameters
6. Click "Run Detection"

### Algorithms

| Algorithm | Best For | Speed |
|-----------|----------|-------|
| Z-Score | Normal distributions | Fast |
| IQR | Skewed data | Fast |
| Mahalanobis | Correlated features | Medium |
| Isolation Forest | General purpose | Medium |
| LOF | Local anomalies | Slow |
| DBSCAN | Clustered data | Medium |

See [ALGORITHMS.md](ALGORITHMS.md) for details.

### Results

After detection:
- **Score histogram**: Distribution of anomaly scores
- **Threshold line**: Anomaly cutoff
- **Top anomalies table**: Ranked list
- **Per-metric charts**: Feature contribution

### Actions

- **Apply to Graph**: Add scores as node attribute
- **Highlight Anomalies**: Visual highlighting
- **Export CSV**: Download results

---

## Distribution Analysis

### Histogram View

View distribution of any metric:

1. Open Distributions page
2. Check metrics to visualize
3. Histograms appear in grid

### Scatter Plot

Compare two metrics:

1. Switch to "Scatter" tab
2. Select X and Y metrics
3. Optionally select color metric
4. Click "Create Scatter Plot"

### PCA View

Dimensionality reduction visualization:

1. Switch to "PCA" tab
2. Select metrics for analysis
3. Choose components (2D or 3D)
4. Click "Run PCA"

### Selection Integration

Charts sync with main graph selection:
- Select nodes in main graph → highlighted in charts
- Toggle "Selected Only" to filter charts

---

## Data Explorer

### Overview

Full tabular view of all node data.

### Features

| Feature | Description |
|---------|-------------|
| Pagination | Navigate through pages |
| Sorting | Click column headers |
| Search | Filter by any value |
| Column hiding | Toggle column visibility |
| Export | Download as CSV |

### Access

Click "Data Explorer" button in toolbar.

### Navigation

- **Page size**: 10, 25, 50, 100, 500 rows
- **Navigation**: First, Prev, Next, Last buttons
- **Jump to page**: Direct page entry

### Sorting

Click column header to sort:
- First click: Ascending ↑
- Second click: Descending ↓
- Third click: Clear sort

### Column Management

Click "Columns" button to:
- Show/hide specific columns
- Reorder columns
- Reset to defaults

---

## Auto-Reload

### Purpose

Automatically refresh data from database at intervals.

### Configuration

1. Open Auto-Reload section in sidebar
2. Set interval (60-3600 seconds)
3. Configure SQL files to reload
4. Enable auto-reload

### Behavior

When enabled:
- Data refreshes at configured interval
- New nodes animate into view
- Removed nodes fade out
- Status shows reload progress

### SSE Events

Real-time updates via Server-Sent Events:
- Reload started
- Progress updates
- Completion notification
- Error alerts

### API

```bash
# Start auto-reload
POST /api/auto-reload/start
{
  "enabled": true,
  "interval_seconds": 300,
  "sql_files": ["crc_v2_trusts.sql"]
}

# Stop auto-reload
POST /api/auto-reload/stop

# Get status
GET /api/auto-reload/status

# SSE stream
GET /api/auto-reload/events
```

---

## Export Functions

### Graph Export

Export current graph state:

| Format | Contents | Use Case |
|--------|----------|----------|
| PNG | Graph image | Documentation |
| JSON | Elements + positions | Backup/restore |
| CSV (nodes) | All node data | Analysis |
| CSV (edges) | Edge list | Processing |

### Anomaly Export

Export detection results:
- Node ID
- Anomaly score
- Is anomaly (boolean)
- Per-metric values
- Rank

### Selection Export

Export selected nodes:
- All attributes
- Computed metrics
- Properties

### How to Export

1. **Graph Image**: File → Export PNG
2. **Node Data**: Data Explorer → Export CSV
3. **Selection**: Select nodes → Copy IDs
4. **Anomalies**: Distributions → Anomaly tab → Export

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| F | Fit graph to view |
| C | Center graph |
| Enter (in search) | Execute search |
| Escape | Clear search/selection |
| Delete | Remove selected |
| Ctrl+A | Select all |
| Ctrl+C | Copy selected IDs |

---

## UI Sections

### Sidebar Sections

All sections are collapsible:

| Section | Purpose |
|---------|---------|
| Data Source | Load graphs and properties |
| Filter Nodes | Filter by metric values |
| Visual Style | Node/edge styling |
| Composite Metrics | Create derived metrics |
| Auto-Reload | Configure refresh |

### Toolbar

Left side:
- Graph selector dropdown
- Search input and button

Right side:
- Fit/Center buttons
- Load edges button
- Data Explorer button
- Distributions button

### Status Bar

Bottom of sidebar:
- Current operation status
- Success/error messages
- Progress indicators

---

## Tips and Tricks

### Performance

1. **Large graphs**: Enable performance mode
2. **Initial load**: Load nodes first, edges progressively
3. **Metrics**: Use "basic" mode for quick exploration
4. **Filtering**: Filter before applying expensive operations

### Workflow

1. **Exploration**: Load → Basic metrics → Visual styling → Explore
2. **Analysis**: Load → Full metrics → Anomaly detection → Export
3. **Monitoring**: Configure auto-reload → Watch for changes

### Common Tasks

**Find influential nodes**:
```
1. Load graph
2. Run essential metrics
3. Color by pagerank
4. Filter pagerank > 0.001
```

**Detect anomalies**:
```
1. Run all metrics
2. Open Distributions
3. Select multiple metrics
4. Run Isolation Forest
5. Highlight anomalies
```

**Export for external analysis**:
```
1. Run desired metrics
2. Open Data Explorer
3. Verify columns
4. Export CSV
```