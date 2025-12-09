# System Architecture

This document describes the architecture of the Graph Analyzer application, including its components, data flow, and design decisions.

## Overview

Graph Analyzer follows a modular architecture with clear separation between:

- **Backend (FastAPI)**: REST API, business logic, computation engines
- **Frontend (JavaScript)**: Interactive visualization, user interface
- **External Services**: Layout computation, Cytoscape Desktop integration, External APIs

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
│  │         │               │               │                │           │  │
│  │  ┌──────┴───────────────┴───────────────┴────────────────┴────────┐  │  │
│  │  │                   api_properties_service                        │  │  │
│  │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │  │  │
│  │  │  │ BlacklistProvider│  │  (Future APIs)  │  │  Provider Base  │  │  │  │
│  │  │  └─────────────────┘  └─────────────────┘  └─────────────────┘  │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
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
         │
         │              ┌──────────────┐
         │              │ External APIs│
         └──────────────│ (Blacklist)  │
                        └──────────────┘
```

## Backend Architecture

### Directory Structure

```
backend/
├── __init__.py              # Package initialization
├── config.py                # Configuration settings
├── main.py                  # FastAPI application entry point
├── models/
│   ├── requests.py          # Pydantic request models
│   └── responses.py         # Pydantic response models
├── routers/
│   ├── network.py           # Network/graph endpoints
│   ├── metrics.py           # Metrics endpoints
│   ├── anomaly.py           # Anomaly detection endpoints
│   ├── composite.py         # Composite metrics endpoints
│   └── auto_reload.py       # Auto-reload SSE endpoints
├── services/
│   ├── network_service.py       # Main network management
│   ├── layout_service.py        # Layout computation
│   ├── cache_service.py         # Caching logic
│   ├── auto_reload_service.py   # Background reload
│   └── api_properties_service.py# External API properties
└── utils/
    └── helpers.py           # Utility functions
```

### Key Components

#### NetworkService

Central service managing all network data:

```python
class NetworkService:
    # Data storage
    edge_layers: Dict[str, pd.DataFrame]      # Edge data by layer
    metrics_dfs: Dict[str, pd.DataFrame]      # Computed metrics
    node_properties_dfs: Dict[str, pd.DataFrame]  # SQL properties
    layouts: Dict[str, Dict[str, Dict[str, float]]]  # Position data
    graphs: Dict[str, nx.DiGraph]             # NetworkX graphs
    
    # API properties tracking
    _api_properties_loaded: Dict[str, List[str]]  # provider → columns
    _api_properties_source: Optional[str]         # "api", "cache", etc.
    
    # Services
    cache_service: CacheService
    layout_service: LayoutService
    anomaly_engine: AnomalyEngine
    composite_engine: CompositeMetricEngine
```

#### APIPropertiesService

Manages external API property fetching with provider pattern:

```python
class APIPropertiesService:
    """Coordinates multiple external API providers."""
    
    providers: Dict[str, ExternalPropertyProvider]  # Registered providers
    
    @property
    def available_providers(self) -> List[dict]:
        """List enabled providers with metadata."""
    
    def fetch_all_providers(self, version: str) -> Tuple[pd.DataFrame, Dict]:
        """Fetch from all enabled providers, merge results."""

class ExternalPropertyProvider(ABC):
    """Abstract base for API providers."""
    
    name: str           # Internal identifier
    display_name: str   # Human-readable name
    
    @abstractmethod
    def columns_provided(self) -> List[str]: ...
    
    @abstractmethod
    def fetch_all(self, version: str) -> pd.DataFrame: ...
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

#### CacheService

Handles all caching operations:

```python
class CacheService:
    # Edge data caching
    def save_data_cache(layer_id, df): ...
    def load_data_cache(layer_id) -> pd.DataFrame: ...
    
    # Layout caching
    def save_layout_cache(graph_id, positions): ...
    def load_layout_cache(graph_id) -> Dict: ...
    
    # API properties caching (with TTL)
    def save_api_properties_cache(provider, version, df): ...
    def load_api_properties_cache(provider, version, ttl) -> pd.DataFrame: ...
    def is_api_cache_valid(provider, version, ttl) -> bool: ...
```

---

## Data Flow

### Network Loading Flow

```
LoadConfig Request
       │
       ▼
┌──────────────────┐
│ Phase 1: Edges   │
│ - SQL query      │
│ - Parse columns  │
│ - Cache results  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Phase 2: SQL     │
│ Properties       │
│ - Load from SQL  │
│ - Map to version │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Phase 3: API     │
│ Properties       │
│ - Check cache    │
│ - Fetch if needed│
│ - Merge with SQL │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Phase 4: Build   │
│ - NetworkX graph │
│ - Compute metrics│
│ - Get layout     │
│ - Merge props    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Phase 5: State   │
│ - Atomic swap    │
│ - Return result  │
└──────────────────┘
```

### API Properties Flow

```
load_api_properties(version, providers)
              │
              ▼
    ┌─────────────────┐
    │ Check TTL Cache │
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
 Cache Hit        Cache Miss/Expired
    │                 │
    │                 ▼
    │        ┌─────────────────┐
    │        │ Fetch from API  │
    │        │ (with retries)  │
    │        └────────┬────────┘
    │                 │
    │        ┌────────┴────────┐
    │        │                 │
    │        ▼                 ▼
    │     Success           Failure
    │        │                 │
    │        │                 ▼
    │        │        ┌─────────────────┐
    │        │        │ Fallback: stale │
    │        │        │ cache (any age) │
    │        │        └────────┬────────┘
    │        │                 │
    │        ▼                 │
    │   Save to Cache          │
    │        │                 │
    └────────┴─────────────────┘
                    │
                    ▼
           Return DataFrame
```

### Metrics Computation Flow

```
Graph → CategoryCalculators → MetricsDF → GraphNodes
          │
          ├─ TopologyCalculator (in/out degree)
          ├─ CentralityCalculator (pagerank, betweenness)
          ├─ ClusteringCalculator (coefficients, triangles)
          └─ CommunityCalculator (components, modularity)
```

### Anomaly Detection Flow

```
Input Metrics → Preprocessing → Algorithm → Scoring → Thresholding
                     │              │           │            │
                     │              │           │            │
              ┌──────┴──────┐ ┌────┴────┐ ┌────┴────┐ ┌─────┴─────┐
              │ NaN handling│ │ Z-Score │ │ Normalize│ │ Percentile│
              │ Scaling     │ │ IQR     │ │ 0-1     │ │ Fixed     │
              │ Transforms  │ │ IF/LOF  │ │ ranking │ │ MAD       │
              └─────────────┘ └─────────┘ └─────────┘ └───────────┘
```

---

## Frontend Architecture

### Module Structure

```
static/
├── js/
│   ├── app.js              # Main application logic
│   ├── api.js              # Backend API communication
│   ├── state.js            # Global state management
│   ├── cytoscape-manager.js# Cytoscape.js wrapper
│   ├── graph-loader.js     # Network loading UI
│   ├── info-panel.js       # Node details sidebar
│   ├── metrics.js          # Metrics display
│   ├── search.js           # Node search/filter
│   ├── export.js           # Export functionality
│   ├── icons.js            # SVG icon definitions
│   ├── toast.js            # Notification system
│   ├── utils.js            # Utility functions
│   ├── auto-reload.js      # Auto-reload UI
│   ├── composite-metrics.js# Composite metric builder
│   └── distributions-popup.js # Distribution charts
├── css/
│   └── style.css           # All styles
└── *.html                  # Page templates
```

### State Management

```javascript
const State = {
    cy: null,                    // Cytoscape instance
    currentGraph: null,          // Active graph ID
    loadedGraphs: [],            // All loaded graph IDs
    metricsData: {},             // Metric values by node
    edgesLoading: false,         // Edge loading status
    autoReloadEnabled: false,    // Auto-reload state
    selectedNodes: new Set(),    // Current selection
};
```

### Component Communication

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  GraphLoader│────▶│    State    │◀────│  InfoPanel  │
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
       ┌───────────┐ ┌───────────┐ ┌───────────┐
       │ Cytoscape │ │  Metrics  │ │  Export   │
       │  Manager  │ │  Display  │ │  Handler  │
       └───────────┘ └───────────┘ └───────────┘
```

---

## Caching Strategy

### Cache Types

| Type | Location | Format | TTL |
|------|----------|--------|-----|
| Edge Data | `cache/data/{layer_id}.parquet` | Parquet | Permanent |
| Layouts | `cache/layouts/{graph_id}.parquet` | Parquet | Permanent |
| SQL Properties | `cache/data/properties_{version}.parquet` | Parquet | Permanent |
| API Properties | `cache/data/api_properties_{provider}_{version}.parquet` | Parquet | Configurable (default 1hr) |

### Cache Invalidation

- **Edge Data**: Manually cleared or on SQL file change
- **Layouts**: Cleared when graph structure changes significantly
- **SQL Properties**: Refreshed on reload with `skip_sql=false`
- **API Properties**: TTL-based expiration, manual refresh endpoint

---

## External Integrations

### PostgreSQL Database

Primary data source for edge and property data:

```python
# Connection via psycopg2
conn = psycopg2.connect(
    host=settings.DB_HOST,
    port=settings.DB_PORT,
    database=settings.DB_NAME,
    user=settings.DB_USER,
    password=settings.DB_PASSWORD
)
```

### External APIs (API Properties)

REST API integration for additional node properties:

```python
# Provider pattern for extensibility
class BlacklistProvider(ExternalPropertyProvider):
    name = "blacklist"
    
    def fetch_all(self, version: str) -> pd.DataFrame:
        # Paginated fetch from external API
        # Returns DataFrame with 'avatar' column for joining
```

**Supported Providers**:
- `blacklist`: Bot detection data (isBlacklisted, blacklistReason)
- Extensible for future providers

### Cytoscape Desktop

Optional integration for high-quality layouts:

```python
# Via py4cytoscape library
import py4cytoscape as p4c

# Check availability
p4c.cytoscape_ping()

# Import network
network_suid = p4c.create_network_from_data_frames(
    nodes=nodes_df, edges=edges_df
)

# Apply layout
p4c.layout_network('force-directed')

# Export positions
positions = p4c.get_node_position()
```

### Node.js Layout Service

External service for force-directed layouts:

```javascript
// layout_server.js
app.post('/layout', async (req, res) => {
    const { nodes, edges } = req.body;
    const positions = await computeLayout(nodes, edges);
    res.json({ positions });
});
```

---

## Performance Considerations

### Large Graph Handling

| Graph Size | Strategy |
|------------|----------|
| < 10K nodes | Full rendering, all features |
| 10K - 50K | Progressive edge loading |
| 50K - 100K | Performance mode recommended |
| > 100K | Sampling or filtering required |

### Memory Management

- Edge data streamed in chunks (50K default)
- Layouts computed incrementally
- Metrics parallelized with joblib
- API properties cached to disk

### Computation Optimization

- NumPy vectorization for metrics
- Parallel processing (N_JOBS config)
- Incremental layout updates
- WebGL rendering in frontend

---

## Security Considerations

### Database Access

- Read-only user recommended
- Credentials via environment variables
- No direct SQL injection (parameterized queries)

### External API Access

- Configurable base URL
- Timeout and retry limits
- No sensitive data in API requests
- Cache avoids repeated external calls

### CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Extensibility

### Adding New Metrics

1. Add calculator function in `graph_metrics.py`
2. Register in appropriate category
3. Metrics automatically available in UI

### Adding New Anomaly Algorithms

1. Implement algorithm class in `engines/algorithms/`
2. Register in `AnomalyEngine.ALGORITHMS`
3. Define parameter schema

### Adding New API Property Providers

1. Create provider class extending `ExternalPropertyProvider`
2. Implement required methods: `endpoint`, `enabled`, `columns_provided`, `fetch_all`, `transform_to_df`
3. Register in `PROVIDER_CLASSES` dictionary
4. Add environment variable configuration
5. Enable via `EXTERNAL_API_PROVIDERS`

Example:

```python
class NewProvider(ExternalPropertyProvider):
    name = "newprovider"
    display_name = "New Provider"
    
    @property
    def columns_provided(self) -> List[str]:
        return ['score', 'category']
    
    def transform_to_df(self, response_data: Dict) -> pd.DataFrame:
        # Must return DataFrame with 'avatar' column
        return pd.DataFrame(response_data['items'])
```