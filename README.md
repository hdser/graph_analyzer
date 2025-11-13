# Graph Analyzer

Computes multiple graph metrics for directed trust networks using NetworkX with **selective metric computation** for faster targeted analysis.

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env with your database credentials and metrics mode

# 4. Run main graph metrics
python graph_metrics.py

# 5. Get help on available modes
python graph_metrics.py --help
````

For blacklist / whitelist analysis commands, see **Blacklist / Whitelist Management** below.

## Configuration

Create `.env` file:

```env
DB_USER=your_username
DB_PASSWORD=your_password
DB_HOST=your_host
DB_NAME=your_database
OUTPUT_FILE=graph_metrics.csv
N_JOBS=2  # Number of parallel workers (optional)
METRICS_MODE=all  # Which metrics to compute (see below)
```

## NEW: Selective Metrics Computation

You can now choose which metrics to compute, dramatically reducing computation time for targeted analysis.

### Metrics Modes

#### Preset Modes

* **`basic`**: Quick topology + clustering analysis (~5 seconds)

  * Basic Topology (in/out degree, degree imbalance)
  * Clustering Metrics (clustering coefficient, triangles)

* **`essential`**: Most important metrics (~1-5 minutes)

  * Basic Topology
  * Centrality Measures (PageRank, betweenness, etc.)
  * Clustering Metrics
  * Community Detection

* **`moderate`**: Balanced comprehensive analysis (~5-15 minutes)

  * Everything in 'essential' plus:
  * Path Metrics (shortest paths, reachability)
  * Structural Metrics (structural holes, articulation points)

* **`all`**: Complete analysis (default, ~15-60+ minutes)

  * All 120+ metrics across all 15 categories

#### Individual Categories

You can specify individual categories or combine them:

```env
# Single category
METRICS_MODE=topology

# Multiple categories (comma-separated)
METRICS_MODE=topology,clustering,community

# Mix and match as needed
METRICS_MODE=topology,centrality,paths,reach
```

### Available Categories

| Category      | Description                              | Metrics Count |
| ------------- | ---------------------------------------- | ------------- |
| `topology`    | Basic degree metrics                     | 4             |
| `centrality`  | PageRank, betweenness, eigenvector, etc. | 20+           |
| `clustering`  | Triangles, clustering coefficients       | 6             |
| `community`   | Louvain communities, k-core              | 5             |
| `paths`       | Shortest paths, eccentricity             | 10            |
| `distances`   | Radius, diameter, center/periphery       | 4             |
| `structural`  | Structural holes, bridges                | 12            |
| `reciprocity` | Mutual connections                       | 5             |
| `reach`       | N-hop reach analysis                     | 8             |
| `components`  | Connected components                     | 3             |
| `vitality`    | Node importance                          | 1             |
| `dispersion`  | Node dispersion                          | 2             |
| `efficiency`  | Network efficiency                       | 1             |
| `flow`        | Flow hierarchy                           | 1             |
| `dominance`   | Dominance metrics                        | 2             |

### Usage Examples

```bash
# Quick topology check (seconds)
METRICS_MODE=basic python graph_metrics.py

# Just need centrality measures
METRICS_MODE=centrality python graph_metrics.py  

# Custom analysis for community structure
METRICS_MODE=topology,clustering,community python graph_metrics.py

# Full analysis (default)
METRICS_MODE=all python graph_metrics.py
```

## Blacklist / Whitelist Management

Two helper scripts manage blacklist/whitelist data in `data/blacklist.db` and CSV files.

### 1. Compare CSV blacklists vs existing DB

This script:

* Reads multiple `blacklist_Sheet*.csv` files
* Applies v1/v2 rules (using `graph_metrics_v1.csv` and `graph_metrics_v2.csv`)
* Checks overlaps with existing `Blacklist` and `Whitelist` tables
* Produces a **full updated blacklist CSV** (existing DB reasons take priority)

```bash
python compare_blacklist_csvs.py \
  --csv data/blacklist_Sheet1.csv data/blacklist_Sheet2.csv data/blacklist_Sheet3.csv data/blacklist_Sheet4.csv \
  --db data/blacklist.db \
  --graph-metrics-v1 data/graph_metrics_v1.csv \
  --graph-metrics-v2 data/graph_metrics_v2.csv \
  --output-csv data/blacklist_full_updated.csv \
  --verbose
```

Key arguments:

* `--csv` – one or more blacklist CSV sheets
* `--db` – path to `blacklist.db` (default `data/blacklist.db`)
* `--graph-metrics-v1` – `graph_metrics_v1.csv` (for v1/v2 logic)
* `--graph-metrics-v2` – `graph_metrics_v2.csv` (for reporting)
* `--output-csv` – where to write the merged blacklist
* `--verbose` – more detailed logging

### 2. Apply CSV updates to Blacklist / Whitelist

This script updates the SQLite DB from a CSV:

* Update **Blacklist** or **Whitelist**
* Add/update entries (default)
* Or remove entries with `--remove`

#### Update Blacklist from merged CSV

Typically used after `compare_blacklist_csvs.py`:

```bash
python update_blacklist.py \
  --db data/blacklist.db \
  --list-type blacklist \
  --csv data/blacklist_full_updated.csv
```

#### Add to Whitelist

```bash
python update_blacklist.py \
  --db data/blacklist.db \
  --list-type whitelist \
  --csv data/my_whitelist.csv
```

#### Remove addresses from Blacklist or Whitelist

```bash
# Remove from blacklist
python update_blacklist.py \
  --db data/blacklist.db \
  --list-type blacklist \
  --csv data/remove_from_blacklist.csv \
  --remove

# Remove from whitelist
python update_blacklist.py \
  --db data/blacklist.db \
  --list-type whitelist \
  --csv data/remove_from_whitelist.csv \
  --remove
```

## Performance Guide

| Mode         | Time Estimate  | Use Case                  |
| ------------ | -------------- | ------------------------- |
| `basic`      | 2-5 seconds    | Quick graph overview      |
| `topology`   | <1 second      | Degree distribution only  |
| `centrality` | 1-3 minutes    | Node importance analysis  |
| `essential`  | 1-5 minutes    | Standard network analysis |
| `moderate`   | 5-15 minutes   | Comprehensive analysis    |
| `all`        | 15-60+ minutes | Complete metrics suite    |

Times vary based on graph size:

* Small (<1K nodes): Lower estimates
* Medium (1-10K nodes): Middle estimates
* Large (>10K nodes): Upper estimates

## What Gets Computed

**120+ metrics across 15 categories:**

### 1. Basic Topology (4 metrics)

* in_degree, out_degree, total_degree
* degree_imbalance

### 2. Centrality (20+ metrics)

* Degree: in_degree_centrality, out_degree_centrality
* Closeness: closeness_centrality
* Betweenness: betweenness_centrality
* Eigenvector: eigenvector_centrality
* Katz: katz_centrality
* PageRank: pagerank
* HITS: hub_score, authority_score
* Harmonic: harmonic_centrality
* Load: load_centrality
* Subgraph: subgraph_centrality
* Second order: second_order_centrality
* Percolation: percolation_centrality
* Trophic: trophic_level
* Current flow, Information, Communicability, VoteRank, Edge betweenness

### 3. Clustering (6 metrics)

* clustering_coefficient (directed & undirected)
* triangle_count (directed & undirected)
* square_clustering
* local_transitivity

### 4. Community (5 metrics)

* community_id (Louvain)
* community_size
* core_number (k-core)
* onion_layer
* local_reaching_centrality

### 5. Path Metrics (10 metrics)

* avg_shortest_path, median_shortest_path, max_shortest_path
* path_variance, path_sum
* reachable_nodes
* paths_length_1, paths_length_2_targets
* eccentricity
* wiener_contribution (small graphs only)

### 6. Distance Measures (4 metrics)

* graph_radius, graph_diameter
* is_center, is_periphery

### 7. Structural (12 metrics)

* constraint, effective_size, redundancy (Burt's structural holes)
* is_articulation_point, bridge_count
* avg_neighbor_degree, min_neighbor_degree, max_neighbor_degree, std_neighbor_degree
* avg_neighbor_degree_directed, avg_neighbor_degree_undirected

### 8. Reciprocity (5 metrics)

* mutual_count, mutual_ratio, mutual_received_ratio
* one_way_out, one_way_in

### 9. Reach (8 metrics)

* reach_hop_1 through reach_hop_6
* total_reach
* network_penetration

### 10. Components (3 metrics)

* weak_component_size, strong_component_size
* in_largest_component

### 11-15. Additional Categories

* Vitality: closeness_vitality
* Dispersion: avg_dispersion, max_dispersion
* Efficiency: local_efficiency
* Flow: flow_hierarchy
* Dominance: dominated_nodes_count, dominance_ratio

## Output

CSV file with one row per node:

```csv
avatar,in_degree,pagerank,betweenness_centrality,...
0x123...,45,0.0023,0.0145,...
```

All metrics are numeric (booleans converted to 0/1, NaN filled with 0).

## Parallel Processing

The calculator uses parallel processing for expensive operations:

* Path computations
* Reach analysis
* Dominance metrics
* Local reaching centrality

Configure workers in `.env`:

```env
N_JOBS=4  # Use 4 CPU cores
N_JOBS=0  # Use all available cores minus 1 (default)
```

## Files

```text
graph_metrics.py           # Main script with selective metrics
compare_blacklist_csvs.py  # Analyze & merge blacklist CSVs with existing DB
update_blacklist.py        # Apply CSV changes to SQLite Blacklist / Whitelist
requirements.txt           # Dependencies
.env.example               # Configuration template
README.md                  # This file
.gitignore                 # Protects sensitive files
```

## Requirements

* Python 3.8+
* PostgreSQL database with trust relations
* 4GB+ RAM for large graphs
* NetworkX 3.0+

## Command Line Options

```bash
# Show help and available metrics modes (graph analysis)
python graph_metrics.py --help
python graph_metrics.py -h

# Show help for blacklist / whitelist comparison
python compare_blacklist_csvs.py --help

# Show help for applying CSV updates to blacklist/whitelist
python update_blacklist.py --help
```

## Customization Tips

### Creating Custom Metric Sets

Edit the script to define your own presets:

```python
METRIC_PRESETS = {
    'basic': ['topology', 'clustering'],
    'my_custom': ['topology', 'centrality', 'reciprocity'],  # Add your own
    # ...
}
```

### Adjusting Performance

For very large graphs, consider:

1. Use `basic` or `essential` mode first
2. Run expensive metrics separately
3. Increase `N_JOBS` for more parallelism
4. Comment out metrics with size restrictions in the code

**Want to see what modes are available?**

```bash
python graph_metrics.py --help
```

