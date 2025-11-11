# Graph Analyzer

Computes many graph metrics for directed trust networks using NetworkX.

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
# Edit .env with your database credentials

# 4. Run
python graph_metrics.py
```

## Configuration

Create `.env` file:
```env
DB_USER=your_username
DB_PASSWORD=your_password
DB_HOST=your_host
DB_NAME=your_database
OUTPUT_FILE=graph_metrics.csv
```

## What Gets Computed

**93+ metrics across 15 categories:**

### 1. Basic Topology (4 metrics)
- `in_degree` - Number of incoming edges
- `out_degree` - Number of outgoing edges
- `total_degree` - Sum of in and out degree
- `degree_imbalance` - Normalized difference between in and out degree

### 2. Centrality Measures (30 metrics)
**Degree Centrality:**
- `in_degree_centrality` - Normalized in-degree
- `out_degree_centrality` - Normalized out-degree
- `degree_centrality_undirected` - Undirected degree centrality

**Closeness Centrality:**
- `closeness_centrality` - Average distance to all other nodes (outgoing)
- `closeness_centrality_in` - Average distance to all other nodes (incoming)
- `closeness_centrality_undirected` - Closeness in undirected graph

**Betweenness Centrality:**
- `betweenness_centrality` - Fraction of shortest paths passing through node (directed)
- `betweenness_centrality_undirected` - Betweenness in undirected graph

**Eigenvector Centrality:**
- `eigenvector_centrality` - Influence based on connected neighbors (directed)
- `eigenvector_centrality_undirected` - Eigenvector in undirected graph

**Katz Centrality:**
- `katz_centrality` - Influence with attenuation factor (directed)
- `katz_centrality_undirected` - Katz in undirected graph

**PageRank:**
- `pagerank` - Google's PageRank algorithm (directed)
- `pagerank_undirected` - PageRank in undirected graph

**HITS Algorithm:**
- `hub_score` - Quality as a hub (pointing to good authorities)
- `authority_score` - Quality as an authority (pointed to by good hubs)

**Other Centrality Measures:**
- `harmonic_centrality` - Sum of inverse distances (directed)
- `harmonic_centrality_undirected` - Harmonic in undirected graph
- `load_centrality` - Fraction of shortest paths through node with endpoints
- `load_centrality_undirected` - Load in undirected graph
- `subgraph_centrality` - Sum of closed walks weighted by inverse factorials
- `second_order_centrality` - Sum of inverse eigenvalues of neighbors (n<1000)
- `percolation_centrality` - Importance in percolation processes
- `trophic_level` - Position in directed food web hierarchy
- `current_flow_betweenness` - Betweenness using current flow (n<1000, connected)
- `current_flow_closeness` - Closeness using current flow (n<1000, connected)
- `information_centrality` - Harmonic mean of resistance distances (n<1000, connected)
- `communicability_betweenness` - Betweenness via matrix exponential (n<500)
- `voterank` - Voting-based node ranking
- `edge_betweenness_sum` - Sum of edge betweenness for edges incident to node

### 3. Clustering (6 metrics)
- `clustering_coefficient` - Fraction of triangles around node (undirected)
- `clustering_coefficient_directed` - Clustering in directed graph
- `triangle_count` - Number of triangles node participates in (undirected)
- `triangle_count_directed` - Triangles in directed graph
- `square_clustering` - Fraction of squares (4-cycles) around node
- `local_transitivity` - Same as clustering coefficient

### 4. Community Detection (5 metrics)
- `community_id` - Community assignment from Louvain algorithm
- `community_size` - Size of node's community
- `core_number` - k-core number (largest k where node is in k-core)
- `onion_layer` - Layer in onion decomposition
- `local_reaching_centrality` - Proportion of nodes reachable via neighbors

### 5. Path Metrics (10 metrics)
- `avg_shortest_path` - Average shortest path length from node
- `median_shortest_path` - Median shortest path length from node
- `max_shortest_path` - Maximum shortest path length from node
- `path_variance` - Variance in shortest path lengths from node
- `path_sum` - Sum of all shortest path lengths from node
- `reachable_nodes` - Number of nodes reachable from node
- `paths_length_1` - Number of direct connections (out-degree)
- `paths_length_2_targets` - Number of unique 2-hop targets
- `eccentricity` - Maximum distance to any other node
- `wiener_contribution` - Node's contribution to Wiener index (n<500)

### 6. Distance Measures (4 metrics)
- `graph_radius` - Minimum eccentricity in graph
- `graph_diameter` - Maximum eccentricity in graph
- `is_center` - Whether node is in graph center (1/0)
- `is_periphery` - Whether node is in graph periphery (1/0)

### 7. Structural Properties (11 metrics)
**Structural Holes (Burt):**
- `constraint` - Burt's constraint measure (lack of holes)
- `effective_size` - Effective network size (non-redundant contacts)
- `redundancy` - Network redundancy (1 - effective_size/degree)

**Network Robustness:**
- `is_articulation_point` - Whether removal disconnects graph (1/0)
- `bridge_count` - Number of bridges incident to node

**Neighbor Properties:**
- `avg_neighbor_degree` - Average degree of neighbors
- `min_neighbor_degree` - Minimum neighbor degree
- `max_neighbor_degree` - Maximum neighbor degree
- `std_neighbor_degree` - Standard deviation of neighbor degrees
- `avg_neighbor_degree_undirected` - NetworkX's undirected version
- `avg_neighbor_degree_directed` - NetworkX's directed version

### 8. Reciprocity (5 metrics)
- `mutual_count` - Number of reciprocated connections
- `mutual_ratio` - Fraction of outgoing edges that are reciprocated
- `mutual_received_ratio` - Fraction of incoming edges that are reciprocated
- `one_way_out` - Number of non-reciprocated outgoing edges
- `one_way_in` - Number of non-reciprocated incoming edges

### 9. Reach Metrics (8 metrics)
- `reach_hop_1` - Nodes reachable in exactly 1 hop
- `reach_hop_2` - New nodes reachable in exactly 2 hops
- `reach_hop_3` - New nodes reachable in exactly 3 hops
- `reach_hop_4` - New nodes reachable in exactly 4 hops
- `reach_hop_5` - New nodes reachable in exactly 5 hops
- `reach_hop_6` - New nodes reachable in exactly 6 hops
- `total_reach` - Total nodes reachable within 6 hops
- `network_penetration` - Fraction of network reachable within 6 hops

### 10. Component Membership (3 metrics)
- `weak_component_size` - Size of weakly connected component
- `strong_component_size` - Size of strongly connected component
- `in_largest_component` - Whether in largest weak component (1/0)

### 11. Vitality (1 metric)
- `closeness_vitality` - Change in sum of distances if node removed (n<500, connected)

### 12. Dispersion (2 metrics)
- `avg_dispersion` - Average dispersion to connected nodes (sampled)
- `max_dispersion` - Maximum dispersion to connected nodes (sampled)

### 13. Efficiency (1 metric)
- `local_efficiency` - Efficiency of node's neighborhood (constant for all nodes)

### 14. Flow (1 metric)
- `flow_hierarchy` - Hierarchical organization measure (constant for all nodes)

### 15. Dominance (2 metrics)
- `dominated_nodes_count` - Number of nodes reachable via directed paths
- `dominance_ratio` - Fraction of graph dominated by node

**Total: 93+ metrics** (some metrics are conditional on graph size and connectivity)

## Output

CSV file with one row per node:
```
avatar,in_degree,pagerank,betweenness_centrality,...
0x123...,45,0.0023,0.0145,...
```

All metrics are numeric (booleans converted to 0/1, NaN filled with 0).

## Performance

- Small graphs (<1K nodes): 2-5 minutes
- Medium graphs (1-10K nodes): 5-20 minutes
- Large graphs (>10K nodes): 20-60 minutes

## Files

```
graph_metrics.py     # Main script
requirements.txt     # Dependencies
.env.example         # Config template
.gitignore          # Protects sensitive files
README.md           # This file
```

## NetworkX Algorithms Used

- `networkx.algorithms.centrality.*` - All centrality measures
- `networkx.algorithms.distance_measures.*` - Radius, diameter, eccentricity
- `networkx.algorithms.community.*` - Louvain community detection
- `networkx.algorithms.core.*` - k-core decomposition
- `networkx.algorithms.components.*` - Connected components
- `networkx.algorithms.cluster.*` - Clustering coefficients
- `networkx.algorithms.shortest_paths.*` - Path computations

## Requirements

- Python 3.8+
- PostgreSQL database with trust relations
- 4GB+ RAM for large graphs

## Customization

To compute only specific metrics, comment out method calls in `compute_all()`:

```python
def compute_all(self):
    # ...
    self._topology(metrics)
    self._centrality(metrics)
    # self._clustering(metrics)  # Skip this
    # ...
```

## Virtual Environment Tips

```bash
# Activate
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Deactivate
deactivate

# Delete and recreate
rm -rf venv && python3 -m venv venv
```

## Troubleshooting

**Error: Missing DB credentials**
- Make sure `.env` file exists with all required variables

**Error: Module not found**
- Activate venv: `source venv/bin/activate`
- Install deps: `pip install -r requirements.txt`

**Script is slow**
- Comment out expensive metrics (subgraph_centrality, second_order_centrality)
- Reduce max_hops in `_reach()` method

**Memory error**
- Close other applications
- Use a machine with more RAM
- Process graph in batches

**Katz centrality warnings/failures**
- Katz centrality can fail on large or dense graphs due to numerical overflow
- The script automatically retries with smaller alpha values (0.5λ, 0.05λ, 0.0001)
- If all attempts fail, Katz metrics will be skipped but other metrics continue
- This is expected behavior for certain graph structures

