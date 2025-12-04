# Graph Metrics Documentation

Graph Analyzer computes 120+ graph metrics using NetworkX for directed networks. Metrics are organized into categories that can be selectively computed based on analysis needs.

## Metric Categories

| Category | Description | Metrics Count | Computation Cost |
|----------|-------------|---------------|------------------|
| topology | Basic degree metrics | ~8 | Low |
| centrality | Importance measures | ~12 | Medium |
| clustering | Local connectivity | ~6 | Medium |
| community | Group detection | ~4 | Medium-High |
| paths | Shortest path analysis | ~8 | High |
| distances | Network distances | ~6 | High |
| structural | Structural holes, bridges | ~10 | Medium |
| reciprocity | Mutual connections | ~4 | Low |
| reach | N-hop reachability | ~6 | Medium |
| components | Component membership | ~4 | Low |
| vitality | Removal impact | ~2 | Very High |
| dispersion | Spread patterns | ~2 | High |
| efficiency | Communication efficiency | ~2 | High |
| flow | Flow hierarchy | ~2 | Medium |
| dominance | Dominance patterns | ~4 | Medium |

## Computation Modes

### Presets

| Preset | Categories Included | Use Case |
|--------|---------------------|----------|
| `basic` | topology, community | Quick overview |
| `essential` | topology, centrality, clustering, community | Standard analysis |
| `moderate` | essential + paths, structural | Detailed analysis |
| `all` | All categories | Comprehensive (slow) |

### Usage

```python
# In API request
{
  "metrics_mode": "essential"  # or "basic", "moderate", "all"
}
```

---

## Metric Reference

### Topology Category

Basic structural metrics computed from the graph's degree distribution.

| Metric | Type | Description | Range |
|--------|------|-------------|-------|
| `in_degree` | int | Number of incoming edges | [0, ∞) |
| `out_degree` | int | Number of outgoing edges | [0, ∞) |
| `total_degree` | int | Sum of in and out degree | [0, ∞) |
| `degree_ratio` | float | in_degree / out_degree | [0, ∞) |
| `degree_imbalance` | float | (in - out) / total | [-1, 1] |
| `is_source` | bool | Has only outgoing edges | 0 or 1 |
| `is_sink` | bool | Has only incoming edges | 0 or 1 |
| `is_isolated` | bool | Has no edges | 0 or 1 |

**Interpretation**:
- **High in_degree**: Popular/trusted node
- **High out_degree**: Active/trusting node
- **degree_imbalance > 0**: More incoming than outgoing (receiver)
- **degree_imbalance < 0**: More outgoing than incoming (giver)

---

### Centrality Category

Measures of node importance in the network.

| Metric | Type | Description | Range |
|--------|------|-------------|-------|
| `pagerank` | float | Google PageRank score | [0, 1] |
| `in_degree_centrality` | float | Normalized in-degree | [0, 1] |
| `out_degree_centrality` | float | Normalized out-degree | [0, 1] |
| `eigenvector_centrality` | float | Influence based on neighbor importance | [0, 1] |
| `katz_centrality` | float | Generalized eigenvector with attenuation | [0, 1] |
| `betweenness_centrality` | float | Fraction of shortest paths through node | [0, 1] |
| `closeness_centrality` | float | Inverse mean distance to all nodes | [0, 1] |
| `harmonic_centrality` | float | Sum of inverse distances | [0, ∞) |
| `load_centrality` | float | Traffic flow through node | [0, 1] |
| `hub_score` | float | HITS hub score (links to authorities) | [0, 1] |
| `authority_score` | float | HITS authority score (linked by hubs) | [0, 1] |
| `voterank` | int | VoteRank influence measure | [0, ∞) |

**Key Centralities**:

**PageRank**: Probability of random walker landing on node. Higher = more "important" based on incoming links from important nodes.

**Betweenness**: Nodes that bridge different parts of network. High betweenness = potential bottleneck or broker.

**Eigenvector**: Importance inherited from neighbors. Connected to important nodes = important.

**Closeness**: How quickly can node reach all others. High closeness = central position.

---

### Clustering Category

Measures of local connectivity and triangle formation.

| Metric | Type | Description | Range |
|--------|------|-------------|-------|
| `clustering_coefficient` | float | Fraction of possible triangles | [0, 1] |
| `triangles` | int | Number of triangles involving node | [0, ∞) |
| `squares` | int | Number of squares involving node | [0, ∞) |
| `local_reaching_centrality` | float | Proportion of network reachable | [0, 1] |
| `generalized_degree` | int | Number of triangles + degree | [0, ∞) |
| `constraint` | float | Burt's constraint (lack of structural holes) | [0, 1] |

**Interpretation**:
- **High clustering**: Node's neighbors also connected (tight-knit group)
- **Low clustering**: Node bridges different groups
- **High constraint**: Few options for brokerage (embedded in single cluster)

---

### Community Category

Community and core structure detection.

| Metric | Type | Description | Range |
|--------|------|-------------|-------|
| `community_id` | int | Louvain community assignment | [0, n_communities) |
| `core_number` | int | K-core membership | [0, max_core] |
| `onion_layer` | int | Onion decomposition layer | [0, max_layer] |
| `community_size` | int | Size of node's community | [1, n] |

**K-Core**: Maximal subgraph where all nodes have degree ≥ k. Higher core number = more densely connected.

**Onion Layer**: Extension of k-core showing distance from network periphery.

---

### Paths Category

Shortest path and reachability metrics.

| Metric | Type | Description | Range |
|--------|------|-------------|-------|
| `avg_shortest_path` | float | Mean shortest path to reachable nodes | [0, ∞) |
| `median_shortest_path` | float | Median shortest path | [0, ∞) |
| `max_shortest_path` | float | Maximum shortest path (eccentricity) | [0, ∞) |
| `reachable_nodes` | int | Number of reachable nodes | [0, n-1] |
| `reaching_nodes` | int | Number of nodes that can reach this node | [0, n-1] |
| `reachability_ratio` | float | Fraction of network reachable | [0, 1] |
| `is_central` | bool | In the center (minimum eccentricity) | 0 or 1 |
| `is_peripheral` | bool | On the periphery (maximum eccentricity) | 0 or 1 |

**Note**: Path metrics are expensive to compute for large graphs.

---

### Structural Category

Structural holes and neighbor statistics.

| Metric | Type | Description | Range |
|--------|------|-------------|-------|
| `effective_size` | float | Non-redundant contacts | [0, degree] |
| `efficiency` | float | effective_size / degree | [0, 1] |
| `constraint` | float | Burt's network constraint | [0, 1] |
| `hierarchy` | float | Concentration of constraint | [0, 1] |
| `neighbor_avg_in_degree` | float | Mean in-degree of neighbors | [0, ∞) |
| `neighbor_avg_out_degree` | float | Mean out-degree of neighbors | [0, ∞) |
| `neighbor_degree_correlation` | float | Assortativity with neighbors | [-1, 1] |
| `is_articulation_point` | bool | Removal disconnects graph | 0 or 1 |
| `is_bridge_node` | bool | Has bridge edges | 0 or 1 |
| `bridge_edge_count` | int | Number of bridge edges | [0, degree] |

**Structural Holes**:
- **High effective_size**: Many non-redundant contacts
- **Low constraint**: Access to diverse information
- **Good for brokerage** opportunities

---

### Reciprocity Category

Mutual connection patterns.

| Metric | Type | Description | Range |
|--------|------|-------------|-------|
| `mutual_in` | int | Edges that are reciprocated | [0, min(in,out)] |
| `one_way_in` | int | Incoming edges not reciprocated | [0, in_degree] |
| `one_way_out` | int | Outgoing edges not reciprocated | [0, out_degree] |
| `reciprocity_ratio` | float | Fraction of edges reciprocated | [0, 1] |

**Interpretation**:
- **High reciprocity**: Mutual relationships
- **Low reciprocity**: Asymmetric relationships

---

### Reach Category

N-hop neighborhood statistics.

| Metric | Type | Description | Range |
|--------|------|-------------|-------|
| `reach_1_hop` | int | Nodes reachable in 1 hop | [0, n-1] |
| `reach_2_hop` | int | Nodes reachable in 2 hops | [0, n-1] |
| `reach_3_hop` | int | Nodes reachable in 3 hops | [0, n-1] |
| `network_penetration_1` | float | 1-hop reach / total nodes | [0, 1] |
| `network_penetration_2` | float | 2-hop reach / total nodes | [0, 1] |
| `network_penetration_3` | float | 3-hop reach / total nodes | [0, 1] |

**Use Cases**:
- Information spread potential
- Influence reach
- Network coverage

---

### Components Category

Connected component membership.

| Metric | Type | Description | Range |
|--------|------|-------------|-------|
| `weakly_connected_component_id` | int | WCC identifier | [0, n_wcc) |
| `weakly_connected_component_size` | int | Size of WCC | [1, n] |
| `strongly_connected_component_id` | int | SCC identifier | [0, n_scc) |
| `strongly_connected_component_size` | int | Size of SCC | [1, n] |

**WCC vs SCC**:
- **Weakly Connected**: Ignores edge direction
- **Strongly Connected**: Respects edge direction (all nodes mutually reachable)

---

### Vitality Category

Impact of node removal.

| Metric | Type | Description | Range |
|--------|------|-------------|-------|
| `closeness_vitality` | float | Change in closeness if removed | (-∞, ∞) |
| `betweenness_vitality` | float | Change in betweenness if removed | (-∞, ∞) |

**Warning**: Very expensive to compute (requires recomputation for each node).

---

### Efficiency Category

Communication efficiency metrics.

| Metric | Type | Description | Range |
|--------|------|-------------|-------|
| `local_efficiency` | float | Efficiency of immediate neighborhood | [0, 1] |
| `global_efficiency_contribution` | float | Node's contribution to global efficiency | [0, 1] |

---

### Flow Category

Flow and hierarchy metrics.

| Metric | Type | Description | Range |
|--------|------|-------------|-------|
| `flow_hierarchy` | float | Hierarchical flow structure | [0, 1] |
| `flow_centrality` | float | Role in flow patterns | [0, 1] |

---

### Dominance Category

Dominance relationship patterns.

| Metric | Type | Description | Range |
|--------|------|-------------|-------|
| `immediate_dominators` | int | Nodes that dominate this node | [0, n-1] |
| `dominated_nodes` | int | Nodes dominated by this node | [0, n-1] |
| `dominance_ratio` | float | dominated / (dominated + dominators) | [0, 1] |
| `dominance_frontier_size` | int | Size of dominance frontier | [0, n-1] |

---

## Performance Considerations

### Computation Times (approximate)

| Category | 1K nodes | 10K nodes | 100K nodes |
|----------|----------|-----------|------------|
| topology | < 1s | < 1s | 1-2s |
| centrality | 1-2s | 10-30s | 2-5 min |
| clustering | < 1s | 5-10s | 1-2 min |
| community | 1-2s | 5-15s | 1-3 min |
| paths | 2-5s | 30s-2min | 10-30 min |
| structural | 1-2s | 10-30s | 2-5 min |
| vitality | 5-10s | 5-15 min | hours |

### Recommendations

```
Small graphs (< 5K nodes):    Use "all" preset
Medium graphs (5K-50K nodes): Use "moderate" preset
Large graphs (50K-500K nodes): Use "essential" preset
Very large graphs (> 500K):    Use "basic" preset
```

### Parallel Processing

Graph metrics computation uses parallel processing:

```bash
# Environment variable
N_JOBS=-1  # Use all CPU cores
N_JOBS=4   # Use 4 cores
N_JOBS=1   # Single-threaded
```

---

## Common Metric Patterns

### Bot Detection

```
High indicators:
- Very high or very low in_degree/out_degree ratio
- Low clustering_coefficient (not part of organic communities)
- High degree with low reciprocity
- Unusual community assignment

Example filter:
out_degree > 100 AND reciprocity_ratio < 0.1 AND clustering_coefficient < 0.01
```

### Influential Nodes

```
High indicators:
- High pagerank
- High betweenness_centrality
- High eigenvector_centrality
- Low constraint (structural holes)

Example filter:
pagerank > 0.001 AND betweenness_centrality > 0.01
```

### Bridge Nodes

```
High indicators:
- High betweenness_centrality
- Low clustering_coefficient
- is_articulation_point = true
- Low constraint, high effective_size

Example filter:
is_articulation_point = 1 OR (betweenness_centrality > 0.05 AND clustering_coefficient < 0.1)
```

### Peripheral Nodes

```
High indicators:
- core_number = 1 (lowest k-core)
- High onion_layer
- Low reachability_ratio
- is_peripheral = true

Example filter:
core_number <= 2 AND reachability_ratio < 0.1
```

---

## API Usage

### Computing Metrics

```bash
# Compute essential metrics
curl -X POST "http://localhost:8000/api/metrics/run" \
  -H "Content-Type: application/json" \
  -d '{"categories": ["topology", "centrality", "clustering"]}'

# Compute using preset
curl -X POST "http://localhost:8000/api/metrics/run" \
  -H "Content-Type: application/json" \
  -d '{"preset": "essential"}'
```

### Response Format

```json
{
  "metrics_computed": [
    "in_degree", "out_degree", "pagerank", "clustering_coefficient", ...
  ],
  "node_count": 10000,
  "computation_time": 15.3,
  "categories_run": ["topology", "centrality", "clustering", "community"]
}
```

---

## Adding Custom Metrics

Custom metrics can be added to `graph_metrics.py`:

```python
# In GraphMetrics class
def compute_custom_metric(self, G):
    """Compute custom metric for all nodes."""
    results = {}
    for node in G.nodes():
        # Your computation here
        results[node] = value
    return results

# Add to appropriate category
METRIC_CATEGORIES['custom'] = 'Custom Metrics (description)'
```