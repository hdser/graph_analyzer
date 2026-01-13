# Graph Metrics Documentation

Graph Analyzer computes **150+ graph metrics** using NetworkX, igraph, SciPy, and custom algorithms for directed and undirected networks. Metrics are organized into **25 categories** that can be selectively computed based on analysis needs.

## Table of Contents

1. [Metric Categories Overview](#metric-categories-overview)
2. [Computation Presets](#computation-presets)
3. [Metric Reference by Category](#metric-reference-by-category)
4. [Performance Considerations](#performance-considerations)
5. [API Usage](#api-usage)
6. [Common Metric Patterns](#common-metric-patterns)

---

## Metric Categories Overview

| Category | Description | Metrics Count | Computation Cost |
|----------|-------------|---------------|------------------|
| topology | Basic degree metrics | 4 | Low |
| centrality | Importance measures | 22 | Low-Very High |
| clustering | Local connectivity | 6 | Medium |
| community | Group detection | 9 | Low-High |
| paths | Shortest path analysis | 4 | High |
| distances | Network distances | 1 | High |
| structural | Structural holes, bridges | 5 | Medium |
| reciprocity | Mutual connections | 1 | Low |
| reach | N-hop reachability | 1 | High |
| components | Component membership | 5 | Low-Medium |
| vitality | Removal impact | 1 | Very High |
| dispersion | Spread patterns | 1 | Very High |
| efficiency | Communication efficiency | 5 | High-Very High |
| flow | Flow hierarchy, cycles | 6 | Medium-Very High |
| dominance | Dominance patterns | 1 | High |
| trust | Trust network algorithms | 2 | Medium |
| similarity | Node similarity measures | 4 | High |
| link_prediction | Link prediction metrics | 3 | Medium-High |
| robustness | Network resilience | 3 | High |
| spectral | Spectral graph theory | 2 | High |
| influence | Influence maximization | 2 | Medium |
| bipartite | Bipartite graph metrics | 1 | Medium |
| graph_coloring | Graph coloring analysis | 1 | Low |
| igraph_community | igraph community detection | 5 | Medium-High |
| igraph_centrality | igraph centrality algorithms | 1 | Medium |
| motifs | Network motif analysis | 1 | High |

---

## Computation Presets

Presets provide predefined metric combinations for common use cases.

### Standard Presets

| Preset | Description | Use Case |
|--------|-------------|----------|
| `basic` | Quick overview (topology + community basics) | Initial exploration |
| `essential` | Standard analysis (topology, clustering, key centralities) | General analysis |
| `moderate` | Detailed analysis (essential + paths, structural) | In-depth study |
| `comprehensive` | Full analysis excluding very expensive metrics | Thorough analysis |
| `all` | All available metrics | Complete analysis (slow) |

### Specialized Presets

| Preset | Description | Categories/Metrics Included |
|--------|-------------|----------------------------|
| `trust_analysis` | Trust network focused | pagerank, eigentrust, appleseed, reciprocity |
| `influence` | Node influence and importance | pagerank, betweenness, katz, hits, voterank, collective_influence |
| `structure` | Network structure analysis | components, condensation, articulation points |
| `community_detection` | Compare community methods | louvain, label_propagation, greedy_modularity |
| `robustness_analysis` | Network resilience | node/edge connectivity, resilience_score |
| `link_prediction` | Link prediction focused | common_neighbors, preferential_attachment, similarity metrics |
| `hierarchy` | Network hierarchy analysis | flow_hierarchy, hierarchy_level, trophic_level |
| `igraph_advanced` | Advanced igraph algorithms | leiden, infomap, walktrap, alpha_centrality, motifs |
| `spectral` | Spectral graph analysis | fiedler_vector, spectral_centrality, laplacian_centrality |

### Usage

```python
# In API request
{
  "metrics_mode": "essential"  # or any preset name
}

# Or specify categories
{
  "categories": ["topology", "centrality", "clustering"]
}

# Or specific metrics
{
  "metrics": ["pagerank", "betweenness_centrality", "louvain_community"]
}
```

---

## Metric Reference by Category

### Topology Category

Basic structural metrics computed from the graph's degree distribution.

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `in_degree` | in_degree | Number of incoming edges | [0, ∞) | Low |
| `out_degree` | out_degree | Number of outgoing edges | [0, ∞) | Low |
| `total_degree` | total_degree | Sum of in and out degree | [0, ∞) | Low |
| `degree_imbalance` | degree_imbalance | (in - out) / total | [-1, 1] | Low |

**Interpretation**:
- **High in_degree**: Popular/trusted node (receiver)
- **High out_degree**: Active/trusting node (giver)
- **degree_imbalance > 0**: More incoming than outgoing
- **degree_imbalance < 0**: More outgoing than incoming

---

### Centrality Category

Measures of node importance in the network. This is the largest category with 22 metrics.

#### Core Centrality Metrics

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `degree_centrality` | in_degree_centrality, out_degree_centrality, degree_centrality_undirected | Normalized degree | [0, 1] | Low |
| `closeness_centrality` | closeness_centrality, closeness_centrality_in, closeness_centrality_undirected | Inverse mean distance | [0, 1] | Medium |
| `betweenness_centrality` | betweenness_centrality, betweenness_centrality_undirected | Fraction of shortest paths through node | [0, 1] | High |
| `eigenvector_centrality` | eigenvector_centrality, eigenvector_centrality_undirected | Influence from neighbor importance | [0, 1] | Medium |
| `katz_centrality` | katz_centrality, katz_centrality_undirected | Generalized eigenvector with attenuation | [0, 1] | Medium |
| `pagerank` | pagerank, pagerank_undirected | Google PageRank score | [0, 1] | Low |
| `hits` | hub_score, authority_score | HITS hub and authority scores | [0, 1] | Medium |

**Parameters for PageRank**:
- `alpha` (float, default=0.85): Damping parameter
- `max_iter` (int, default=100): Maximum iterations
- `tol` (float, default=1e-6): Convergence tolerance

**Parameters for Betweenness**:
- `normalized` (bool, default=True): Normalize values
- `endpoints` (bool, default=False): Include endpoints in counts

#### Extended Centrality Metrics

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `harmonic_centrality` | harmonic_centrality, harmonic_centrality_undirected | Sum of inverse distances | [0, ∞) | Medium |
| `load_centrality` | load_centrality, load_centrality_undirected | Traffic flow through node | [0, 1] | High |
| `subgraph_centrality` | subgraph_centrality | Based on closed walks | [0, ∞) | Medium |
| `second_order_centrality` | second_order_centrality | Random walk variance | [0, ∞) | High |
| `percolation_centrality` | percolation_centrality | Percolation with random states | [0, 1] | Medium |
| `trophic_level` | trophic_level | Hierarchical level in directed graph | [1, ∞) | Medium |
| `voterank` | voterank | VoteRank influence measure | [0, ∞) | Medium |
| `edge_betweenness_sum` | edge_betweenness_sum | Sum of incident edge betweenness | [0, ∞) | High |

#### Advanced Centrality Metrics

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `current_flow_centrality` | current_flow_betweenness, current_flow_closeness | Electric current flow model | [0, 1] | Very High |
| `information_centrality` | information_centrality | Information flow efficiency | [0, 1] | Very High |
| `communicability_betweenness` | communicability_betweenness | Based on graph communicability | [0, 1] | Very High |
| `laplacian_centrality` | laplacian_centrality | Based on Laplacian energy drop | [0, 1] | Medium |
| `leverage_centrality` | leverage_centrality | Degree advantage over neighbors | [-1, 1] | Low |
| `semi_local_centrality` | semi_local_centrality | 2-hop neighborhood influence | [0, ∞) | Medium |
| `decay_centrality` | decay_centrality | Distance-weighted influence | [0, ∞) | High |

**Parameters for Decay Centrality**:
- `delta` (float, default=0.5): Decay factor (0.1-0.9)

**Key Centrality Interpretations**:

- **PageRank**: Probability of random walker landing on node. Higher = more "important" based on incoming links from important nodes.
- **Betweenness**: Nodes that bridge different parts of network. High = potential bottleneck or broker.
- **Eigenvector**: Importance inherited from neighbors. Connected to important nodes = important.
- **Closeness**: How quickly node can reach all others. High = central position.
- **Katz**: Like eigenvector but accounts for all walks with attenuation.
- **Leverage**: Positive = more connected than neighbors, Negative = less connected.

---

### Clustering Category

Measures of local connectivity and triangle formation.

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `clustering_coefficient` | clustering_coefficient, clustering_coefficient_directed | Fraction of possible triangles | [0, 1] | Medium |
| `triangles` | triangle_count, triangle_count_directed | Number of triangles involving node | [0, ∞) | Medium |
| `square_clustering` | square_clustering | Square clustering coefficient | [0, 1] | Medium |
| `local_transitivity` | local_transitivity | Same as clustering coefficient | [0, 1] | Medium |
| `clique_count` | clique_count, max_clique_size | Maximal cliques containing node | [0, ∞) | High |
| `average_neighbor_clustering` | average_neighbor_clustering | Average clustering of neighbors | [0, 1] | Medium |

**Interpretation**:
- **High clustering**: Node's neighbors also connected (tight-knit group)
- **Low clustering**: Node bridges different groups
- **High clique_count**: Participates in many dense subgraphs

---

### Community Category

Community detection and core structure.

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `louvain_community` | community_id, community_size | Louvain community detection | [0, n_communities) | Medium |
| `core_number` | core_number | K-core membership | [0, max_core] | Low |
| `onion_layer` | onion_layer | Onion decomposition layer | [0, max_layer] | Medium |
| `local_reaching_centrality` | local_reaching_centrality | Proportion of network reachable | [0, 1] | High |
| `label_propagation` | lp_community_id, lp_community_size | Semi-synchronous label propagation | [0, n_communities) | Low |
| `async_label_propagation` | async_lp_community_id, async_lp_community_size | Asynchronous label propagation | [0, n_communities) | Low |
| `greedy_modularity_community` | gm_community_id, gm_community_size, graph_modularity | Clauset-Newman-Moore greedy | [0, n_communities) | Medium |
| `participation_coefficient` | participation_coefficient | Inter-community connectivity | [0, 1] | Medium |
| `within_module_degree` | within_module_degree_z | Z-score of within-community degree | (-∞, ∞) | Medium |

**Parameters for Louvain**:
- `resolution` (float, default=1.0): Resolution parameter (0.1-2.0)
- `seed` (int, default=42): Random seed

**K-Core Interpretation**: Maximal subgraph where all nodes have degree ≥ k. Higher core number = more densely connected.

**Participation Coefficient**: 0 = all connections within community, 1 = evenly distributed across communities.

---

### Paths Category

Shortest path and reachability metrics.

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `shortest_paths` | avg_shortest_path, median_shortest_path, max_shortest_path, path_variance, path_sum, reachable_nodes | Path statistics | [0, ∞) | High |
| `hop_paths` | paths_length_1, paths_length_2_targets | Direct and 2-hop path counts | [0, ∞) | Medium |
| `eccentricity` | eccentricity | Maximum distance to any node | [0, ∞) | High |
| `wiener_contribution` | wiener_contribution | Contribution to Wiener index | [0, ∞) | High |

**Note**: Path metrics are expensive to compute for large graphs.

---

### Distances Category

Network distance measures.

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `graph_distances` | graph_radius, graph_diameter, is_center, is_periphery | Graph radius, diameter, center/periphery membership | Various | High |

---

### Structural Category

Structural holes and robustness indicators.

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `structural_holes` | constraint, effective_size, redundancy | Burt's structural holes | [0, 1] | Medium |
| `articulation_points` | is_articulation_point | Removal disconnects graph | 0 or 1 | Medium |
| `bridges` | bridge_count | Number of bridge edges | [0, degree] | Medium |
| `neighbor_degree_stats` | avg_neighbor_degree, min_neighbor_degree, max_neighbor_degree, std_neighbor_degree | Statistics of neighbor degrees | [0, ∞) | Medium |
| `biconnected_component` | biconnected_component_id, biconnected_component_size | Biconnected component membership | [0, n_bcc) | Medium |

**Structural Holes Interpretation**:
- **High effective_size**: Many non-redundant contacts
- **Low constraint**: Access to diverse information (good for brokerage)
- **High redundancy**: Contacts are well-connected to each other

---

### Reciprocity Category

Mutual connection patterns.

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `reciprocity` | mutual_count, mutual_ratio, mutual_received_ratio, one_way_out, one_way_in | Mutual connection statistics | [0, 1] | Low |

**Interpretation**:
- **High mutual_ratio**: Most relationships are bidirectional
- **Low mutual_ratio**: Asymmetric relationships dominate

---

### Reach Category

N-hop neighborhood statistics.

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `reach` | reach_hop_1 through reach_hop_6, total_reach, network_penetration | N-hop reachability | [0, n-1] | High |

**Use Cases**:
- Information spread potential
- Influence reach analysis
- Network coverage assessment

---

### Components Category

Connected component membership and hierarchy.

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `components` | weak_component_size, strong_component_size, in_largest_component | Basic component membership | [1, n] | Low |
| `strongly_connected_components` | strong_component_id, scc_size, in_largest_scc, scc_is_trivial, scc_condensation_id | Detailed SCC analysis | Various | Low |
| `condensation_graph` | condensation_in_degree, condensation_out_degree, condensation_is_root, condensation_is_leaf, condensation_depth | Condensation DAG analysis | Various | Medium |
| `attracting_components` | in_attracting_component, attracting_component_id, attracting_component_size | Attracting (sink) components | Various | Low |
| `biconnected_components` | biconnected_component_count, max_biconnected_size | Biconnected components analysis | [0, ∞) | Medium |

**WCC vs SCC**:
- **Weakly Connected**: Ignores edge direction
- **Strongly Connected**: Respects direction (all nodes mutually reachable)

---

### Vitality Category

Impact of node removal.

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `closeness_vitality` | closeness_vitality | Change in Wiener index when removed | (-∞, ∞) | Very High |

**Warning**: Very expensive - requires recomputation for each node.

---

### Efficiency Category

Communication efficiency metrics.

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `local_efficiency` | local_efficiency | Efficiency of immediate neighborhood | [0, 1] | High |
| `global_efficiency_contribution` | global_efficiency_contribution, global_efficiency_ratio | Node's contribution to global efficiency | [0, 1] | Very High |
| `node_efficiency` | node_efficiency, node_efficiency_in | Per-node communication efficiency | [0, 1] | High |
| `robustness_efficiency` | efficiency_criticality, efficiency_redundancy | Efficiency-based robustness | [0, 1] | High |
| `routing_efficiency` | routing_efficiency, path_diversity | Routing efficiency through node | [0, 1] | High |

---

### Flow Category

Flow, hierarchy, and cycle metrics.

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `flow_hierarchy` | flow_hierarchy | Hierarchical flow structure | [0, 1] | Medium |
| `max_flow_centrality` | max_flow_in, max_flow_out, max_flow_centrality | Maximum flow capacity | [0, ∞) | Very High |
| `flow_betweenness` | flow_betweenness, flow_bottleneck_score | Flow-based betweenness | [0, 1] | High |
| `hierarchy_level` | hierarchy_level, is_source, is_sink, hierarchy_depth | Hierarchical level in DAG | [0, ∞) | Medium |
| `cycle_participation` | in_cycle, cycle_count_estimate, scc_participation | Cycle participation analysis | [0, 1] | Medium |
| `min_cut_centrality` | min_cut_frequency, cut_vertex_centrality | Minimum cut centrality | [0, 1] | Very High |

**Parameters for Max Flow Centrality**:
- `sample_size` (int, default=20): Number of source/sink samples

---

### Dominance Category

Dominance relationship patterns.

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `dominance` | dominated_nodes_count, dominance_ratio | Dominated nodes count and ratio | [0, n-1] | High |

---

### Trust Category

Trust network algorithms for social networks.

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `eigentrust` | eigentrust | EigenTrust score | [0, 1] | Medium |
| `appleseed` | appleseed | Appleseed trust propagation | [0, 1] | Medium |

**Parameters for EigenTrust**:
- `epsilon` (float, default=0.01): Convergence threshold
- `max_iter` (int, default=100): Maximum iterations

**Citations**:
- EigenTrust: Kamvar et al., 2003
- Appleseed: Ziegler and Lausen, 2005

---

### Similarity Category

Node similarity measures.

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `jaccard_similarity` | jaccard_similarity_avg, jaccard_similarity_max | Jaccard similarity with neighbors | [0, 1] | High |
| `cosine_similarity` | cosine_similarity_avg, cosine_similarity_max | Cosine similarity with neighbors | [0, 1] | High |
| `adamic_adar` | adamic_adar_sum, adamic_adar_avg | Adamic-Adar similarity | [0, ∞) | High |
| `resource_allocation` | resource_allocation_sum, resource_allocation_avg | Resource allocation index | [0, ∞) | High |

---

### Link Prediction Category

Metrics useful for predicting future links.

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `common_neighbors` | common_neighbors_sum, common_neighbors_max | Common neighbors score | [0, ∞) | High |
| `preferential_attachment` | preferential_attachment_score | Preferential attachment score | [0, ∞) | Medium |
| `link_prediction_scores` | link_pred_adamic_adar, link_pred_resource_alloc, link_pred_jaccard | Aggregated link prediction scores | [0, ∞) | High |

**Parameters for Link Prediction Scores**:
- `top_k` (int, default=10): Number of top predictions per node

---

### Robustness Category

Network resilience and connectivity.

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `node_connectivity` | node_connectivity | Minimum nodes to disconnect | [0, ∞) | High |
| `edge_connectivity` | edge_connectivity | Minimum edges to disconnect | [0, ∞) | High |
| `resilience_score` | resilience_score, lcc_reduction | Impact on largest component when removed | [0, 1] | High |

---

### Spectral Category

Spectral graph theory metrics.

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `fiedler_vector` | fiedler_component, fiedler_partition | Second eigenvector of Laplacian | (-∞, ∞) | High |
| `spectral_centrality` | spectral_centrality | Top eigenvector of adjacency | [0, 1] | High |

**Note**: Requires connected graph for Fiedler vector.

---

### Influence Category

Influence maximization metrics.

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `collective_influence` | collective_influence | Morone & Makse for network dismantling | [0, ∞) | Medium |
| `spreading_activation` | spreading_activation, influence_reach | Simulated information spreading | [0, ∞) | Medium |

**Parameters for Collective Influence**:
- `ball_radius` (int, default=2): Radius for ball neighborhood (1-4)

**Parameters for Spreading Activation**:
- `steps` (int, default=3): Number of spreading steps (1-10)
- `decay` (float, default=0.5): Decay factor per step (0.1-0.9)

**Citation**: Morone & Makse, 2015

---

### Bipartite Category

Bipartite graph metrics.

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `bipartite_projection_degree` | bipartite_projection_degree, bipartite_redundancy | Projection degree (or 2-hop proxy) | [0, ∞) | Medium |

---

### Graph Coloring Category

Graph coloring analysis.

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `greedy_color` | greedy_color, color_class_size, chromatic_estimate | Greedy graph coloring | [0, χ) | Low |

**Parameters**:
- `strategy` (choice, default="largest_first"): Coloring strategy
  - Options: largest_first, smallest_last, independent_set, connected_sequential

---

### igraph Community Category

Advanced community detection algorithms (requires igraph library).

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `leiden_community` | leiden_community_id, leiden_community_size | Leiden algorithm (improved Louvain) | [0, n_communities) | Medium |
| `infomap_community` | infomap_community_id, infomap_community_size, infomap_codelength | Infomap (information flow) | [0, n_communities) | Medium |
| `walktrap_community` | walktrap_community_id, walktrap_community_size | Walktrap (random walks) | [0, n_communities) | Medium |
| `fast_greedy_community` | fast_greedy_community_id, fast_greedy_community_size | Fast greedy modularity | [0, n_communities) | Medium |
| `spinglass_community` | spinglass_community_id, spinglass_community_size | Spinglass (statistical physics) | [0, n_communities) | High |

**Parameters for Leiden**:
- `resolution` (float, default=1.0): Resolution parameter

**Parameters for Walktrap**:
- `steps` (int, default=4): Length of random walks (2-10)

---

### igraph Centrality Category

igraph centrality algorithms.

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `alpha_centrality` | alpha_centrality | Alpha centrality (generalized eigenvector) | [0, ∞) | Medium |

---

### Motifs Category

Network motif analysis.

| Metric | Output Columns | Description | Range | Cost |
|--------|----------------|-------------|-------|------|
| `motif_count` | motif3_count, motif4_count | 3-node and 4-node motif counting | [0, ∞) | High |

---

## Performance Considerations

### Computation Times (approximate)

| Category | 1K nodes | 10K nodes | 100K nodes |
|----------|----------|-----------|------------|
| topology | < 1s | < 1s | 1-2s |
| centrality (basic) | 1-2s | 10-30s | 2-5 min |
| centrality (advanced) | 5-10s | 1-5 min | 10-30 min |
| clustering | < 1s | 5-10s | 1-2 min |
| community | 1-2s | 5-15s | 1-3 min |
| paths | 2-5s | 30s-2min | 10-30 min |
| structural | 1-2s | 10-30s | 2-5 min |
| vitality | 5-10s | 5-15 min | hours |
| igraph algorithms | 1-2s | 5-30s | 1-5 min |

### Recommendations by Graph Size

```
Small graphs (< 5K nodes):    Use "all" or "comprehensive" preset
Medium graphs (5K-50K nodes): Use "moderate" or "essential" preset
Large graphs (50K-500K nodes): Use "essential" or "basic" preset
Very large graphs (> 500K):    Use "basic" preset with specific metrics
```

### Max Node Limits

Some metrics have built-in node limits to prevent excessive computation:

| Metric | Max Nodes |
|--------|-----------|
| second_order_centrality | 100,000 |
| current_flow_centrality | 100,000 |
| information_centrality | 100,000 |
| communicability_betweenness | 50,000 |
| dispersion | 10,000 |
| closeness_vitality | 50,000 |
| resilience_score | 20,000 |
| motif_count | 50,000 |

### Parallel Processing

```bash
# Environment variable
N_JOBS=-1  # Use all CPU cores
N_JOBS=4   # Use 4 cores
N_JOBS=1   # Single-threaded
```

---

## API Usage

### Computing Metrics

```bash
# Compute using preset
curl -X POST "http://localhost:8000/api/metrics/run" \
  -H "Content-Type: application/json" \
  -d '{"preset": "essential"}'

# Compute specific categories
curl -X POST "http://localhost:8000/api/metrics/run" \
  -H "Content-Type: application/json" \
  -d '{"categories": ["topology", "centrality", "clustering"]}'

# Compute specific metrics
curl -X POST "http://localhost:8000/api/metrics/run" \
  -H "Content-Type: application/json" \
  -d '{"metrics": ["pagerank", "louvain_community", "betweenness_centrality"]}'

# With custom parameters
curl -X POST "http://localhost:8000/api/metrics/run" \
  -H "Content-Type: application/json" \
  -d '{
    "metrics": ["pagerank", "louvain_community"],
    "parameters": {
      "pagerank": {"alpha": 0.9},
      "louvain_community": {"resolution": 1.5}
    }
  }'
```

### List Available Metrics

```bash
# List all metrics
curl "http://localhost:8000/api/metrics/available"

# List categories
curl "http://localhost:8000/api/metrics/categories"

# List presets
curl "http://localhost:8000/api/metrics/presets"

# Get metric details
curl "http://localhost:8000/api/metrics/info/pagerank"
```

### Response Format

```json
{
  "metrics_computed": [
    "in_degree", "out_degree", "pagerank", "clustering_coefficient"
  ],
  "node_count": 10000,
  "computation_time": 15.3,
  "categories_run": ["topology", "centrality", "clustering"],
  "skipped": {
    "closeness_vitality": "exceeded max_nodes (50000)"
  }
}
```

---

## Common Metric Patterns

### Bot Detection

```
High indicators:
- Very high or very low in_degree/out_degree ratio
- Low clustering_coefficient (not part of organic communities)
- High degree with low reciprocity
- Low constraint (many structural holes)

Example filter:
out_degree > 100 AND mutual_ratio < 0.1 AND clustering_coefficient < 0.01
```

### Influential Nodes

```
High indicators:
- High pagerank
- High betweenness_centrality
- High eigenvector_centrality
- Low constraint (structural holes)
- High collective_influence

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
- High participation_coefficient

Example filter:
is_articulation_point = 1 OR (betweenness_centrality > 0.05 AND clustering_coefficient < 0.1)
```

### Peripheral Nodes

```
High indicators:
- core_number = 1 (lowest k-core)
- High onion_layer
- Low network_penetration
- In small component

Example filter:
core_number <= 2 AND network_penetration < 0.1
```

### Community Leaders

```
High indicators:
- High within_module_degree_z (central within community)
- Low participation_coefficient (focused on own community)
- High pagerank within community
- High local_reaching_centrality

Example filter:
within_module_degree_z > 2 AND participation_coefficient < 0.3
```

### Inter-Community Connectors

```
High indicators:
- High participation_coefficient
- High betweenness_centrality
- Moderate clustering_coefficient
- Connects multiple communities

Example filter:
participation_coefficient > 0.6 AND betweenness_centrality > 0.01
```

---

## Metric Sources

Metrics are implemented using:

| Source | Description |
|--------|-------------|
| `networkx` | Python NetworkX library |
| `igraph` | igraph library (optional, higher performance) |
| `scipy` | SciPy for spectral computations |
| `custom` | Custom implementations |

To check which metrics require igraph:

```bash
curl "http://localhost:8000/api/metrics/igraph-required"
```