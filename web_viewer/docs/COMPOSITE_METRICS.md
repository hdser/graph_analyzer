# Composite Metrics

Composite metrics allow you to create new metrics by combining existing ones using mathematical operations. This is useful for creating custom scores, risk indicators, and derived measurements.

## Overview

Composite metrics combine two source metrics using an operation:

```
New Metric = Metric1 [operation] Metric2
```

For example:
```
influence_score = pagerank × betweenness_centrality
activity_ratio = out_degree / in_degree
```

## Available Operations

| Operation | Symbol | Formula | Description |
|-----------|--------|---------|-------------|
| Multiply | × | M1 × M2 | Product of metrics |
| Add | + | M1 + M2 | Sum of metrics |
| Subtract | - | M1 - M2 | Difference of metrics |
| Divide | / | M1 / M2 | Ratio of metrics |
| Average | avg | (M1 + M2) / 2 | Mean of metrics |
| Maximum | max | max(M1, M2) | Higher value |
| Minimum | min | min(M1, M2) | Lower value |
| Weighted Sum | wsum | w1×M1 + w2×M2 | Weighted combination |
| Norm Multiply | normx | norm(M1) × norm(M2) | Normalized product |

## Creating Composite Metrics

### Via Main UI

1. Open the "Composite Metrics" section in the sidebar
2. Select two source metrics from the dropdowns
3. Choose an operation
4. (Optional) Check "Normalize" to scale inputs to [0,1]
5. Enter a name for the new metric
6. Click "Create"

### Via Distributions Page

1. Open the Distributions page (📊 button)
2. Switch to the "Composite" tab
3. Configure and preview the composite
4. Click "Create" to save

### Via API

```bash
curl -X POST "http://localhost:8000/api/metrics/composite/create" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "influence_score",
    "metrics": ["pagerank", "betweenness_centrality"],
    "operation": "multiply",
    "normalize": true,
    "save": true
  }'
```

## Operations in Detail

### Multiply (×)

**Formula**: `result = metric1 × metric2`

**Use Cases**:
- Combining importance measures
- Creating joint probability scores
- Intersection of characteristics

**Example**:
```
trust_influence = pagerank × eigenvector_centrality
```

High values indicate nodes that are both highly ranked and connected to important nodes.

---

### Add (+)

**Formula**: `result = metric1 + metric2`

**Use Cases**:
- Combining counts
- Aggregating complementary measures
- Creating total scores

**Example**:
```
total_connections = in_degree + out_degree
```

---

### Subtract (-)

**Formula**: `result = metric1 - metric2`

**Use Cases**:
- Imbalance detection
- Comparing metrics
- Identifying asymmetry

**Example**:
```
degree_difference = in_degree - out_degree
```

Positive = more incoming; Negative = more outgoing.

---

### Divide (/)

**Formula**: `result = metric1 / metric2`

**Use Cases**:
- Creating ratios
- Normalizing by baseline
- Efficiency measures

**Example**:
```
give_receive_ratio = out_degree / in_degree
```

**Note**: Division by zero is handled by returning 0.

---

### Average (avg)

**Formula**: `result = (metric1 + metric2) / 2`

**Use Cases**:
- Balanced scoring
- Reducing noise
- Central tendency

**Example**:
```
avg_centrality = average(pagerank, betweenness_centrality)
```

---

### Maximum (max)

**Formula**: `result = max(metric1, metric2)`

**Use Cases**:
- Conservative anomaly scoring
- Upper bound estimates
- OR-like combination

**Example**:
```
max_influence = max(hub_score, authority_score)
```

---

### Minimum (min)

**Formula**: `result = min(metric1, metric2)`

**Use Cases**:
- Conservative scoring
- Lower bound estimates
- AND-like combination

**Example**:
```
min_centrality = min(pagerank, eigenvector_centrality)
```

---

### Weighted Sum (wsum)

**Formula**: `result = w1 × metric1 + w2 × metric2`

**Use Cases**:
- Custom importance weighting
- Domain-specific scoring
- Adjustable combinations

**Parameters**:
- `weights`: [w1, w2] array (should sum to 1 for normalized result)

**Example**:
```json
{
  "operation": "weighted_sum",
  "weights": [0.7, 0.3]
}
```

Result = 0.7 × pagerank + 0.3 × betweenness

---

### Norm Multiply (normx)

**Formula**: `result = normalize(metric1) × normalize(metric2)`

**Normalization**: Each metric is scaled to [0, 1] before multiplication.

**Use Cases**:
- Combining metrics with different scales
- Fair comparison
- Probability-like products

**Example**:
```
normalized_influence = norm(pagerank) × norm(clustering_coefficient)
```

---

## Normalization Option

When "Normalize" is checked, both input metrics are scaled to [0, 1] before the operation:

```
normalized_value = (value - min) / (max - min)
```

**When to use normalization**:
- Metrics have different scales (e.g., pagerank [0,1] vs degree [0,1000])
- You want fair contribution from each metric
- Creating probability-like scores

**When NOT to normalize**:
- Metrics already on same scale
- Ratio calculations (divide)
- When original scale matters

---

## Preview Feature

Before creating a composite, you can preview its effect:

### Preview Information

1. **Formula**: Visual representation (e.g., "pagerank × clustering")
2. **Statistics**: min, max, mean, std, median
3. **Histogram**: Distribution of new metric
4. **Scatter Plot**: Source metrics colored by composite value
5. **Correlations**: Input correlation and metric-composite correlations

### Preview API

```bash
curl -X POST "http://localhost:8000/api/metrics/composite/preview" \
  -H "Content-Type: application/json" \
  -d '{
    "metrics": ["pagerank", "betweenness_centrality"],
    "operation": "multiply",
    "normalize": true
  }'
```

**Response**:
```json
{
  "formula": "norm(pagerank) × norm(betweenness_centrality)",
  "statistics": {
    "min": 0.0,
    "max": 0.847,
    "mean": 0.042,
    "std": 0.089,
    "median": 0.012
  },
  "correlations": {
    "input_correlation": 0.456,
    "m1_composite": 0.823,
    "m2_composite": 0.712
  },
  "histogram": {
    "bins": [0, 0.1, 0.2, ...],
    "counts": [8234, 1245, 342, ...]
  },
  "values": [
    {"id": "node1", "metric1": 0.001, "metric2": 0.05, "composite": 0.012},
    ...
  ]
}
```

---

## Saving and Managing Composites

### Saved Composites

Composites can be saved for reuse across sessions:

```json
{
  "save": true  // Include in create request
}
```

### List Saved Composites

```bash
curl "http://localhost:8000/api/metrics/composite/saved"
```

**Response**:
```json
{
  "composites": [
    {
      "id": "abc123",
      "name": "influence_score",
      "formula": "norm(pagerank) × norm(betweenness)",
      "operation": "multiply",
      "source_metrics": ["pagerank", "betweenness_centrality"],
      "normalize": true,
      "created_at": "2024-01-15T10:30:00Z"
    }
  ]
}
```

### Delete Composite

```bash
curl -X DELETE "http://localhost:8000/api/metrics/composite/saved/influence_score"
```

---

## Common Composite Patterns

### Influence Score

**Purpose**: Identify nodes with high overall network importance.

```json
{
  "name": "influence_score",
  "metrics": ["pagerank", "betweenness_centrality"],
  "operation": "multiply",
  "normalize": true
}
```

### Activity Ratio

**Purpose**: Identify giving vs receiving behavior.

```json
{
  "name": "activity_ratio",
  "metrics": ["out_degree", "in_degree"],
  "operation": "divide",
  "normalize": false
}
```

### Centrality Balance

**Purpose**: Find nodes central in multiple ways.

```json
{
  "name": "multi_centrality",
  "metrics": ["eigenvector_centrality", "closeness_centrality"],
  "operation": "average",
  "normalize": true
}
```

### Community Importance

**Purpose**: Find important nodes within their communities.

```json
{
  "name": "community_importance",
  "metrics": ["pagerank", "clustering_coefficient"],
  "operation": "multiply",
  "normalize": true
}
```

### Bridge Score

**Purpose**: Identify nodes that bridge communities.

```json
{
  "name": "bridge_score",
  "metrics": ["betweenness_centrality", "constraint"],
  "operation": "subtract",
  "normalize": true
}
```

(High betweenness, low constraint = good bridge)

### Risk Score

**Purpose**: Weighted risk indicator.

```json
{
  "name": "risk_score",
  "metrics": ["anomaly_score", "degree_imbalance"],
  "operation": "weighted_sum",
  "weights": [0.8, 0.2],
  "normalize": true
}
```

---

## Using Composites for Visualization

Once created, composite metrics appear in:

1. **Node Size**: Size nodes by composite value
2. **Node Color**: Color nodes by composite value
3. **Filters**: Filter by composite value
4. **Distributions**: Analyze composite distribution
5. **Data Explorer**: View and sort by composite

### Workflow Example

```
1. Create influence_score = pagerank × betweenness (normalized)
2. Set Node Size → influence_score
3. Set Node Color → influence_score (Spectral gradient)
4. Filter → influence_score > 0.5
5. Analyze selected high-influence nodes
```

---

## API Reference

### Get Available Operations

```bash
GET /api/metrics/composite/operations
```

### Create Composite

```bash
POST /api/metrics/composite/create
Content-Type: application/json

{
  "name": "string",
  "metrics": ["metric1", "metric2"],
  "operation": "multiply|add|subtract|divide|average|maximum|minimum|weighted_sum|norm_multiply",
  "weights": [0.5, 0.5],  // For weighted_sum only
  "normalize": true,
  "save": true,
  "version": "v2"  // Optional
}
```

### Preview Composite

```bash
POST /api/metrics/composite/preview
Content-Type: application/json

{
  "metrics": ["metric1", "metric2"],
  "operation": "multiply",
  "normalize": true,
  "node_ids": ["id1", "id2"]  // Optional filter
}
```

### List Saved

```bash
GET /api/metrics/composite/saved?version=v2
```

### Delete Saved

```bash
DELETE /api/metrics/composite/saved/{name}
```

---

## Best Practices

1. **Normalize for different scales**: When combining metrics with vastly different ranges

2. **Use meaningful names**: `influence_score` not `m1_x_m2`

3. **Preview first**: Check distribution before applying

4. **Consider correlations**: High input correlation = less information gain

5. **Document purpose**: Comment why this combination is useful

6. **Test edge cases**: Check behavior at 0, negative, or extreme values

7. **Save for reuse**: Don't recreate common composites