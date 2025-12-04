# Anomaly Detection Algorithms

Graph Analyzer provides eight anomaly detection algorithms, ranging from simple statistical methods to advanced machine learning approaches. This document explains each algorithm, its parameters, and when to use it.

## Algorithm Overview

| Algorithm | Type | Complexity | Best For |
|-----------|------|------------|----------|
| Z-Score | Statistical | O(n × d) | Normally distributed data |
| IQR | Statistical | O(n × d × log n) | Skewed distributions |
| Mahalanobis | Distance | O(n × d²) | Correlated features |
| Isolation Forest | ML (Ensemble) | O(n × t × log n) | High-dimensional, general purpose |
| LOF | ML (Density) | O(n² × d) | Local deviations |
| DBSCAN | ML (Clustering) | O(n × log n) | Clustered data |
| PCA Reconstruction | ML (Manifold) | O(n × d²) | Linear relationships |
| One-Class SVM | ML (Boundary) | O(n² × d) to O(n³) | Clean training data |

Where:
- `n` = number of nodes
- `d` = number of features (metrics)
- `t` = number of trees (Isolation Forest)

---

## Statistical Algorithms

### Z-Score

**Description**: Computes z-scores for each feature and aggregates them to identify outliers. Points with high aggregated z-scores are marked as anomalies.

**How it works**:
1. For each metric, compute: z = (x - μ) / σ
2. Take absolute value: |z|
3. Aggregate across metrics using chosen method
4. Mark as anomaly if aggregated score > threshold

**Parameters**:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| threshold | float | 3.0 | 1.0 - 10.0 | Z-score threshold for anomaly detection |
| aggregation | string | "max" | max, mean, l2, weighted | How to combine z-scores across features |

**Aggregation Methods**:
- **max**: Maximum z-score across all features (most sensitive)
- **mean**: Average z-score (balanced)
- **l2**: Euclidean norm √(Σz²) (geometric)
- **weighted**: Weighted average (use with metric weights)

**Example**:
```json
{
  "algorithm": "zscore",
  "parameters": {
    "threshold": 3.0,
    "aggregation": "max"
  }
}
```

**When to use**:
- Quick initial analysis
- Normally distributed metrics
- When interpretability is important

**Limitations**:
- Assumes normal distribution
- Sensitive to extreme outliers affecting mean/std

---

### IQR (Interquartile Range)

**Description**: Uses Tukey's method to identify outliers based on quartiles. Points outside [Q1 - k×IQR, Q3 + k×IQR] are flagged as outliers.

**How it works**:
1. Compute Q1 (25th percentile) and Q3 (75th percentile)
2. Calculate IQR = Q3 - Q1
3. Define fences: lower = Q1 - k×IQR, upper = Q3 + k×IQR
4. Compute distance outside fences (normalized by IQR)
5. Aggregate across metrics

**Parameters**:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| multiplier | float | 1.5 | 1.0 - 5.0 | IQR multiplier (1.5 = outlier, 3.0 = extreme) |
| side | string | "both" | both, high, low | Which tail(s) to consider |
| aggregation | string | "max" | max, mean, any | How to aggregate across features |

**Multiplier Guidelines**:
- **1.5**: Standard outlier threshold (Tukey's mild outlier)
- **3.0**: Extreme outlier threshold (Tukey's far outlier)
- **2.0-2.5**: Common middle ground

**Example**:
```json
{
  "algorithm": "iqr",
  "parameters": {
    "multiplier": 1.5,
    "side": "both",
    "aggregation": "max"
  }
}
```

**When to use**:
- Skewed distributions
- When robustness to extreme values is needed
- Non-normally distributed data

**Limitations**:
- May miss outliers in multimodal distributions
- Less effective for high-dimensional data

---

## Distance-Based Algorithms

### Mahalanobis Distance

**Description**: Measures distance from the centroid while accounting for feature correlations. Points far from the center in Mahalanobis space are anomalies.

**How it works**:
1. Compute mean vector μ and covariance matrix Σ
2. For each point x, compute: D = √((x-μ)ᵀ Σ⁻¹ (x-μ))
3. Use chi-squared distribution for threshold
4. Points exceeding threshold are anomalies

**Parameters**:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| alpha | float | 0.99 | 0.9 - 0.9999 | Confidence level (0.99 = 99%) |
| robust | bool | false | - | Use robust covariance (MinCovDet) |
| regularization | float | 1e-5 | 1e-10 - 0.1 | Regularization for matrix inversion |
| support_fraction | float | 0.75 | 0.5 - 1.0 | Fraction for MinCovDet |

**Robust Mode**:
When `robust=true`, uses Minimum Covariance Determinant (MinCovDet) which:
- Finds subset of points with minimum covariance determinant
- More resistant to outliers affecting covariance estimation
- Slower but more accurate for contaminated data

**Example**:
```json
{
  "algorithm": "mahalanobis",
  "parameters": {
    "alpha": 0.99,
    "robust": true,
    "regularization": 1e-5
  }
}
```

**When to use**:
- Correlated features
- Elliptical/Gaussian distributed data
- When correlation structure matters

**Limitations**:
- Requires invertible covariance matrix
- Assumes roughly elliptical distribution
- Computationally expensive for many features

---

## Machine Learning Algorithms

### Isolation Forest

**Description**: Tree-based anomaly detection using random isolation. Anomalies are easier to isolate, requiring fewer splits on average.

**How it works**:
1. Build ensemble of isolation trees
2. Each tree randomly selects feature and split point
3. Record path length (number of splits) to isolate each point
4. Anomalies have shorter average path lengths
5. Score = 2^(-E(h)/c(n)) where h = path length

**Parameters**:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| n_estimators | int | 100 | 10 - 1000 | Number of isolation trees |
| contamination | float | 0.1 | 0.001 - 0.5 | Expected proportion of outliers |
| max_samples | string | "auto" | auto, 256, 512, 1024, all | Samples per tree |
| random_state | int | 42 | 0 - 99999 | Random seed |
| bootstrap | bool | false | - | Use bootstrap sampling |

**Contamination Guidelines**:
- Use domain knowledge if available
- Start with 0.1 (10%) as default
- Lower values = fewer anomalies flagged
- "auto" uses heuristic based on algorithm

**Example**:
```json
{
  "algorithm": "isolation_forest",
  "parameters": {
    "n_estimators": 200,
    "contamination": 0.1,
    "max_samples": "auto"
  }
}
```

**When to use**:
- High-dimensional data
- Large datasets (scales well)
- Unknown distribution
- General-purpose anomaly detection

**Limitations**:
- May struggle with local anomalies
- Contamination must be set appropriately

---

### LOF (Local Outlier Factor)

**Description**: Density-based algorithm measuring local deviation. Points with substantially lower density than their neighbors are anomalies.

**How it works**:
1. Find k-nearest neighbors for each point
2. Compute local reachability density (LRD)
3. LOF = average ratio of LRD(neighbors) / LRD(point)
4. LOF >> 1 indicates anomaly (lower density than neighbors)

**Parameters**:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| n_neighbors | int | 20 | 2 - 200 | Neighbors for density estimation |
| contamination | float | 0.1 | 0.001 - 0.5 | Expected outlier proportion |
| algorithm | string | "auto" | auto, ball_tree, kd_tree, brute | Nearest neighbor algorithm |
| leaf_size | int | 30 | 10 - 100 | Leaf size for tree algorithms |
| metric | string | "euclidean" | euclidean, manhattan, minkowski, chebyshev | Distance metric |

**n_neighbors Guidelines**:
- Small (5-10): Sensitive to local variations
- Medium (20-50): Balanced approach
- Large (100+): Captures broader patterns

**Example**:
```json
{
  "algorithm": "lof",
  "parameters": {
    "n_neighbors": 20,
    "contamination": 0.1,
    "algorithm": "auto"
  }
}
```

**When to use**:
- Local anomaly detection
- Varying density clusters
- Small to medium datasets

**Limitations**:
- O(n²) complexity - slow for large datasets
- Sensitive to n_neighbors choice
- Performance degrades in high dimensions

---

### DBSCAN

**Description**: Density-based clustering where points not belonging to any cluster (noise points) are considered anomalies.

**How it works**:
1. Find ε-neighborhood of each point
2. Core points: ≥ min_samples neighbors within ε
3. Expand clusters from core points
4. Points not in any cluster are noise (anomalies)
5. Score = distance to nearest core point

**Parameters**:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| eps | float | 0.5 | 0.01 - 10.0 | Maximum distance between points in cluster |
| min_samples | int | 5 | 2 - 100 | Minimum points to form cluster |
| algorithm | string | "auto" | auto, ball_tree, kd_tree, brute | Nearest neighbor algorithm |
| metric | string | "euclidean" | euclidean, manhattan, minkowski, chebyshev | Distance metric |

**Parameter Tuning**:
- **eps**: Use k-distance plot to find elbow
- **min_samples**: rule of thumb = d × 2 (where d = dimensions)
- Standardize data before applying

**Example**:
```json
{
  "algorithm": "dbscan",
  "parameters": {
    "eps": 0.5,
    "min_samples": 5,
    "algorithm": "auto"
  }
}
```

**When to use**:
- Well-defined clusters expected
- Unknown number of clusters
- Varying density data

**Limitations**:
- Sensitive to eps parameter
- Struggles with varying density clusters
- Requires data standardization

---

### PCA Reconstruction

**Description**: Detects anomalies based on reconstruction error after projecting to lower-dimensional space and back.

**How it works**:
1. Fit PCA to retain n_components (or variance ratio)
2. Project data to lower dimensions
3. Reconstruct back to original space
4. Compute reconstruction error for each point
5. High error = anomaly (doesn't fit linear structure)

**Parameters**:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| n_components | string/int | "auto" | auto, 2, 3, 5, 10, 0.95 | Components or variance ratio |
| contamination | float | 0.1 | 0.001 - 0.5 | Expected outlier proportion |
| whiten | bool | false | - | Whiten components |
| standardize | bool | true | - | Standardize before PCA |

**n_components Options**:
- Integer (2, 3, 5, 10): Fixed number of components
- Float (0.95, 0.99): Variance ratio to retain
- "auto": Retains 95% variance

**Example**:
```json
{
  "algorithm": "pca_reconstruction",
  "parameters": {
    "n_components": "auto",
    "contamination": 0.1,
    "standardize": true
  }
}
```

**When to use**:
- High-dimensional data with linear structure
- Correlated features
- When dimensionality reduction is beneficial

**Limitations**:
- Assumes linear relationships
- May miss non-linear patterns
- Sensitive to n_components choice

---

### One-Class SVM

**Description**: Learns a decision boundary around normal data. Points outside the boundary are anomalies.

**How it works**:
1. Map data to high-dimensional space via kernel
2. Find hyperplane separating origin from data
3. Maximize margin from origin
4. Points on wrong side of boundary are anomalies

**Parameters**:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| nu | float | 0.1 | 0.01 - 0.5 | Upper bound on outlier fraction |
| kernel | string | "rbf" | linear, poly, rbf, sigmoid | Kernel function |
| gamma | string | "scale" | scale, auto, 0.001-100 | Kernel coefficient |
| degree | int | 3 | 2 - 5 | Polynomial degree |
| coef0 | float | 0.0 | 0.0 - 1.0 | Kernel offset |

**Kernel Selection**:
- **rbf**: General purpose, handles non-linear boundaries
- **linear**: Fast, assumes linear separation
- **poly**: Polynomial boundaries
- **sigmoid**: Similar to neural network

**Example**:
```json
{
  "algorithm": "one_class_svm",
  "parameters": {
    "nu": 0.1,
    "kernel": "rbf",
    "gamma": "scale"
  }
}
```

**When to use**:
- Small to medium datasets
- When clean training data available
- Non-linear decision boundaries needed

**Limitations**:
- O(n² × d) to O(n³) complexity - very slow for large data
- Requires parameter tuning
- Sensitive to scale

---

## Preprocessing Configuration

All algorithms can use preprocessing configuration:

```json
{
  "config": {
    "id_column": "avatar",
    "nan_strategy": "zero",
    "global_scaling": "standard",
    "min_group_size": 3,
    "per_metric": {
      "pagerank": {
        "log": true,
        "clip_min": null,
        "clip_max": 0.5
      }
    }
  }
}
```

### NaN Strategies
- **zero**: Replace NaN with 0
- **mean**: Replace with column mean
- **median**: Replace with column median
- **drop**: Remove rows with NaN

### Global Scaling
- **none**: No scaling
- **standard**: (x - μ) / σ
- **robust**: (x - median) / IQR
- **minmax**: (x - min) / (max - min)

### Per-Metric Transforms
- **log**: Apply log1p transform
- **clip_min/clip_max**: Bound values
- **weight**: Importance weight
- **drop**: Exclude from analysis

---

## Algorithm Selection Guide

```
                    ┌─────────────────────────┐
                    │    How big is your      │
                    │        dataset?         │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
          Small (<1K)      Medium (1K-50K)    Large (>50K)
              │                 │                 │
              ▼                 ▼                 ▼
         Any algorithm    LOF, OCSVM          Isolation Forest
                          DBSCAN, Mah         (best scaling)
                                │
                                │
                    ┌───────────┴───────────┐
                    │  What's your data     │
                    │     distribution?     │
                    └───────────┬───────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
     Gaussian              Skewed/Heavy          Clustered
         │                    Tails                    │
         ▼                      ▼                      ▼
     Mahalanobis              IQR                   DBSCAN
     Z-Score           Isolation Forest              LOF
```

## Performance Recommendations

| Dataset Size | Recommended Algorithms | Notes |
|--------------|----------------------|-------|
| < 1,000 | Any | All perform well |
| 1,000 - 10,000 | IF, LOF, DBSCAN, Mahalanobis | LOF starts to slow |
| 10,000 - 50,000 | IF, DBSCAN, Mahalanobis | Avoid One-Class SVM |
| > 50,000 | Isolation Forest | Use sampling for others |
| > 100,000 | Isolation Forest with sampling | Consider chunked processing |