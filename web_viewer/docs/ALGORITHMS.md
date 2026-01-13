# Anomaly Detection Algorithms

Graph Analyzer provides **eight anomaly detection algorithms**, ranging from simple statistical methods to advanced machine learning approaches. This document explains each algorithm, its parameters, and when to use it.

## Table of Contents

1. [Algorithm Overview](#algorithm-overview)
2. [Statistical Algorithms](#statistical-algorithms)
3. [Distance-Based Algorithms](#distance-based-algorithms)
4. [Machine Learning Algorithms](#machine-learning-algorithms)
5. [Preprocessing Configuration](#preprocessing-configuration)
6. [Algorithm Selection Guide](#algorithm-selection-guide)
7. [Threshold Methods](#threshold-methods)
8. [Score Normalization](#score-normalization)
9. [API Reference](#api-reference)
10. [Best Practices](#best-practices)

---

## Algorithm Overview

| Algorithm | Type | Complexity | Best For | Interpretability |
|-----------|------|------------|----------|------------------|
| Z-Score | Statistical | O(n × d) | Normally distributed data | High |
| IQR | Statistical | O(n × d × log n) | Skewed distributions | High |
| Mahalanobis | Distance | O(n × d²) | Correlated features | Medium |
| Isolation Forest | ML (Ensemble) | O(n × t × log n) | High-dimensional, general purpose | Low |
| LOF | ML (Density) | O(n² × d) | Local deviations | Medium |
| DBSCAN | ML (Clustering) | O(n × log n) | Clustered data | Medium |
| PCA Reconstruction | ML (Manifold) | O(n × d²) | Linear relationships | Medium |
| One-Class SVM | ML (Boundary) | O(n² × d) to O(n³) | Clean training data | Low |

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
- **max**: Maximum z-score across all features (most sensitive to any single outlier metric)
- **mean**: Average z-score (balanced, requires multiple metrics to be unusual)
- **l2**: Euclidean norm √(Σz²) (geometric combination)
- **weighted**: Weighted average (use with metric weights for domain knowledge)

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

**Statistical Interpretation**:
- threshold = 2.0: ~5% expected outliers (2 standard deviations)
- threshold = 3.0: ~0.3% expected outliers (3 standard deviations)
- threshold = 4.0: ~0.006% expected outliers (4 standard deviations)

**When to use**:
- Quick initial analysis
- Normally distributed metrics
- When interpretability is important
- Small to medium datasets

**Limitations**:
- Assumes normal distribution
- Sensitive to extreme outliers affecting mean/std
- May miss multivariate outliers

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
- **2.0-2.5**: Common middle ground
- **3.0**: Extreme outlier threshold (Tukey's far outlier)

**Side Options**:
- **both**: Flag high and low outliers
- **high**: Only flag values above Q3 + k×IQR
- **low**: Only flag values below Q1 - k×IQR

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
- Quick screening

**Limitations**:
- May miss outliers in multimodal distributions
- Less effective for high-dimensional data
- Single-metric focus

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

**Alpha Interpretation**:
- alpha = 0.95: 5% expected outliers
- alpha = 0.99: 1% expected outliers
- alpha = 0.999: 0.1% expected outliers

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
- Medium-sized datasets

**Limitations**:
- Requires invertible covariance matrix
- Assumes roughly elliptical distribution
- Computationally expensive for many features
- Sensitive to non-Gaussian distributions

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
| max_samples | string/int | "auto" | auto, 256, 512, 1024, all | Samples per tree |
| random_state | int | 42 | 0 - 99999 | Random seed for reproducibility |
| bootstrap | bool | false | - | Use bootstrap sampling |
| max_features | float | 1.0 | 0.1 - 1.0 | Fraction of features per tree |

**Contamination Guidelines**:
- Use domain knowledge if available
- Start with 0.1 (10%) as default
- Lower values = fewer anomalies flagged
- "auto" uses heuristic based on algorithm

**Max Samples**:
- "auto": min(256, n_samples)
- Integer: exact number of samples
- Lower values = faster, more variance

**Example**:
```json
{
  "algorithm": "isolation_forest",
  "parameters": {
    "n_estimators": 200,
    "contamination": 0.1,
    "max_samples": "auto",
    "random_state": 42
  }
}
```

**When to use**:
- High-dimensional data
- Large datasets (scales well)
- Unknown distribution
- General-purpose anomaly detection
- When interpretability is less important

**Limitations**:
- May struggle with local anomalies
- Contamination must be set appropriately
- Less interpretable than statistical methods

---

### LOF (Local Outlier Factor)

**Description**: Density-based algorithm measuring local deviation. Points with substantially lower density than their neighbors are anomalies.

**How it works**:
1. Find k-nearest neighbors for each point
2. Compute local reachability density (LRD)
3. LOF = average ratio of LRD(neighbors) / LRD(point)
4. LOF >> 1 indicates anomaly (lower density than neighbors)
5. LOF ≈ 1 indicates normal point

**Parameters**:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| n_neighbors | int | 20 | 2 - 200 | Neighbors for density estimation |
| contamination | float | 0.1 | 0.001 - 0.5 | Expected outlier proportion |
| algorithm | string | "auto" | auto, ball_tree, kd_tree, brute | Nearest neighbor algorithm |
| leaf_size | int | 30 | 10 - 100 | Leaf size for tree algorithms |
| metric | string | "euclidean" | euclidean, manhattan, minkowski, chebyshev | Distance metric |
| p | int | 2 | 1 - 5 | Power for Minkowski metric |

**n_neighbors Guidelines**:
- Small (5-10): Sensitive to local variations, may be noisy
- Medium (20-50): Balanced approach (recommended)
- Large (100+): Captures broader patterns, smoother

**Metric Options**:
- **euclidean**: Standard L2 distance
- **manhattan**: L1 distance (city-block)
- **chebyshev**: Maximum coordinate difference
- **minkowski**: Generalized (p=1: manhattan, p=2: euclidean)

**Example**:
```json
{
  "algorithm": "lof",
  "parameters": {
    "n_neighbors": 20,
    "contamination": 0.1,
    "algorithm": "auto",
    "metric": "euclidean"
  }
}
```

**When to use**:
- Local anomaly detection
- Varying density clusters
- Small to medium datasets
- When global methods miss local outliers

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
| leaf_size | int | 30 | 10 - 100 | Leaf size for tree algorithms |

**Parameter Tuning**:
- **eps**: Use k-distance plot to find elbow
  - Too small: everything is noise
  - Too large: everything is one cluster
- **min_samples**: rule of thumb = d × 2 (where d = dimensions)
- **Important**: Standardize data before applying

**Example**:
```json
{
  "algorithm": "dbscan",
  "parameters": {
    "eps": 0.5,
    "min_samples": 5,
    "algorithm": "auto",
    "metric": "euclidean"
  }
}
```

**When to use**:
- Well-defined clusters expected
- Unknown number of clusters
- Varying density data (with care)
- When cluster membership matters

**Limitations**:
- Very sensitive to eps parameter
- Struggles with varying density clusters
- Requires data standardization
- May produce no anomalies or all anomalies with wrong parameters

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
| n_components | string/int/float | "auto" | auto, 2-100, 0.5-0.99 | Components or variance ratio |
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
    "standardize": true,
    "whiten": false
  }
}
```

**When to use**:
- High-dimensional data with linear structure
- Correlated features
- When dimensionality reduction is beneficial
- Quick anomaly screening

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
| gamma | string/float | "scale" | scale, auto, 0.001-100 | Kernel coefficient |
| degree | int | 3 | 2 - 5 | Polynomial degree (for poly kernel) |
| coef0 | float | 0.0 | 0.0 - 1.0 | Kernel offset (for poly/sigmoid) |
| tol | float | 1e-3 | 1e-6 - 1e-1 | Convergence tolerance |
| shrinking | bool | true | - | Use shrinking heuristic |
| cache_size | int | 200 | 50 - 1000 | Kernel cache size (MB) |

**Kernel Selection**:
- **rbf**: General purpose, handles non-linear boundaries (recommended)
- **linear**: Fast, assumes linear separation
- **poly**: Polynomial boundaries
- **sigmoid**: Similar to neural network activation

**Gamma Options**:
- "scale": 1 / (n_features × X.var())
- "auto": 1 / n_features
- Float: explicit value

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
- Complex anomaly patterns

**Limitations**:
- O(n² × d) to O(n³) complexity - very slow for large data
- Requires careful parameter tuning
- Sensitive to scale
- Memory intensive

---

## Preprocessing Configuration

All algorithms can use preprocessing configuration to handle data issues:

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
        "clip_max": 0.5,
        "weight": 2.0
      },
      "in_degree": {
        "drop": true
      }
    }
  }
}
```

### NaN Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| zero | Replace NaN with 0 | When missing = no value |
| mean | Replace with column mean | Normal distributions |
| median | Replace with column median | Skewed distributions |
| drop | Remove rows with NaN | When completeness required |

### Global Scaling

| Scaling | Formula | When to Use |
|---------|---------|-------------|
| none | No scaling | Metrics already comparable |
| standard | (x - μ) / σ | Normal distributions |
| robust | (x - median) / IQR | Outlier-robust scaling |
| minmax | (x - min) / (max - min) | Bounded [0,1] output |

### Per-Metric Transforms

| Transform | Description |
|-----------|-------------|
| log | Apply log1p(x) transform for skewed data |
| clip_min | Lower bound clipping |
| clip_max | Upper bound clipping |
| weight | Importance weight for aggregation |
| drop | Exclude from analysis |

---

## Algorithm Selection Guide

### Decision Tree

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
                          DBSCAN, Mah         Z-Score, IQR
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

### Detailed Recommendations

| Scenario | Primary | Secondary | Avoid |
|----------|---------|-----------|-------|
| Quick screening | Z-Score | IQR | OCSVM |
| Small clean data | Mahalanobis | LOF | - |
| Large datasets | Isolation Forest | Z-Score | LOF, OCSVM |
| Clustered data | DBSCAN | LOF | Z-Score |
| High dimensions | Isolation Forest | PCA | LOF |
| Correlated features | Mahalanobis | PCA | Z-Score |
| Unknown distribution | Isolation Forest | LOF | Z-Score |
| Local outliers | LOF | DBSCAN | Mahalanobis |

---

## Threshold Methods

After computing anomaly scores, a threshold determines which nodes are flagged.

| Method | Description | Parameters |
|--------|-------------|------------|
| percentile | Top X% of scores | percentile (0-100) |
| std | Mean + k×std | num_std (1-5) |
| iqr | Q3 + k×IQR | multiplier (1-5) |
| fixed | Fixed score threshold | threshold (0-1) |
| contamination | Algorithm's contamination | - |

**Example**:
```json
{
  "threshold_method": "percentile",
  "threshold_params": {
    "percentile": 95
  }
}
```

---

## Score Normalization

Raw anomaly scores can be normalized for comparison:

| Method | Output Range | Description |
|--------|--------------|-------------|
| none | Raw scores | No normalization |
| minmax | [0, 1] | Linear scaling |
| zscore | (-∞, ∞) | Standard normalization |
| rank | [0, 1] | Percentile rank |

---

## API Reference

### Run Anomaly Detection

```bash
POST /api/anomaly/detect
Content-Type: application/json

{
  "algorithm": "isolation_forest",
  "metrics": ["pagerank", "betweenness_centrality", "in_degree", "out_degree"],
  "parameters": {
    "n_estimators": 100,
    "contamination": 0.1
  },
  "config": {
    "nan_strategy": "zero",
    "global_scaling": "standard"
  },
  "threshold_method": "percentile",
  "threshold_params": {
    "percentile": 95
  },
  "score_normalization": "minmax"
}
```

### Get Algorithm Info

```bash
GET /api/anomaly/algorithms
GET /api/anomaly/algorithms/{algorithm_name}
```

### Get Metric Profiles

```bash
POST /api/anomaly/profile
Content-Type: application/json

{
  "metrics": ["pagerank", "betweenness_centrality"]
}
```

---

## Best Practices

### 1. Start Simple
Begin with Z-Score or IQR to understand your data before moving to ML methods.

### 2. Check Distributions
Profile your metrics first. Skewed data may need log transforms.

### 3. Scale Appropriately
Always standardize features for distance-based and ML algorithms.

### 4. Validate Results
- Compare multiple algorithms
- Check flagged nodes manually
- Use domain knowledge

### 5. Tune Contamination
- Start with expected anomaly rate
- Adjust based on results
- Consider business context

### 6. Handle Missing Data
- Choose appropriate NaN strategy
- Consider dropping metrics with >50% missing

### 7. Feature Selection
- Don't use too many metrics (curse of dimensionality)
- Select uncorrelated, meaningful metrics
- 5-15 metrics is usually sufficient

### 8. Document Settings
Record algorithm and parameters for reproducibility.

---

## Performance Benchmarks

| Dataset Size | Z-Score | IQR | Mahalanobis | IF | LOF | DBSCAN | OCSVM |
|--------------|---------|-----|-------------|----|----|--------|-------|
| 1,000 | <1s | <1s | <1s | 1s | 1s | <1s | 1s |
| 10,000 | <1s | <1s | 1s | 2s | 10s | 2s | 30s |
| 50,000 | 1s | 1s | 5s | 5s | 2min | 10s | 5min |
| 100,000 | 2s | 2s | 15s | 10s | 10min | 30s | 20min+ |
| 500,000 | 10s | 10s | 1min | 30s | hours | 2min | hours |

*Times are approximate and depend on hardware and number of features.*