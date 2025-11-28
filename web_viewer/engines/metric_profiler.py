"""
Metric Profiler for Anomaly Detection

Analyzes metrics and suggests preprocessing configurations.
This is an explicit utility - must be called separately from detection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .anomaly_config import MetricConfig, MetricTransform, GlobalScaling


@dataclass
class MetricProfile:
    """
    Statistical profile of a single metric.
    
    Contains descriptive statistics and suggested preprocessing.
    """
    name: str
    dtype: str
    n_samples: int
    n_unique: int
    n_missing: int
    n_zeros: int
    n_negative: int
    n_inf: int
    
    # Descriptive statistics
    min_val: float
    max_val: float
    mean: float
    median: float
    std: float
    variance: float
    
    # Distribution shape
    skewness: float
    kurtosis: float
    
    # Percentiles
    p01: float
    p05: float
    p25: float
    p75: float
    p95: float
    p99: float
    
    # Computed properties
    iqr: float
    coefficient_of_variation: float
    
    # Suggested preprocessing
    suggested_transform: MetricTransform = field(default_factory=MetricTransform)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'dtype': self.dtype,
            'n_samples': self.n_samples,
            'n_unique': self.n_unique,
            'n_missing': self.n_missing,
            'n_zeros': self.n_zeros,
            'n_negative': self.n_negative,
            'n_inf': self.n_inf,
            'min': self.min_val,
            'max': self.max_val,
            'mean': self.mean,
            'median': self.median,
            'std': self.std,
            'variance': self.variance,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'p01': self.p01,
            'p05': self.p05,
            'p25': self.p25,
            'p75': self.p75,
            'p95': self.p95,
            'p99': self.p99,
            'iqr': self.iqr,
            'coefficient_of_variation': self.coefficient_of_variation,
            'suggested_transform': self.suggested_transform.to_dict(),
            'warnings': self.warnings,
        }


class MetricProfiler:
    """
    Profiles metrics and suggests preprocessing configurations.
    
    This is an explicit utility - must be called separately from detection.
    The engine itself remains metric-agnostic.
    
    Usage:
        profiler = MetricProfiler()
        profiles = profiler.profile(df, metrics)
        config = profiler.suggest_config(profiles)
        
        # Review and modify config as needed
        config.per_metric['my_metric'].log = False
        
        # Then use config in detection
        result = engine.detect_anomalies(df, metrics, algorithm, config=config)
    """
    
    # Thresholds for automatic suggestions
    HIGH_SKEWNESS_THRESHOLD = 2.0
    EXTREME_OUTLIER_RATIO = 10.0
    HIGH_ZERO_RATIO = 0.5
    LOW_UNIQUE_THRESHOLD = 10
    HIGH_MISSING_RATIO = 0.1
    SCALE_DIFFERENCE_THRESHOLD = 100.0
    
    def __init__(self):
        pass
    
    def profile(
        self,
        df: pd.DataFrame,
        metrics: List[str],
    ) -> Dict[str, MetricProfile]:
        """
        Profile all specified metrics.
        
        Args:
            df: Input DataFrame
            metrics: List of metric column names
            
        Returns:
            Dictionary mapping metric names to profiles
        """
        profiles = {}
        
        for metric in metrics:
            if metric not in df.columns:
                continue
            
            profile = self._profile_metric(df[metric], metric)
            profiles[metric] = profile
        
        return profiles
    
    def suggest_config(
        self,
        profiles: Dict[str, MetricProfile],
    ) -> MetricConfig:
        """
        Suggest a MetricConfig based on metric profiles.
        
        Args:
            profiles: Dictionary of metric profiles
            
        Returns:
            Suggested MetricConfig with per-metric transforms
        """
        config = MetricConfig()
        
        # Check if global scaling is needed
        scales = []
        for name, profile in profiles.items():
            if profile.std > 0:
                scales.append(profile.std)
        
        if scales:
            scale_ratio = max(scales) / min(scales) if min(scales) > 0 else 0
            if scale_ratio > self.SCALE_DIFFERENCE_THRESHOLD:
                config.global_scaling = GlobalScaling.STANDARD
        
        # Set per-metric transforms
        for name, profile in profiles.items():
            config.per_metric[name] = profile.suggested_transform
        
        return config
    
    def generate_report(
        self,
        profiles: Dict[str, MetricProfile],
    ) -> str:
        """
        Generate human-readable report of metric profiles.
        
        Args:
            profiles: Dictionary of metric profiles
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("METRIC PROFILER REPORT")
        lines.append("=" * 70)
        lines.append("")
        
        for name, profile in profiles.items():
            lines.append(f"Metric: {name}")
            lines.append("-" * 40)
            lines.append(f"  Samples: {profile.n_samples:,}")
            lines.append(f"  Unique values: {profile.n_unique:,}")
            lines.append(f"  Missing: {profile.n_missing:,} ({100*profile.n_missing/profile.n_samples:.1f}%)")
            lines.append(f"  Zeros: {profile.n_zeros:,} ({100*profile.n_zeros/profile.n_samples:.1f}%)")
            lines.append("")
            lines.append(f"  Range: [{profile.min_val:.4g}, {profile.max_val:.4g}]")
            lines.append(f"  Mean: {profile.mean:.4g} ± {profile.std:.4g}")
            lines.append(f"  Median: {profile.median:.4g}")
            lines.append(f"  IQR: {profile.iqr:.4g}")
            lines.append("")
            lines.append(f"  Skewness: {profile.skewness:.4g}")
            lines.append(f"  Kurtosis: {profile.kurtosis:.4g}")
            lines.append("")
            
            if profile.warnings:
                lines.append("  Warnings:")
                for warning in profile.warnings:
                    lines.append(f"    ⚠ {warning}")
                lines.append("")
            
            transform = profile.suggested_transform
            suggestions = []
            if transform.log:
                suggestions.append("log transform")
            if transform.clip_min is not None or transform.clip_max is not None:
                suggestions.append(f"clip to [{transform.clip_min}, {transform.clip_max}]")
            if transform.drop:
                suggestions.append("DROP (not suitable)")
            
            if suggestions:
                lines.append(f"  Suggested: {', '.join(suggestions)}")
            else:
                lines.append("  Suggested: no preprocessing needed")
            
            lines.append("")
        
        lines.append("=" * 70)
        return "\n".join(lines)
    
    def _profile_metric(self, series: pd.Series, name: str) -> MetricProfile:
        """Profile a single metric."""
        warnings = []
        
        # Basic counts
        n_samples = len(series)
        n_missing = series.isna().sum()
        n_unique = series.nunique(dropna=True)
        
        # Convert to numeric for analysis
        numeric_series = pd.to_numeric(series, errors='coerce')
        valid_series = numeric_series.dropna()
        
        n_zeros = (valid_series == 0).sum()
        n_negative = (valid_series < 0).sum()
        n_inf = np.isinf(valid_series).sum()
        
        # Handle empty or constant case
        if len(valid_series) == 0 or valid_series.nunique() <= 1:
            warnings.append("Constant or empty column")
            return MetricProfile(
                name=name,
                dtype=str(series.dtype),
                n_samples=n_samples,
                n_unique=n_unique,
                n_missing=n_missing,
                n_zeros=n_zeros,
                n_negative=n_negative,
                n_inf=n_inf,
                min_val=0.0,
                max_val=0.0,
                mean=0.0,
                median=0.0,
                std=0.0,
                variance=0.0,
                skewness=0.0,
                kurtosis=0.0,
                p01=0.0, p05=0.0, p25=0.0, p75=0.0, p95=0.0, p99=0.0,
                iqr=0.0,
                coefficient_of_variation=0.0,
                suggested_transform=MetricTransform(drop=True),
                warnings=warnings,
            )
        
        # Remove infinities for statistics
        finite_series = valid_series.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Descriptive statistics
        min_val = float(finite_series.min())
        max_val = float(finite_series.max())
        mean = float(finite_series.mean())
        median = float(finite_series.median())
        std = float(finite_series.std())
        variance = float(finite_series.var())
        
        # Percentiles
        percentiles = finite_series.quantile([0.01, 0.05, 0.25, 0.75, 0.95, 0.99]).values
        p01, p05, p25, p75, p95, p99 = percentiles
        
        iqr = p75 - p25
        
        # Shape statistics
        skewness = float(scipy_stats.skew(finite_series)) if len(finite_series) > 2 else 0.0
        kurtosis = float(scipy_stats.kurtosis(finite_series)) if len(finite_series) > 3 else 0.0
        
        # Coefficient of variation
        cv = std / abs(mean) if abs(mean) > 1e-10 else 0.0
        
        # Generate warnings and suggestions
        transform = MetricTransform()
        
        # Check for high missing ratio
        if n_missing / n_samples > self.HIGH_MISSING_RATIO:
            warnings.append(f"High missing ratio: {100*n_missing/n_samples:.1f}%")
        
        # Check for low unique values (potential categorical)
        if n_unique <= self.LOW_UNIQUE_THRESHOLD:
            warnings.append(f"Low unique values ({n_unique}), may be categorical")
        
        # Check for high zero ratio
        if n_zeros / n_samples > self.HIGH_ZERO_RATIO:
            warnings.append(f"High zero ratio: {100*n_zeros/n_samples:.1f}%")
        
        # Check for high skewness (suggest log transform)
        if skewness > self.HIGH_SKEWNESS_THRESHOLD and n_negative == 0:
            transform.log = True
            warnings.append(f"High positive skewness ({skewness:.2f}), log transform suggested")
        
        # Check for extreme outliers (suggest clipping)
        if iqr > 0:
            outlier_ratio = (p99 - median) / iqr
            if outlier_ratio > self.EXTREME_OUTLIER_RATIO:
                # Suggest clipping at 99th percentile
                transform.clip_max = float(p99)
                warnings.append(f"Extreme high outliers (99th percentile {outlier_ratio:.1f}x IQR from median)")
            
            outlier_ratio_low = (median - p01) / iqr
            if outlier_ratio_low > self.EXTREME_OUTLIER_RATIO:
                transform.clip_min = float(p01)
                warnings.append(f"Extreme low outliers")
        
        # Check for infinities
        if n_inf > 0:
            warnings.append(f"{n_inf} infinite values will be replaced")
        
        return MetricProfile(
            name=name,
            dtype=str(series.dtype),
            n_samples=n_samples,
            n_unique=n_unique,
            n_missing=n_missing,
            n_zeros=n_zeros,
            n_negative=n_negative,
            n_inf=n_inf,
            min_val=min_val,
            max_val=max_val,
            mean=mean,
            median=median,
            std=std,
            variance=variance,
            skewness=skewness,
            kurtosis=kurtosis,
            p01=float(p01),
            p05=float(p05),
            p25=float(p25),
            p75=float(p75),
            p95=float(p95),
            p99=float(p99),
            iqr=float(iqr),
            coefficient_of_variation=cv,
            suggested_transform=transform,
            warnings=warnings,
        )
    
    def get_problematic_metrics(
        self,
        profiles: Dict[str, MetricProfile],
    ) -> Dict[str, List[str]]:
        """
        Get metrics with potential issues.
        
        Returns:
            Dictionary with issue type as key and list of metric names as value
        """
        issues = {
            'constant': [],
            'high_missing': [],
            'high_skewness': [],
            'extreme_outliers': [],
            'low_unique': [],
        }
        
        for name, profile in profiles.items():
            if profile.suggested_transform.drop:
                issues['constant'].append(name)
            if profile.n_missing / profile.n_samples > self.HIGH_MISSING_RATIO:
                issues['high_missing'].append(name)
            if profile.skewness > self.HIGH_SKEWNESS_THRESHOLD:
                issues['high_skewness'].append(name)
            if profile.suggested_transform.clip_max is not None:
                issues['extreme_outliers'].append(name)
            if profile.n_unique <= self.LOW_UNIQUE_THRESHOLD:
                issues['low_unique'].append(name)
        
        return {k: v for k, v in issues.items() if v}