# scripts/ab_testing_framework.py

"""
A/B Testing Framework Foundation for InsightDesk AI

This module provides the foundation for A/B testing different models
and configurations in production environments.

Features:
- Traffic splitting and routing
- Statistical significance testing
- Performance monitoring
- Automated decision making
"""

import sys
import json
import time
import random
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from scipy import stats
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))


class TestStatus(Enum):
    """A/B test status enumeration."""
    PLANNED = "planned"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class VariantType(Enum):
    """Test variant types."""
    CONTROL = "control"
    TREATMENT = "treatment"


@dataclass
class TestMetric:
    """Metric definition for A/B testing."""
    name: str
    metric_type: str  # 'ratio', 'count', 'continuous'
    primary: bool = False
    improvement_direction: str = "increase"  # 'increase', 'decrease'
    minimum_detectable_effect: float = 0.05  # 5% minimum effect
    
    def __post_init__(self):
        if self.improvement_direction not in ["increase", "decrease"]:
            raise ValueError("improvement_direction must be 'increase' or 'decrease'")


@dataclass
class TestVariant:
    """A/B test variant configuration."""
    name: str
    variant_type: VariantType
    traffic_allocation: float  # 0.0 to 1.0
    model_config: Dict[str, Any]
    description: str = ""
    
    def __post_init__(self):
        if not 0 <= self.traffic_allocation <= 1:
            raise ValueError("traffic_allocation must be between 0 and 1")


@dataclass
class TestConfiguration:
    """Complete A/B test configuration."""
    test_id: str
    name: str
    description: str
    variants: List[TestVariant]
    metrics: List[TestMetric]
    start_date: datetime
    end_date: datetime
    minimum_sample_size: int = 1000
    confidence_level: float = 0.95
    power: float = 0.8
    status: TestStatus = TestStatus.PLANNED
    
    def __post_init__(self):
        # Validate traffic allocation sums to 1
        total_allocation = sum(v.traffic_allocation for v in self.variants)
        if abs(total_allocation - 1.0) > 0.001:
            raise ValueError(f"Variant traffic allocations must sum to 1.0, got {total_allocation}")
        
        # Ensure exactly one control variant
        control_count = sum(1 for v in self.variants if v.variant_type == VariantType.CONTROL)
        if control_count != 1:
            raise ValueError("Exactly one control variant is required")


@dataclass
class MetricResult:
    """Result for a single metric in A/B test."""
    metric_name: str
    variant_name: str
    sample_size: int
    mean_value: float
    std_deviation: float
    confidence_interval: Tuple[float, float]
    timestamp: datetime


@dataclass
class StatisticalTestResult:
    """Statistical test result comparing variants."""
    metric_name: str
    control_variant: str
    treatment_variant: str
    control_mean: float
    treatment_mean: float
    effect_size: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    statistical_power: float
    test_type: str
    sample_sizes: Dict[str, int]


class TrafficRouter:
    """Routes traffic between test variants."""
    
    def __init__(self, test_config: TestConfiguration):
        self.test_config = test_config
        self.variants = {v.name: v for v in test_config.variants}
        
        # Build cumulative allocation for routing
        self.allocation_ranges = {}
        cumulative = 0.0
        
        for variant in test_config.variants:
            start = cumulative
            end = cumulative + variant.traffic_allocation
            self.allocation_ranges[variant.name] = (start, end)
            cumulative = end
    
    def route_request(self, user_id: str) -> str:
        """Route a request to a variant based on user ID."""
        # Use consistent hashing for stable assignment
        hash_input = f"{self.test_config.test_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        assignment_value = (hash_value % 10000) / 10000.0  # 0.0 to 1.0
        
        # Find variant based on allocation ranges
        for variant_name, (start, end) in self.allocation_ranges.items():
            if start <= assignment_value < end:
                return variant_name
        
        # Fallback to control (shouldn't happen with proper allocation)
        control_variant = next(v.name for v in self.test_config.variants 
                             if v.variant_type == VariantType.CONTROL)
        return control_variant
    
    def get_variant_config(self, variant_name: str) -> Dict[str, Any]:
        """Get model configuration for a variant."""
        return self.variants[variant_name].model_config


class MetricsCollector:
    """Collects and stores metrics for A/B testing."""
    
    def __init__(self, storage_dir: str = "ab_test_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # In-memory cache for recent metrics
        self.metrics_cache = {}
    
    def record_metric(self, test_id: str, variant_name: str, metric_name: str, 
                     value: float, user_id: str = None, metadata: Dict = None):
        """Record a metric value for a test variant."""
        timestamp = datetime.now()
        
        metric_record = {
            "test_id": test_id,
            "variant_name": variant_name,
            "metric_name": metric_name,
            "value": value,
            "user_id": user_id,
            "metadata": metadata or {},
            "timestamp": timestamp.isoformat()
        }
        
        # Store to file (append mode)
        metrics_file = self.storage_dir / f"{test_id}_metrics.jsonl"
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metric_record) + '\n')
        
        # Update cache
        cache_key = f"{test_id}:{variant_name}:{metric_name}"
        if cache_key not in self.metrics_cache:
            self.metrics_cache[cache_key] = []
        
        self.metrics_cache[cache_key].append({
            "value": value,
            "timestamp": timestamp,
            "user_id": user_id
        })
        
        # Limit cache size
        if len(self.metrics_cache[cache_key]) > 10000:
            self.metrics_cache[cache_key] = self.metrics_cache[cache_key][-5000:]
    
    def get_metrics(self, test_id: str, variant_name: str = None, 
                   metric_name: str = None, since: datetime = None) -> List[Dict]:
        """Retrieve metrics for analysis."""
        metrics_file = self.storage_dir / f"{test_id}_metrics.jsonl"
        
        if not metrics_file.exists():
            return []
        
        metrics = []
        with open(metrics_file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    
                    # Apply filters
                    if variant_name and record["variant_name"] != variant_name:
                        continue
                    
                    if metric_name and record["metric_name"] != metric_name:
                        continue
                    
                    if since:
                        record_time = datetime.fromisoformat(record["timestamp"])
                        if record_time < since:
                            continue
                    
                    metrics.append(record)
                    
                except json.JSONDecodeError:
                    continue
        
        return metrics


class StatisticalAnalyzer:
    """Performs statistical analysis for A/B tests."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def analyze_metric(self, test_config: TestConfiguration, metric_name: str,
                      metrics_data: List[Dict]) -> List[StatisticalTestResult]:
        """Analyze a metric across test variants."""
        
        # Group metrics by variant
        variant_data = {}
        for record in metrics_data:
            if record["metric_name"] == metric_name:
                variant = record["variant_name"]
                if variant not in variant_data:
                    variant_data[variant] = []
                variant_data[variant].append(record["value"])
        
        # Find control and treatment variants
        control_variant = next(v.name for v in test_config.variants 
                             if v.variant_type == VariantType.CONTROL)
        treatment_variants = [v.name for v in test_config.variants 
                            if v.variant_type == VariantType.TREATMENT]
        
        results = []
        
        # Compare each treatment to control
        for treatment_variant in treatment_variants:
            if control_variant not in variant_data or treatment_variant not in variant_data:
                continue
            
            control_values = variant_data[control_variant]
            treatment_values = variant_data[treatment_variant]
            
            if len(control_values) < 30 or len(treatment_values) < 30:
                # Not enough data for reliable testing
                continue
            
            result = self._perform_statistical_test(
                control_values, treatment_values,
                control_variant, treatment_variant, metric_name
            )
            
            results.append(result)
        
        return results
    
    def _perform_statistical_test(self, control_values: List[float], 
                                 treatment_values: List[float],
                                 control_name: str, treatment_name: str,
                                 metric_name: str) -> StatisticalTestResult:
        """Perform statistical test between two groups."""
        
        control_array = np.array(control_values)
        treatment_array = np.array(treatment_values)
        
        # Calculate basic statistics
        control_mean = np.mean(control_array)
        treatment_mean = np.mean(treatment_array)
        control_std = np.std(control_array, ddof=1)
        treatment_std = np.std(treatment_array, ddof=1)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control_array) - 1) * control_std**2 + 
                             (len(treatment_array) - 1) * treatment_std**2) / 
                            (len(control_array) + len(treatment_array) - 2))
        
        effect_size = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        # Statistical test
        if self._is_normal_distributed(control_array) and self._is_normal_distributed(treatment_array):
            # Use t-test for normal distributions
            statistic, p_value = stats.ttest_ind(control_array, treatment_array)
            test_type = "welch_t_test"
            
            # Confidence interval for difference of means
            se_diff = np.sqrt(control_std**2/len(control_array) + treatment_std**2/len(treatment_array))
            df = len(control_array) + len(treatment_array) - 2
            t_critical = stats.t.ppf(1 - self.alpha/2, df)
            margin_error = t_critical * se_diff
            diff = treatment_mean - control_mean
            ci_lower = diff - margin_error
            ci_upper = diff + margin_error
            
        else:
            # Use Mann-Whitney U test for non-normal distributions
            statistic, p_value = stats.mannwhitneyu(control_array, treatment_array, 
                                                   alternative='two-sided')
            test_type = "mann_whitney_u"
            
            # Bootstrap confidence interval for non-parametric case
            ci_lower, ci_upper = self._bootstrap_ci(control_array, treatment_array)
        
        # Calculate statistical power
        statistical_power = self._calculate_power(
            len(control_array), len(treatment_array), effect_size
        )
        
        # Determine significance
        is_significant = p_value < self.alpha
        
        return StatisticalTestResult(
            metric_name=metric_name,
            control_variant=control_name,
            treatment_variant=treatment_name,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            effect_size=effect_size,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=is_significant,
            statistical_power=statistical_power,
            test_type=test_type,
            sample_sizes={control_name: len(control_array), treatment_name: len(treatment_array)}
        )
    
    def _is_normal_distributed(self, data: np.ndarray, p_threshold: float = 0.05) -> bool:
        """Check if data is normally distributed using Shapiro-Wilk test."""
        if len(data) < 8:
            return False  # Too few samples for reliable test
        
        # Use smaller sample for large datasets to avoid overly sensitive test
        if len(data) > 5000:
            sample_data = np.random.choice(data, 5000, replace=False)
        else:
            sample_data = data
        
        _, p_value = stats.shapiro(sample_data)
        return p_value > p_threshold
    
    def _bootstrap_ci(self, control: np.ndarray, treatment: np.ndarray, 
                     n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for difference of means."""
        differences = []
        
        for _ in range(n_bootstrap):
            control_sample = np.random.choice(control, len(control), replace=True)
            treatment_sample = np.random.choice(treatment, len(treatment), replace=True)
            diff = np.mean(treatment_sample) - np.mean(control_sample)
            differences.append(diff)
        
        differences = np.array(differences)
        ci_lower = np.percentile(differences, (self.alpha/2) * 100)
        ci_upper = np.percentile(differences, (1 - self.alpha/2) * 100)
        
        return ci_lower, ci_upper
    
    def _calculate_power(self, n1: int, n2: int, effect_size: float) -> float:
        """Calculate statistical power for the test."""
        # Simplified power calculation for t-test
        # In practice, you might want to use more sophisticated methods
        
        if effect_size == 0:
            return self.alpha  # Power equals alpha when effect size is 0
        
        # Harmonic mean of sample sizes
        n_harmonic = 2 * n1 * n2 / (n1 + n2)
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(n_harmonic / 2)
        
        # Critical value
        df = n1 + n2 - 2
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        
        # Power calculation (simplified)
        power = 1 - stats.t.cdf(t_critical, df, ncp) + stats.t.cdf(-t_critical, df, ncp)
        
        return min(max(power, 0), 1)  # Clamp between 0 and 1


class ABTestManager:
    """Main class for managing A/B tests."""
    
    def __init__(self, storage_dir: str = "ab_test_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.metrics_collector = MetricsCollector(storage_dir)
        self.analyzer = StatisticalAnalyzer()
        
        # Load existing tests
        self.active_tests = {}
        self._load_existing_tests()
    
    def create_test(self, test_config: TestConfiguration) -> bool:
        """Create a new A/B test."""
        try:
            # Validate configuration
            self._validate_test_config(test_config)
            
            # Save test configuration
            config_file = self.storage_dir / f"{test_config.test_id}_config.json"
            with open(config_file, 'w') as f:
                # Convert dataclass to dict for JSON serialization
                config_dict = asdict(test_config)
                # Convert enums to strings
                config_dict['status'] = test_config.status.value
                config_dict['start_date'] = test_config.start_date.isoformat()
                config_dict['end_date'] = test_config.end_date.isoformat()
                
                # Convert variant enums
                for variant in config_dict['variants']:
                    variant['variant_type'] = variant['variant_type'].value
                
                json.dump(config_dict, f, indent=2)
            
            # Add to active tests if within date range
            now = datetime.now()
            if test_config.start_date <= now <= test_config.end_date:
                self.active_tests[test_config.test_id] = {
                    'config': test_config,
                    'router': TrafficRouter(test_config)
                }
            
            print(f"✅ A/B test '{test_config.name}' created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create A/B test: {e}")
            return False
    
    def start_test(self, test_id: str) -> bool:
        """Start an A/B test."""
        try:
            config_file = self.storage_dir / f"{test_id}_config.json"
            if not config_file.exists():
                raise ValueError(f"Test {test_id} not found")
            
            # Load and update config
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            
            config_dict['status'] = TestStatus.RUNNING.value
            
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            # Add to active tests
            self._load_test(test_id)
            
            print(f"✅ A/B test {test_id} started")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start test: {e}")
            return False
    
    def stop_test(self, test_id: str) -> bool:
        """Stop an A/B test."""
        try:
            if test_id in self.active_tests:
                del self.active_tests[test_id]
            
            # Update config file
            config_file = self.storage_dir / f"{test_id}_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                
                config_dict['status'] = TestStatus.COMPLETED.value
                
                with open(config_file, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            
            print(f"✅ A/B test {test_id} stopped")
            return True
            
        except Exception as e:
            print(f"❌ Failed to stop test: {e}")
            return False
    
    def route_request(self, test_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Route a request through A/B test."""
        if test_id not in self.active_tests:
            return None
        
        test_info = self.active_tests[test_id]
        variant_name = test_info['router'].route_request(user_id)
        variant_config = test_info['router'].get_variant_config(variant_name)
        
        return {
            'variant_name': variant_name,
            'config': variant_config,
            'test_id': test_id
        }
    
    def record_metric(self, test_id: str, variant_name: str, metric_name: str,
                     value: float, user_id: str = None, metadata: Dict = None):
        """Record a metric for A/B testing."""
        self.metrics_collector.record_metric(
            test_id, variant_name, metric_name, value, user_id, metadata
        )
    
    def analyze_test(self, test_id: str, since: datetime = None) -> Dict[str, Any]:
        """Analyze A/B test results."""
        try:
            # Load test configuration
            config_file = self.storage_dir / f"{test_id}_config.json"
            if not config_file.exists():
                raise ValueError(f"Test {test_id} not found")
            
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            
            # Get metrics data
            metrics_data = self.metrics_collector.get_metrics(test_id, since=since)
            
            if not metrics_data:
                return {"error": "No metrics data available"}
            
            # Analyze each metric
            results = {}
            metric_names = set(record["metric_name"] for record in metrics_data)
            
            for metric_name in metric_names:
                # Reconstruct test config for analysis
                test_config = self._dict_to_test_config(config_dict)
                
                # Perform statistical analysis
                statistical_results = self.analyzer.analyze_metric(
                    test_config, metric_name, metrics_data
                )
                
                results[metric_name] = [asdict(result) for result in statistical_results]
            
            # Calculate summary statistics
            summary = self._calculate_summary_stats(metrics_data, config_dict)
            
            return {
                "test_id": test_id,
                "analysis_timestamp": datetime.now().isoformat(),
                "summary": summary,
                "statistical_results": results,
                "recommendations": self._generate_recommendations(results, config_dict)
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {e}"}
    
    def _validate_test_config(self, config: TestConfiguration):
        """Validate A/B test configuration."""
        if config.start_date >= config.end_date:
            raise ValueError("Start date must be before end date")
        
        if len(config.variants) < 2:
            raise ValueError("At least 2 variants required")
        
        if not config.metrics:
            raise ValueError("At least one metric must be defined")
        
        primary_metrics = sum(1 for m in config.metrics if m.primary)
        if primary_metrics != 1:
            raise ValueError("Exactly one primary metric required")
    
    def _load_existing_tests(self):
        """Load existing test configurations."""
        for config_file in self.storage_dir.glob("*_config.json"):
            test_id = config_file.stem.replace("_config", "")
            self._load_test(test_id)
    
    def _load_test(self, test_id: str):
        """Load a specific test configuration."""
        try:
            config_file = self.storage_dir / f"{test_id}_config.json"
            
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            
            test_config = self._dict_to_test_config(config_dict)
            
            # Only add to active tests if running and within date range
            now = datetime.now()
            if (test_config.status == TestStatus.RUNNING and 
                test_config.start_date <= now <= test_config.end_date):
                
                self.active_tests[test_id] = {
                    'config': test_config,
                    'router': TrafficRouter(test_config)
                }
                
        except Exception as e:
            print(f"⚠️ Failed to load test {test_id}: {e}")
    
    def _dict_to_test_config(self, config_dict: Dict) -> TestConfiguration:
        """Convert dictionary back to TestConfiguration object."""
        # Convert date strings back to datetime objects
        config_dict['start_date'] = datetime.fromisoformat(config_dict['start_date'])
        config_dict['end_date'] = datetime.fromisoformat(config_dict['end_date'])
        config_dict['status'] = TestStatus(config_dict['status'])
        
        # Convert variants
        variants = []
        for v_dict in config_dict['variants']:
            v_dict['variant_type'] = VariantType(v_dict['variant_type'])
            variants.append(TestVariant(**v_dict))
        config_dict['variants'] = variants
        
        # Convert metrics
        metrics = []
        for m_dict in config_dict['metrics']:
            metrics.append(TestMetric(**m_dict))
        config_dict['metrics'] = metrics
        
        return TestConfiguration(**config_dict)
    
    def _calculate_summary_stats(self, metrics_data: List[Dict], 
                                config_dict: Dict) -> Dict[str, Any]:
        """Calculate summary statistics for the test."""
        summary = {
            "total_samples": len(metrics_data),
            "start_time": min(record["timestamp"] for record in metrics_data) if metrics_data else None,
            "end_time": max(record["timestamp"] for record in metrics_data) if metrics_data else None,
            "variants": {},
            "metrics": {}
        }
        
        # Per-variant statistics
        for variant_dict in config_dict['variants']:
            variant_name = variant_dict['name']
            variant_data = [r for r in metrics_data if r["variant_name"] == variant_name]
            
            summary["variants"][variant_name] = {
                "sample_count": len(variant_data),
                "allocation": variant_dict['traffic_allocation']
            }
        
        # Per-metric statistics
        metric_names = set(record["metric_name"] for record in metrics_data)
        for metric_name in metric_names:
            metric_data = [r for r in metrics_data if r["metric_name"] == metric_name]
            values = [r["value"] for r in metric_data]
            
            summary["metrics"][metric_name] = {
                "sample_count": len(values),
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values)
            }
        
        return summary
    
    def _generate_recommendations(self, results: Dict[str, List], 
                                 config_dict: Dict) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check if we have enough data
        for metric_name, metric_results in results.items():
            if not metric_results:
                recommendations.append(f"⚠️ Insufficient data for {metric_name} - continue collecting")
                continue
            
            for result in metric_results:
                sample_sizes = result['sample_sizes']
                min_sample = min(sample_sizes.values())
                
                if min_sample < 1000:
                    recommendations.append(
                        f"⚠️ Sample size too small for {metric_name} "
                        f"({min_sample} < 1000) - continue test"
                    )
                
                # Check statistical significance
                if result['is_significant']:
                    effect_direction = "increase" if result['effect_size'] > 0 else "decrease"
                    recommendations.append(
                        f"✅ Significant {effect_direction} in {metric_name} "
                        f"for {result['treatment_variant']} (p={result['p_value']:.4f})"
                    )
                
                # Check statistical power
                if result['statistical_power'] < 0.8:
                    recommendations.append(
                        f"⚠️ Low statistical power for {metric_name} "
                        f"({result['statistical_power']:.2f} < 0.8) - increase sample size"
                    )
        
        # Overall recommendation
        significant_results = []
        for metric_results in results.values():
            for result in metric_results:
                if result['is_significant']:
                    significant_results.append(result)
        
        if significant_results:
            recommendations.append(
                f"🎯 Consider implementing winning variant based on "
                f"{len(significant_results)} significant results"
            )
        else:
            recommendations.append(
                "📊 No significant differences detected - consider longer test duration"
            )
        
        return recommendations


# Example usage and testing functions
def create_sample_test() -> TestConfiguration:
    """Create a sample A/B test configuration."""
    return TestConfiguration(
        test_id="model_comparison_001",
        name="XGBoost vs TensorFlow Model Test",
        description="Compare XGBoost and TensorFlow models for ticket classification",
        variants=[
            TestVariant(
                name="control_xgboost",
                variant_type=VariantType.CONTROL,
                traffic_allocation=0.5,
                model_config={"model_type": "xgboost", "confidence_threshold": 0.7},
                description="Current XGBoost model"
            ),
            TestVariant(
                name="treatment_tensorflow",
                variant_type=VariantType.TREATMENT,
                traffic_allocation=0.5,
                model_config={"model_type": "tensorflow", "confidence_threshold": 0.7},
                description="New TensorFlow model"
            )
        ],
        metrics=[
            TestMetric(
                name="classification_accuracy",
                metric_type="ratio",
                primary=True,
                improvement_direction="increase",
                minimum_detectable_effect=0.02
            ),
            TestMetric(
                name="inference_latency_ms",
                metric_type="continuous",
                primary=False,
                improvement_direction="decrease",
                minimum_detectable_effect=0.1
            ),
            TestMetric(
                name="agent_satisfaction",
                metric_type="ratio",
                primary=False,
                improvement_direction="increase",
                minimum_detectable_effect=0.05
            )
        ],
        start_date=datetime.now(),
        end_date=datetime.now() + timedelta(days=14),
        minimum_sample_size=2000,
        confidence_level=0.95,
        power=0.8
    )


def main():
    """Main function for A/B testing framework demo."""
    print("🧪 A/B Testing Framework for InsightDesk AI")
    print("=" * 45)
    
    # Initialize A/B test manager
    ab_manager = ABTestManager()
    
    # Create sample test
    print("\n📋 Creating sample A/B test...")
    test_config = create_sample_test()
    
    if ab_manager.create_test(test_config):
        print(f"✅ Test '{test_config.name}' created successfully")
        
        # Start the test
        ab_manager.start_test(test_config.test_id)
        
        # Simulate some traffic and metrics
        print("\n🔄 Simulating test traffic and metrics...")
        
        for i in range(100):
            user_id = f"user_{i}"
            
            # Route request
            routing_result = ab_manager.route_request(test_config.test_id, user_id)
            
            if routing_result:
                variant_name = routing_result['variant_name']
                
                # Simulate metrics based on variant
                if variant_name == "control_xgboost":
                    accuracy = random.gauss(0.82, 0.05)  # 82% accuracy
                    latency = random.gauss(50, 10)       # 50ms latency
                    satisfaction = random.gauss(0.75, 0.1)  # 75% satisfaction
                else:
                    accuracy = random.gauss(0.85, 0.05)  # 85% accuracy (better)
                    latency = random.gauss(80, 15)       # 80ms latency (worse)
                    satisfaction = random.gauss(0.78, 0.1)  # 78% satisfaction
                
                # Record metrics
                ab_manager.record_metric(
                    test_config.test_id, variant_name, 
                    "classification_accuracy", max(0, min(1, accuracy)), user_id
                )
                ab_manager.record_metric(
                    test_config.test_id, variant_name,
                    "inference_latency_ms", max(1, latency), user_id
                )
                ab_manager.record_metric(
                    test_config.test_id, variant_name,
                    "agent_satisfaction", max(0, min(1, satisfaction)), user_id
                )
        
        print("✅ Simulation complete")
        
        # Analyze results
        print("\n📊 Analyzing test results...")
        analysis = ab_manager.analyze_test(test_config.test_id)
        
        if "error" not in analysis:
            print(f"📈 Analysis completed for test: {analysis['test_id']}")
            print(f"📊 Total samples: {analysis['summary']['total_samples']}")
            
            print("\n🎯 Recommendations:")
            for rec in analysis['recommendations']:
                print(f"   {rec}")
        else:
            print(f"❌ Analysis failed: {analysis['error']}")
        
        # Stop the test
        ab_manager.stop_test(test_config.test_id)
    
    print("\n🎉 A/B Testing Framework Demo Complete!")
    print("This framework provides the foundation for production A/B testing.")


if __name__ == "__main__":
    main()