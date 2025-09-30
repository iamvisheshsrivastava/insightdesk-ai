# src/monitoring/drift_detector.py

"""
Data Drift Detection for the Intelligent Support System.

This module detects distribution changes in incoming data compared to training data.
It monitors both categorical features (product, priority, category) and text embeddings
using statistical methods and machine learning techniques.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json
import warnings

# Statistical imports with fallbacks
try:
    from scipy import stats
    from scipy.spatial.distance import jensen_shannon_distance
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Some drift detection methods will be limited.")

# ML imports with fallbacks
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Text drift detection will be limited.")

# Advanced drift detection imports with fallbacks
try:
    import evidently
    from evidently.report import Report
    from evidently.metrics import DataDriftPreset, TextDescriptorsDistribution
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logging.warning("Evidently not available. Advanced drift detection will be limited.")

try:
    from river import drift
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    logging.warning("River not available. Online drift detection will be limited.")

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class FeatureDrift:
    """Container for feature-specific drift metrics."""
    
    feature_name: str
    drift_score: float
    drift_detected: bool
    statistical_test: str
    p_value: Optional[float] = None
    threshold: float = 0.05
    
    # Distribution statistics
    reference_stats: Optional[Dict[str, Any]] = None
    current_stats: Optional[Dict[str, Any]] = None
    
    # For categorical features
    reference_distribution: Optional[Dict[str, float]] = None
    current_distribution: Optional[Dict[str, float]] = None
    
    # Additional metrics
    jensen_shannon_distance: Optional[float] = None
    kolmogorov_smirnov_stat: Optional[float] = None
    chi_square_stat: Optional[float] = None
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "feature_name": self.feature_name,
            "drift_score": self.drift_score,
            "drift_detected": self.drift_detected,
            "statistical_test": self.statistical_test,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat()
        }
        
        # Add optional fields
        optional_fields = [
            "p_value", "reference_stats", "current_stats",
            "reference_distribution", "current_distribution",
            "jensen_shannon_distance", "kolmogorov_smirnov_stat", "chi_square_stat"
        ]
        
        for field_name in optional_fields:
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        
        return result


@dataclass
class DriftResult:
    """Container for overall drift detection results."""
    
    overall_drift_detected: bool
    drift_score: float
    num_drifted_features: int
    total_features: int
    
    # Feature-specific results
    feature_drifts: List[FeatureDrift]
    
    # Summary statistics
    max_drift_score: float
    avg_drift_score: float
    most_drifted_feature: Optional[str] = None
    
    # Metadata
    detection_method: str = "statistical"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    sample_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "overall_drift_detected": self.overall_drift_detected,
            "drift_score": self.drift_score,
            "num_drifted_features": self.num_drifted_features,
            "total_features": self.total_features,
            "max_drift_score": self.max_drift_score,
            "avg_drift_score": self.avg_drift_score,
            "most_drifted_feature": self.most_drifted_feature,
            "detection_method": self.detection_method,
            "timestamp": self.timestamp.isoformat(),
            "sample_size": self.sample_size,
            "feature_drifts": [fd.to_dict() for fd in self.feature_drifts]
        }


@dataclass
class DriftMetrics:
    """Container for drift monitoring metrics over time."""
    
    model_name: str
    historical_drift_scores: List[float] = field(default_factory=list)
    historical_timestamps: List[datetime] = field(default_factory=list)
    recent_drift_results: List[DriftResult] = field(default_factory=list)
    
    # Alert thresholds
    drift_alert_threshold: float = 0.3
    consecutive_drift_threshold: int = 3
    
    # Current status
    current_drift_status: str = "stable"  # stable, warning, critical
    consecutive_drift_count: int = 0
    
    def add_drift_result(self, result: DriftResult):
        """Add new drift detection result."""
        self.recent_drift_results.append(result)
        self.historical_drift_scores.append(result.drift_score)
        self.historical_timestamps.append(result.timestamp)
        
        # Keep only recent results (last 100)
        if len(self.recent_drift_results) > 100:
            self.recent_drift_results.pop(0)
            self.historical_drift_scores.pop(0)
            self.historical_timestamps.pop(0)
        
        # Update drift status
        self._update_drift_status(result)
    
    def _update_drift_status(self, result: DriftResult):
        """Update current drift status based on recent results."""
        if result.drift_score > self.drift_alert_threshold:
            self.consecutive_drift_count += 1
        else:
            self.consecutive_drift_count = 0
        
        # Determine status
        if self.consecutive_drift_count >= self.consecutive_drift_threshold:
            self.current_drift_status = "critical"
        elif result.drift_score > self.drift_alert_threshold:
            self.current_drift_status = "warning"
        else:
            self.current_drift_status = "stable"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get drift monitoring summary."""
        summary = {
            "model_name": self.model_name,
            "current_status": self.current_drift_status,
            "consecutive_drift_count": self.consecutive_drift_count,
            "alert_threshold": self.drift_alert_threshold,
            "total_evaluations": len(self.historical_drift_scores)
        }
        
        if self.historical_drift_scores:
            summary.update({
                "latest_drift_score": self.historical_drift_scores[-1],
                "avg_drift_score": np.mean(self.historical_drift_scores),
                "max_drift_score": np.max(self.historical_drift_scores),
                "drift_trend": self._calculate_drift_trend()
            })
        
        if self.recent_drift_results:
            latest_result = self.recent_drift_results[-1]
            summary.update({
                "latest_evaluation": latest_result.timestamp.isoformat(),
                "num_drifted_features": latest_result.num_drifted_features,
                "most_drifted_feature": latest_result.most_drifted_feature
            })
        
        return summary
    
    def _calculate_drift_trend(self) -> float:
        """Calculate drift trend over time."""
        if len(self.historical_drift_scores) < 2:
            return 0.0
        
        x = np.arange(len(self.historical_drift_scores))
        y = np.array(self.historical_drift_scores)
        
        # Simple linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)


class DataDriftDetector:
    """Detect data drift in categorical and text features."""
    
    def __init__(self,
                 drift_threshold: float = 0.3,
                 statistical_threshold: float = 0.05,
                 min_sample_size: int = 50,
                 text_similarity_threshold: float = 0.8):
        """
        Initialize drift detector.
        
        Args:
            drift_threshold: Threshold for overall drift detection
            statistical_threshold: P-value threshold for statistical tests
            min_sample_size: Minimum samples needed for drift detection
            text_similarity_threshold: Threshold for text similarity
        """
        self.drift_threshold = drift_threshold
        self.statistical_threshold = statistical_threshold
        self.min_sample_size = min_sample_size
        self.text_similarity_threshold = text_similarity_threshold
        
        # Reference data storage
        self.reference_data: Dict[str, Any] = {}
        self.reference_text_features: Dict[str, Any] = {}
        
        # Drift metrics storage
        self.drift_metrics: Dict[str, DriftMetrics] = {}
        
        # Text vectorizers for consistency
        self.text_vectorizers: Dict[str, Any] = {}
        
        logger.info(f"Drift detector initialized with threshold={drift_threshold}")
    
    def set_reference_data(self, 
                          reference_data: pd.DataFrame,
                          categorical_columns: List[str],
                          text_columns: List[str],
                          model_name: str = "default"):
        """
        Set reference (training) data for drift detection.
        
        Args:
            reference_data: Reference dataset
            categorical_columns: List of categorical column names
            text_columns: List of text column names
            model_name: Name of the model for tracking
        """
        logger.info(f"Setting reference data for {model_name}")
        
        if model_name not in self.drift_metrics:
            self.drift_metrics[model_name] = DriftMetrics(
                model_name=model_name,
                drift_alert_threshold=self.drift_threshold
            )
        
        # Store reference statistics for categorical features
        categorical_stats = {}
        for col in categorical_columns:
            if col in reference_data.columns:
                value_counts = reference_data[col].value_counts(normalize=True)
                categorical_stats[col] = {
                    "distribution": value_counts.to_dict(),
                    "unique_values": set(reference_data[col].unique()),
                    "null_rate": reference_data[col].isnull().mean()
                }
        
        # Store reference statistics for text features
        text_stats = {}
        for col in text_columns:
            if col in reference_data.columns:
                # Clean and prepare text
                texts = reference_data[col].dropna().astype(str).tolist()
                
                if texts and SKLEARN_AVAILABLE:
                    # Create TF-IDF vectorizer
                    vectorizer = TfidfVectorizer(
                        max_features=1000,
                        stop_words='english',
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95
                    )
                    
                    try:
                        tfidf_matrix = vectorizer.fit_transform(texts)
                        
                        text_stats[col] = {
                            "avg_length": np.mean([len(text) for text in texts]),
                            "vocab_size": len(vectorizer.vocabulary_),
                            "document_count": len(texts),
                            "tfidf_mean": np.mean(tfidf_matrix.toarray()),
                            "tfidf_std": np.std(tfidf_matrix.toarray())
                        }
                        
                        # Store vectorizer for consistent comparison
                        self.text_vectorizers[f"{model_name}_{col}"] = vectorizer
                        
                    except Exception as e:
                        logger.warning(f"Could not process text column {col}: {e}")
                        text_stats[col] = {
                            "avg_length": np.mean([len(text) for text in texts]),
                            "document_count": len(texts)
                        }
        
        # Store reference data
        self.reference_data[model_name] = {
            "categorical_stats": categorical_stats,
            "text_stats": text_stats,
            "sample_size": len(reference_data),
            "timestamp": datetime.utcnow()
        }
        
        logger.info(f"Reference data set for {model_name}: "
                   f"{len(categorical_stats)} categorical, {len(text_stats)} text features")
    
    def detect_drift(self,
                    current_data: pd.DataFrame,
                    categorical_columns: List[str],
                    text_columns: List[str],
                    model_name: str = "default") -> DriftResult:
        """
        Detect drift in current data compared to reference data.
        
        Args:
            current_data: Current dataset to check for drift
            categorical_columns: List of categorical column names
            text_columns: List of text column names
            model_name: Name of the model
            
        Returns:
            DriftResult with drift detection results
        """
        if model_name not in self.reference_data:
            raise ValueError(f"No reference data set for model {model_name}")
        
        if len(current_data) < self.min_sample_size:
            logger.warning(f"Current data size ({len(current_data)}) below minimum ({self.min_sample_size})")
        
        feature_drifts = []
        drift_scores = []
        
        # Detect categorical drift
        for col in categorical_columns:
            if col in current_data.columns:
                drift = self._detect_categorical_drift(
                    current_data[col], col, model_name
                )
                if drift:
                    feature_drifts.append(drift)
                    drift_scores.append(drift.drift_score)
        
        # Detect text drift
        for col in text_columns:
            if col in current_data.columns:
                drift = self._detect_text_drift(
                    current_data[col], col, model_name
                )
                if drift:
                    feature_drifts.append(drift)
                    drift_scores.append(drift.drift_score)
        
        # Calculate overall drift metrics
        if drift_scores:
            max_drift_score = max(drift_scores)
            avg_drift_score = np.mean(drift_scores)
            overall_drift_detected = max_drift_score > self.drift_threshold
            
            most_drifted_feature = None
            if feature_drifts:
                most_drifted = max(feature_drifts, key=lambda x: x.drift_score)
                most_drifted_feature = most_drifted.feature_name
        else:
            max_drift_score = 0.0
            avg_drift_score = 0.0
            overall_drift_detected = False
            most_drifted_feature = None
        
        # Count drifted features
        num_drifted_features = sum(1 for fd in feature_drifts if fd.drift_detected)
        
        # Create result
        result = DriftResult(
            overall_drift_detected=overall_drift_detected,
            drift_score=max_drift_score,
            num_drifted_features=num_drifted_features,
            total_features=len(feature_drifts),
            feature_drifts=feature_drifts,
            max_drift_score=max_drift_score,
            avg_drift_score=avg_drift_score,
            most_drifted_feature=most_drifted_feature,
            detection_method="statistical",
            sample_size=len(current_data)
        )
        
        # Store result
        self.drift_metrics[model_name].add_drift_result(result)
        
        logger.info(f"Drift detection completed for {model_name}: "
                   f"drift_score={max_drift_score:.3f}, "
                   f"drifted_features={num_drifted_features}/{len(feature_drifts)}")
        
        return result
    
    def _detect_categorical_drift(self,
                                 current_series: pd.Series,
                                 feature_name: str,
                                 model_name: str) -> Optional[FeatureDrift]:
        """Detect drift in categorical feature."""
        if model_name not in self.reference_data:
            return None
        
        ref_stats = self.reference_data[model_name]["categorical_stats"]
        if feature_name not in ref_stats:
            return None
        
        ref_dist = ref_stats[feature_name]["distribution"]
        
        # Calculate current distribution
        current_counts = current_series.value_counts(normalize=True)
        current_dist = current_counts.to_dict()
        
        # Align distributions (handle new/missing categories)
        all_categories = set(ref_dist.keys()) | set(current_dist.keys())
        
        ref_aligned = np.array([ref_dist.get(cat, 0.0) for cat in all_categories])
        current_aligned = np.array([current_dist.get(cat, 0.0) for cat in all_categories])
        
        # Calculate drift metrics
        drift_score = 0.0
        statistical_test = "chi_square"
        p_value = None
        js_distance = None
        chi_square_stat = None
        
        # Jensen-Shannon Distance
        if SCIPY_AVAILABLE:
            try:
                # Add small epsilon to avoid log(0)
                epsilon = 1e-10
                ref_smooth = ref_aligned + epsilon
                current_smooth = current_aligned + epsilon
                
                # Normalize
                ref_smooth = ref_smooth / ref_smooth.sum()
                current_smooth = current_smooth / current_smooth.sum()
                
                js_distance = jensen_shannon_distance(ref_smooth, current_smooth)
                drift_score = js_distance
                statistical_test = "jensen_shannon"
                
            except Exception as e:
                logger.warning(f"Could not calculate Jensen-Shannon distance: {e}")
        
        # Chi-square test
        if drift_score == 0.0:  # Fallback if JS distance failed
            try:
                # Expected frequencies (from reference)
                total_current = len(current_series)
                expected = ref_aligned * total_current
                observed = np.array([current_series.value_counts().get(cat, 0) 
                                   for cat in all_categories])
                
                # Avoid categories with zero expected frequency
                mask = expected > 5
                if mask.sum() > 1:
                    if SCIPY_AVAILABLE:
                        chi_square_stat, p_value = stats.chisquare(
                            observed[mask], expected[mask]
                        )
                        drift_score = min(1.0, chi_square_stat / 100.0)  # Normalize
                    else:
                        # Simple alternative: sum of squared differences
                        drift_score = np.sum((observed[mask] - expected[mask]) ** 2) / np.sum(expected[mask])
                        drift_score = min(1.0, drift_score / 10.0)  # Normalize
                
            except Exception as e:
                logger.warning(f"Could not calculate chi-square test: {e}")
                drift_score = 0.0
        
        # Determine if drift detected
        drift_detected = drift_score > self.drift_threshold
        
        return FeatureDrift(
            feature_name=feature_name,
            drift_score=drift_score,
            drift_detected=drift_detected,
            statistical_test=statistical_test,
            p_value=p_value,
            threshold=self.drift_threshold,
            reference_distribution=ref_dist,
            current_distribution=current_dist,
            jensen_shannon_distance=js_distance,
            chi_square_stat=chi_square_stat
        )
    
    def _detect_text_drift(self,
                          current_series: pd.Series,
                          feature_name: str,
                          model_name: str) -> Optional[FeatureDrift]:
        """Detect drift in text feature."""
        if model_name not in self.reference_data:
            return None
        
        ref_stats = self.reference_data[model_name]["text_stats"]
        if feature_name not in ref_stats:
            return None
        
        # Get current text data
        current_texts = current_series.dropna().astype(str).tolist()
        if not current_texts:
            return None
        
        # Calculate basic statistics
        current_avg_length = np.mean([len(text) for text in current_texts])
        ref_avg_length = ref_stats[feature_name]["avg_length"]
        
        # Length-based drift score
        length_drift = abs(current_avg_length - ref_avg_length) / ref_avg_length
        
        drift_score = length_drift
        statistical_test = "length_comparison"
        
        # Advanced text drift detection if sklearn available
        vectorizer_key = f"{model_name}_{feature_name}"
        if (SKLEARN_AVAILABLE and 
            vectorizer_key in self.text_vectorizers and
            len(current_texts) >= 10):
            
            try:
                vectorizer = self.text_vectorizers[vectorizer_key]
                
                # Transform current texts
                current_tfidf = vectorizer.transform(current_texts)
                current_tfidf_mean = np.mean(current_tfidf.toarray())
                current_tfidf_std = np.std(current_tfidf.toarray())
                
                # Compare with reference statistics
                ref_tfidf_mean = ref_stats[feature_name].get("tfidf_mean", 0)
                ref_tfidf_std = ref_stats[feature_name].get("tfidf_std", 1)
                
                # Calculate drift based on TF-IDF statistics
                mean_drift = abs(current_tfidf_mean - ref_tfidf_mean) / (ref_tfidf_mean + 1e-10)
                std_drift = abs(current_tfidf_std - ref_tfidf_std) / (ref_tfidf_std + 1e-10)
                
                tfidf_drift = (mean_drift + std_drift) / 2
                drift_score = max(drift_score, tfidf_drift)
                statistical_test = "tfidf_comparison"
                
            except Exception as e:
                logger.warning(f"Could not calculate TF-IDF drift for {feature_name}: {e}")
        
        # Determine if drift detected
        drift_detected = drift_score > self.drift_threshold
        
        # Current statistics
        current_stats = {
            "avg_length": current_avg_length,
            "document_count": len(current_texts)
        }
        
        return FeatureDrift(
            feature_name=feature_name,
            drift_score=drift_score,
            drift_detected=drift_detected,
            statistical_test=statistical_test,
            threshold=self.drift_threshold,
            reference_stats=ref_stats[feature_name],
            current_stats=current_stats
        )
    
    def get_drift_status(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get current drift status for a model."""
        if model_name not in self.drift_metrics:
            return None
        
        return self.drift_metrics[model_name].get_summary()
    
    def get_all_drift_status(self) -> Dict[str, Any]:
        """Get drift status for all monitored models."""
        return {
            "models": {
                name: metrics.get_summary()
                for name, metrics in self.drift_metrics.items()
            },
            "total_models": len(self.drift_metrics),
            "monitoring_timestamp": datetime.utcnow().isoformat()
        }
    
    def reset_drift_metrics(self, model_name: str):
        """Reset drift metrics for a specific model."""
        if model_name in self.drift_metrics:
            self.drift_metrics[model_name] = DriftMetrics(
                model_name=model_name,
                drift_alert_threshold=self.drift_threshold
            )
            logger.info(f"Reset drift metrics for model: {model_name}")


# Utility functions for drift detection

def detect_categorical_drift(reference_data: List[str],
                            current_data: List[str],
                            threshold: float = 0.3) -> FeatureDrift:
    """
    Detect drift in categorical data using distribution comparison.
    
    Args:
        reference_data: Reference categorical values
        current_data: Current categorical values
        threshold: Drift detection threshold
        
    Returns:
        FeatureDrift result
    """
    # Calculate distributions
    ref_counts = Counter(reference_data)
    current_counts = Counter(current_data)
    
    total_ref = len(reference_data)
    total_current = len(current_data)
    
    ref_dist = {k: v/total_ref for k, v in ref_counts.items()}
    current_dist = {k: v/total_current for k, v in current_counts.items()}
    
    # Get all categories
    all_categories = set(ref_dist.keys()) | set(current_dist.keys())
    
    # Align distributions
    ref_aligned = np.array([ref_dist.get(cat, 0.0) for cat in all_categories])
    current_aligned = np.array([current_dist.get(cat, 0.0) for cat in all_categories])
    
    # Calculate Jensen-Shannon distance
    drift_score = 0.0
    if SCIPY_AVAILABLE:
        try:
            epsilon = 1e-10
            ref_smooth = ref_aligned + epsilon
            current_smooth = current_aligned + epsilon
            
            ref_smooth = ref_smooth / ref_smooth.sum()
            current_smooth = current_smooth / current_smooth.sum()
            
            drift_score = jensen_shannon_distance(ref_smooth, current_smooth)
        except Exception:
            # Fallback: simple sum of squared differences
            drift_score = np.sum((ref_aligned - current_aligned) ** 2)
    else:
        # Simple sum of squared differences
        drift_score = np.sum((ref_aligned - current_aligned) ** 2)
    
    return FeatureDrift(
        feature_name="categorical_feature",
        drift_score=drift_score,
        drift_detected=drift_score > threshold,
        statistical_test="jensen_shannon" if SCIPY_AVAILABLE else "sum_squared_diff",
        threshold=threshold,
        reference_distribution=ref_dist,
        current_distribution=current_dist,
        jensen_shannon_distance=drift_score if SCIPY_AVAILABLE else None
    )


def detect_text_drift(reference_texts: List[str],
                     current_texts: List[str],
                     threshold: float = 0.3) -> FeatureDrift:
    """
    Detect drift in text data using length and vocabulary comparison.
    
    Args:
        reference_texts: Reference text samples
        current_texts: Current text samples
        threshold: Drift detection threshold
        
    Returns:
        FeatureDrift result
    """
    # Calculate length statistics
    ref_lengths = [len(text) for text in reference_texts]
    current_lengths = [len(text) for text in current_texts]
    
    ref_avg_length = np.mean(ref_lengths)
    current_avg_length = np.mean(current_lengths)
    
    # Length-based drift
    length_drift = abs(current_avg_length - ref_avg_length) / (ref_avg_length + 1e-10)
    
    drift_score = length_drift
    statistical_test = "length_comparison"
    
    # Advanced comparison if sklearn available
    if SKLEARN_AVAILABLE and len(reference_texts) >= 10 and len(current_texts) >= 10:
        try:
            # Create vocabularies
            all_texts = reference_texts + current_texts
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            vectorizer.fit(all_texts)
            
            # Transform reference and current
            ref_tfidf = vectorizer.transform(reference_texts)
            current_tfidf = vectorizer.transform(current_texts)
            
            # Calculate similarity
            ref_centroid = np.mean(ref_tfidf.toarray(), axis=0)
            current_centroid = np.mean(current_tfidf.toarray(), axis=0)
            
            similarity = cosine_similarity([ref_centroid], [current_centroid])[0][0]
            tfidf_drift = 1.0 - similarity
            
            drift_score = max(drift_score, tfidf_drift)
            statistical_test = "tfidf_cosine_similarity"
            
        except Exception as e:
            logger.warning(f"Could not calculate advanced text drift: {e}")
    
    return FeatureDrift(
        feature_name="text_feature",
        drift_score=drift_score,
        drift_detected=drift_score > threshold,
        statistical_test=statistical_test,
        threshold=threshold,
        reference_stats={"avg_length": ref_avg_length, "document_count": len(reference_texts)},
        current_stats={"avg_length": current_avg_length, "document_count": len(current_texts)}
    )


def calculate_drift_severity(drift_score: float, thresholds: Dict[str, float] = None) -> str:
    """
    Calculate drift severity based on score.
    
    Args:
        drift_score: Drift score (0-1)
        thresholds: Custom thresholds for severity levels
        
    Returns:
        Severity level string
    """
    if thresholds is None:
        thresholds = {
            "low": 0.1,
            "medium": 0.3,
            "high": 0.5,
            "critical": 0.7
        }
    
    if drift_score < thresholds["low"]:
        return "stable"
    elif drift_score < thresholds["medium"]:
        return "low"
    elif drift_score < thresholds["high"]:
        return "medium"
    elif drift_score < thresholds["critical"]:
        return "high"
    else:
        return "critical"