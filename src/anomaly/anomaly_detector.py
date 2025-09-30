# src/anomaly/anomaly_detector.py

"""
Core anomaly detection classes for the InsightDesk AI system.
Implements volume spikes, sentiment shifts, new issues, and outlier detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import defaultdict, Counter
import logging
from abc import ABC, abstractmethod

# ML imports with fallbacks
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Some anomaly detection features will be limited.")

from .anomaly_models import (
    AnomalyRecord, AnomalyType, AnomalySeverity, AnomalyThresholds,
    AnomalyDetectionResult, VolumeSpike, SentimentShift, NewIssue, OutlierInfo,
    create_anomaly_record, determine_severity
)

logger = logging.getLogger(__name__)


class BaseAnomalyDetector(ABC):
    """Base class for all anomaly detectors."""
    
    def __init__(self, thresholds: Optional[AnomalyThresholds] = None):
        """
        Initialize base anomaly detector.
        
        Args:
            thresholds: Configuration thresholds for detection
        """
        self.thresholds = thresholds or AnomalyThresholds()
        self.detection_history: List[AnomalyRecord] = []
        
    @abstractmethod
    def detect(self, tickets_data: List[Dict[str, Any]]) -> List[AnomalyRecord]:
        """
        Detect anomalies in the given ticket data.
        
        Args:
            tickets_data: List of ticket dictionaries
            
        Returns:
            List of detected anomalies
        """
        pass
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string to datetime object."""
        try:
            # Try multiple common formats
            for fmt in [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%d",
                "%m/%d/%Y %H:%M:%S",
                "%m/%d/%Y"
            ]:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue
            
            # If all formats fail, try pandas
            return pd.to_datetime(timestamp_str)
            
        except Exception as e:
            logger.warning(f"Failed to parse timestamp '{timestamp_str}': {e}")
            return datetime.utcnow()
    
    def _filter_recent_tickets(
        self, 
        tickets_data: List[Dict[str, Any]], 
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Filter tickets to include only recent ones."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_tickets = []
        
        for ticket in tickets_data:
            created_at = self._parse_timestamp(ticket.get("created_at", ""))
            if created_at >= cutoff_date:
                recent_tickets.append(ticket)
        
        logger.info(f"Filtered {len(recent_tickets)} recent tickets from {len(tickets_data)} total")
        return recent_tickets


class VolumeAnomalyDetector(BaseAnomalyDetector):
    """Detects volume spikes in ticket categories."""
    
    def detect(self, tickets_data: List[Dict[str, Any]]) -> List[AnomalyRecord]:
        """
        Detect volume spikes using rolling statistics.
        
        Args:
            tickets_data: List of ticket dictionaries
            
        Returns:
            List of volume spike anomalies
        """
        logger.info("Starting volume spike detection...")
        anomalies = []
        
        if len(tickets_data) < self.thresholds.min_tickets_for_analysis:
            logger.warning(f"Not enough tickets for volume analysis: {len(tickets_data)}")
            return anomalies
        
        try:
            # Create DataFrame with parsed timestamps
            df_data = []
            for ticket in tickets_data:
                df_data.append({
                    'created_at': self._parse_timestamp(ticket.get("created_at", "")),
                    'category': ticket.get("category", "unknown"),
                    'product': ticket.get("product", "unknown"),
                    'ticket_id': ticket.get("ticket_id", "")
                })
            
            df = pd.DataFrame(df_data)
            
            # Group by category and date
            df['date'] = df['created_at'].dt.date
            daily_counts = df.groupby(['category', 'date']).size().reset_index(name='count')
            
            # Detect spikes for each category
            for category in daily_counts['category'].unique():
                category_data = daily_counts[daily_counts['category'] == category]
                spike_anomalies = self._detect_category_volume_spikes(category_data, category)
                anomalies.extend(spike_anomalies)
            
            logger.info(f"Detected {len(anomalies)} volume spike anomalies")
            
        except Exception as e:
            logger.error(f"Error in volume spike detection: {e}")
        
        return anomalies
    
    def _detect_category_volume_spikes(
        self, 
        category_data: pd.DataFrame, 
        category: str
    ) -> List[AnomalyRecord]:
        """Detect volume spikes for a specific category."""
        anomalies = []
        
        if len(category_data) < self.thresholds.volume_rolling_window:
            return anomalies
        
        # Calculate rolling statistics
        counts = category_data['count'].values
        rolling_mean = pd.Series(counts).rolling(
            window=self.thresholds.volume_rolling_window, 
            min_periods=1
        ).mean()
        rolling_std = pd.Series(counts).rolling(
            window=self.thresholds.volume_rolling_window, 
            min_periods=1
        ).std()
        
        # Detect spikes
        for i, (_, row) in enumerate(category_data.iterrows()):
            if i >= self.thresholds.volume_rolling_window:
                mean_val = rolling_mean.iloc[i-1]  # Use previous window mean
                std_val = rolling_std.iloc[i-1]   # Use previous window std
                
                if std_val > 0:  # Avoid division by zero
                    threshold = mean_val + (self.thresholds.volume_spike_sigma * std_val)
                    actual_count = row['count']
                    
                    if actual_count > threshold:
                        # Calculate spike metrics
                        spike_ratio = actual_count / mean_val if mean_val > 0 else float('inf')
                        score = min(1.0, (actual_count - threshold) / (threshold + 1))  # Normalize score
                        
                        # Determine severity based on spike magnitude
                        severity = determine_severity(spike_ratio / 5.0)  # Normalize spike ratio
                        
                        # Create volume spike details
                        volume_spike = VolumeSpike(
                            category=category,
                            date=datetime.combine(row['date'], datetime.min.time()),
                            actual_count=int(actual_count),
                            expected_count=float(mean_val),
                            spike_ratio=float(spike_ratio),
                            historical_mean=float(mean_val),
                            historical_std=float(std_val)
                        )
                        
                        anomaly = create_anomaly_record(
                            anomaly_type=AnomalyType.VOLUME_SPIKE,
                            description=f"Volume spike detected in {category}: {actual_count} tickets "
                                       f"(expected ~{mean_val:.1f}, {spike_ratio:.1f}x normal)",
                            score=score,
                            severity=severity,
                            category=category,
                            details={
                                "volume_spike": volume_spike.dict(),
                                "threshold_exceeded": float(actual_count - threshold),
                                "sigma_multiplier": self.thresholds.volume_spike_sigma
                            },
                            baseline_value=float(mean_val),
                            actual_value=float(actual_count),
                            threshold=float(threshold)
                        )
                        
                        anomalies.append(anomaly)
                        
                        logger.info(
                            f"Volume spike detected: {category} on {row['date']} "
                            f"({actual_count} vs expected {mean_val:.1f})"
                        )
        
        return anomalies


class SentimentAnomalyDetector(BaseAnomalyDetector):
    """Detects sentiment shifts in ticket categories."""
    
    def __init__(self, thresholds: Optional[AnomalyThresholds] = None):
        super().__init__(thresholds)
        self.sentiment_mapping = {
            'frustrated': -1.0,
            'angry': -1.0,
            'disappointed': -0.5,
            'neutral': 0.0,
            'satisfied': 0.5,
            'happy': 1.0,
            'positive': 1.0
        }
    
    def detect(self, tickets_data: List[Dict[str, Any]]) -> List[AnomalyRecord]:
        """
        Detect sentiment shifts by category.
        
        Args:
            tickets_data: List of ticket dictionaries
            
        Returns:
            List of sentiment shift anomalies
        """
        logger.info("Starting sentiment shift detection...")
        anomalies = []
        
        if len(tickets_data) < self.thresholds.min_tickets_for_analysis:
            logger.warning(f"Not enough tickets for sentiment analysis: {len(tickets_data)}")
            return anomalies
        
        try:
            # Parse sentiment data
            sentiment_data = []
            for ticket in tickets_data:
                sentiment_str = ticket.get("customer_sentiment", "neutral").lower()
                sentiment_score = self.sentiment_mapping.get(sentiment_str, 0.0)
                
                sentiment_data.append({
                    'created_at': self._parse_timestamp(ticket.get("created_at", "")),
                    'category': ticket.get("category", "unknown"),
                    'product': ticket.get("product", "unknown"),
                    'sentiment_score': sentiment_score,
                    'ticket_id': ticket.get("ticket_id", "")
                })
            
            df = pd.DataFrame(sentiment_data)
            
            # Detect sentiment shifts for each category
            for category in df['category'].unique():
                category_data = df[df['category'] == category]
                if len(category_data) >= 10:  # Need minimum samples
                    shift_anomalies = self._detect_category_sentiment_shifts(category_data, category)
                    anomalies.extend(shift_anomalies)
            
            logger.info(f"Detected {len(anomalies)} sentiment shift anomalies")
            
        except Exception as e:
            logger.error(f"Error in sentiment shift detection: {e}")
        
        return anomalies
    
    def _detect_category_sentiment_shifts(
        self, 
        category_data: pd.DataFrame, 
        category: str
    ) -> List[AnomalyRecord]:
        """Detect sentiment shifts for a specific category."""
        anomalies = []
        
        # Calculate baseline sentiment (historical average)
        cutoff_date = datetime.utcnow() - timedelta(days=self.thresholds.sentiment_window_days)
        historical_data = category_data[category_data['created_at'] < cutoff_date]
        recent_data = category_data[category_data['created_at'] >= cutoff_date]
        
        if len(historical_data) < 5 or len(recent_data) < 5:
            return anomalies
        
        # Calculate sentiment metrics
        baseline_sentiment = historical_data['sentiment_score'].mean()
        recent_sentiment = recent_data['sentiment_score'].mean()
        shift_magnitude = abs(recent_sentiment - baseline_sentiment)
        
        # Check if shift exceeds threshold
        if shift_magnitude > self.thresholds.sentiment_shift_threshold:
            # Calculate additional metrics
            historical_std = historical_data['sentiment_score'].std()
            z_score = abs(recent_sentiment - baseline_sentiment) / (historical_std + 0.001)  # Avoid div by zero
            
            score = min(1.0, shift_magnitude / 2.0)  # Normalize score (max shift is 2.0)
            severity = determine_severity(z_score / 3.0)  # Normalize z-score
            
            # Determine shift direction
            shift_direction = "negative" if recent_sentiment < baseline_sentiment else "positive"
            
            # Create sentiment shift details
            sentiment_shift = SentimentShift(
                category=category,
                current_sentiment=float(recent_sentiment),
                baseline_sentiment=float(baseline_sentiment),
                shift_magnitude=float(shift_magnitude),
                affected_period={
                    "start": cutoff_date,
                    "end": datetime.utcnow()
                },
                sample_size=len(recent_data)
            )
            
            anomaly = create_anomaly_record(
                anomaly_type=AnomalyType.SENTIMENT_SHIFT,
                description=f"Sentiment shift detected in {category}: {shift_direction} shift of "
                           f"{shift_magnitude:.2f} points (baseline: {baseline_sentiment:.2f}, "
                           f"current: {recent_sentiment:.2f})",
                score=score,
                severity=severity,
                category=category,
                details={
                    "sentiment_shift": sentiment_shift.dict(),
                    "shift_direction": shift_direction,
                    "z_score": float(z_score),
                    "historical_samples": len(historical_data),
                    "recent_samples": len(recent_data)
                },
                baseline_value=float(baseline_sentiment),
                actual_value=float(recent_sentiment),
                threshold=self.thresholds.sentiment_shift_threshold,
                affected_tickets=recent_data['ticket_id'].tolist()
            )
            
            anomalies.append(anomaly)
            
            logger.info(
                f"Sentiment shift detected: {category} - {shift_direction} shift of "
                f"{shift_magnitude:.2f} points"
            )
        
        return anomalies


class NewIssueDetector(BaseAnomalyDetector):
    """Detects new/emerging issues using text pattern analysis."""
    
    def __init__(self, thresholds: Optional[AnomalyThresholds] = None):
        super().__init__(thresholds)
        self.known_patterns: Dict[str, List[str]] = {}
        self.vectorizer = None
        
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
    
    def detect(self, tickets_data: List[Dict[str, Any]]) -> List[AnomalyRecord]:
        """
        Detect new issues using text similarity analysis.
        
        Args:
            tickets_data: List of ticket dictionaries
            
        Returns:
            List of new issue anomalies
        """
        logger.info("Starting new issue detection...")
        anomalies = []
        
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. New issue detection disabled.")
            return anomalies
        
        if len(tickets_data) < self.thresholds.min_tickets_for_analysis:
            logger.warning(f"Not enough tickets for new issue analysis: {len(tickets_data)}")
            return anomalies
        
        try:
            # Prepare text data
            recent_tickets = self._filter_recent_tickets(
                tickets_data, 
                days=self.thresholds.lookback_days
            )
            
            if len(recent_tickets) < 10:
                return anomalies
            
            # Extract text features
            ticket_texts = []
            ticket_metadata = []
            
            for ticket in recent_tickets:
                # Combine subject, description, and error logs
                text_parts = []
                if ticket.get("subject"):
                    text_parts.append(ticket["subject"])
                if ticket.get("description"):
                    text_parts.append(ticket["description"])
                if ticket.get("error_logs"):
                    text_parts.append(ticket["error_logs"])
                
                combined_text = " ".join(text_parts)
                ticket_texts.append(combined_text)
                ticket_metadata.append({
                    'ticket_id': ticket.get("ticket_id", ""),
                    'category': ticket.get("category", "unknown"),
                    'created_at': self._parse_timestamp(ticket.get("created_at", "")),
                    'text': combined_text
                })
            
            # Detect new patterns
            new_issue_anomalies = self._detect_new_text_patterns(
                ticket_texts, 
                ticket_metadata
            )
            anomalies.extend(new_issue_anomalies)
            
            logger.info(f"Detected {len(anomalies)} new issue anomalies")
            
        except Exception as e:
            logger.error(f"Error in new issue detection: {e}")
        
        return anomalies
    
    def _detect_new_text_patterns(
        self, 
        ticket_texts: List[str], 
        ticket_metadata: List[Dict[str, Any]]
    ) -> List[AnomalyRecord]:
        """Detect new text patterns in tickets."""
        anomalies = []
        
        try:
            # Create TF-IDF vectors
            if len(ticket_texts) < 5:
                return anomalies
            
            tfidf_matrix = self.vectorizer.fit_transform(ticket_texts)
            
            # Calculate pairwise similarities
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find potential new issues (tickets with low similarity to others)
            for i, ticket_meta in enumerate(ticket_metadata):
                similarities = similarity_matrix[i]
                max_similarity = np.max(similarities[similarities < 1.0])  # Exclude self-similarity
                
                # If similarity is below threshold, it might be a new issue
                if max_similarity < self.thresholds.new_issue_similarity_threshold:
                    # Check if this pattern appears in multiple tickets
                    similar_tickets = [
                        ticket_metadata[j] for j, sim in enumerate(similarities)
                        if sim >= max_similarity * 0.8 and j != i  # Find somewhat similar tickets
                    ]
                    
                    if len(similar_tickets) >= self.thresholds.new_issue_min_frequency - 1:
                        # Extract key terms
                        ticket_vector = tfidf_matrix[i]
                        feature_names = self.vectorizer.get_feature_names_out()
                        top_indices = ticket_vector.toarray()[0].argsort()[-10:][::-1]
                        keywords = [feature_names[idx] for idx in top_indices if ticket_vector[0, idx] > 0]
                        
                        score = 1.0 - max_similarity  # Higher score for lower similarity
                        severity = determine_severity(score)
                        
                        # Create new issue details
                        new_issue = NewIssue(
                            pattern_description=f"New issue pattern detected with keywords: {', '.join(keywords[:5])}",
                            first_occurrence=ticket_meta['created_at'],
                            frequency=len(similar_tickets) + 1,
                            similarity_to_known=float(max_similarity),
                            example_tickets=[ticket_meta['ticket_id']] + [t['ticket_id'] for t in similar_tickets],
                            keywords=keywords[:10]
                        )
                        
                        anomaly = create_anomaly_record(
                            anomaly_type=AnomalyType.NEW_ISSUE,
                            description=f"New issue pattern detected: {new_issue.pattern_description}",
                            score=score,
                            severity=severity,
                            category=ticket_meta['category'],
                            details={
                                "new_issue": new_issue.dict(),
                                "max_similarity_to_existing": float(max_similarity),
                                "pattern_frequency": len(similar_tickets) + 1
                            },
                            affected_tickets=new_issue.example_tickets,
                            confidence=score
                        )
                        
                        anomalies.append(anomaly)
                        
                        logger.info(
                            f"New issue detected: {ticket_meta['category']} - "
                            f"similarity {max_similarity:.2f}, frequency {len(similar_tickets) + 1}"
                        )
            
        except Exception as e:
            logger.error(f"Error in text pattern detection: {e}")
        
        return anomalies


class OutlierDetector(BaseAnomalyDetector):
    """Detects outlier tickets using IsolationForest."""
    
    def detect(self, tickets_data: List[Dict[str, Any]]) -> List[AnomalyRecord]:
        """
        Detect outlier tickets using isolation forest.
        
        Args:
            tickets_data: List of ticket dictionaries
            
        Returns:
            List of outlier anomalies
        """
        logger.info("Starting outlier detection...")
        anomalies = []
        
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Outlier detection disabled.")
            return anomalies
        
        if len(tickets_data) < self.thresholds.outlier_min_samples:
            logger.warning(f"Not enough tickets for outlier analysis: {len(tickets_data)}")
            return anomalies
        
        try:
            # Extract numerical features
            feature_data = []
            ticket_metadata = []
            
            for ticket in tickets_data:
                features = self._extract_numerical_features(ticket)
                if features is not None:
                    feature_data.append(features)
                    ticket_metadata.append({
                        'ticket_id': ticket.get("ticket_id", ""),
                        'category': ticket.get("category", "unknown"),
                        'product': ticket.get("product", "unknown"),
                        'created_at': self._parse_timestamp(ticket.get("created_at", "")),
                        'original_features': features
                    })
            
            if len(feature_data) < self.thresholds.outlier_min_samples:
                logger.warning("Not enough valid feature data for outlier detection")
                return anomalies
            
            # Detect outliers
            outlier_anomalies = self._detect_isolation_forest_outliers(
                feature_data, 
                ticket_metadata
            )
            anomalies.extend(outlier_anomalies)
            
            logger.info(f"Detected {len(anomalies)} outlier anomalies")
            
        except Exception as e:
            logger.error(f"Error in outlier detection: {e}")
        
        return anomalies
    
    def _extract_numerical_features(self, ticket: Dict[str, Any]) -> Optional[List[float]]:
        """Extract numerical features from a ticket."""
        try:
            features = []
            
            # Priority encoding
            priority_mapping = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            priority = priority_mapping.get(ticket.get("priority", "medium").lower(), 2)
            features.append(float(priority))
            
            # Sentiment encoding
            sentiment_mapping = {'frustrated': -1, 'neutral': 0, 'positive': 1}
            sentiment = sentiment_mapping.get(ticket.get("customer_sentiment", "neutral").lower(), 0)
            features.append(float(sentiment))
            
            # Text length features
            subject_len = len(ticket.get("subject", ""))
            description_len = len(ticket.get("description", ""))
            error_logs_len = len(ticket.get("error_logs", ""))
            features.extend([float(subject_len), float(description_len), float(error_logs_len)])
            
            # Customer features
            previous_tickets = ticket.get("previous_tickets", 0)
            account_age = ticket.get("account_age_days", 0)
            features.extend([float(previous_tickets), float(account_age)])
            
            # Resolution time (if available)
            resolution_time = ticket.get("resolution_time_hours", 0)
            features.append(float(resolution_time))
            
            return features
            
        except Exception as e:
            logger.warning(f"Failed to extract features for ticket: {e}")
            return None
    
    def _detect_isolation_forest_outliers(
        self, 
        feature_data: List[List[float]], 
        ticket_metadata: List[Dict[str, Any]]
    ) -> List[AnomalyRecord]:
        """Detect outliers using Isolation Forest."""
        anomalies = []
        
        try:
            # Prepare feature matrix
            X = np.array(feature_data)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply Isolation Forest
            isolation_forest = IsolationForest(
                contamination=self.thresholds.outlier_contamination,
                random_state=42,
                n_estimators=100
            )
            
            outlier_predictions = isolation_forest.fit_predict(X_scaled)
            outlier_scores = isolation_forest.score_samples(X_scaled)
            
            # Process outliers
            for i, (prediction, score, metadata) in enumerate(
                zip(outlier_predictions, outlier_scores, ticket_metadata)
            ):
                if prediction == -1:  # Outlier detected
                    # Calculate normalized anomaly score
                    anomaly_score = (0.5 - score) / 0.5  # Normalize to 0-1 range
                    anomaly_score = max(0, min(1, anomaly_score))
                    
                    severity = determine_severity(anomaly_score)
                    
                    # Analyze which features contributed to outlier status
                    feature_values = feature_data[i]
                    feature_names = [
                        "priority", "sentiment", "subject_length", "description_length",
                        "error_logs_length", "previous_tickets", "account_age", "resolution_time"
                    ]
                    
                    feature_dict = dict(zip(feature_names, feature_values))
                    
                    # Calculate feature deviations from mean
                    feature_means = np.mean(X, axis=0)
                    feature_stds = np.std(X, axis=0)
                    deviations = {}
                    
                    for j, (name, value) in enumerate(zip(feature_names, feature_values)):
                        if feature_stds[j] > 0:
                            deviation = abs(value - feature_means[j]) / feature_stds[j]
                            deviations[name] = float(deviation)
                    
                    # Create outlier details
                    outlier_info = OutlierInfo(
                        ticket_id=metadata['ticket_id'],
                        outlier_score=float(score),
                        feature_values=feature_dict,
                        feature_deviations=deviations
                    )
                    
                    # Find the most anomalous features
                    top_anomalous_features = sorted(
                        deviations.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:3]
                    
                    description = (
                        f"Outlier ticket detected: {metadata['ticket_id']} in {metadata['category']} "
                        f"(top anomalous features: {', '.join([f[0] for f in top_anomalous_features])})"
                    )
                    
                    anomaly = create_anomaly_record(
                        anomaly_type=AnomalyType.OUTLIER,
                        description=description,
                        score=anomaly_score,
                        severity=severity,
                        category=metadata['category'],
                        product=metadata['product'],
                        details={
                            "outlier_info": outlier_info.dict(),
                            "top_anomalous_features": dict(top_anomalous_features),
                            "isolation_forest_score": float(score)
                        },
                        affected_tickets=[metadata['ticket_id']],
                        confidence=anomaly_score
                    )
                    
                    anomalies.append(anomaly)
                    
                    logger.info(
                        f"Outlier detected: {metadata['ticket_id']} "
                        f"(score: {score:.3f}, features: {[f[0] for f in top_anomalous_features]})"
                    )
        
        except Exception as e:
            logger.error(f"Error in isolation forest outlier detection: {e}")
        
        return anomalies


class AnomalyDetector:
    """Main anomaly detection orchestrator."""
    
    def __init__(self, thresholds: Optional[AnomalyThresholds] = None):
        """
        Initialize the main anomaly detector.
        
        Args:
            thresholds: Configuration thresholds for all detectors
        """
        self.thresholds = thresholds or AnomalyThresholds()
        
        # Initialize individual detectors
        self.volume_detector = VolumeAnomalyDetector(self.thresholds)
        self.sentiment_detector = SentimentAnomalyDetector(self.thresholds)
        self.new_issue_detector = NewIssueDetector(self.thresholds)
        self.outlier_detector = OutlierDetector(self.thresholds)
        
        self.detection_history: List[AnomalyDetectionResult] = []
    
    def detect_all_anomalies(
        self, 
        tickets_data: List[Dict[str, Any]],
        detection_types: Optional[List[AnomalyType]] = None
    ) -> AnomalyDetectionResult:
        """
        Run all anomaly detection methods.
        
        Args:
            tickets_data: List of ticket dictionaries
            detection_types: Specific types to detect (default: all)
            
        Returns:
            AnomalyDetectionResult with all detected anomalies
        """
        start_time = datetime.utcnow()
        processing_start = start_time.timestamp()
        
        logger.info(f"Starting comprehensive anomaly detection on {len(tickets_data)} tickets")
        
        all_anomalies = []
        
        # Default to all detection types
        if detection_types is None:
            detection_types = [
                AnomalyType.VOLUME_SPIKE,
                AnomalyType.SENTIMENT_SHIFT,
                AnomalyType.NEW_ISSUE,
                AnomalyType.OUTLIER
            ]
        
        # Run each detector
        if AnomalyType.VOLUME_SPIKE in detection_types:
            try:
                volume_anomalies = self.volume_detector.detect(tickets_data)
                all_anomalies.extend(volume_anomalies)
                logger.info(f"Volume detector found {len(volume_anomalies)} anomalies")
            except Exception as e:
                logger.error(f"Volume detection failed: {e}")
        
        if AnomalyType.SENTIMENT_SHIFT in detection_types:
            try:
                sentiment_anomalies = self.sentiment_detector.detect(tickets_data)
                all_anomalies.extend(sentiment_anomalies)
                logger.info(f"Sentiment detector found {len(sentiment_anomalies)} anomalies")
            except Exception as e:
                logger.error(f"Sentiment detection failed: {e}")
        
        if AnomalyType.NEW_ISSUE in detection_types:
            try:
                new_issue_anomalies = self.new_issue_detector.detect(tickets_data)
                all_anomalies.extend(new_issue_anomalies)
                logger.info(f"New issue detector found {len(new_issue_anomalies)} anomalies")
            except Exception as e:
                logger.error(f"New issue detection failed: {e}")
        
        if AnomalyType.OUTLIER in detection_types:
            try:
                outlier_anomalies = self.outlier_detector.detect(tickets_data)
                all_anomalies.extend(outlier_anomalies)
                logger.info(f"Outlier detector found {len(outlier_anomalies)} anomalies")
            except Exception as e:
                logger.error(f"Outlier detection failed: {e}")
        
        # Calculate processing time
        processing_time = datetime.utcnow().timestamp() - processing_start
        
        # Create summary statistics
        severity_breakdown = defaultdict(int)
        type_breakdown = defaultdict(int)
        
        for anomaly in all_anomalies:
            severity_breakdown[anomaly.severity] += 1
            type_breakdown[anomaly.type] += 1
        
        # Create detection result
        result = AnomalyDetectionResult(
            total_anomalies=len(all_anomalies),
            detection_period={
                "start": start_time - timedelta(days=self.thresholds.lookback_days),
                "end": start_time
            },
            anomalies=all_anomalies,
            severity_breakdown=dict(severity_breakdown),
            type_breakdown=dict(type_breakdown),
            processing_time=processing_time,
            tickets_analyzed=len(tickets_data)
        )
        
        # Store in history
        self.detection_history.append(result)
        
        logger.info(
            f"Anomaly detection completed: {len(all_anomalies)} total anomalies "
            f"in {processing_time:.2f}s"
        )
        
        return result
    
    def get_recent_anomalies(self, days: int = 7) -> List[AnomalyRecord]:
        """
        Get anomalies from the last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of recent anomalies
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_anomalies = []
        
        for result in self.detection_history:
            for anomaly in result.anomalies:
                if anomaly.timestamp >= cutoff_date:
                    recent_anomalies.append(anomaly)
        
        # Sort by timestamp (most recent first)
        recent_anomalies.sort(key=lambda x: x.timestamp, reverse=True)
        
        return recent_anomalies
    
    def update_thresholds(self, new_thresholds: AnomalyThresholds):
        """Update detection thresholds for all detectors."""
        self.thresholds = new_thresholds
        self.volume_detector.thresholds = new_thresholds
        self.sentiment_detector.thresholds = new_thresholds
        self.new_issue_detector.thresholds = new_thresholds
        self.outlier_detector.thresholds = new_thresholds
        
        logger.info("Updated anomaly detection thresholds")
    
    def load_demo_anomalies(self, demo_file_path: str = "anomaly_detection_demo_results.json") -> bool:
        """
        Load demo anomalies from JSON file into detection history.
        This allows the dashboard to display demo anomalies.
        
        Args:
            demo_file_path: Path to the demo anomalies JSON file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        import json
        from pathlib import Path
        from .anomaly_models import AnomalyRecord, AnomalyType, AnomalySeverity
        
        try:
            demo_file = Path(demo_file_path)
            if not demo_file.exists():
                logger.warning(f"Demo anomalies file not found: {demo_file_path}")
                return False
            
            with open(demo_file, 'r') as f:
                demo_data = json.load(f)
            
            # Convert JSON anomalies back to AnomalyRecord objects
            demo_anomalies = []
            for anomaly_data in demo_data.get('anomalies', []):
                try:
                    # Parse timestamp
                    timestamp = datetime.fromisoformat(anomaly_data['timestamp'].replace('Z', '+00:00'))
                    
                    # Create anomaly record
                    anomaly = AnomalyRecord(
                        id=anomaly_data['id'],
                        type=AnomalyType(anomaly_data['type']),
                        severity=AnomalySeverity(anomaly_data['severity']),
                        timestamp=timestamp,
                        description=anomaly_data['description'],
                        category=anomaly_data.get('category'),
                        product=anomaly_data.get('product'),
                        score=anomaly_data.get('score', 1.0),
                        confidence=anomaly_data.get('confidence', 1.0),
                        affected_tickets=anomaly_data.get('affected_tickets', []),
                        details=anomaly_data.get('details', {}),
                        detection_method=f"{anomaly_data['type']}_detector"
                    )
                    demo_anomalies.append(anomaly)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse anomaly {anomaly_data.get('id', 'unknown')}: {e}")
                    continue
            
            # Create a mock detection result and add to history
            from .anomaly_models import AnomalyDetectionResult
            
            detection_result = AnomalyDetectionResult(
                anomalies=demo_anomalies,
                total_anomalies=len(demo_anomalies),
                tickets_analyzed=demo_data['summary']['tickets_analyzed'],
                processing_time=demo_data['summary']['processing_time'],
                severity_breakdown=demo_data['summary']['severity_breakdown'],
                type_breakdown=demo_data['summary']['type_breakdown'],
                detection_period={
                    'start': datetime.fromisoformat(demo_data['summary']['detection_period']['start']),
                    'end': datetime.fromisoformat(demo_data['summary']['detection_period']['end'])
                }
            )
            
            # Add to detection history
            self.detection_history.append(detection_result)
            
            logger.info(f"✅ Loaded {len(demo_anomalies)} demo anomalies from {demo_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load demo anomalies: {e}")
            return False


# Utility functions for common anomaly detection tasks

def detect_volume_spikes(
    tickets_data: List[Dict[str, Any]], 
    thresholds: Optional[AnomalyThresholds] = None
) -> List[AnomalyRecord]:
    """Convenience function for volume spike detection only."""
    detector = VolumeAnomalyDetector(thresholds)
    return detector.detect(tickets_data)


def detect_sentiment_shifts(
    tickets_data: List[Dict[str, Any]], 
    thresholds: Optional[AnomalyThresholds] = None
) -> List[AnomalyRecord]:
    """Convenience function for sentiment shift detection only."""
    detector = SentimentAnomalyDetector(thresholds)
    return detector.detect(tickets_data)


def detect_outliers(
    tickets_data: List[Dict[str, Any]], 
    thresholds: Optional[AnomalyThresholds] = None
) -> List[AnomalyRecord]:
    """Convenience function for outlier detection only."""
    detector = OutlierDetector(thresholds)
    return detector.detect(tickets_data)