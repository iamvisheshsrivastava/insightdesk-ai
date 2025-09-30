# src/feedback/feedback_manager.py

"""
Feedback Manager for the Feedback Loop system.

This module provides the main interface for managing feedback data,
including storage, analytics, and learning triggers.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from collections import defaultdict, Counter
import statistics

from .feedback_models import (
    FeedbackEvent, AgentCorrection, CustomerFeedback, FeedbackSummary,
    FeedbackType, FeedbackSeverity, PredictionQuality,
    generate_feedback_id, calculate_accuracy
)
from .feedback_storage import FeedbackStorage, JSONFeedbackStorage, SQLiteFeedbackStorage

logger = logging.getLogger(__name__)


class FeedbackManager:
    """Main manager for feedback data and analytics."""
    
    def __init__(
        self,
        storage: Optional[FeedbackStorage] = None,
        storage_type: str = "json",
        storage_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the feedback manager.
        
        Args:
            storage: Custom storage backend
            storage_type: Type of storage ('json' or 'sqlite')
            storage_config: Configuration for storage backend
        """
        if storage:
            self.storage = storage
        else:
            storage_config = storage_config or {}
            
            if storage_type == "sqlite":
                self.storage = SQLiteFeedbackStorage(**storage_config)
            else:
                self.storage = JSONFeedbackStorage(**storage_config)
        
        self.retraining_thresholds = {
            "min_corrections": 10,
            "accuracy_drop_threshold": 0.1,
            "low_confidence_threshold": 0.6,
            "high_severity_threshold": 0.3,
            "time_window_hours": 24
        }
        
        logger.info(f"Feedback manager initialized with {type(self.storage).__name__}")
    
    # Agent Correction Methods
    
    def record_agent_correction(
        self,
        ticket_id: str,
        agent_id: str,
        original_prediction: str,
        original_confidence: float,
        model_type: str,
        corrected_label: str,
        correction_reason: str,
        correction_notes: Optional[str] = None,
        prediction_quality: PredictionQuality = PredictionQuality.POOR,
        severity: FeedbackSeverity = FeedbackSeverity.MEDIUM,
        ticket_data: Optional[Dict[str, Any]] = None,
        should_retrain: bool = True,
        correction_confidence: float = 1.0
    ) -> str:
        """
        Record an agent correction.
        
        Returns:
            str: Correction ID
        """
        try:
            correction = AgentCorrection(
                correction_id=generate_feedback_id("correction"),
                ticket_id=ticket_id,
                agent_id=agent_id,
                original_prediction=original_prediction,
                original_confidence=original_confidence,
                model_type=model_type,
                corrected_label=corrected_label,
                correction_reason=correction_reason,
                correction_notes=correction_notes,
                prediction_quality=prediction_quality,
                severity=severity,
                correction_timestamp=datetime.now(),
                ticket_data=ticket_data,
                should_retrain=should_retrain,
                correction_confidence=correction_confidence
            )
            
            success = self.storage.store_correction(correction)
            if success:
                # Create feedback event
                event = FeedbackEvent(
                    event_id=generate_feedback_id("event"),
                    event_type=FeedbackType.AGENT_CORRECTION,
                    ticket_id=ticket_id,
                    event_data={
                        "correction_id": correction.correction_id,
                        "original_prediction": original_prediction,
                        "corrected_label": corrected_label,
                        "model_type": model_type,
                        "agent_id": agent_id
                    },
                    severity=severity,
                    timestamp=datetime.now()
                )
                
                self.storage.store_event(event)
                
                logger.info(f"Recorded agent correction: {correction.correction_id}")
                return correction.correction_id
            else:
                logger.error(f"Failed to store agent correction for ticket {ticket_id}")
                return ""
                
        except Exception as e:
            logger.error(f"Error recording agent correction: {e}")
            return ""
    
    def record_customer_feedback(
        self,
        ticket_id: str,
        rating: int,
        customer_id: Optional[str] = None,
        comments: Optional[str] = None,
        feedback_type: str = "resolution_quality",
        resolution_helpful: bool = True,
        resolution_accurate: bool = True,
        would_recommend: Optional[bool] = None,
        ai_suggestions_used: bool = False,
        ai_suggestions_helpful: Optional[bool] = None,
        ai_accuracy_rating: Optional[int] = None,
        resolution_time_hours: Optional[float] = None,
        agent_id: Optional[str] = None,
        needs_followup: bool = False
    ) -> str:
        """
        Record customer feedback.
        
        Returns:
            str: Feedback ID
        """
        try:
            feedback = CustomerFeedback(
                feedback_id=generate_feedback_id("feedback"),
                ticket_id=ticket_id,
                rating=rating,
                feedback_timestamp=datetime.now(),
                customer_id=customer_id,
                comments=comments,
                feedback_type=feedback_type,
                resolution_helpful=resolution_helpful,
                resolution_accurate=resolution_accurate,
                would_recommend=would_recommend,
                ai_suggestions_used=ai_suggestions_used,
                ai_suggestions_helpful=ai_suggestions_helpful,
                ai_accuracy_rating=ai_accuracy_rating,
                resolution_time_hours=resolution_time_hours,
                agent_id=agent_id
            )
            
            # Set needs_followup after creation if different from computed value
            if needs_followup != feedback.needs_followup:
                feedback.needs_followup = needs_followup
            
            success = self.storage.store_customer_feedback(feedback)
            if success:
                # Determine severity based on rating
                if rating <= 2:
                    severity = FeedbackSeverity.HIGH
                elif rating <= 3:
                    severity = FeedbackSeverity.MEDIUM
                else:
                    severity = FeedbackSeverity.LOW
                
                # Create feedback event
                event = FeedbackEvent(
                    event_id=generate_feedback_id("event"),
                    event_type=FeedbackType.CUSTOMER_FEEDBACK,
                    ticket_id=ticket_id,
                    event_data={
                        "feedback_id": feedback.feedback_id,
                        "rating": rating,
                        "satisfaction_level": feedback.satisfaction_level,
                        "ai_suggestions_used": ai_suggestions_used,
                        "customer_id": customer_id
                    },
                    severity=severity,
                    timestamp=datetime.now()
                )
                
                self.storage.store_event(event)
                
                logger.info(f"Recorded customer feedback: {feedback.feedback_id}")
                return feedback.feedback_id
            else:
                logger.error(f"Failed to store customer feedback for ticket {ticket_id}")
                return ""
                
        except Exception as e:
            logger.error(f"Error recording customer feedback: {e}")
            return ""
    
    def record_feedback_event(
        self,
        event_type: FeedbackType,
        ticket_id: str,
        event_data: Dict[str, Any],
        severity: FeedbackSeverity = FeedbackSeverity.LOW
    ) -> str:
        """
        Record a generic feedback event.
        
        Returns:
            str: Event ID
        """
        try:
            event = FeedbackEvent(
                event_id=generate_feedback_id("event"),
                event_type=event_type,
                ticket_id=ticket_id,
                event_data=event_data,
                severity=severity,
                timestamp=datetime.now()
            )
            
            success = self.storage.store_event(event)
            if success:
                logger.info(f"Recorded feedback event: {event.event_id}")
                return event.event_id
            else:
                logger.error(f"Failed to store feedback event for ticket {ticket_id}")
                return ""
                
        except Exception as e:
            logger.error(f"Error recording feedback event: {e}")
            return ""
    
    # Retrieval Methods
    
    def get_corrections(
        self,
        days_back: int = 30,
        ticket_id: Optional[str] = None,
        model_type: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> List[AgentCorrection]:
        """Get agent corrections with filters."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        corrections = self.storage.get_corrections(
            start_date=start_date,
            end_date=end_date,
            ticket_id=ticket_id,
            model_type=model_type
        )
        
        # Filter by agent_id if provided (storage doesn't have this filter)
        if agent_id:
            corrections = [c for c in corrections if c.agent_id == agent_id]
        
        return corrections
    
    def get_customer_feedback(
        self,
        days_back: int = 30,
        ticket_id: Optional[str] = None,
        rating_min: Optional[int] = None,
        customer_id: Optional[str] = None
    ) -> List[CustomerFeedback]:
        """Get customer feedback with filters."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        feedback_list = self.storage.get_customer_feedback(
            start_date=start_date,
            end_date=end_date,
            ticket_id=ticket_id,
            rating_min=rating_min
        )
        
        # Filter by customer_id if provided
        if customer_id:
            feedback_list = [f for f in feedback_list if f.customer_id == customer_id]
        
        return feedback_list
    
    def get_events(
        self,
        days_back: int = 30,
        event_type: Optional[FeedbackType] = None,
        ticket_id: Optional[str] = None
    ) -> List[FeedbackEvent]:
        """Get feedback events with filters."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        return self.storage.get_events(
            start_date=start_date,
            end_date=end_date,
            event_type=event_type
        )
    
    # Analytics Methods
    
    def generate_feedback_summary(
        self,
        days_back: int = 30,
        model_type: Optional[str] = None
    ) -> FeedbackSummary:
        """Generate comprehensive feedback summary."""
        try:
            corrections = self.get_corrections(days_back=days_back, model_type=model_type)
            feedback_list = self.get_customer_feedback(days_back=days_back)
            events = self.get_events(days_back=days_back)
            
            # Calculate periods
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Correction analysis
            total_corrections = len(corrections)
            corrections_by_model = Counter(c.model_type for c in corrections)
            corrections_by_severity = Counter(c.severity.value for c in corrections)
            avg_confidence = statistics.mean([c.original_confidence for c in corrections]) if corrections else 0.0
            
            # Customer feedback analysis
            total_feedback = len(feedback_list)
            avg_rating = statistics.mean([f.rating for f in feedback_list]) if feedback_list else 0.0
            satisfaction_breakdown = Counter(f.satisfaction_level for f in feedback_list)
            ai_usage_rate = sum(1 for f in feedback_list if f.ai_suggestions_used) / len(feedback_list) if feedback_list else 0.0
            
            # Events analysis
            events_by_type = Counter(e.event_type.value for e in events)
            events_by_severity = Counter(e.severity.value for e in events)
            
            # Quality metrics
            accuracy_metrics = self._calculate_accuracy_metrics(corrections, feedback_list)
            
            # Retraining recommendations
            retraining_needed = self._assess_retraining_need(corrections, feedback_list)
            
            summary = FeedbackSummary(
                summary_id=generate_feedback_id("summary"),
                period_start=start_date,
                period_end=end_date,
                total_corrections=total_corrections,
                corrections_by_model=dict(corrections_by_model),
                corrections_by_severity=dict(corrections_by_severity),
                avg_original_confidence=avg_confidence,
                total_customer_feedback=total_feedback,
                avg_customer_rating=avg_rating,
                satisfaction_breakdown=dict(satisfaction_breakdown),
                ai_usage_rate=ai_usage_rate,
                events_by_type=dict(events_by_type),
                events_by_severity=dict(events_by_severity),
                quality_metrics=accuracy_metrics,
                retraining_recommendations=retraining_needed,
                generated_at=datetime.now()
            )
            
            logger.info(f"Generated feedback summary: {summary.summary_id}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating feedback summary: {e}")
            # Return empty summary
            return FeedbackSummary(
                summary_id=generate_feedback_id("summary"),
                period_start=datetime.now() - timedelta(days=days_back),
                period_end=datetime.now(),
                total_corrections=0,
                corrections_by_model={},
                corrections_by_severity={},
                avg_original_confidence=0.0,
                total_customer_feedback=0,
                avg_customer_rating=0.0,
                satisfaction_breakdown={},
                ai_usage_rate=0.0,
                events_by_type={},
                events_by_severity={},
                quality_metrics={},
                retraining_recommendations={},
                generated_at=datetime.now()
            )
    
    def _calculate_accuracy_metrics(
        self,
        corrections: List[AgentCorrection],
        feedback_list: List[CustomerFeedback]
    ) -> Dict[str, float]:
        """Calculate accuracy and quality metrics."""
        metrics = {}
        
        try:
            # Model accuracy from corrections
            if corrections:
                model_accuracies = defaultdict(list)
                for correction in corrections:
                    # Inverse accuracy: if corrected, original was wrong
                    accuracy = 1.0 - correction.original_confidence
                    model_accuracies[correction.model_type].append(accuracy)
                
                for model, accuracies in model_accuracies.items():
                    metrics[f"{model}_accuracy"] = statistics.mean(accuracies)
                
                # Overall accuracy
                all_accuracies = [acc for accs in model_accuracies.values() for acc in accs]
                metrics["overall_accuracy"] = statistics.mean(all_accuracies)
                
                # Severity-based accuracy
                severe_corrections = [c for c in corrections if c.severity == FeedbackSeverity.HIGH]
                if severe_corrections:
                    severe_accuracies = [1.0 - c.original_confidence for c in severe_corrections]
                    metrics["severe_issues_accuracy"] = statistics.mean(severe_accuracies)
            
            # Customer satisfaction correlation
            if feedback_list:
                ai_feedback = [f for f in feedback_list if f.ai_suggestions_used]
                non_ai_feedback = [f for f in feedback_list if not f.ai_suggestions_used]
                
                if ai_feedback:
                    metrics["ai_assisted_satisfaction"] = statistics.mean([f.rating for f in ai_feedback])
                
                if non_ai_feedback:
                    metrics["non_ai_satisfaction"] = statistics.mean([f.rating for f in non_ai_feedback])
                
                # AI accuracy from customer feedback
                ai_accuracy_ratings = [f.ai_accuracy_rating for f in feedback_list 
                                     if f.ai_accuracy_rating is not None]
                if ai_accuracy_ratings:
                    metrics["customer_reported_ai_accuracy"] = statistics.mean(ai_accuracy_ratings)
            
        except Exception as e:
            logger.error(f"Error calculating accuracy metrics: {e}")
        
        return metrics
    
    def _assess_retraining_need(
        self,
        corrections: List[AgentCorrection],
        feedback_list: List[CustomerFeedback]
    ) -> Dict[str, Any]:
        """Assess if model retraining is needed."""
        recommendations = {
            "needs_retraining": False,
            "reasons": [],
            "priority": "low",
            "affected_models": [],
            "suggested_actions": []
        }
        
        try:
            # Check correction thresholds
            if len(corrections) >= self.retraining_thresholds["min_corrections"]:
                recommendations["reasons"].append(f"High correction volume: {len(corrections)} corrections")
                recommendations["needs_retraining"] = True
            
            # Check severe corrections
            severe_corrections = [c for c in corrections if c.severity == FeedbackSeverity.HIGH]
            severe_ratio = len(severe_corrections) / len(corrections) if corrections else 0
            
            if severe_ratio >= self.retraining_thresholds["high_severity_threshold"]:
                recommendations["reasons"].append(f"High severe correction ratio: {severe_ratio:.2%}")
                recommendations["needs_retraining"] = True
                recommendations["priority"] = "high"
            
            # Check low confidence predictions
            low_confidence_corrections = [c for c in corrections 
                                        if c.original_confidence < self.retraining_thresholds["low_confidence_threshold"]]
            low_confidence_ratio = len(low_confidence_corrections) / len(corrections) if corrections else 0
            
            if low_confidence_ratio >= 0.5:  # More than 50% low confidence corrections
                recommendations["reasons"].append(f"High low-confidence correction ratio: {low_confidence_ratio:.2%}")
                recommendations["needs_retraining"] = True
            
            # Model-specific analysis
            model_corrections = defaultdict(list)
            for correction in corrections:
                model_corrections[correction.model_type].append(correction)
            
            for model, model_corr_list in model_corrections.items():
                model_severe_ratio = len([c for c in model_corr_list if c.severity == FeedbackSeverity.HIGH]) / len(model_corr_list)
                
                if model_severe_ratio >= self.retraining_thresholds["high_severity_threshold"]:
                    recommendations["affected_models"].append(model)
                    recommendations["suggested_actions"].append(f"Retrain {model} model due to {model_severe_ratio:.2%} severe corrections")
            
            # Check customer satisfaction
            if feedback_list:
                low_rating_feedback = [f for f in feedback_list if f.rating <= 2]
                if len(low_rating_feedback) / len(feedback_list) >= 0.3:  # 30% low ratings
                    recommendations["reasons"].append("High customer dissatisfaction")
                    recommendations["needs_retraining"] = True
            
            # Set priority
            if len(recommendations["reasons"]) >= 3:
                recommendations["priority"] = "high"
            elif len(recommendations["reasons"]) >= 2:
                recommendations["priority"] = "medium"
            
        except Exception as e:
            logger.error(f"Error assessing retraining need: {e}")
        
        return recommendations
    
    def get_model_performance_trends(
        self,
        model_type: str,
        days_back: int = 30,
        granularity: str = "daily"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get performance trends for a specific model."""
        try:
            corrections = self.get_corrections(days_back=days_back, model_type=model_type)
            feedback_list = self.get_customer_feedback(days_back=days_back)
            
            # Group by time periods
            if granularity == "daily":
                period_delta = timedelta(days=1)
            elif granularity == "weekly":
                period_delta = timedelta(weeks=1)
            else:
                period_delta = timedelta(hours=1)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            trends = {
                "correction_trends": [],
                "satisfaction_trends": [],
                "confidence_trends": []
            }
            
            current_date = start_date
            while current_date < end_date:
                period_end = current_date + period_delta
                
                # Corrections in this period
                period_corrections = [c for c in corrections 
                                    if current_date <= c.correction_timestamp < period_end]
                
                # Feedback in this period  
                period_feedback = [f for f in feedback_list 
                                 if current_date <= f.feedback_timestamp < period_end]
                
                # Calculate metrics
                correction_count = len(period_corrections)
                avg_confidence = statistics.mean([c.original_confidence for c in period_corrections]) if period_corrections else 0.0
                avg_rating = statistics.mean([f.rating for f in period_feedback]) if period_feedback else 0.0
                
                trends["correction_trends"].append({
                    "period": current_date.isoformat(),
                    "count": correction_count,
                    "severe_count": len([c for c in period_corrections if c.severity == FeedbackSeverity.HIGH])
                })
                
                trends["confidence_trends"].append({
                    "period": current_date.isoformat(),
                    "avg_confidence": avg_confidence,
                    "low_confidence_count": len([c for c in period_corrections if c.original_confidence < 0.6])
                })
                
                trends["satisfaction_trends"].append({
                    "period": current_date.isoformat(),
                    "avg_rating": avg_rating,
                    "feedback_count": len(period_feedback)
                })
                
                current_date = period_end
            
            return trends
            
        except Exception as e:
            logger.error(f"Error getting model performance trends: {e}")
            return {"correction_trends": [], "satisfaction_trends": [], "confidence_trends": []}
    
    def get_agent_performance_analysis(
        self,
        agent_id: str,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Analyze individual agent performance."""
        try:
            corrections = self.get_corrections(days_back=days_back, agent_id=agent_id)
            feedback_list = [f for f in self.get_customer_feedback(days_back=days_back) 
                           if f.agent_id == agent_id]
            
            analysis = {
                "agent_id": agent_id,
                "total_corrections": len(corrections),
                "corrections_by_model": dict(Counter(c.model_type for c in corrections)),
                "corrections_by_severity": dict(Counter(c.severity.value for c in corrections)),
                "avg_correction_confidence": statistics.mean([c.correction_confidence for c in corrections]) if corrections else 0.0,
                "total_feedback": len(feedback_list),
                "avg_customer_rating": statistics.mean([f.rating for f in feedback_list]) if feedback_list else 0.0,
                "ai_usage_rate": sum(1 for f in feedback_list if f.ai_suggestions_used) / len(feedback_list) if feedback_list else 0.0
            }
            
            # Calculate accuracy improvement
            if corrections:
                improvement_scores = []
                for correction in corrections:
                    # Improvement = correction_confidence - original_confidence
                    improvement = correction.correction_confidence - correction.original_confidence
                    improvement_scores.append(improvement)
                
                analysis["avg_improvement_score"] = statistics.mean(improvement_scores)
                analysis["improvement_consistency"] = 1.0 - (statistics.stdev(improvement_scores) if len(improvement_scores) > 1 else 0.0)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing agent performance: {e}")
            return {"agent_id": agent_id, "error": str(e)}
    
    # Configuration Methods
    
    def update_retraining_thresholds(self, thresholds: Dict[str, Any]):
        """Update retraining thresholds."""
        self.retraining_thresholds.update(thresholds)
        logger.info(f"Updated retraining thresholds: {thresholds}")
    
    def get_retraining_thresholds(self) -> Dict[str, Any]:
        """Get current retraining thresholds."""
        return self.retraining_thresholds.copy()
    
    # Health Check Methods
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on feedback system."""
        try:
            # Test storage
            test_event = FeedbackEvent(
                event_id="health_check_test",
                event_type=FeedbackType.SYSTEM_EVENT,
                ticket_id="health_check",
                event_data={"test": True},
                severity=FeedbackSeverity.LOW,
                timestamp=datetime.now()
            )
            
            storage_ok = self.storage.store_event(test_event)
            
            # Get recent stats
            recent_corrections = len(self.get_corrections(days_back=1))
            recent_feedback = len(self.get_customer_feedback(days_back=1))
            recent_events = len(self.get_events(days_back=1))
            
            return {
                "status": "healthy" if storage_ok else "degraded",
                "storage_backend": type(self.storage).__name__,
                "storage_operational": storage_ok,
                "recent_activity": {
                    "corrections_24h": recent_corrections,
                    "feedback_24h": recent_feedback,
                    "events_24h": recent_events
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }