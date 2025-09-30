# scripts/demo_feedback.py

"""
Demo script for the Feedback Loop system.

This script demonstrates:
1. Recording agent corrections
2. Recording customer feedback
3. Generating analytics and insights
4. Testing retraining triggers
"""

import sys
import json
import time
import random
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.feedback.feedback_manager import FeedbackManager
from src.feedback.feedback_models import FeedbackType, FeedbackSeverity, PredictionQuality


def demo_agent_corrections(feedback_manager: FeedbackManager, num_corrections: int = 15):
    """Demo agent corrections with realistic scenarios."""
    print(f"\n🔧 Recording {num_corrections} agent corrections...")
    
    # Sample ticket scenarios
    scenarios = [
        {
            "ticket_id": "TICKET-001",
            "agent_id": "agent_sarah",
            "original_prediction": "technical_issue",
            "corrected_label": "billing_inquiry",
            "correction_reason": "Customer's actual issue was billing dispute, not technical",
            "original_confidence": 0.85,
            "model_type": "xgboost",
            "prediction_quality": PredictionQuality.POOR,
            "severity": FeedbackSeverity.HIGH
        },
        {
            "ticket_id": "TICKET-002", 
            "agent_id": "agent_mike",
            "original_prediction": "general_inquiry",
            "corrected_label": "feature_request",
            "correction_reason": "Customer requested new feature, model missed context",
            "original_confidence": 0.62,
            "model_type": "tensorflow",
            "prediction_quality": PredictionQuality.ACCEPTABLE,
            "severity": FeedbackSeverity.MEDIUM
        },
        {
            "ticket_id": "TICKET-003",
            "agent_id": "agent_alex",
            "original_prediction": "login_issue",
            "corrected_label": "account_suspended",
            "correction_reason": "Account was suspended due to policy violation",
            "original_confidence": 0.78,
            "model_type": "xgboost",
            "prediction_quality": PredictionQuality.POOR,
            "severity": FeedbackSeverity.HIGH
        },
        {
            "ticket_id": "TICKET-004",
            "agent_id": "agent_sarah",
            "original_prediction": "password_reset",
            "corrected_label": "login_issue",
            "correction_reason": "Minor misclassification, very similar categories",
            "original_confidence": 0.71,
            "model_type": "tensorflow",
            "prediction_quality": PredictionQuality.GOOD,
            "severity": FeedbackSeverity.LOW
        },
        {
            "ticket_id": "TICKET-005",
            "agent_id": "agent_emma",
            "original_prediction": "bug_report",
            "corrected_label": "configuration_error",
            "correction_reason": "User misconfiguration, not a bug",
            "original_confidence": 0.89,
            "model_type": "xgboost",
            "prediction_quality": PredictionQuality.ACCEPTABLE,
            "severity": FeedbackSeverity.MEDIUM
        }
    ]
    
    correction_ids = []
    
    # Generate corrections based on scenarios
    for i in range(num_corrections):
        scenario = scenarios[i % len(scenarios)]
        
        # Add some variation
        ticket_id = f"{scenario['ticket_id']}-{i}"
        confidence_variation = random.uniform(-0.15, 0.15)
        original_confidence = max(0.1, min(0.99, scenario['original_confidence'] + confidence_variation))
        
        # Add ticket data
        ticket_data = {
            "subject": f"Sample ticket {i+1}",
            "description": f"Detailed description for ticket {i+1}",
            "category": scenario['corrected_label'],
            "priority": random.choice(["low", "medium", "high"]),
            "created_at": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat()
        }
        
        correction_id = feedback_manager.record_agent_correction(
            ticket_id=ticket_id,
            agent_id=scenario['agent_id'],
            original_prediction=scenario['original_prediction'],
            original_confidence=original_confidence,
            model_type=scenario['model_type'],
            corrected_label=scenario['corrected_label'],
            correction_reason=scenario['correction_reason'],
            correction_notes=f"Additional notes for correction {i+1}",
            prediction_quality=scenario['prediction_quality'],
            severity=scenario['severity'],
            ticket_data=ticket_data,
            should_retrain=random.choice([True, False]),
            correction_confidence=random.uniform(0.8, 1.0)
        )
        
        if correction_id:
            correction_ids.append(correction_id)
            print(f"  ✅ Recorded correction {i+1}: {correction_id[:8]}...")
        else:
            print(f"  ❌ Failed to record correction {i+1}")
        
        # Small delay to simulate real usage
        time.sleep(0.1)
    
    print(f"✅ Recorded {len(correction_ids)} agent corrections")
    return correction_ids


def demo_customer_feedback(feedback_manager: FeedbackManager, num_feedback: int = 20):
    """Demo customer feedback with realistic scenarios."""
    print(f"\n⭐ Recording {num_feedback} customer feedback entries...")
    
    feedback_scenarios = [
        {
            "rating": 5,
            "satisfaction": "very_satisfied",
            "ai_suggestions_used": True,
            "ai_helpful": True,
            "resolution_helpful": True,
            "resolution_accurate": True
        },
        {
            "rating": 4,
            "satisfaction": "satisfied", 
            "ai_suggestions_used": True,
            "ai_helpful": True,
            "resolution_helpful": True,
            "resolution_accurate": True
        },
        {
            "rating": 3,
            "satisfaction": "neutral",
            "ai_suggestions_used": False,
            "ai_helpful": None,
            "resolution_helpful": True,
            "resolution_accurate": True
        },
        {
            "rating": 2,
            "satisfaction": "dissatisfied",
            "ai_suggestions_used": True,
            "ai_helpful": False,
            "resolution_helpful": False,
            "resolution_accurate": False
        },
        {
            "rating": 1,
            "satisfaction": "very_dissatisfied",
            "ai_suggestions_used": False,
            "ai_helpful": None,
            "resolution_helpful": False,
            "resolution_accurate": False
        }
    ]
    
    feedback_ids = []
    
    for i in range(num_feedback):
        scenario = feedback_scenarios[i % len(feedback_scenarios)]
        
        # Add some natural variation to ratings
        rating_variation = random.randint(-1, 1)
        rating = max(1, min(5, scenario['rating'] + rating_variation))
        
        # Generate feedback
        feedback_id = feedback_manager.record_customer_feedback(
            ticket_id=f"TICKET-{i+1:03d}",
            customer_id=f"customer_{random.randint(1000, 9999)}",
            rating=rating,
            comments=f"Sample feedback comment {i+1}. {'Great service!' if rating >= 4 else 'Could be better.' if rating == 3 else 'Not satisfied.'}",
            feedback_type="resolution_quality",
            resolution_helpful=scenario['resolution_helpful'],
            resolution_accurate=scenario['resolution_accurate'],
            would_recommend=rating >= 4,
            ai_suggestions_used=scenario['ai_suggestions_used'],
            ai_suggestions_helpful=scenario['ai_helpful'],
            ai_accuracy_rating=rating if scenario['ai_suggestions_used'] else None,
            resolution_time_hours=random.uniform(0.5, 48.0),
            agent_id=random.choice(["agent_sarah", "agent_mike", "agent_alex", "agent_emma"]),
            needs_followup=rating <= 2
        )
        
        if feedback_id:
            feedback_ids.append(feedback_id)
            print(f"  ⭐ Recorded feedback {i+1}: Rating {rating}/5, ID {feedback_id[:8]}...")
        else:
            print(f"  ❌ Failed to record feedback {i+1}")
        
        time.sleep(0.05)
    
    print(f"✅ Recorded {len(feedback_ids)} customer feedback entries")
    return feedback_ids


def demo_analytics(feedback_manager: FeedbackManager):
    """Demo analytics and insights generation."""
    print(f"\n📊 Generating feedback analytics...")
    
    # Generate comprehensive summary
    summary = feedback_manager.generate_feedback_summary(days_back=30)
    
    print(f"\n📈 Feedback Summary ({summary.period_start.strftime('%Y-%m-%d')} to {summary.period_end.strftime('%Y-%m-%d')}):")
    print(f"  📋 Agent Corrections: {summary.total_corrections}")
    print(f"     └─ By Model: {summary.corrections_by_model}")
    print(f"     └─ By Severity: {summary.corrections_by_severity}")
    print(f"     └─ Avg Original Confidence: {summary.avg_original_confidence:.3f}")
    
    print(f"  ⭐ Customer Feedback: {summary.total_customer_feedback}")
    print(f"     └─ Average Rating: {summary.avg_customer_rating:.2f}/5")
    print(f"     └─ Satisfaction: {summary.satisfaction_breakdown}")
    print(f"     └─ AI Usage Rate: {summary.ai_usage_rate:.1%}")
    
    print(f"  📊 Quality Metrics:")
    for metric, value in summary.quality_metrics.items():
        print(f"     └─ {metric}: {value:.3f}")
    
    print(f"  🔄 Retraining Assessment:")
    retraining = summary.retraining_recommendations
    print(f"     └─ Needs Retraining: {retraining.get('needs_retraining', False)}")
    print(f"     └─ Priority: {retraining.get('priority', 'low')}")
    if retraining.get('reasons'):
        print(f"     └─ Reasons: {retraining['reasons']}")
    if retraining.get('affected_models'):
        print(f"     └─ Affected Models: {retraining['affected_models']}")
    
    # Agent performance analysis
    print(f"\n👥 Agent Performance Analysis:")
    agents = ["agent_sarah", "agent_mike", "agent_alex", "agent_emma"]
    
    for agent in agents:
        analysis = feedback_manager.get_agent_performance_analysis(agent, days_back=30)
        
        print(f"  👤 {agent.replace('_', ' ').title()}:")
        print(f"     └─ Corrections: {analysis.get('total_corrections', 0)}")
        print(f"     └─ Customer Feedback: {analysis.get('total_feedback', 0)}")
        print(f"     └─ Avg Customer Rating: {analysis.get('avg_customer_rating', 0):.2f}/5")
        print(f"     └─ AI Usage Rate: {analysis.get('ai_usage_rate', 0):.1%}")
        if 'avg_improvement_score' in analysis:
            print(f"     └─ Improvement Score: {analysis['avg_improvement_score']:.3f}")
    
    # Model performance trends
    print(f"\n📈 Model Performance Trends (Last 7 days):")
    for model in ["xgboost", "tensorflow"]:
        trends = feedback_manager.get_model_performance_trends(
            model_type=model,
            days_back=7,
            granularity="daily"
        )
        
        correction_counts = [t['count'] for t in trends['correction_trends']]
        avg_confidence = [t['avg_confidence'] for t in trends['confidence_trends'] if t['avg_confidence'] > 0]
        
        print(f"  🤖 {model.upper()}:")
        print(f"     └─ Total Corrections: {sum(correction_counts)}")
        print(f"     └─ Avg Daily Corrections: {sum(correction_counts) / 7:.1f}")
        if avg_confidence:
            print(f"     └─ Avg Confidence: {sum(avg_confidence) / len(avg_confidence):.3f}")
    
    return summary


def demo_feedback_retrieval(feedback_manager: FeedbackManager):
    """Demo feedback data retrieval with filters."""
    print(f"\n🔍 Testing feedback data retrieval...")
    
    # Get recent corrections
    recent_corrections = feedback_manager.get_corrections(days_back=7)
    print(f"  📋 Recent corrections (7 days): {len(recent_corrections)}")
    
    # Get high-severity corrections
    all_corrections = feedback_manager.get_corrections(days_back=30)
    high_severity_corrections = [c for c in all_corrections if c.severity == FeedbackSeverity.HIGH]
    print(f"  🚨 High severity corrections: {len(high_severity_corrections)}")
    
    # Get recent customer feedback
    recent_feedback = feedback_manager.get_customer_feedback(days_back=7)
    print(f"  ⭐ Recent customer feedback (7 days): {len(recent_feedback)}")
    
    # Get low-rated feedback
    low_rated_feedback = feedback_manager.get_customer_feedback(days_back=30, rating_min=1)
    very_low_rated = [f for f in low_rated_feedback if f.rating <= 2]
    print(f"  👎 Low-rated feedback (≤2): {len(very_low_rated)}")
    
    # Get model-specific corrections
    xgb_corrections = feedback_manager.get_corrections(days_back=30, model_type="xgboost")
    tf_corrections = feedback_manager.get_corrections(days_back=30, model_type="tensorflow")
    print(f"  🤖 XGBoost corrections: {len(xgb_corrections)}")
    print(f"  🤖 TensorFlow corrections: {len(tf_corrections)}")
    
    return {
        "recent_corrections": len(recent_corrections),
        "high_severity_corrections": len(high_severity_corrections),
        "recent_feedback": len(recent_feedback),
        "low_rated_feedback": len(very_low_rated),
        "xgb_corrections": len(xgb_corrections),
        "tf_corrections": len(tf_corrections)
    }


def demo_system_health(feedback_manager: FeedbackManager):
    """Demo system health monitoring."""
    print(f"\n🔧 System Health Check...")
    
    health = feedback_manager.health_check()
    
    print(f"  📊 Status: {health.get('status', 'unknown')}")
    print(f"  💾 Storage: {health.get('storage_backend', 'unknown')}")
    print(f"  ✅ Storage Operational: {health.get('storage_operational', False)}")
    
    if 'recent_activity' in health:
        activity = health['recent_activity']
        print(f"  📈 Recent Activity (24h):")
        print(f"     └─ Corrections: {activity.get('corrections_24h', 0)}")
        print(f"     └─ Feedback: {activity.get('feedback_24h', 0)}")
        print(f"     └─ Events: {activity.get('events_24h', 0)}")
    
    return health


def main():
    """Run the feedback loop system demo."""
    print("🚀 InsightDesk AI - Feedback Loop System Demo")
    print("=" * 50)
    
    try:
        # Initialize feedback manager
        print("📋 Initializing Feedback Manager...")
        feedback_manager = FeedbackManager(
            storage_type="json",
            storage_config={"storage_dir": "feedback_demo_data"}
        )
        print("✅ Feedback Manager initialized with JSON storage")
        
        # Run health check
        demo_system_health(feedback_manager)
        
        # Demo agent corrections
        correction_ids = demo_agent_corrections(feedback_manager, num_corrections=15)
        
        # Demo customer feedback
        feedback_ids = demo_customer_feedback(feedback_manager, num_feedback=20)
        
        # Demo analytics
        summary = demo_analytics(feedback_manager)
        
        # Demo data retrieval
        retrieval_stats = demo_feedback_retrieval(feedback_manager)
        
        # Final health check
        print(f"\n🔍 Final System Status:")
        final_health = demo_system_health(feedback_manager)
        
        # Summary
        print(f"\n📊 Demo Summary:")
        print(f"  ✅ Agent Corrections Recorded: {len(correction_ids)}")
        print(f"  ✅ Customer Feedback Recorded: {len(feedback_ids)}")
        print(f"  ✅ Analytics Generated: {summary.summary_id[:8]}...")
        print(f"  ✅ System Health: {final_health.get('status', 'unknown')}")
        
        # Retraining recommendation
        if summary.retraining_recommendations.get('needs_retraining'):
            print(f"\n🔄 RETRAINING RECOMMENDED:")
            print(f"  Priority: {summary.retraining_recommendations.get('priority', 'unknown').upper()}")
            print(f"  Reasons: {summary.retraining_recommendations.get('reasons', [])}")
        else:
            print(f"\n✅ No immediate retraining needed")
        
        print(f"\n🎉 Feedback Loop Demo completed successfully!")
        print(f"📁 Data stored in: feedback_demo_data/")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()