# scripts/demo_anomaly.py

"""
Demo script for testing the Anomaly Detection module.
Shows volume spikes, sentiment shifts, new issues, and outlier detection.
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    from anomaly.anomaly_detector import AnomalyDetector
    from anomaly.anomaly_models import AnomalyThresholds, AnomalyType
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root and dependencies are installed")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_demo_data():
    """Create synthetic ticket data for anomaly detection testing."""
    logger.info("🎲 Creating synthetic demo data...")
    
    # Base date
    base_date = datetime.utcnow() - timedelta(days=60)
    
    # Categories and products
    categories = ["authentication", "payment", "api", "ui", "billing", "account"]
    products = ["web_application", "mobile_app", "payment_gateway", "admin_panel"]
    sentiments = ["frustrated", "neutral", "positive", "angry", "satisfied"]
    priorities = ["low", "medium", "high", "critical"]
    
    tickets = []
    ticket_id = 1000
    
    # Generate normal baseline data (days 0-45)
    for day in range(45):
        current_date = base_date + timedelta(days=day)
        
        # Normal volume per category (5-15 tickets per day per category)
        for category in categories:
            daily_volume = np.random.poisson(8)  # Average 8 tickets per category
            
            for _ in range(daily_volume):
                ticket = {
                    "ticket_id": f"T{ticket_id:06d}",
                    "category": category,
                    "product": np.random.choice(products),
                    "subject": f"Issue with {category} in {np.random.choice(products)}",
                    "description": f"Customer experiencing problems with {category} functionality. "
                                 f"This is affecting their ability to use the system properly.",
                    "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "customer_sentiment": np.random.choice(sentiments, p=[0.2, 0.4, 0.2, 0.1, 0.1]),
                    "priority": np.random.choice(priorities, p=[0.3, 0.4, 0.2, 0.1]),
                    "previous_tickets": np.random.poisson(2),
                    "account_age_days": np.random.randint(30, 1000),
                    "resolution_time_hours": np.random.exponential(24),
                    "error_logs": f"Error in {category} module at line {np.random.randint(100, 999)}"
                }
                tickets.append(ticket)
                ticket_id += 1
    
    logger.info(f"Generated {len(tickets)} baseline tickets")
    
    # Generate anomalous data (days 46-60)
    anomaly_start_date = base_date + timedelta(days=46)
    
    # VOLUME SPIKE ANOMALY - Authentication category spike
    logger.info("🔥 Adding volume spike anomaly for authentication...")
    for day in range(46, 50):  # 4 days of high volume
        current_date = base_date + timedelta(days=day)
        
        # Create volume spike in authentication (30-50 tickets instead of normal 8)
        spike_volume = np.random.randint(30, 50)
        for _ in range(spike_volume):
            ticket = {
                "ticket_id": f"T{ticket_id:06d}",
                "category": "authentication",
                "product": "web_application",
                "subject": f"Cannot login - new authentication issue {ticket_id}",
                "description": "Sudden login failures started happening for multiple users. "
                             "Authentication service seems to be having issues.",
                "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                "customer_sentiment": np.random.choice(["frustrated", "angry"], p=[0.6, 0.4]),
                "priority": np.random.choice(["high", "critical"], p=[0.7, 0.3]),
                "previous_tickets": np.random.poisson(2),
                "account_age_days": np.random.randint(30, 1000),
                "resolution_time_hours": np.random.exponential(24),
                "error_logs": "AuthenticationError: Token validation failed"
            }
            tickets.append(ticket)
            ticket_id += 1
        
        # Add normal volume for other categories
        for category in [c for c in categories if c != "authentication"]:
            daily_volume = np.random.poisson(8)
            for _ in range(daily_volume):
                ticket = {
                    "ticket_id": f"T{ticket_id:06d}",
                    "category": category,
                    "product": np.random.choice(products),
                    "subject": f"Regular issue with {category}",
                    "description": f"Standard issue in {category} functionality.",
                    "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "customer_sentiment": np.random.choice(sentiments, p=[0.2, 0.4, 0.2, 0.1, 0.1]),
                    "priority": np.random.choice(priorities, p=[0.3, 0.4, 0.2, 0.1]),
                    "previous_tickets": np.random.poisson(2),
                    "account_age_days": np.random.randint(30, 1000),
                    "resolution_time_hours": np.random.exponential(24),
                    "error_logs": f"Error in {category} module"
                }
                tickets.append(ticket)
                ticket_id += 1
    
    # SENTIMENT SHIFT ANOMALY - Payment category becomes more negative
    logger.info("😠 Adding sentiment shift anomaly for payment...")
    for day in range(50, 55):  # 5 days of negative sentiment
        current_date = base_date + timedelta(days=day)
        
        # Create sentiment shift in payment category
        for category in categories:
            daily_volume = np.random.poisson(12 if category == "payment" else 8)
            
            for _ in range(daily_volume):
                if category == "payment":
                    # Very negative sentiment for payment issues
                    sentiment = np.random.choice(["frustrated", "angry"], p=[0.4, 0.6])
                else:
                    sentiment = np.random.choice(sentiments, p=[0.2, 0.4, 0.2, 0.1, 0.1])
                
                ticket = {
                    "ticket_id": f"T{ticket_id:06d}",
                    "category": category,
                    "product": "payment_gateway" if category == "payment" else np.random.choice(products),
                    "subject": f"Payment processing failure {ticket_id}" if category == "payment" 
                             else f"Issue with {category}",
                    "description": "Payment gateway is charging customers but not processing orders correctly. "
                                 "This is causing major customer dissatisfaction." if category == "payment"
                                 else f"Regular issue in {category} functionality.",
                    "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "customer_sentiment": sentiment,
                    "priority": "critical" if category == "payment" and sentiment == "angry" 
                              else np.random.choice(priorities, p=[0.3, 0.4, 0.2, 0.1]),
                    "previous_tickets": np.random.poisson(2),
                    "account_age_days": np.random.randint(30, 1000),
                    "resolution_time_hours": np.random.exponential(24),
                    "error_logs": "PaymentGatewayError: Transaction failed with code 500" if category == "payment"
                              else f"Error in {category} module"
                }
                tickets.append(ticket)
                ticket_id += 1
    
    # NEW ISSUE ANOMALY - Unique error pattern
    logger.info("🆕 Adding new issue anomaly...")
    for day in range(55, 58):  # 3 days of new issue pattern
        current_date = base_date + timedelta(days=day)
        
        # Create new issue pattern - database connection issues
        for _ in range(8):  # New pattern appears multiple times
            ticket = {
                "ticket_id": f"T{ticket_id:06d}",
                "category": "api",
                "product": "web_application",
                "subject": f"Database connection timeout in API endpoint {ticket_id}",
                "description": "API endpoints are timing out when trying to connect to the database. "
                             "This appears to be a new issue not seen before. Connection pool exhaustion "
                             "is happening during peak hours.",
                "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                "customer_sentiment": "frustrated",
                "priority": "high",
                "previous_tickets": np.random.poisson(2),
                "account_age_days": np.random.randint(30, 1000),
                "resolution_time_hours": np.random.exponential(36),
                "error_logs": "DatabaseConnectionError: Connection timeout after 30s. Pool exhausted. "
                             "Max connections: 100, Active: 100, Idle: 0"
            }
            tickets.append(ticket)
            ticket_id += 1
        
        # Add normal tickets for other categories
        for category in [c for c in categories if c != "api"]:
            daily_volume = np.random.poisson(8)
            for _ in range(daily_volume):
                ticket = {
                    "ticket_id": f"T{ticket_id:06d}",
                    "category": category,
                    "product": np.random.choice(products),
                    "subject": f"Regular issue with {category}",
                    "description": f"Standard issue in {category} functionality.",
                    "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "customer_sentiment": np.random.choice(sentiments, p=[0.2, 0.4, 0.2, 0.1, 0.1]),
                    "priority": np.random.choice(priorities, p=[0.3, 0.4, 0.2, 0.1]),
                    "previous_tickets": np.random.poisson(2),
                    "account_age_days": np.random.randint(30, 1000),
                    "resolution_time_hours": np.random.exponential(24),
                    "error_logs": f"Error in {category} module"
                }
                tickets.append(ticket)
                ticket_id += 1
    
    # OUTLIER ANOMALY - Some very unusual tickets
    logger.info("🎯 Adding outlier anomalies...")
    for day in range(58, 60):  # 2 days with outliers
        current_date = base_date + timedelta(days=day)
        
        # Create statistical outliers
        outlier_tickets = [
            {
                "ticket_id": f"T{ticket_id:06d}",
                "category": "billing",
                "product": "admin_panel",
                "subject": "Extremely long subject line that goes on and on and describes a very complex billing issue that affects multiple accounts and involves integration with external payment systems and requires immediate escalation to senior engineering team",
                "description": "This is an extremely detailed description of a billing issue that spans multiple paragraphs and contains extensive technical details about the problem. " * 20,  # Very long description
                "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                "customer_sentiment": "angry",
                "priority": "critical",
                "previous_tickets": 50,  # Outlier: very high number of previous tickets
                "account_age_days": 5,   # Outlier: very new account but many tickets
                "resolution_time_hours": 200,  # Outlier: very long resolution time
                "error_logs": "Critical system failure: " + "Error details. " * 100
            },
            {
                "ticket_id": f"T{ticket_id + 1:06d}",
                "category": "ui",
                "product": "mobile_app",
                "subject": "",  # Outlier: empty subject
                "description": "UI",  # Outlier: very short description
                "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                "customer_sentiment": "positive",  # Outlier: positive sentiment for a problem
                "priority": "low",
                "previous_tickets": 0,
                "account_age_days": 3000,  # Outlier: very old account
                "resolution_time_hours": 0.1,  # Outlier: very fast resolution
                "error_logs": ""  # Outlier: no error logs
            }
        ]
        
        for outlier in outlier_tickets:
            tickets.append(outlier)
            ticket_id += 1
        
        # Add normal tickets
        for category in categories:
            daily_volume = np.random.poisson(8)
            for _ in range(daily_volume):
                ticket = {
                    "ticket_id": f"T{ticket_id:06d}",
                    "category": category,
                    "product": np.random.choice(products),
                    "subject": f"Regular issue with {category}",
                    "description": f"Standard issue in {category} functionality.",
                    "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "customer_sentiment": np.random.choice(sentiments, p=[0.2, 0.4, 0.2, 0.1, 0.1]),
                    "priority": np.random.choice(priorities, p=[0.3, 0.4, 0.2, 0.1]),
                    "previous_tickets": np.random.poisson(2),
                    "account_age_days": np.random.randint(30, 1000),
                    "resolution_time_hours": np.random.exponential(24),
                    "error_logs": f"Error in {category} module"
                }
                tickets.append(ticket)
                ticket_id += 1
    
    logger.info(f"✅ Created {len(tickets)} total demo tickets with embedded anomalies")
    return tickets


def run_anomaly_detection_demo():
    """Run comprehensive anomaly detection demo."""
    logger.info("🚀 Starting Anomaly Detection Demo")
    
    try:
        # Create demo data
        tickets_data = create_demo_data()
        
        # Initialize anomaly detector with custom thresholds
        logger.info("🔧 Initializing anomaly detector...")
        custom_thresholds = AnomalyThresholds(
            volume_spike_sigma=2.0,           # Lower threshold for demo
            sentiment_shift_threshold=0.2,    # Lower threshold for demo
            new_issue_similarity_threshold=0.7,  # Adjust for demo patterns
            outlier_contamination=0.05        # Expect 5% outliers
        )
        
        detector = AnomalyDetector(custom_thresholds)
        
        # Test individual detectors
        logger.info("\n" + "="*60)
        logger.info("🔍 TESTING INDIVIDUAL ANOMALY DETECTORS")
        logger.info("="*60)
        
        # Volume spike detection
        logger.info("\n📊 Testing Volume Spike Detection...")
        volume_anomalies = detector.volume_detector.detect(tickets_data)
        logger.info(f"Found {len(volume_anomalies)} volume spike anomalies:")
        for anomaly in volume_anomalies:
            logger.info(f"  - {anomaly.description} (severity: {anomaly.severity.value})")
        
        # Sentiment shift detection
        logger.info("\n😔 Testing Sentiment Shift Detection...")
        sentiment_anomalies = detector.sentiment_detector.detect(tickets_data)
        logger.info(f"Found {len(sentiment_anomalies)} sentiment shift anomalies:")
        for anomaly in sentiment_anomalies:
            logger.info(f"  - {anomaly.description} (severity: {anomaly.severity.value})")
        
        # New issue detection
        logger.info("\n🆕 Testing New Issue Detection...")
        new_issue_anomalies = detector.new_issue_detector.detect(tickets_data)
        logger.info(f"Found {len(new_issue_anomalies)} new issue anomalies:")
        for anomaly in new_issue_anomalies:
            logger.info(f"  - {anomaly.description} (severity: {anomaly.severity.value})")
        
        # Outlier detection
        logger.info("\n🎯 Testing Outlier Detection...")
        outlier_anomalies = detector.outlier_detector.detect(tickets_data)
        logger.info(f"Found {len(outlier_anomalies)} outlier anomalies:")
        for anomaly in outlier_anomalies:
            logger.info(f"  - {anomaly.description} (severity: {anomaly.severity.value})")
        
        # Full detection run
        logger.info("\n" + "="*60)
        logger.info("🔍 RUNNING COMPREHENSIVE ANOMALY DETECTION")
        logger.info("="*60)
        
        result = detector.detect_all_anomalies(tickets_data)
        
        # Display results
        logger.info(f"\n✅ ANOMALY DETECTION COMPLETE")
        logger.info(f"📋 Analysis Summary:")
        logger.info(f"   • Total tickets analyzed: {result.tickets_analyzed}")
        logger.info(f"   • Total anomalies found: {result.total_anomalies}")
        logger.info(f"   • Processing time: {result.processing_time:.2f} seconds")
        
        # Severity breakdown
        logger.info(f"\n📊 Severity Breakdown:")
        for severity, count in result.severity_breakdown.items():
            logger.info(f"   • {severity.title()}: {count}")
        
        # Type breakdown
        logger.info(f"\n🏷️  Type Breakdown:")
        for anomaly_type, count in result.type_breakdown.items():
            logger.info(f"   • {anomaly_type.replace('_', ' ').title()}: {count}")
        
        # Detailed anomaly list
        logger.info(f"\n🔍 Detailed Anomaly Reports:")
        logger.info("-" * 60)
        
        for i, anomaly in enumerate(result.anomalies, 1):
            logger.info(f"\n{i}. {anomaly.type.value.upper()} - {anomaly.severity.value.upper()}")
            logger.info(f"   📝 Description: {anomaly.description}")
            logger.info(f"   🕒 Timestamp: {anomaly.timestamp}")
            logger.info(f"   📊 Score: {anomaly.score:.3f}")
            logger.info(f"   🏷️  Category: {anomaly.category}")
            
            if anomaly.confidence:
                logger.info(f"   🎯 Confidence: {anomaly.confidence:.3f}")
            
            if anomaly.affected_tickets:
                logger.info(f"   🎫 Affected tickets: {len(anomaly.affected_tickets)}")
            
            # Show specific details based on anomaly type
            if anomaly.type == AnomalyType.VOLUME_SPIKE and "volume_spike" in anomaly.details:
                spike_info = anomaly.details["volume_spike"]
                logger.info(f"   📈 Volume: {spike_info['actual_count']} vs expected {spike_info['expected_count']:.1f}")
                logger.info(f"   📊 Spike ratio: {spike_info['spike_ratio']:.1f}x normal")
            
            elif anomaly.type == AnomalyType.SENTIMENT_SHIFT and "sentiment_shift" in anomaly.details:
                sentiment_info = anomaly.details["sentiment_shift"]
                logger.info(f"   😔 Sentiment: {sentiment_info['current_sentiment']:.2f} vs baseline {sentiment_info['baseline_sentiment']:.2f}")
                logger.info(f"   📊 Shift magnitude: {sentiment_info['shift_magnitude']:.2f}")
            
            elif anomaly.type == AnomalyType.NEW_ISSUE and "new_issue" in anomaly.details:
                new_issue_info = anomaly.details["new_issue"]
                logger.info(f"   🆕 Pattern: {new_issue_info['pattern_description']}")
                logger.info(f"   🔁 Frequency: {new_issue_info['frequency']}")
                logger.info(f"   🎯 Similarity to known: {new_issue_info['similarity_to_known']:.2f}")
            
            elif anomaly.type == AnomalyType.OUTLIER and "outlier_info" in anomaly.details:
                outlier_info = anomaly.details["outlier_info"]
                logger.info(f"   🎯 Outlier score: {outlier_info['outlier_score']:.3f}")
                top_features = anomaly.details.get("top_anomalous_features", {})
                if top_features:
                    logger.info(f"   🔍 Top anomalous features: {list(top_features.keys())[:3]}")
        
        # Save results to file
        logger.info(f"\n💾 Saving results to file...")
        output_file = Path("anomaly_detection_demo_results.json")
        
        # Convert result to JSON-serializable format
        result_dict = {
            "summary": {
                "total_anomalies": result.total_anomalies,
                "tickets_analyzed": result.tickets_analyzed,
                "processing_time": result.processing_time,
                "severity_breakdown": result.severity_breakdown,
                "type_breakdown": result.type_breakdown,
                "detection_period": {
                    "start": result.detection_period["start"].isoformat(),
                    "end": result.detection_period["end"].isoformat()
                }
            },
            "anomalies": []
        }
        
        for anomaly in result.anomalies:
            anomaly_dict = anomaly.dict()
            anomaly_dict["timestamp"] = anomaly.timestamp.isoformat()
            result_dict["anomalies"].append(anomaly_dict)
        
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        logger.info(f"✅ Results saved to {output_file}")
        
        # Performance statistics
        tickets_per_second = result.tickets_analyzed / result.processing_time
        logger.info(f"\n⚡ Performance Statistics:")
        logger.info(f"   • Processing rate: {tickets_per_second:.1f} tickets/second")
        logger.info(f"   • Average time per ticket: {(result.processing_time * 1000 / result.tickets_analyzed):.2f} ms")
        logger.info(f"   • Anomaly detection rate: {(result.total_anomalies / result.tickets_analyzed * 100):.1f}%")
        
        logger.info(f"\n🎉 Anomaly Detection Demo Complete!")
        return result
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    try:
        result = run_anomaly_detection_demo()
        print(f"\n✅ Demo completed successfully! Found {result.total_anomalies} anomalies.")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)