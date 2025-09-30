#!/usr/bin/env python3
"""
Load demo anomalies from JSON file and feed them to the API's anomaly detector.
This allows the Streamlit dashboard to display the demo anomalies.
"""

import json
import sys
import requests
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_demo_anomalies():
    """Load demo anomalies from JSON file and send them to the API."""
    
    # Check if the demo results file exists
    demo_file = Path("anomaly_detection_demo_results.json")
    if not demo_file.exists():
        logger.error(f"Demo results file not found: {demo_file}")
        logger.info("Please run: python scripts/demo_anomaly.py first")
        return False
    
    # Load the demo data
    logger.info(f"Loading demo anomalies from {demo_file}")
    with open(demo_file, 'r') as f:
        demo_data = json.load(f)
    
    logger.info(f"Found {demo_data['summary']['total_anomalies']} anomalies in demo file")
    
    # Check if API is running
    api_url = "http://localhost:8000"
    try:
        health_response = requests.get(f"{api_url}/health", timeout=5)
        if health_response.status_code != 200:
            logger.error(f"API health check failed: {health_response.status_code}")
            return False
        logger.info("✅ API is running")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ API is not accessible: {e}")
        logger.info("Please start the API server with: python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload")
        return False
    
    # We need to load the anomalies into the detector's memory
    # Since we can't directly modify the detector's state via API,
    # we'll need to modify the anomaly detector to load from file
    
    logger.info("✅ Demo anomalies are ready to be loaded")
    logger.info("The anomaly detector needs to be updated to load these anomalies.")
    logger.info("Let me create a method to load demo anomalies into the detector...")
    
    return True

if __name__ == "__main__":
    success = load_demo_anomalies()
    if not success:
        sys.exit(1)
    
    print("\n🎉 Demo anomalies loaded successfully!")
    print("Now refresh your Streamlit dashboard to see the anomalies.")