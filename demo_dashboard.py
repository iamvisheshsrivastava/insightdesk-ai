#!/usr/bin/env python3
"""
Streamlit Dashboard Demo Script
==============================

This script demonstrates the Streamlit dashboard functionality with mock data
when the FastAPI backend is not available.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import random
import json


def generate_mock_api_response(endpoint: str, method: str = "GET", data=None):
    """Generate mock API responses for testing"""
    
    if endpoint == "/health":
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    elif endpoint == "/predict/category":
        categories = ["Technical Issue", "Billing", "Feature Request", "Bug Report", "Account"]
        selected_category = random.choice(categories)
        return {
            "category": selected_category,
            "confidence": random.uniform(0.75, 0.95),
            "predictions": {
                "xgboost": {
                    "category": selected_category,
                    "confidence": random.uniform(0.7, 0.9)
                },
                "tensorflow": {
                    "category": random.choice(categories),
                    "confidence": random.uniform(0.6, 0.85)
                }
            },
            "probabilities": {cat: random.uniform(0.1, 0.9) if cat == selected_category else random.uniform(0.05, 0.3) 
                           for cat in categories}
        }
    
    elif endpoint == "/retrieve/solutions":
        solutions = []
        for i in range(data.get("top_k", 5)):
            solutions.append({
                "resolution": f"This is a sample resolution #{i+1} for your query. "
                             f"You can find detailed steps in our knowledge base. "
                             f"This solution has been verified by our expert team.",
                "score": random.uniform(0.5, 0.95),
                "category": random.choice(["Technical Issue", "Billing", "Feature Request"]),
                "source": f"KB-{random.randint(1000, 9999)}",
                "ticket_id": f"TKT-{random.randint(100000, 999999)}"
            })
        return {"solutions": sorted(solutions, key=lambda x: x["score"], reverse=True)}
    
    elif endpoint == "/anomalies/recent":
        anomalies = []
        severities = ["high", "medium", "low"]
        types = ["Data Drift", "Performance Degradation", "Unusual Traffic", "Model Error", "System Alert"]
        
        for i in range(random.randint(0, 8)):
            severity = random.choice(severities)
            anomaly_type = random.choice(types)
            
            anomalies.append({
                "type": anomaly_type,
                "severity": severity,
                "details": f"Detected {anomaly_type.lower()} with {severity} severity. "
                          f"Threshold exceeded by {random.randint(10, 50)}%. Investigate immediately.",
                "timestamp": (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat()
            })
        
        return {"anomalies": sorted(anomalies, key=lambda x: x["timestamp"], reverse=True)}
    
    elif endpoint == "/monitoring/status":
        return {
            "accuracy": random.uniform(0.8, 0.95),
            "weighted_f1": random.uniform(0.75, 0.92),
            "drift_score": random.uniform(0.05, 0.4),
            "avg_latency": random.uniform(50, 200),
            "accuracy_change": random.uniform(-0.05, 0.05),
            "f1_change": random.uniform(-0.03, 0.03),
            "latency_change": random.uniform(-20, 20),
            "metrics_history": [
                {
                    "timestamp": (datetime.now() - timedelta(days=i)).isoformat(),
                    "accuracy": random.uniform(0.8, 0.95),
                    "drift_score": random.uniform(0.05, 0.3)
                }
                for i in range(7, 0, -1)
            ],
            "model_comparison": [
                {"Model": "XGBoost", "Accuracy": random.uniform(0.85, 0.92), "F1": random.uniform(0.8, 0.9), "Latency": random.uniform(30, 60)},
                {"Model": "TensorFlow", "Accuracy": random.uniform(0.82, 0.90), "F1": random.uniform(0.78, 0.88), "Latency": random.uniform(80, 120)}
            ],
            "system_health": {
                "cpu_usage": random.uniform(20, 80),
                "memory_usage": random.uniform(40, 85),
                "requests_per_minute": random.randint(50, 200)
            }
        }
    
    elif endpoint.startswith("/feedback/"):
        if endpoint == "/feedback/stats":
            return {
                "total_feedback": random.randint(100, 500),
                "avg_customer_satisfaction": random.uniform(3.5, 4.8),
                "prediction_accuracy": random.uniform(0.75, 0.92),
                "avg_resolution_time": random.uniform(2.5, 8.0),
                "feedback_trends": [
                    {
                        "date": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"),
                        "satisfaction": random.uniform(3.0, 5.0),
                        "accuracy": random.uniform(0.7, 0.95)
                    }
                    for i in range(30, 0, -1)
                ],
                "category_performance": [
                    {"Category": cat, "Accuracy": random.uniform(0.7, 0.95), "Satisfaction": random.uniform(3.5, 4.8)}
                    for cat in ["Technical Issue", "Billing", "Feature Request", "Bug Report", "Account"]
                ]
            }
        else:
            return {"status": "success", "message": "Feedback submitted successfully"}
    
    return None


def run_demo():
    """Run the Streamlit dashboard demo"""
    
    st.set_page_config(
        page_title="InsightDesk AI Demo",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 InsightDesk AI Dashboard - Demo Mode")
    
    st.warning("""
    **🧪 Demo Mode Active**  
    This is a demonstration of the Streamlit dashboard with mock data.
    To use with real data, start the FastAPI backend server first:
    ```bash
    cd src && python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    ```
    """)
    
    # Simple demo interface
    tab1, tab2, tab3 = st.tabs(["📊 Mock Data Preview", "🎯 Features Overview", "🚀 Getting Started"])
    
    with tab1:
        st.subheader("📊 Sample API Responses")
        
        # Demo prediction
        if st.button("🔍 Test Ticket Prediction"):
            mock_response = generate_mock_api_response("/predict/category")
            st.json(mock_response)
        
        # Demo solutions
        if st.button("🔎 Test Solution Retrieval"):
            mock_response = generate_mock_api_response("/retrieve/solutions", data={"top_k": 3})
            st.json(mock_response)
        
        # Demo anomalies
        if st.button("🚨 Test Anomaly Detection"):
            mock_response = generate_mock_api_response("/anomalies/recent")
            st.json(mock_response)
    
    with tab2:
        st.subheader("🎯 Dashboard Features")
        
        features = [
            {"Feature": "📨 Ticket Categorization", "Description": "AI-powered ticket classification with XGBoost + TensorFlow"},
            {"Feature": "🔎 Solution Retrieval", "Description": "RAG-based solution search with semantic similarity"},
            {"Feature": "🚨 Anomaly Detection", "Description": "Real-time anomaly monitoring with severity alerts"},
            {"Feature": "📊 Performance Monitoring", "Description": "Model drift detection and performance tracking"},
            {"Feature": "🔄 Feedback Collection", "Description": "Agent and customer feedback for continuous improvement"}
        ]
        
        features_df = pd.DataFrame(features)
        st.dataframe(features_df, use_container_width=True)
    
    with tab3:
        st.subheader("🚀 Getting Started")
        
        st.markdown("""
        ### 1. Install Dependencies
        ```bash
        pip install streamlit requests plotly pandas
        ```
        
        ### 2. Start FastAPI Backend
        ```bash
        cd src
        python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
        ```
        
        ### 3. Launch Streamlit Dashboard
        ```bash
        streamlit run app.py
        ```
        
        ### 4. Access the Dashboard
        - **Dashboard**: http://localhost:8501
        - **API Docs**: http://localhost:8000/docs
        - **API Health**: http://localhost:8000/health
        """)
        
        st.info("💡 **Tip**: Make sure both servers are running simultaneously for full functionality!")


if __name__ == "__main__":
    run_demo()