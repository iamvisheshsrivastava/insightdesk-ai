#!/usr/bin/env python3
"""
Demo Streamlit Dashboard for InsightDesk AI
==========================================

A simplified demo version that works without backend dependencies.
Perfect for deployment platforms like Streamlit Cloud.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import random
import time

# Configuration for demo mode
DEMO_CONFIG = {
    "page_title": "InsightDesk AI - Demo Dashboard",
    "page_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

def setup_demo_page_config():
    """Configure Streamlit page settings for demo"""
    st.set_page_config(**DEMO_CONFIG)
    
    # Custom CSS for polished styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

def generate_demo_data():
    """Generate realistic demo data for the dashboard"""
    
    # Demo ticket for classification
    demo_ticket = {
        "ticket_id": "DEMO-001",
        "subject": "Cannot login to application",
        "description": "User is unable to authenticate with correct credentials. Getting timeout error after 30 seconds.",
        "error_logs": "Authentication timeout after 30 seconds",
        "product": "web_application",
        "channel": "email",
        "priority": "high",
        "customer_tier": "premium"
    }
    
    # Demo predictions
    demo_predictions = {
        "xgboost": {
            "predicted_category": "authentication",
            "confidence": 0.92,
            "probabilities": {
                "authentication": 0.92,
                "database": 0.05,
                "api": 0.03
            },
            "inference_time_ms": 45.2
        },
        "tensorflow": {
            "predicted_category": "authentication",
            "confidence": 0.89,
            "probabilities": {
                "authentication": 0.89,
                "database": 0.07,
                "api": 0.04
            },
            "inference_time_ms": 67.8
        }
    }
    
    # Demo solutions
    demo_solutions = [
        {
            "resolution_id": "RES-AUTH-001",
            "category": "authentication",
            "product": "web_application",
            "resolution": "Reset user password and clear authentication cache",
            "resolution_steps": "1. Reset password in admin panel\n2. Clear browser cache and cookies\n3. Restart authentication service",
            "similarity_score": 0.94,
            "success_rate": 0.89,
            "usage_count": 47
        },
        {
            "resolution_id": "RES-AUTH-002", 
            "category": "authentication",
            "product": "web_application",
            "resolution": "Check authentication service configuration",
            "resolution_steps": "1. Verify service endpoints\n2. Check timeout settings\n3. Restart auth microservice",
            "similarity_score": 0.87,
            "success_rate": 0.82,
            "usage_count": 31
        }
    ]
    
    # Demo anomalies
    demo_anomalies = []
    for i in range(10):
        demo_anomalies.append({
            "anomaly_id": f"ANO-{i+1:03d}",
            "timestamp": datetime.now() - timedelta(hours=random.randint(1, 48)),
            "severity": random.choice(["high", "medium", "low"]),
            "type": random.choice(["spike_in_tickets", "unusual_category", "performance_degradation"]),
            "details": f"Anomaly detected in system behavior - pattern {i+1}",
            "affected_component": random.choice(["API", "Database", "Authentication", "Payment"])
        })
    
    return demo_ticket, demo_predictions, demo_solutions, demo_anomalies

def main():
    """Main demo dashboard application"""
    setup_demo_page_config()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 InsightDesk AI - Demo Dashboard</h1>
        <p>Intelligent Support Ticket Classification & Solution Retrieval</p>
        <p><em>Demo Mode - Using Simulated Data</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate demo data
    demo_ticket, demo_predictions, demo_solutions, demo_anomalies = generate_demo_data()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    selected_tab = st.sidebar.radio(
        "Choose a section:",
        ["📨 Ticket Classification", "🔎 Solution Retrieval", "🚨 Anomaly Detection", "📊 System Metrics", "🔄 Feedback Demo"]
    )
    
    # Display connection status
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔗 System Status")
        st.markdown('<div class="info-box">🟡 Demo Mode Active<br/>Using simulated data</div>', unsafe_allow_html=True)
    
    if selected_tab == "📨 Ticket Classification":
        st.header("📨 Ticket Classification")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📝 Ticket Information")
            
            # Pre-populate with demo data
            ticket_id = st.text_input("Ticket ID", value=demo_ticket["ticket_id"])
            subject = st.text_area("Subject", value=demo_ticket["subject"], height=100)
            description = st.text_area("Description", value=demo_ticket["description"], height=150)
            error_logs = st.text_area("Error Logs", value=demo_ticket["error_logs"], height=100)
            
            col1a, col1b = st.columns(2)
            with col1a:
                product = st.selectbox("Product", ["web_application", "mobile_app", "api_server", "payment_gateway"], 
                                     index=0)
                priority = st.selectbox("Priority", ["low", "medium", "high", "critical"], index=2)
            with col1b:
                channel = st.selectbox("Channel", ["email", "chat", "phone", "api"], index=0)
                customer_tier = st.selectbox("Customer Tier", ["basic", "premium", "enterprise"], index=1)
            
            if st.button("🔮 Classify Ticket", type="primary"):
                with st.spinner("Classifying ticket..."):
                    time.sleep(1.5)  # Simulate processing time
                    st.session_state.demo_classification_done = True
        
        with col2:
            st.subheader("🤖 AI Predictions")
            
            if hasattr(st.session_state, 'demo_classification_done'):
                st.success("✅ Classification Complete!")
                
                # Display predictions
                for model_name, prediction in demo_predictions.items():
                    with st.expander(f"📊 {model_name.upper()} Model Results", expanded=True):
                        col2a, col2b, col2c = st.columns(3)
                        with col2a:
                            st.metric("Predicted Category", prediction["predicted_category"].title())
                        with col2b:
                            st.metric("Confidence", f"{prediction['confidence']:.1%}")
                        with col2c:
                            st.metric("Inference Time", f"{prediction['inference_time_ms']:.1f}ms")
                        
                        # Probability distribution
                        prob_df = pd.DataFrame(list(prediction["probabilities"].items()), 
                                             columns=["Category", "Probability"])
                        fig = px.bar(prob_df, x="Category", y="Probability", 
                                   title=f"{model_name.upper()} Prediction Probabilities")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Model comparison
                st.subheader("⚖️ Model Comparison")
                comparison_data = {
                    "Model": ["XGBoost", "TensorFlow"],
                    "Accuracy": [83.4, 84.7],
                    "Confidence": [92.0, 89.0],
                    "Latency (ms)": [45.2, 67.8]
                }
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
            
            else:
                st.info("👆 Click 'Classify Ticket' to see AI predictions")
    
    elif selected_tab == "🔎 Solution Retrieval":
        st.header("🔎 Solution Retrieval")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🔍 Search for Solutions")
            
            query_subject = st.text_input("Problem Subject", value="Cannot login to application")
            query_description = st.text_area("Problem Description", 
                                            value="User authentication failing with timeout errors", height=100)
            query_category = st.selectbox("Category (optional)", 
                                        ["", "authentication", "database", "api", "payment"], index=1)
            
            col1a, col1b = st.columns(2)
            with col1a:
                search_type = st.selectbox("Search Type", ["hybrid", "semantic", "keyword"], index=0)
                k_results = st.slider("Number of Results", 1, 10, 5)
            with col1b:
                product_filter = st.selectbox("Product Filter", ["", "web_application", "mobile_app"], index=1)
            
            if st.button("🔍 Search Solutions", type="primary"):
                with st.spinner("Searching knowledge base..."):
                    time.sleep(1.2)
                    st.session_state.demo_search_done = True
        
        with col2:
            st.subheader("💡 Relevant Solutions")
            
            if hasattr(st.session_state, 'demo_search_done'):
                st.success(f"✅ Found {len(demo_solutions)} relevant solutions!")
                
                for i, solution in enumerate(demo_solutions, 1):
                    with st.expander(f"🎯 Solution #{i} - {solution['resolution_id']}", expanded=i==1):
                        col2a, col2b, col2c = st.columns(3)
                        with col2a:
                            st.metric("Similarity Score", f"{solution['similarity_score']:.2f}")
                        with col2b:
                            st.metric("Success Rate", f"{solution['success_rate']:.1%}")
                        with col2c:
                            st.metric("Usage Count", solution['usage_count'])
                        
                        st.markdown(f"**Resolution:** {solution['resolution']}")
                        st.markdown("**Steps:**")
                        st.code(solution['resolution_steps'])
                        
                        if st.button(f"👍 Mark as Helpful #{i}", key=f"helpful_{i}"):
                            st.success("Feedback recorded! Thank you.")
            else:
                st.info("👆 Enter a problem description and search for solutions")
    
    elif selected_tab == "🚨 Anomaly Detection":
        st.header("🚨 Anomaly Detection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Recent Anomalies")
            
            # Filter controls
            col1a, col1b = st.columns(2)
            with col1a:
                severity_filter = st.multiselect("Severity Filter", 
                                               ["high", "medium", "low"], default=["high", "medium"])
            with col1b:
                hours_back = st.slider("Hours Back", 1, 48, 24)
            
            # Filter and display anomalies
            filtered_anomalies = [a for a in demo_anomalies 
                                if a["severity"] in severity_filter and 
                                   a["timestamp"] >= datetime.now() - timedelta(hours=hours_back)]
            
            if filtered_anomalies:
                for anomaly in filtered_anomalies:
                    severity_color = {"high": "🔴", "medium": "🟡", "low": "🔵"}[anomaly["severity"]]
                    st.markdown(f"""
                    <div class="metric-card">
                        {severity_color} <strong>{anomaly['anomaly_id']}</strong> - {anomaly['type'].replace('_', ' ').title()}<br/>
                        <small>🕐 {anomaly['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} | 🔧 {anomaly['affected_component']}</small><br/>
                        📝 {anomaly['details']}
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("<br/>", unsafe_allow_html=True)
            else:
                st.info("No anomalies found for the selected criteria.")
        
        with col2:
            st.subheader("📈 Anomaly Statistics")
            
            # Anomaly distribution by severity
            severity_counts = pd.DataFrame({
                'Severity': ['High', 'Medium', 'Low'],
                'Count': [3, 4, 3]
            })
            fig_severity = px.pie(severity_counts, values='Count', names='Severity', 
                                title="Anomalies by Severity")
            st.plotly_chart(fig_severity, use_container_width=True)
            
            # Trend over time
            trend_data = pd.DataFrame({
                'Hour': list(range(-24, 1)),
                'Anomalies': [random.randint(0, 3) for _ in range(25)]
            })
            fig_trend = px.line(trend_data, x='Hour', y='Anomalies', 
                              title="Anomaly Trend (24h)")
            st.plotly_chart(fig_trend, use_container_width=True)
    
    elif selected_tab == "📊 System Metrics":
        st.header("📊 System Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Model Accuracy", "84.7%", "↑ 2.3%")
        with col2:
            st.metric("⚡ Avg Response Time", "56ms", "↓ 12ms")
        with col3:
            st.metric("📈 Requests/Hour", "2,847", "↑ 15%")
        with col4:
            st.metric("😊 Satisfaction Score", "4.6/5", "↑ 0.2")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏃‍♂️ Model Performance Over Time")
            
            # Generate performance data
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
            performance_data = pd.DataFrame({
                'Date': dates,
                'XGBoost_Accuracy': [0.83 + random.uniform(-0.02, 0.02) for _ in dates],
                'TensorFlow_Accuracy': [0.847 + random.uniform(-0.02, 0.02) for _ in dates]
            })
            
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(x=performance_data['Date'], y=performance_data['XGBoost_Accuracy'],
                                        mode='lines', name='XGBoost', line=dict(color='#667eea')))
            fig_perf.add_trace(go.Scatter(x=performance_data['Date'], y=performance_data['TensorFlow_Accuracy'],
                                        mode='lines', name='TensorFlow', line=dict(color='#764ba2')))
            fig_perf.update_layout(title="Model Accuracy Trends", yaxis_title="Accuracy")
            st.plotly_chart(fig_perf, use_container_width=True)
        
        with col2:
            st.subheader("📊 Category Distribution")
            
            category_data = pd.DataFrame({
                'Category': ['Authentication', 'Database', 'API', 'Payment', 'UI/UX'],
                'Count': [245, 189, 167, 134, 98]
            })
            fig_cat = px.bar(category_data, x='Category', y='Count', 
                           title="Ticket Categories (Last 30 Days)")
            st.plotly_chart(fig_cat, use_container_width=True)
        
        # System health indicators
        st.subheader("🔋 System Health")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="success-box">✅ API Status: Healthy</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="success-box">✅ Models: Loaded & Active</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="success-box">✅ Database: Connected</div>', unsafe_allow_html=True)
    
    elif selected_tab == "🔄 Feedback Demo":
        st.header("🔄 Feedback Collection")
        
        tab1, tab2 = st.tabs(["👨‍💼 Agent Feedback", "👥 Customer Feedback"])
        
        with tab1:
            st.subheader("📝 Agent Experience Feedback")
            
            col1, col2 = st.columns(2)
            with col1:
                ticket_id_feedback = st.text_input("Ticket ID", "DEMO-001")
                predicted_category = st.selectbox("AI Predicted Category", 
                                                ["authentication", "database", "api"], index=0)
                actual_category = st.selectbox("Actual Category", 
                                             ["authentication", "database", "api"], index=0)
                resolution_time = st.number_input("Resolution Time (minutes)", min_value=1, value=15)
            
            with col2:
                prediction_helpful = st.slider("Prediction Helpfulness (1-5)", 1, 5, 4)
                solution_quality = st.slider("Solution Quality (1-5)", 1, 5, 4)
                overall_satisfaction = st.slider("Overall Satisfaction (1-5)", 1, 5, 4)
            
            agent_comments = st.text_area("Additional Comments", 
                                        "The prediction was accurate and helped resolve the issue quickly.")
            
            if st.button("📨 Submit Agent Feedback", type="primary"):
                st.success("✅ Agent feedback submitted successfully!")
        
        with tab2:
            st.subheader("💭 Customer Satisfaction Feedback")
            
            col1, col2 = st.columns(2)
            with col1:
                customer_ticket = st.text_input("Customer Ticket ID", "DEMO-001")
                resolution_rating = st.slider("Resolution Quality (1-5)", 1, 5, 5)
                response_time_rating = st.slider("Response Time (1-5)", 1, 5, 4)
            
            with col2:
                communication_rating = st.slider("Communication Quality (1-5)", 1, 5, 4)
                would_recommend = st.selectbox("Would you recommend our support?", 
                                             ["Yes", "No", "Maybe"], index=0)
            
            customer_comments = st.text_area("Customer Comments", 
                                           "Great support! The issue was resolved quickly and professionally.")
            
            if st.button("💌 Submit Customer Feedback", type="primary"):
                st.success("✅ Customer feedback submitted successfully!")
        
        # Feedback analytics
        st.subheader("📊 Feedback Analytics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📈 Average Satisfaction", "4.6/5", "↑ 0.2")
        with col2:
            st.metric("🎯 Prediction Accuracy", "89.3%", "↑ 3.1%")
        with col3:
            st.metric("⚡ Avg Resolution Time", "18 min", "↓ 5 min")

if __name__ == "__main__":
    main()