#!/usr/bin/env python3
"""
Streamlit Dashboard for Intelligent Product Support System
=========================================================

A comprehensive web interface for the InsightDesk AI platform featuring:
- Ticket classification and categorization
- RAG-powered solution retrieval
- Anomaly detection and monitoring
- Model drift tracking
- Feedback collection and analysis

Author: Vishesh Srivastava
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Any
import time


# Configuration
API_BASE_URL = "http://localhost:8000"
STREAMLIT_CONFIG = {
    "page_title": "InsightDesk AI Dashboard",
    "page_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}


def setup_page_config():
    """Configure Streamlit page settings and custom CSS"""
    st.set_page_config(**STREAMLIT_CONFIG)
    
    # Custom CSS for polished styling
    st.markdown("""
    <style>
    /* Main theme styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .anomaly-high {
        background: #fee;
        border-left: 4px solid #dc3545;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
    }
    
    .anomaly-medium {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
    }
    
    .anomaly-low {
        background: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
    }
    
    /* Form styling */
    .stSelectbox > label, .stTextInput > label, .stTextArea > label {
        font-weight: 600;
        color: #333;
    }
    
    /* Success/Error styling */
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
    """, unsafe_allow_html=True)


def create_header():
    """Create the main dashboard header"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Intelligent Product Support System</h1>
        <p>AI-Powered Ticket Classification • RAG Solution Retrieval • Anomaly Detection • Performance Monitoring</p>
    </div>
    """, unsafe_allow_html=True)


def make_api_request(endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Optional[Dict]:
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            st.error(f"Unsupported HTTP method: {method}")
            return None
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error ({response.status_code}): {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("🔌 Unable to connect to API server. Please ensure the FastAPI server is running on http://localhost:8000")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The server might be overloaded.")
        return None
    except Exception as e:
        st.error(f"🚨 Unexpected error: {str(e)}")
        return None


def predict_category():
    """Ticket Categorization Tab"""
    st.header("📨 Ticket Categorization")
    st.markdown("Classify support tickets using AI models (XGBoost + TensorFlow)")
    
    with st.form("ticket_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            subject = st.text_input("Subject", placeholder="Brief description of the issue")
            description = st.text_area("Description", placeholder="Detailed description of the problem", height=100)
            error_logs = st.text_area("Error Logs", placeholder="Any error messages or logs", height=80)
        
        with col2:
            stack_trace = st.text_area("Stack Trace", placeholder="Stack trace if available", height=80)
            product = st.selectbox("Product", ["Product A", "Product B", "Product C", "Other"])
            priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
            severity = st.selectbox("Severity", ["1 - Low", "2 - Medium", "3 - High", "4 - Critical"])
        
        submitted = st.form_submit_button("🔍 Classify Ticket", use_container_width=True)
    
    if submitted and subject and description:
        with st.spinner("🤖 Analyzing ticket..."):
            # Prepare request data
            ticket_data = {
                "subject": subject,
                "description": description,
                "error_logs": error_logs,
                "stack_trace": stack_trace,
                "product": product,
                "priority": priority,
                "severity": severity
            }
            
            start_time = time.time()
            result = make_api_request("/predict/category", "POST", ticket_data)
            inference_time = time.time() - start_time
            
            if result:
                st.success("✅ Classification completed successfully!")

                # Derive summary values from model predictions when top-level fields are absent.
                primary_prediction = result.get("predictions", {}).get("xgboost", {})
                recommended_category = result.get("category") or primary_prediction.get("category", "Unknown")
                confidence_value = result.get("confidence")
                if confidence_value is None:
                    confidence_value = primary_prediction.get("confidence", 0)
                
                # Display results in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🎯 Recommended Category", recommended_category)
                
                with col2:
                    st.metric("⚡ Inference Time", f"{inference_time:.2f}s")
                
                with col3:
                    st.metric("🎯 Confidence", f"{confidence_value:.1%}")
                
                # Model predictions comparison
                st.subheader("🤖 Model Predictions Comparison")
                
                if "predictions" in result:
                    predictions_df = pd.DataFrame([
                        {"Model": "XGBoost", "Category": result["predictions"].get("xgboost", {}).get("category", "N/A"), 
                         "Confidence": result["predictions"].get("xgboost", {}).get("confidence", 0)},
                        {"Model": "TensorFlow", "Category": result["predictions"].get("tensorflow", {}).get("category", "N/A"), 
                         "Confidence": result["predictions"].get("tensorflow", {}).get("confidence", 0)}
                    ])
                    
                    # Bar chart of model confidences
                    fig = px.bar(predictions_df, x="Model", y="Confidence", 
                               color="Model", title="Model Confidence Comparison",
                               color_discrete_map={"XGBoost": "#1f77b4", "TensorFlow": "#ff7f0e"})
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed predictions table
                    st.subheader("📊 Detailed Predictions")
                    st.dataframe(predictions_df, use_container_width=True)
                    
                    # Probabilities breakdown
                    if "probabilities" in result:
                        st.subheader("📈 Category Probabilities")
                        prob_data = result["probabilities"]
                        prob_df = pd.DataFrame(list(prob_data.items()), columns=["Category", "Probability"])
                        prob_df = prob_df.sort_values("Probability", ascending=False)
                        
                        fig_prob = px.bar(prob_df.head(5), x="Category", y="Probability", 
                                        title="Top 5 Category Probabilities")
                        st.plotly_chart(fig_prob, use_container_width=True)
    
    elif submitted:
        st.warning("⚠️ Please fill in at least the Subject and Description fields.")


def retrieve_solutions():
    """Solution Retrieval Tab"""
    st.header("🔎 Solution Retrieval")
    st.markdown("Find relevant solutions using RAG (Retrieval-Augmented Generation)")
    
    # Input form
    with st.form("solution_form"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query_subject = st.text_input("Subject", placeholder="Brief issue description")
            query_description = st.text_area("Description", placeholder="Detailed problem description", height=100)
        
        with col2:
            top_k = st.slider("Number of Solutions", min_value=1, max_value=10, value=5)
            search_type = st.selectbox("Search Type", ["hybrid", "semantic", "keyword"])
        
        search_submitted = st.form_submit_button("🔍 Find Solutions", use_container_width=True)
    
    if search_submitted and (query_subject or query_description):
        with st.spinner("🔍 Searching knowledge base..."):
            search_data = {
                "subject": query_subject or "General Issue",
                "description": query_description or "No detailed description provided",
                "k": top_k,
                "search_type": search_type
            }
            
            result = make_api_request("/retrieve/solutions", "POST", search_data)
            
            if result and "solutions" in result:
                st.success(f"✅ Found {len(result['solutions'])} relevant solutions!")
                
                # Display solutions
                for i, solution in enumerate(result["solutions"], 1):
                    with st.expander(f"Solution #{i} - Similarity: {solution.get('score', 0):.3f}"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown("**Resolution:**")
                            st.write(solution.get("resolution", "No resolution available"))
                            
                            if solution.get("category"):
                                st.markdown(f"**Category:** {solution['category']}")
                        
                        with col2:
                            st.metric("Similarity Score", f"{solution.get('score', 0):.3f}")
                            if solution.get("source"):
                                st.markdown(f"**Source:** {solution['source']}")
                            if solution.get("ticket_id"):
                                st.markdown(f"**Ticket ID:** {solution['ticket_id']}")
                
                # Solutions summary table
                st.subheader("📊 Solutions Summary")
                solutions_df = pd.DataFrame([
                    {
                        "Rank": i+1,
                        "Score": sol.get("score", 0),
                        "Category": sol.get("category", "Unknown"),
                        "Source": sol.get("source", "N/A"),
                        "Preview": sol.get("resolution", "")[:100] + "..." if len(sol.get("resolution", "")) > 100 else sol.get("resolution", "")
                    }
                    for i, sol in enumerate(result["solutions"])
                ])
                st.dataframe(solutions_df, use_container_width=True)
                
                # TODO: Add Graph-RAG visualization
                st.info("🚧 **Coming Soon:** Graph visualization of knowledge relationships and citation paths")
            
            elif result:
                st.warning("🤷 No solutions found for your query. Try rephrasing or using different keywords.")
    
    elif search_submitted:
        st.warning("⚠️ Please enter at least a subject or description to search.")


def view_anomalies():
    """Anomaly Detection Tab"""
    st.header("🚨 Anomaly Detection")
    st.markdown("Monitor and analyze system anomalies in real-time")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        time_range = st.selectbox("Time Range", ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last Week"])
    
    with col2:
        severity_filter = st.selectbox("Severity Filter", ["All", "High", "Medium", "Low"])
    
    with col3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Fetch anomalies
    with st.spinner("🔍 Fetching anomalies..."):
        result = make_api_request("/anomalies/recent")
        
        if result and "anomalies" in result:
            anomalies = result["anomalies"]
            
            # Filter by severity if specified
            if severity_filter != "All":
                anomalies = [a for a in anomalies if a.get("severity", "").lower() == severity_filter.lower()]
            
            if anomalies:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Anomalies", len(anomalies))
                
                with col2:
                    high_severity = sum(1 for a in anomalies if a.get("severity", "").lower() == "high")
                    st.metric("High Severity", high_severity, delta=None if high_severity == 0 else "🚨")
                
                with col3:
                    latest_time = max(anomalies, key=lambda x: x.get("timestamp", ""))["timestamp"] if anomalies else "N/A"
                    st.metric("Latest Detection", latest_time[:19] if latest_time != "N/A" else "N/A")
                
                with col4:
                    unique_types = len(set(a.get("type", "") for a in anomalies))
                    st.metric("Anomaly Types", unique_types)
                
                # Anomalies display
                st.subheader("🚨 Recent Anomalies")
                
                for anomaly in anomalies:
                    severity = anomaly.get("severity", "low").lower()
                    
                    # Apply severity-based styling
                    if severity == "high":
                        style_class = "anomaly-high"
                        icon = "🔴"
                    elif severity == "medium":
                        style_class = "anomaly-medium"
                        icon = "🟡"
                    else:
                        style_class = "anomaly-low"
                        icon = "🔵"
                    
                    st.markdown(f"""
                    <div class="{style_class}">
                        <strong>{icon} {anomaly.get('type', 'Unknown Type')}</strong><br>
                        <em>Severity: {anomaly.get('severity', 'Unknown')}</em><br>
                        {anomaly.get('details', 'No details available')}<br>
                        <small>📅 {anomaly.get('timestamp', 'Unknown time')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Anomalies table
                st.subheader("📊 Anomalies Table")
                anomalies_df = pd.DataFrame(anomalies)
                st.dataframe(anomalies_df, use_container_width=True)
                
                # Anomaly distribution chart
                if len(anomalies) > 1:
                    st.subheader("📈 Anomaly Distribution")
                    severity_counts = pd.Series([a.get("severity", "Unknown") for a in anomalies]).value_counts()
                    fig = px.pie(values=severity_counts.values, names=severity_counts.index, 
                               title="Anomalies by Severity")
                    st.plotly_chart(fig, use_container_width=True)
                    
            else:
                st.success("✅ No anomalies detected in the selected time range!")
        
        else:
            st.info("📊 Unable to fetch anomaly data. API might be unavailable.")
    
    # TODO: Real-time streaming
    st.info("🚧 **Coming Soon:** Real-time anomaly streaming with WebSocket connection")


def monitor_status():
    """Monitoring & Drift Tab"""
    st.header("📊 Monitoring & Drift Detection")
    st.markdown("Track model performance and detect data drift")
    
    # Fetch monitoring status
    with st.spinner("📈 Loading monitoring data..."):
        result = make_api_request("/monitoring/status")
        
        if result:
            # Key metrics cards
            st.subheader("🎯 Key Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                accuracy = result.get("accuracy", 0)
                st.metric("Model Accuracy", f"{accuracy:.1%}", 
                         delta=f"{result.get('accuracy_change', 0):+.1%}" if result.get('accuracy_change') else None)
            
            with col2:
                f1_score = result.get("weighted_f1", 0)
                st.metric("Weighted F1", f"{f1_score:.3f}",
                         delta=f"{result.get('f1_change', 0):+.3f}" if result.get('f1_change') else None)
            
            with col3:
                drift_score = result.get("drift_score", 0)
                drift_status = "🟢 Normal" if drift_score < 0.1 else "🟡 Warning" if drift_score < 0.3 else "🔴 Critical"
                st.metric("Drift Score", f"{drift_score:.3f}", delta=drift_status)
            
            with col4:
                latency = result.get("avg_latency", 0)
                st.metric("Avg Latency", f"{latency:.0f}ms",
                         delta=f"{result.get('latency_change', 0):+.0f}ms" if result.get('latency_change') else None)
            
            # Drift alerts
            if drift_score > 0.3:
                st.error("🚨 **Critical Drift Detected!** Model performance may be degraded. Consider retraining.")
            elif drift_score > 0.1:
                st.warning("⚠️ **Drift Warning:** Monitor model performance closely.")
            
            # Performance over time
            if "metrics_history" in result:
                st.subheader("📈 Performance Trends")
                
                history = result["metrics_history"]
                if history:
                    # Create time series chart
                    df_history = pd.DataFrame(history)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_history.get("timestamp", []), 
                                           y=df_history.get("accuracy", []),
                                           mode='lines+markers', name='Accuracy',
                                           line=dict(color='#1f77b4')))
                    fig.add_trace(go.Scatter(x=df_history.get("timestamp", []), 
                                           y=df_history.get("drift_score", []),
                                           mode='lines+markers', name='Drift Score',
                                           line=dict(color='#ff7f0e'), yaxis='y2'))
                    
                    fig.update_layout(
                        title="Model Performance Over Time",
                        xaxis_title="Time",
                        yaxis=dict(title="Accuracy", side="left"),
                        yaxis2=dict(title="Drift Score", side="right", overlaying="y"),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Model comparison
            if "model_comparison" in result:
                st.subheader("🤖 Model Performance Comparison")
                comparison_df = pd.DataFrame(result["model_comparison"])
                st.dataframe(comparison_df, use_container_width=True)
            
            # System health
            st.subheader("🔧 System Health")
            health_data = result.get("system_health", {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                cpu_usage = health_data.get("cpu_usage", 0)
                st.metric("CPU Usage", f"{cpu_usage:.1f}%")
            
            with col2:
                memory_usage = health_data.get("memory_usage", 0)
                st.metric("Memory Usage", f"{memory_usage:.1f}%")
            
            with col3:
                api_requests = health_data.get("requests_per_minute", 0)
                st.metric("Requests/min", f"{api_requests:.0f}")
                
        else:
            st.error("📊 Unable to fetch monitoring data. Please check API connectivity.")


def submit_feedback():
    """Feedback Tab"""
    st.header("🔄 Feedback Collection")
    st.markdown("Collect feedback to improve model performance")
    
    # Feedback forms
    tab1, tab2, tab3 = st.tabs(["👤 Agent Feedback", "🙋 Customer Feedback", "📊 Feedback Stats"])
    
    with tab1:
        st.subheader("👤 Agent Feedback")
        with st.form("agent_feedback_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                ticket_id = st.text_input("Ticket ID", placeholder="TKT-123456")
                predicted_category = st.text_input("Predicted Category", placeholder="As shown by the model")
                actual_category = st.selectbox("Actual Category", 
                                             ["Technical Issue", "Billing", "Feature Request", "Bug Report", "Account", "Other"])
            
            with col2:
                prediction_correct = st.radio("Was the prediction correct?", ["Yes", "No", "Partially"])
                resolution_time = st.number_input("Resolution Time (hours)", min_value=0.0, step=0.5)
                agent_satisfaction = st.slider("Agent Satisfaction (1-5)", 1, 5, 3)
            
            feedback_notes = st.text_area("Additional Notes", placeholder="Any specific observations or suggestions")
            
            if st.form_submit_button("📝 Submit Agent Feedback", use_container_width=True):
                feedback_data = {
                    "ticket_id": ticket_id,
                    "predicted_category": predicted_category,
                    "actual_category": actual_category,
                    "prediction_correct": prediction_correct,
                    "resolution_time": resolution_time,
                    "agent_satisfaction": agent_satisfaction,
                    "notes": feedback_notes,
                    "feedback_type": "agent"
                }
                
                result = make_api_request("/feedback/agent", "POST", feedback_data)
                if result:
                    st.success("✅ Agent feedback submitted successfully!")
                else:
                    st.error("❌ Failed to submit feedback. Please try again.")
    
    with tab2:
        st.subheader("🙋 Customer Feedback")
        with st.form("customer_feedback_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                customer_ticket_id = st.text_input("Ticket ID", placeholder="TKT-123456", key="customer_ticket")
                customer_satisfaction = st.slider("Customer Satisfaction (1-5)", 1, 5, 3, key="customer_sat")
                resolution_rating = st.slider("Resolution Quality (1-5)", 1, 5, 3)
            
            with col2:
                response_time_rating = st.slider("Response Time Rating (1-5)", 1, 5, 3)
                would_recommend = st.radio("Would recommend our support?", ["Yes", "No", "Maybe"])
                issue_resolved = st.radio("Was your issue resolved?", ["Yes", "No", "Partially"])
            
            customer_comments = st.text_area("Customer Comments", placeholder="Customer's feedback or suggestions")
            
            if st.form_submit_button("📝 Submit Customer Feedback", use_container_width=True):
                customer_feedback_data = {
                    "ticket_id": customer_ticket_id,
                    "satisfaction_score": customer_satisfaction,
                    "resolution_rating": resolution_rating,
                    "response_time_rating": response_time_rating,
                    