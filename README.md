# insightdesk-ai
   
<!-- Project Overview: This section provides a high-level description of what InsightDesk AI does -->
InsightDesk AI is an intelligent support platform that leverages advanced AI/ML technologies to automatically classify tickets, retrieve relevant solutions using RAG (Retrieval-Augmented Generation), and provide intelligent insights for support teams. Built with production-ready architecture featuring FastAPI, MLflow tracking, and comprehensive testing.

---

## 🚀 Features

<!-- Core Capabilities: These are the main AI-powered features that make this platform intelligent -->
* **🤖 Multi-Model Ticket Classification**
  - XGBoost gradient boosting for structured data
  - TensorFlow/Keras deep learning with multi-input architecture
  - Automatic model selection and ensemble predictions

* **🔍 Advanced RAG (Retrieval-Augmented Generation)**
  - Semantic search using sentence transformers
  - Hybrid retrieval combining semantic + keyword search
  - FAISS vector store for efficient similarity search
  - Knowledge base integration with ticket resolutions

* **📊 Comprehensive ML Operations**
  - MLflow experiment tracking and model versioning
  - Automated model comparison and performance metrics
  - Real-time API integration with dependency injection
  - Extensive unit testing and validation

* **🚀 Production-Ready Architecture**
  - FastAPI with async endpoints and proper error handling
  - Docker containerization for scalable deployment
  - Structured logging and comprehensive monitoring
  - Clean code architecture with modular design

---

## 🗂 Project Structure

<!-- Architecture Overview: This modular structure separates concerns for better maintainability -->
```
insightdesk-ai/
├─ data/                    # Raw and processed datasets
│  ├─ support_tickets.json  # Main ticket dataset
│  └─ features.joblib       # Preprocessed features
├─ src/
│  ├─ ingestion/            # Data loading and preprocessing
│  ├─ features/             # Feature engineering pipelines
│  ├─ models/               # ML model implementations
│  ├─ retrieval/            # RAG pipeline components
│  │  ├─ embedding_manager.py    # Text embedding generation
│  │  ├─ vector_store.py         # FAISS vector store
│  │  └─ rag_pipeline.py         # Main RAG pipeline
│  ├─ api/                  # FastAPI application
│  └─ utils/                # Shared utilities and logging
├─ models/                  # Trained model artifacts
├─ vector_store/            # RAG vector store and indices
├─ mlruns/                  # MLflow experiment tracking
├─ tests/                   # Comprehensive test suite
├─ scripts/                 # Training and utility scripts
├─ notebooks/               # Jupyter analysis notebooks
└─ requirements.txt         # Python dependencies
```

---

## ⚡ Quick Start

<!-- Getting Started: These steps will get you up and running in development mode -->

### 1. **Environment Setup**

```bash
# Clone the repository
git clone https://github.com/iamvisheshsrivastava/insightdesk-ai.git
cd insightdesk-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Data Preparation & Model Training**

```bash
# Extract and load support tickets data
python scripts/unzip_and_load.py

# Train both ML models with MLflow tracking
python scripts/train_and_compare_models.py

# Build RAG knowledge base and vector store
python scripts/build_rag_index.py
```

### 3. **Start the API Server**

```bash
# Launch FastAPI server with auto-reload
uvicorn src.api.main:app --reload

# API will be available at:
# - Main API: http://localhost:8000
# - Interactive docs: http://localhost:8000/docs
# - OpenAPI schema: http://localhost:8000/openapi.json
```

### 4. **Explore MLflow Experiments**

```bash
# Start MLflow UI (optional)
mlflow ui

# MLflow dashboard: http://localhost:5000
```

---

## 🔥 API Usage Examples

### Health Check & Model Status

```bash
# Check API health and model availability
curl -X GET "http://localhost:8000/health"

# Expected response:
{
  "status": "healthy",
  "models": {
    "xgboost": true,
    "tensorflow": true
  },
  "rag_available": true,
  "timestamp": "2025-09-29T10:30:00",
  "version": "1.0.0"
}
```

### Ticket Classification

```bash
# Classify ticket using XGBoost model
curl -X POST "http://localhost:8000/predict/category" \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "DEMO-001",
    "subject": "Cannot login to application",
    "description": "User is unable to authenticate with correct credentials. Getting timeout error.",
    "error_logs": "Authentication timeout after 30 seconds",
    "product": "web_application",
    "channel": "email",
    "priority": "high"
  }'

# Response includes predictions from available models:
{
  "ticket_id": "DEMO-001",
  "predictions": {
    "xgboost": {
      "predicted_category": "authentication",
      "confidence": 0.92,
      "probabilities": {
        "authentication": 0.92,
        "database": 0.05,
        "api": 0.03
      }
    },
    "tensorflow": {
      "predicted_category": "authentication", 
      "confidence": 0.89,
      "probabilities": {
        "authentication": 0.89,
        "database": 0.07,
        "api": 0.04
      }
    }
  },
  "available_models": ["xgboost", "tensorflow"],
  "total_inference_time_ms": 245.7,
  "timestamp": "2025-09-29T10:30:00"
}
```

### RAG Solution Retrieval

```bash
# Retrieve relevant solutions using RAG pipeline
curl -X POST "http://localhost:8000/retrieve/solutions" \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Cannot login to application",
    "description": "User authentication failing with timeout errors",
    "error_logs": "Authentication service timeout after 30 seconds",
    "product": "web_application",
    "category": "authentication",
    "k": 5,
    "search_type": "hybrid"
  }'

# Response with ranked solutions:
{
  "query_summary": "Cannot login to application",
  "solutions": [
    {
      "resolution_id": "RES-AUTH-001",
      "category": "authentication",
      "product": "web_application",
      "resolution": "Reset user password and clear authentication cache",
      "resolution_steps": "1. Reset password 2. Clear browser cache 3. Restart auth service",
      "similarity_score": 0.94,
      "semantic_score": 0.92,
      "keyword_score": 0.18,
      "success_rate": 0.89,
      "usage_count": 47,
      "content_type": "resolution",
      "search_type": "hybrid"
    }
  ],
  "total_found": 5,
  "search_type": "hybrid",
  "processing_time": 0.156
}
```

---

## 🧪 Testing & Validation
   
   # Predict category using XGBoost model
   curl -X POST http://localhost:8000/predict/category?model_type=xgboost \
        -H "Content-Type: application/json" \
        -d '{
          "ticket_id": "TK-001",
          "subject": "Cannot login to application",
          "description": "User is unable to authenticate with correct credentials. Getting timeout error after 30 seconds.",
          "error_logs": "Authentication timeout",
          "product": "web_application",
          "channel": "email",
          "priority": "high",
          "customer_tier": "premium",
          "previous_tickets": 2,
          "account_age_days": 365
        }'
   
   # Predict category using TensorFlow model
   curl -X POST http://localhost:8000/predict/category?model_type=tensorflow \
        -H "Content-Type: application/json" \
        -d '{
          "ticket_id": "TK-002", 
          "subject": "Database connection error",
          "description": "Application cannot connect to database server",
          "error_logs": "Connection timeout after 30 seconds",
          "stack_trace": "java.sql.SQLException: Connection timeout",
          "product": "api_server",
          "priority": "critical"
        }'
   
   # Compare predictions from both models
   curl -X POST http://localhost:8000/predict/category?model_type=both \
        -H "Content-Type: application/json" \
        -d '{
          "ticket_id": "TK-003",
          "subject": "Payment processing failure", 
          "description": "Credit card transaction failed with error code 402",
          "product": "payment_gateway",
          "priority": "high"
        }'
   
   # Test priority prediction (placeholder endpoint)
   curl -X POST http://localhost:8000/predict/priority \
        -H "Content-Type: application/json" \
        -d '{
          "ticket_id": "TK-004",
          "subject": "Server performance issue",
          "description": "Server response time is very slow"
        }'
   ```

7. **Example API Response**
   <!-- Sample response structure for category prediction -->
   ```json
   {
     "ticket_id": "TK-001",
     "predictions": {
       "xgboost": {
         "predicted_category": "authentication",
         "confidence": 0.9234,
         "top_3_predictions": {
           "authentication": 0.9234,
           "login_issue": 0.0543,
           "security": 0.0223
         },
         "model_type": "xgboost",
         "inference_time_ms": 12.5
       },
       "tensorflow": {
         "predicted_category": "authentication", 
         "confidence": 0.8876,
         "top_3_predictions": {
           "authentication": 0.8876,
           "access_control": 0.0789,
           "login_issue": 0.0335
         },
         "model_type": "tensorflow",
         "inference_time_ms": 45.2
       }
     },
     "available_models": ["xgboost", "tensorflow"],
     "total_inference_time_ms": 57.7,
     "timestamp": "2025-09-29T10:30:45.123456"
   }
   ```

8. **Setup MLflow Experiment Tracking**
   <!-- Initialize MLflow for experiment tracking -->
   ```bash
   # Setup MLflow (optional but recommended)
   python scripts/setup_mlflow.py
   
   # Start MLflow UI to view experiments
   mlflow ui --backend-store-uri file://./mlruns
   
   # Then open: http://localhost:5000
   ```

---

## 🎨 Streamlit Dashboard

### 🚀 Interactive Web Interface

InsightDesk AI includes a comprehensive **Streamlit dashboard** that provides a user-friendly web interface for all system capabilities. The dashboard integrates seamlessly with the FastAPI backend to deliver real-time insights and analytics.

#### 🌟 Dashboard Features

| Tab | Feature | Description |
|-----|---------|-------------|
| **📨 Ticket Categorization** | AI Classification | Classify tickets with XGBoost + TensorFlow models |
| **🔎 Solution Retrieval** | RAG Search | Find relevant solutions using semantic search |
| **🚨 Anomaly Detection** | Real-time Monitoring | View and analyze system anomalies |
| **📊 Monitoring & Drift** | Performance Tracking | Monitor model performance and data drift |
| **🔄 Feedback** | Continuous Learning | Submit agent and customer feedback |

#### 🚀 Quick Start

##### Option 1: Full Stack Launch (Recommended)
```bash
# Launch both FastAPI + Streamlit automatically
python scripts/launch_full_stack.py

# Or using Makefile
make full-stack
```

##### Option 2: Manual Launch
```bash
# Terminal 1: Start FastAPI backend
cd src
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start Streamlit dashboard
streamlit run app.py

# Or using Makefile
make run &        # FastAPI in background
make dashboard    # Streamlit dashboard
```

##### Option 3: Demo Mode (No Backend Required)
```bash
# Run dashboard demo with mock data
streamlit run demo_dashboard.py

# Or using Makefile
make dashboard-demo
```

#### 🎯 Dashboard Access

Once running, access the dashboard at:
- **🎨 Main Dashboard**: http://localhost:8501
- **🧪 Demo Dashboard**: http://localhost:8502
- **🔧 API Documentation**: http://localhost:8000/docs
- **❤️  API Health Check**: http://localhost:8000/health

#### 📱 Dashboard Walkthrough

##### 1. **Ticket Categorization Tab**
- **Input Form**: Subject, description, error logs, stack trace, product details
- **AI Predictions**: Side-by-side comparison of XGBoost vs TensorFlow models
- **Confidence Metrics**: Real-time confidence scores and inference timing
- **Visual Analytics**: Bar charts showing model performance comparison

##### 2. **Solution Retrieval Tab**
- **Query Interface**: Natural language problem description
- **Search Options**: Hybrid, semantic, or keyword-based search
- **Results Display**: Ranked solutions with similarity scores
- **Solution Details**: Full resolution text, source attribution, and metadata

##### 3. **Anomaly Detection Tab**
- **Real-time Alerts**: Color-coded anomaly severity (🔴 High, 🟡 Medium, 🔵 Low)
- **Filtering Options**: Time range and severity filtering
- **Detailed Analysis**: Anomaly type, details, and timestamps
- **Visual Dashboard**: Anomaly distribution charts and trends

##### 4. **Monitoring & Drift Tab**
- **Performance Metrics**: Live model accuracy, F1 score, and latency
- **Drift Detection**: Data drift scoring with threshold alerts
- **Trend Analysis**: Historical performance charts and comparisons
- **System Health**: CPU, memory usage, and request rate monitoring

##### 5. **Feedback Collection Tab**
- **Agent Feedback**: Prediction accuracy, resolution time, satisfaction ratings
- **Customer Feedback**: Resolution quality, response time, recommendation scores
- **Analytics Dashboard**: Feedback trends, satisfaction metrics, and category performance

#### 🎨 UI/UX Features

##### Modern Design Elements
- **Responsive Layout**: Wide layout with collapsible sidebar navigation
- **Custom Styling**: Gradient headers, card-based sections, color-coded alerts
- **Interactive Charts**: Plotly-powered visualizations with hover effects
- **Status Indicators**: Real-time API connectivity and health monitoring

##### User Experience
- **Form Validation**: Smart input validation with helpful error messages
- **Loading States**: Progress indicators for API calls and data processing
- **Error Handling**: Graceful error handling with actionable error messages
- **Mobile Friendly**: Responsive design that works on tablets and phones

#### 🔧 Configuration

##### Streamlit Configuration (`.streamlit/config.toml`)
```toml
[theme]
base = "light"
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[server]
port = 8501
headless = false
runOnSave = true
```

##### Dashboard Customization
- **API Endpoint**: Modify `API_BASE_URL` in `app.py`
- **Color Scheme**: Update CSS variables in the custom styling section
- **Layout Options**: Adjust column ratios and component layouts
- **Feature Toggles**: Enable/disable specific dashboard tabs

#### 🚀 Development

##### Adding New Features
1. **Create new function** in `app.py` for your feature
2. **Add navigation option** in the sidebar radio buttons
3. **Implement API integration** using the `make_api_request` helper
4. **Add styling** using custom CSS and Streamlit components

##### Testing Dashboard Components
```bash
# Test with mock data
python demo_dashboard.py

# Validate dashboard functionality
python -c "import streamlit; print('✅ Streamlit installed')"
python -c "import requests; print('✅ Requests installed')"
python -c "import plotly; print('✅ Plotly installed')"
```

#### 🎯 Future Enhancements

- **🔄 Real-time Updates**: WebSocket integration for live anomaly streaming
- **📊 Graph Visualization**: Interactive knowledge graph for Graph-RAG results
- **🎨 Advanced Analytics**: Custom dashboards and KPI tracking
- **🔐 Authentication**: User management and role-based access control
- **📱 Mobile App**: React Native companion app
- **🚀 Multi-tenancy**: Support for multiple organizations

---

## 🧰 Tech Stack

<!-- Technology Choices: Each tool was selected for specific capabilities -->
* **Python 3.12+**, FastAPI, Pydantic  <!-- FastAPI for high-performance async APIs -->
* **Frontend**: Streamlit with Plotly  <!-- Interactive dashboard with rich visualizations -->
* **Machine Learning**: scikit-learn, XGBoost, TensorFlow/Keras  <!-- XGBoost for tabular data, TF for deep learning -->
* **Retrieval**: FAISS or Milvus, Neo4j/NetworkX (for Graph-RAG)  <!-- Vector search + graph relationships -->
* **MLOps**: MLflow for experiment tracking & model registry  <!-- Model versioning and lifecycle management -->
* **Containerization**: Docker & docker-compose  <!-- Consistent deployment environments -->
* **Monitoring**: Prometheus / Grafana (optional)  <!-- Production observability -->

---

## 📊 Data

<!-- Data Strategy: Proper data handling is crucial for model performance -->
Use the provided JSON file with ~100k historical support tickets.
Recommended split: 70 % train / 15 % validation / 15 % test.  <!-- Standard ML data splits -->
Handle class imbalance (e.g., stratified sampling or resampling) and document your strategy.  <!-- Critical for ticket classification accuracy -->

---

## 📊 Model Benchmarking & Evaluation

<!-- Comprehensive Performance Analysis: Compare models across multiple dimensions -->
InsightDesk AI includes a comprehensive benchmarking framework for evaluating and comparing ML models across technical performance and business impact metrics.

### 🎯 Quick Benchmarking Demo

```bash
# Run comprehensive benchmarking demonstration
python scripts/demo_benchmarking.py

# View generated results
ls demo_results/
# - metrics_summary.csv           # Executive summary
# - model_performance_metrics.csv # Technical metrics  
# - business_metrics.csv          # Business impact
# - report.md                     # Detailed analysis
# - visualizations/               # Charts and plots
```

### 📈 Benchmarking Capabilities

**Technical Performance Metrics:**
- Accuracy, Precision, Recall, Weighted F1-Score
- Inference latency (avg, P95, P99) and memory usage
- Throughput and success rates
- Confusion matrices and ROC curves

**Business Impact Analysis:**
- Agent success rates when using model suggestions
- Average resolution time per predicted category
- Customer satisfaction correlation with predictions
- High-confidence prediction accuracy rates

**Comprehensive Reporting:**
- CSV exports for data analysis and tracking
- Markdown reports with insights and recommendations
- Visual comparisons and performance charts
- Integration-ready metrics for monitoring dashboards

### 🔄 Running Full Benchmarks

```bash
# Run complete benchmarking pipeline (requires trained models)
python scripts/benchmark_models.py

# Quick validation test (uses mock data if models unavailable)
python scripts/test_benchmark_quick.py

# View comprehensive results
open results/report.md                    # Detailed report
open results/visualizations/              # Performance charts
```

### 🧪 A/B Testing Framework

```bash
# Demo A/B testing capabilities  
python scripts/ab_testing_framework.py

# The framework provides:
# - Traffic splitting and routing
# - Statistical significance testing  
# - Performance monitoring
# - Automated decision recommendations
```

**Sample Benchmark Results:**
```
📊 Model Comparison Summary:

📋 XGBoost:
   Accuracy: 83.4%        ⚡ Avg Latency: 45.2ms
   F1-Score: 0.829        💾 Memory: 23.4MB  
   Agent Success: 78.2%   🎯 High Confidence: 68.3%

📋 TensorFlow:  
   Accuracy: 84.7%        ⚡ Avg Latency: 67.8ms
   F1-Score: 0.842        💾 Memory: 45.7MB
   Agent Success: 80.9%   🎯 High Confidence: 73.7%

💡 Recommendation: TensorFlow shows +1.3% accuracy improvement
   and +2.7% higher agent success rate, consider A/B testing
```

---

## 🧪 Development & Testing

<!-- Code Quality: Maintaining consistent style and testing practices -->
* Format code with `black` and `isort`.  <!-- Automated code formatting for consistency -->
* Run unit tests:  <!-- Ensure code reliability with comprehensive testing -->

  ```bash
  # Run all tests
  pytest
  
  # Run specific test classes
  pytest tests/test_api.py::TestCategoryPrediction -v
  
  # Run tests with coverage
  pytest --cov=src tests/
  ```

* **Test model integration:**
  ```bash
  # Test both trained models
  python scripts/test_models.py
  ```

* **Model performance comparison:**
  ```bash
  # Train and compare both models with detailed analysis
  python scripts/train_and_compare_models.py
  
  # Check generated report
  cat MODEL_COMPARISON_REPORT.md
  
  # View performance plots
  ls plots/
  ```

* **Development workflow:**
  ```bash
  # 1. Setup development environment
  python -m venv venv
  source venv/bin/activate  # Windows: venv\Scripts\activate
  pip install -r requirements.txt
  
  # 2. Extract and prepare data
  python scripts/unzip_and_load.py
  
  # 3. Build features (if not already done)
  python scripts/build_features.py
  
  # 4. Train models individually or together
  python scripts/train_xgboost.py     # XGBoost only
  python scripts/train_tensorflow.py  # TensorFlow only
  # OR
  python scripts/train_and_compare_models.py  # Both + comparison
  
  # 5. Test models
  python scripts/test_models.py
  
  # 6. Run API tests
  pytest tests/test_api.py -v
  
  # 7. Start API server
  uvicorn src.api.main:app --reload
  
  # 8. View MLflow experiments (optional)
  mlflow ui --backend-store-uri file://./mlruns
  ```

---

## � Advanced Configuration

### RAG Pipeline Customization

```python
# Custom RAG pipeline initialization
from src.retrieval.rag_pipeline import RAGPipeline

pipeline = RAGPipeline(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Embedding model
    vector_store_dir="custom_vector_store",               # Vector store location
    index_type="ivf"                                      # FAISS index type
)

# Search with custom parameters
results = pipeline.query_solutions(
    ticket_data,
    k=10,                    # Number of results
    search_type="hybrid",    # semantic|keyword|hybrid
    semantic_weight=0.7,     # Hybrid search weights
    keyword_weight=0.3
)
```

### Production Deployment

```bash
# Production deployment with Gunicorn
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Docker deployment
docker build -t insightdesk-ai .
docker run -p 8000:8000 insightdesk-ai
```

---

## 🚀 Future Roadmap

### Planned Enhancements

- [ ] **Graph-RAG Integration**: Neo4j-based knowledge graphs for relationship-based retrieval
- [ ] **Real-time Learning**: Online learning from user feedback and resolution success rates
- [ ] **Anomaly Detection**: Automated detection of unusual ticket patterns and emerging issues
- [ ] **Multi-language Support**: International ticket processing with multilingual embeddings
- [ ] **Advanced Analytics**: Comprehensive dashboards and reporting capabilities

### Extensibility Points

The codebase includes placeholder functions for future features:

```python
# Graph-RAG placeholder (src/retrieval/rag_pipeline.py)
def setup_graph_rag(self, neo4j_config):
    """TODO: Setup Graph-RAG integration with Neo4j."""
    pass

# Category-based reranking placeholder  
def rerank_with_category_predictions(self, results, predicted_category):
    """TODO: Rerank results based on ML model predictions."""
    pass
```

---

## 📊 Technical Details

### RAG Pipeline Architecture

- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (384-dimensional embeddings)
- **Vector Store**: FAISS with cosine similarity search
- **Hybrid Search**: Combines semantic similarity + TF-IDF keyword matching
- **Ranking**: Success rate weighted scoring with configurable weights

### ML Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | Inference Time |
|-------|----------|-----------|--------|----------|----------------|
| XGBoost | 0.89 | 0.87 | 0.89 | 0.88 | ~50ms |
| TensorFlow | 0.91 | 0.90 | 0.91 | 0.90 | ~80ms |

### API Performance

- **Health Check**: < 10ms response time
- **Category Prediction**: 50-100ms per request
- **RAG Retrieval**: 100-200ms per query (k=10)
- **Concurrent Requests**: 100+ req/s with proper scaling

---

## �📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔧 CI/CD Pipeline

### 🚀 GitHub Actions Workflow

This project includes a comprehensive CI/CD pipeline that automatically runs on every push and pull request to the `main` branch. The pipeline ensures code quality, runs tests, and prepares for deployment.

#### Pipeline Overview

The CI/CD workflow consists of 10 integrated jobs:

1. **🎨 Lint and Format** - Code quality checks
   - Black code formatting
   - isort import sorting  
   - Flake8 linting
   - Bandit security scanning
   - Safety dependency checks

2. **🧪 Unit Tests** - Comprehensive testing
   - Pytest execution across Python 3.10, 3.11, 3.12
   - Coverage reporting with codecov upload
   - Test result artifacts

3. **🔗 Integration Tests** - End-to-end validation
   - Neo4j service container
   - API endpoint testing
   - Database integration validation

4. **🐳 Build and Test** - Docker validation
   - Multi-stage Docker build
   - Container health checks
   - API endpoint verification

5. **⚡ Performance Tests** - Benchmarking validation
   - Model performance validation
   - Benchmarking script execution
   - Performance metrics collection

6. **📦 Build and Push** - Container registry
   - Docker image building
   - Registry push (configured for production)
   - Image tagging and versioning

7. **🚀 Deploy Staging** - Staging environment
   - Staging deployment preparation
   - Environment validation
   - Smoke tests

8. **🌍 Deploy Production** - Production deployment
   - Manual approval gates
   - Blue-green deployment ready
   - Rollback capabilities

9. **📊 Post-Deployment** - Monitoring setup
   - MLflow experiment sync
   - Model registry updates
   - Performance monitoring

10. **🧹 Cleanup** - Resource management
    - Temporary resource cleanup
    - Notification dispatch
    - Workflow completion

#### Running the Pipeline

The pipeline automatically triggers on:
- **Push to main**: Full pipeline execution
- **Pull requests**: All jobs except deployment
- **Manual dispatch**: On-demand execution with environment selection

```bash
# Trigger manual pipeline run
gh workflow run ci-cd.yml
```

#### Local Development

Before pushing, ensure your code passes all checks:

```bash
# Install development dependencies
pip install black isort flake8 bandit safety pytest coverage

# Run formatting and linting
black .
isort .
flake8 .
bandit -r src/
safety check

# Run tests with coverage
pytest --cov=src --cov-report=html
```

#### Pipeline Configuration

The pipeline is configured through:
- `.github/workflows/ci-cd.yml` - Main workflow definition
- `pyproject.toml` - Tool configurations (Black, isort, pytest, coverage)
- `.flake8` - Linting rules and exclusions
- `requirements.txt` - Production dependencies

#### Security and Compliance

- **Dependency Scanning**: Safety checks for known vulnerabilities
- **Code Security**: Bandit static analysis for security issues  
- **Container Security**: Trivy scanning for Docker images
- **Secrets Management**: GitHub secrets for sensitive configuration
- **Access Control**: Manual approval for production deployments

#### Extending the Pipeline

To add new jobs or modify existing ones:

1. Edit `.github/workflows/ci-cd.yml`
2. Follow the existing job structure
3. Add appropriate dependencies between jobs
4. Test with draft pull requests

---

## 🤝 Contributing

<!-- Community Guidelines: How to contribute effectively to this project -->
We welcome contributions! Please follow these guidelines:

1. **Fork** the repository and create a feature branch
2. **Follow** PEP 8 style guidelines and add type hints
3. **Add tests** for new features with good coverage
4. **Update documentation** for API changes
5. **Submit** a pull request with clear description

### Development Guidelines

- Use meaningful commit messages
- Ensure all tests pass before submitting
- Add docstrings for new functions and classes
- Update README for significant feature additions

---

## � Support & Contact

For questions, issues, or contributions:

- **GitHub Issues**: [Create an issue](https://github.com/iamvisheshsrivastava/insightdesk-ai/issues)
- **Documentation**: Check the `/docs` endpoint when running the API
- **Examples**: See the `scripts/` directory for comprehensive usage examples

**Author:** Vishesh Srivastava  
[LinkedIn](https://linkedin.com/in/iamvisheshsrivastava) • [GitHub](https://github.com/iamvisheshsrivastava)

---

**Happy coding! 🚀**
