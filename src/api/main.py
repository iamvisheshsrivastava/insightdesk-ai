from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union, Any
import logging
import time
import json
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import MLflow
try:
    import mlflow
    MLFLOW_AVAILABLE = True
    logger.info("✅ MLflow available for experiment tracking")
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("⚠️ MLflow not available")

# Try to import RAG pipeline
try:
    from src.retrieval.rag_pipeline import RAGPipeline
    RAG_AVAILABLE = True
    logger.info("✅ RAG pipeline available")
except ImportError:
    RAG_AVAILABLE = False
    logger.warning("⚠️ RAG pipeline not available")

# Try to import Anomaly Detection
try:
    from src.anomaly.anomaly_detector import AnomalyDetector
    from src.anomaly.anomaly_models import AnomalyThresholds, AnomalyType
    ANOMALY_AVAILABLE = True
    logger.info("✅ Anomaly detection available")
except ImportError:
    ANOMALY_AVAILABLE = False
    logger.warning("⚠️ Anomaly detection not available")

# Try to import Monitoring & Drift Detection
try:
    from src.monitoring.performance_monitor import ModelPerformanceMonitor, PerformanceMetrics
    from src.monitoring.drift_detector import DataDriftDetector, DriftResult
    from src.monitoring.metrics_logger import MetricsLogger, NotificationConfig
    from src.monitoring.alert_manager import (
        AlertManager, AlertRule, AlertSeverity, AlertType,
        create_default_performance_rules, create_default_drift_rules
    )
    MONITORING_AVAILABLE = True
    logger.info("✅ Monitoring & drift detection available")
except ImportError:
    MONITORING_AVAILABLE = False
    logger.warning("⚠️ Monitoring & drift detection not available")

# Try to import Feedback Loop system
try:
    from src.feedback.feedback_manager import FeedbackManager
    from src.feedback.feedback_models import (
        AgentCorrectionRequest, CustomerFeedbackRequest, FeedbackStatsResponse,
        FeedbackType, FeedbackSeverity, PredictionQuality
    )
    FEEDBACK_AVAILABLE = True
    logger.info("✅ Feedback loop system available")
except ImportError:
    FEEDBACK_AVAILABLE = False
    logger.warning("⚠️ Feedback loop system not available")

# Try to import Graph-RAG system
try:
    from src.retrieval.graph_manager import Neo4jGraphManager
    from src.retrieval.hybrid_rag_pipeline import HybridRAGPipeline
    GRAPH_RAG_AVAILABLE = True
    logger.info("✅ Graph-RAG system available")
except ImportError:
    GRAPH_RAG_AVAILABLE = False
    logger.warning("⚠️ Graph-RAG system not available")

# Try to import model prediction functions
try:
    from src.models.xgboost_classifier import predict_category_xgboost, predict_categories_xgboost_batch
    from src.models.tensorflow_classifier import predict_category_tensorflow, predict_categories_tensorflow_batch
    MODEL_FUNCTIONS_AVAILABLE = True
    logger.info("✅ Model prediction functions available")
except ImportError:
    MODEL_FUNCTIONS_AVAILABLE = False
    logger.warning("⚠️ Model prediction functions not available")

# Try to import Agentic Orchestrator
try:
    from src.agentic.orchestrator import AgentOrchestrator
    AGENTIC_AVAILABLE = True
    logger.info("✅ Agentic Orchestrator available")
except ImportError:
    AGENTIC_AVAILABLE = False
    logger.warning("⚠️ Agentic Orchestrator not available")



class ModelManager:
    """Dependency injection container for ML models and RAG pipeline."""
    
    def __init__(self):
        self.xgb_classifier = None
        self.tf_classifier = None
        self.rag_pipeline = None
        self.anomaly_detector = None
        
        # Monitoring components
        self.performance_monitor = None
        self.drift_detector = None
        self.metrics_logger = None
        self.alert_manager = None
        
        # Feedback system
        self.feedback_manager = None
        
        # Graph-RAG system
        self.graph_manager = None
        self.hybrid_rag_pipeline = None
        
        # Status flags
        self.models_loaded = False
        self.rag_initialized = False
        self.anomaly_initialized = False
        self.monitoring_initialized = False
        self.feedback_initialized = False
        self.graph_rag_initialized = False
        
    def load_models(self):
        """Load both ML models and initialize RAG pipeline."""
        try:
            from src.models.xgboost_classifier import XGBoostCategoryClassifier
            from src.models.tensorflow_classifier import TensorFlowCategoryClassifier
            
            # Load XGBoost model
            try:
                self.xgb_classifier = XGBoostCategoryClassifier()
                self.xgb_classifier.load_model()
                logger.info("✅ XGBoost model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load XGBoost model: {e}")
                self.xgb_classifier = None
            
            # Load TensorFlow model
            try:
                self.tf_classifier = TensorFlowCategoryClassifier()
                self.tf_classifier.load_model()
                logger.info("✅ TensorFlow model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load TensorFlow model: {e}")
                self.tf_classifier = None
            
            # Initialize RAG pipeline
            if RAG_AVAILABLE:
                try:
                    self.rag_pipeline = RAGPipeline(vector_store_dir="vector_store")
                    self.rag_pipeline.initialize()
                    self.rag_initialized = True
                    logger.info("✅ RAG pipeline initialized successfully")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize RAG pipeline: {e}")
                    self.rag_pipeline = None
                    self.rag_initialized = False
            
            # Initialize Anomaly Detection
            if ANOMALY_AVAILABLE:
                try:
                    self.anomaly_detector = AnomalyDetector()
                    self.anomaly_initialized = True
                    logger.info("✅ Anomaly detector initialized successfully")
                    
                    # Load demo anomalies if available
                    try:
                        demo_loaded = self.anomaly_detector.load_demo_anomalies()
                        if demo_loaded:
                            logger.info("✅ Demo anomalies loaded successfully")
                        else:
                            logger.info("ℹ️ No demo anomalies found - run 'python scripts/demo_anomaly.py' to generate them")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load demo anomalies: {e}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize anomaly detector: {e}")
                    self.anomaly_detector = None
                    self.anomaly_initialized = False
            
            # Initialize Monitoring System
            if MONITORING_AVAILABLE:
                try:
                    # Initialize performance monitor
                    self.performance_monitor = ModelPerformanceMonitor(
                        model_name="support_system",
                        target_accuracy=0.85
                    )
                    
                    # Initialize drift detector
                    self.drift_detector = DataDriftDetector(
                        drift_threshold=0.3,
                        min_sample_size=100
                    )
                    
                    # Initialize metrics logger
                    self.metrics_logger = MetricsLogger(
                        log_dir="logs/monitoring",
                        mlflow_experiment_name="support_system_monitoring"
                    )
                    
                    # Initialize alert manager with default rules
                    self.alert_manager = AlertManager()
                    
                    # Add default alert rules
                    for rule in create_default_performance_rules():
                        self.alert_manager.add_alert_rule(rule)
                    
                    for rule in create_default_drift_rules():
                        self.alert_manager.add_alert_rule(rule)
                    
                    self.monitoring_initialized = True
                    logger.info("✅ Monitoring system initialized successfully")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize monitoring system: {e}")
                    self.performance_monitor = None
                    self.drift_detector = None
                    self.metrics_logger = None
                    self.alert_manager = None
                    self.monitoring_initialized = False
            
            # Initialize Feedback Loop System
            if FEEDBACK_AVAILABLE:
                try:
                    # Initialize feedback manager with JSON storage by default
                    self.feedback_manager = FeedbackManager(
                        storage_type="json",
                        storage_config={"storage_dir": "feedback_data"}
                    )
                    
                    self.feedback_initialized = True
                    logger.info("✅ Feedback loop system initialized successfully")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize feedback system: {e}")
                    self.feedback_manager = None
                    self.feedback_initialized = False
            
            # Initialize Graph-RAG System
            if GRAPH_RAG_AVAILABLE:
                try:
                    # Initialize Neo4j graph manager (use environment variables or default config)
                    neo4j_config = {
                        "uri": "bolt://localhost:7687",
                        "user": "neo4j",
                        "password": "password",  # Should come from environment variable
                        "database": "neo4j"
                    }
                    
                    self.graph_manager = Neo4jGraphManager(**neo4j_config)
                    
                    # Initialize hybrid RAG pipeline (requires both traditional RAG and graph)
                    if self.rag_pipeline:
                        self.hybrid_rag_pipeline = HybridRAGPipeline(
                            rag_pipeline=self.rag_pipeline,
                            graph_manager=self.graph_manager
                        )
                        self.graph_rag_initialized = True
                        logger.info("✅ Graph-RAG system initialized successfully")
                    else:
                        logger.warning("⚠️ Traditional RAG not available, skipping hybrid RAG initialization")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize Graph-RAG system: {e}")
                    self.graph_manager = None
                    self.hybrid_rag_pipeline = None
                    self.graph_rag_initialized = False
                
            self.models_loaded = True
            logger.info("📦 Model loading complete")
            
        except Exception as e:
            logger.error(f"❌ Critical error loading models: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        available = []
        if self.xgb_classifier and self.xgb_classifier.is_loaded:
            available.append("xgboost")
        if self.tf_classifier and self.tf_classifier.is_loaded:
            available.append("tensorflow")
        return available
    
    def predict_category(self, ticket_data: Dict, model_type: str = "both") -> Dict:
        """Predict category using specified model(s)."""
        start_time = time.time()
        results = {}
        
        try:
            if model_type in ["xgboost", "both"] and self.xgb_classifier and self.xgb_classifier.is_loaded:
                xgb_start = time.time()
                xgb_result = self.xgb_classifier.predict(ticket_data)
                xgb_time = time.time() - xgb_start
                xgb_result["inference_time_ms"] = round(xgb_time * 1000, 2)
                results["xgboost"] = xgb_result
                
            if model_type in ["tensorflow", "both"] and self.tf_classifier and self.tf_classifier.is_loaded:
                tf_start = time.time()
                tf_result = self.tf_classifier.predict(ticket_data)
                tf_time = time.time() - tf_start
                tf_result["inference_time_ms"] = round(tf_time * 1000, 2)
                results["tensorflow"] = tf_result
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        total_time = time.time() - start_time
        results["total_inference_time_ms"] = round(total_time * 1000, 2)
        results["timestamp"] = datetime.now().isoformat()
        
        return results


def generate_mock_classification(ticket):
    """Generate a mock classification based on ticket content for demo purposes."""
    import random
    
    # Define categories and keywords
    categories = {
        "Technical Issue": ["error", "timeout", "connection", "server", "database", "crash", "bug", "failure"],
        "Feature Request": ["add", "feature", "improve", "enhancement", "request", "suggestion", "new"],
        "Bug Report": ["bug", "issue", "problem", "broken", "not working", "error", "incorrect"],
        "Account Issue": ["login", "password", "account", "access", "permission", "authentication"],
        "Billing": ["payment", "billing", "invoice", "charge", "subscription", "cost", "price"]
    }
    
    # Combine subject and description for analysis
    text_content = f"{ticket.subject or ''} {ticket.description or ''} {ticket.error_logs or ''}".lower()
    
    # Score each category based on keyword matches
    scores = {}
    for category, keywords in categories.items():
        score = sum(1 for keyword in keywords if keyword in text_content)
        scores[category] = score
    
    # If no keywords match, use priority/severity to guess
    if all(score == 0 for score in scores.values()):
        if ticket.priority in ["High", "Critical"] or ticket.severity in ["3 - High", "4 - Critical"]:
            return "Technical Issue"
        else:
            return "Feature Request"
    
    # Return category with highest score, or random if tied
    max_score = max(scores.values())
    best_categories = [cat for cat, score in scores.items() if score == max_score]
    
    return random.choice(best_categories)


# Global model manager instance
model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """Dependency injection for model manager."""
    return model_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("🚀 Starting InsightDesk AI API...")
    
    # Setup MLflow
    if MLFLOW_AVAILABLE:
        try:
            mlflow_dir = Path("mlruns")
            mlflow_dir.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"file://{mlflow_dir.absolute()}")
            logger.info("📊 MLflow tracking initialized")
        except Exception as e:
            logger.warning(f"⚠️ MLflow setup failed: {e}")
    
    # Load models
    try:
        model_manager.load_models()
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
    
    logger.info("✅ API startup complete!")
    yield
    logger.info("🛑 Shutting down API...")


app = FastAPI(
    title="InsightDesk AI",
    description="Intelligent support platform with ML-powered ticket classification and priority prediction",
    version="1.0.0",
    lifespan=lifespan
)


# Pydantic Models
class TicketInput(BaseModel):
    """Input schema for ticket data."""
    ticket_id: Optional[str] = Field(None, description="Unique ticket identifier", min_length=1)
    subject: str = Field(..., description="Ticket subject/title", min_length=1)
    description: str = Field(..., description="Detailed ticket description", min_length=1)
    
    # Optional fields with defaults
    error_logs: Optional[str] = Field("", description="Error logs if available")
    stack_trace: Optional[str] = Field("", description="Stack trace if available")
    product: Optional[str] = Field("unknown", description="Product name")
    product_module: Optional[str] = Field("unknown", description="Product module")
    channel: Optional[str] = Field("email", description="Support channel (email, chat, phone)")
    priority: Optional[str] = Field("medium", description="Ticket priority (low, medium, high, critical)")
    severity: Optional[str] = Field("minor", description="Ticket severity (minor, major, critical)")
    customer_tier: Optional[str] = Field("standard", description="Customer tier (standard, premium, enterprise)")
    region: Optional[str] = Field("US", description="Customer region")
    
    # Numeric fields
    previous_tickets: Optional[int] = Field(0, ge=0, description="Number of previous tickets")
    account_age_days: Optional[int] = Field(0, ge=0, description="Customer account age in days")
    account_monthly_value: Optional[float] = Field(0.0, ge=0, description="Monthly account value")
    ticket_text_length: Optional[int] = Field(0, ge=0, description="Total text length")
    response_count: Optional[int] = Field(0, ge=0, description="Number of responses")
    attachments_count: Optional[int] = Field(0, ge=0, description="Number of attachments")
    affected_users: Optional[int] = Field(1, ge=1, description="Number of affected users")
    resolution_time_hours: Optional[float] = Field(0.0, ge=0, description="Resolution time in hours")
    
    @validator('priority')
    def validate_priority(cls, v):
        valid_priorities = ['low', 'medium', 'high', 'critical']
        if v.lower() not in valid_priorities:
            return 'medium'  # Default fallback
        return v.lower()
    
    @validator('severity')
    def validate_severity(cls, v):
        valid_severities = ['minor', 'major', 'critical']
        if v.lower() not in valid_severities:
            return 'minor'  # Default fallback
        return v.lower()
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticket_id": "TK-2025-001",
                "subject": "Cannot login to application",
                "description": "User is unable to authenticate with correct credentials. Getting timeout error.",
                "error_logs": "Authentication timeout after 30 seconds",
                "product": "web_application",
                "channel": "email",
                "priority": "high",
                "customer_tier": "premium",
                "previous_tickets": 2,
                "account_age_days": 365
            }
        }


class CategoryPredictionResponse(BaseModel):
    """Response schema for category prediction."""
    ticket_id: str
    predictions: Dict[str, Any]
    available_models: List[str]
    total_inference_time_ms: float
    timestamp: str


class PriorityPredictionResponse(BaseModel):
    """Response schema for priority prediction (future implementation)."""
    ticket_id: str
    predicted_priority: str
    confidence: float
    message: str = "Priority prediction not yet implemented"


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    models: Dict[str, bool]
    rag_available: bool
    timestamp: str
    version: str


# RAG-related models
class SolutionResult(BaseModel):
    """Model for a single solution result."""
    resolution_id: Optional[str] = None
    category: Optional[str] = None
    product: Optional[str] = None
    resolution: str
    resolution_steps: Optional[str] = None
    similarity_score: float
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    success_rate: Optional[float] = None
    usage_count: Optional[int] = None
    content_type: Optional[str] = None
    search_type: str


class RetrievalRequest(BaseModel):
    """Request model for solution retrieval."""
    subject: str = Field(..., min_length=1, max_length=200, description="Ticket subject")
    description: str = Field(..., min_length=1, max_length=2000, description="Ticket description")
    error_logs: Optional[str] = Field(None, max_length=5000, description="Error logs or stack trace")
    product: Optional[str] = Field(None, description="Product name")
    category: Optional[str] = Field(None, description="Ticket category")
    stack_trace: Optional[str] = Field(None, max_length=10000, description="Stack trace information")
    k: int = Field(10, ge=1, le=50, description="Number of solutions to return")
    search_type: str = Field("hybrid", pattern="^(semantic|keyword|hybrid)$", description="Type of search to perform")
    
    class Config:
        json_schema_extra = {
            "example": {
                "subject": "Cannot login to application",
                "description": "User is unable to authenticate with correct credentials. Getting timeout error.",
                "error_logs": "Authentication timeout after 30 seconds",
                "product": "web_application",
                "category": "authentication",
                "k": 5,
                "search_type": "hybrid"
            }
        }


class RetrievalResponse(BaseModel):
    """Response model for solution retrieval."""
    query_summary: str
    solutions: List[SolutionResult]
    total_found: int
    search_type: str
    processing_time: float
    status: str
    version: str
    models_loaded: bool
    available_models: List[str]
    mlflow_available: bool
    timestamp: str


# Anomaly Detection models
class AnomalyDetectionRequest(BaseModel):
    """Request model for anomaly detection."""
    tickets_data: List[Dict[str, Any]] = Field(..., description="List of ticket data for anomaly analysis")
    detection_types: Optional[List[str]] = Field(None, description="Specific anomaly types to detect")
    days_lookback: int = Field(30, ge=1, le=365, description="Number of days to look back for analysis")
    
    class Config:
        json_schema_extra = {
            "example": {
                "tickets_data": [
                    {
                        "ticket_id": "T12345",
                        "category": "authentication",
                        "subject": "Login issues",
                        "description": "Cannot login to application",
                        "created_at": "2024-01-15 10:30:00",
                        "customer_sentiment": "frustrated",
                        "priority": "high",
                        "product": "web_application"
                    }
                ],
                "detection_types": ["volume_spike", "sentiment_shift"],
                "days_lookback": 30
            }
        }


class AnomalyDetectionResponse(BaseModel):
    """Response model for anomaly detection."""
    total_anomalies: int
    anomalies: List[Dict[str, Any]]
    severity_breakdown: Dict[str, int]
    type_breakdown: Dict[str, int]
    processing_time: float
    tickets_analyzed: int
    detection_period: Dict[str, datetime]
    timestamp: str
    status: str


# Graph-RAG models
class GraphRetrievalRequest(BaseModel):
    """Request model for graph-enhanced retrieval."""
    subject: str = Field(..., min_length=1, max_length=200, description="Ticket subject")
    description: str = Field(..., min_length=1, max_length=2000, description="Ticket description")
    error_logs: Optional[str] = Field(None, max_length=5000, description="Error logs or stack trace")
    product: Optional[str] = Field(None, description="Product name")
    category: Optional[str] = Field(None, description="Ticket category")
    stack_trace: Optional[str] = Field(None, max_length=10000, description="Stack trace information")
    k: int = Field(10, ge=1, le=50, description="Number of solutions to return")
    semantic_weight: float = Field(0.6, ge=0.0, le=1.0, description="Weight for semantic similarity")
    graph_weight: float = Field(0.4, ge=0.0, le=1.0, description="Weight for graph relevance")
    use_graph_expansion: bool = Field(True, description="Whether to expand search using graph relationships")
    max_depth: int = Field(2, ge=1, le=3, description="Maximum graph traversal depth")
    
    @validator('semantic_weight', 'graph_weight')
    def weights_sum_to_one(cls, v, values):
        """Ensure semantic_weight and graph_weight sum to approximately 1.0."""
        if 'semantic_weight' in values:
            semantic_weight = values['semantic_weight']
            graph_weight = v if 'graph_weight' not in values else values['graph_weight']
            if abs((semantic_weight + graph_weight) - 1.0) > 0.01:
                raise ValueError('semantic_weight and graph_weight must sum to 1.0')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "subject": "Cannot login to application",
                "description": "User is unable to authenticate with correct credentials. Getting timeout error.",
                "error_logs": "Authentication timeout after 30 seconds",
                "product": "web_application",
                "category": "authentication",
                "k": 5,
                "semantic_weight": 0.6,
                "graph_weight": 0.4,
                "use_graph_expansion": True,
                "max_depth": 2
            }
        }


class GraphSolutionResult(SolutionResult):
    """Enhanced solution result with graph information."""
    graph_relevance_score: Optional[float] = Field(None, description="Graph-based relevance score")
    relationship_path: Optional[List[str]] = Field(None, description="Graph relationship path to solution")
    connected_issues: Optional[List[str]] = Field(None, description="Related issues found through graph")
    kb_references: Optional[List[str]] = Field(None, description="Knowledge base articles referenced")
    hybrid_score: Optional[float] = Field(None, description="Combined semantic + graph score")


class GraphRetrievalResponse(BaseModel):
    """Response model for graph-enhanced retrieval."""
    query_summary: str
    solutions: List[GraphSolutionResult]
    total_found: int
    semantic_results: int
    graph_results: int
    hybrid_ranking_applied: bool
    search_weights: Dict[str, float]
    processing_time: float
    graph_stats: Optional[Dict[str, Any]]
    status: str
    version: str
    timestamp: str


class GraphStatsResponse(BaseModel):
    """Response model for graph statistics."""
    nodes: Dict[str, int]
    relationships: Dict[str, int]
    graph_health: Dict[str, Any]
    last_updated: Optional[str]
    connection_status: str
    timestamp: str


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "InsightDesk AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(manager: ModelManager = Depends(get_model_manager)):
    """Comprehensive health check endpoint."""
    return HealthResponse(
        status="healthy",
        models={
            "xgboost": manager.xgb_classifier is not None and getattr(manager.xgb_classifier, 'is_loaded', False),
            "tensorflow": manager.tf_classifier is not None and getattr(manager.tf_classifier, 'is_loaded', False)
        },
        rag_available=manager.rag_initialized and manager.rag_pipeline is not None,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.post("/predict/category", response_model=CategoryPredictionResponse)
async def predict_category(
    ticket: TicketInput,
    model_type: str = "both",
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Predict ticket category using trained ML models.
    
    - **ticket**: Ticket data for classification
    - **model_type**: Which model to use ("xgboost", "tensorflow", or "both")
    """
    # Validate model_type parameter
    valid_models = ["xgboost", "tensorflow", "both"]
    if model_type not in valid_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model_type. Must be one of: {valid_models}"
        )
    
    # Check if any models are available
    available_models = manager.get_available_models()
    if not available_models:
        # Provide mock response when models aren't available
        logger.warning("No ML models available, returning mock classification response")
        
        # Generate a mock classification based on keywords in the ticket
        mock_category = generate_mock_classification(ticket)
        
        return CategoryPredictionResponse(
            ticket_id=ticket.ticket_id or f"demo_{int(time.time())}",
            category=mock_category,
            confidence=0.75,  # Mock confidence
            probabilities={
                mock_category: 0.75,
                "Technical Issue": 0.15,
                "Bug Report": 0.10
            },
            predictions={
                "mock_model": {
                    "category": mock_category,
                    "confidence": 0.75,
                    "model_type": "demo"
                }
            },
            model_performance={
                "inference_time_ms": 50,
                "models_used": ["demo_classifier"],
                "total_models_available": 0
            }
        )
    
    # Log prediction request
    logger.info(f"Category prediction request: ticket_id={ticket.ticket_id}, model_type={model_type}")
    
    try:
        # Convert ticket to dict and ensure ticket_id is set
        ticket_dict = ticket.dict()
        if not ticket_dict.get("ticket_id"):
            ticket_dict["ticket_id"] = f"auto_{int(time.time())}"
        
        # Get predictions
        predictions = manager.predict_category(ticket_dict, model_type)
        
        # Log successful prediction
        logger.info(f"Category prediction successful: ticket_id={ticket_dict['ticket_id']}")
        
        return CategoryPredictionResponse(
            ticket_id=ticket_dict["ticket_id"],
            predictions=predictions,
            available_models=available_models,
            total_inference_time_ms=predictions["total_inference_time_ms"],
            timestamp=predictions["timestamp"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Generate ticket_id for error logging if not available
        error_ticket_id = ticket.ticket_id or f"error_{int(time.time())}"
        logger.error(f"Category prediction failed: ticket_id={error_ticket_id}, error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/priority", response_model=PriorityPredictionResponse)
async def predict_priority(ticket: TicketInput):
    """
    Predict ticket priority (future implementation).
    
    - **ticket**: Ticket data for priority prediction
    """
    logger.info(f"Priority prediction request: ticket_id={ticket.ticket_id}")
    
    # TODO: Implement priority prediction model
    return PriorityPredictionResponse(
        ticket_id=ticket.ticket_id,
        predicted_priority="medium",
        confidence=0.0,
        message="Priority prediction model not yet implemented. This is a placeholder endpoint."
    )


def generate_demo_solutions(subject: str, description: str, k: int, search_type: str) -> List[Dict]:
    """
    Generate demo solutions when RAG pipeline is not available.
    Creates realistic solutions based on the subject and description.
    """
    import random
    
    # Common solution patterns based on keywords
    solution_templates = {
        "login": [
            "Clear browser cache and cookies, then try logging in again.",
            "Reset your password using the 'Forgot Password' link and try again.",
            "Check if Caps Lock is enabled and ensure correct case sensitivity.",
            "Try logging in from an incognito/private browser window.",
            "Contact support to verify your account status and permissions."
        ],
        "password": [
            "Use the 'Forgot Password' link to reset your password securely.",
            "Ensure your new password meets all security requirements (8+ chars, symbols, etc.).",
            "Clear browser cache and try the password reset process again.",
            "Check your email spam folder for the password reset link.",
            "Wait 10-15 minutes after password reset before attempting to login."
        ],
        "payment": [
            "Verify your credit card information and billing address are correct.",
            "Check if your card has sufficient funds and is not expired.",
            "Try using a different payment method (different card or PayPal).",
            "Contact your bank to ensure they're not blocking the transaction.",
            "Clear browser cookies and retry the payment process."
        ],
        "error": [
            "Check the error logs for specific error codes and messages.",
            "Restart the application and try the operation again.",
            "Verify all required fields are filled out correctly.",
            "Update your browser to the latest version and retry.",
            "Contact technical support with the exact error message."
        ],
        "api": [
            "Verify your API key is correct and has not expired.",
            "Check the API endpoint URL and ensure it's properly formatted.",
            "Review API rate limits and ensure you're not exceeding them.",
            "Test the API call with a REST client like Postman first.",
            "Check API documentation for required headers and parameters."
        ]
    }
    
    # Determine relevant solution category based on keywords
    text = f"{subject} {description}".lower()
    relevant_solutions = []
    
    for keyword, solutions in solution_templates.items():
        if keyword in text:
            relevant_solutions.extend(solutions)
    
    # If no specific matches, use general solutions
    if not relevant_solutions:
        relevant_solutions = [
            "Check our knowledge base for similar issues and solutions.",
            "Restart the application and clear your browser cache.",
            "Verify your internet connection is stable and try again.",
            "Update your browser to the latest version.",
            "Contact our support team with detailed error information."
        ]
    
    # Generate solutions with scores
    solutions = []
    for i in range(min(k, len(relevant_solutions))):
        solution = relevant_solutions[i] if i < len(relevant_solutions) else f"Additional solution #{i+1} for your specific issue."
        
        solutions.append({
            "resolution_id": f"DEMO-{random.randint(1000, 9999)}",
            "category": "Technical Support",
            "product": "Demo System",
            "resolution": solution,
            "resolution_steps": f"Step-by-step guide for: {solution[:50]}...",
            "similarity_score": round(random.uniform(0.7, 0.95), 3),
            "semantic_score": round(random.uniform(0.6, 0.9), 3),
            "keyword_score": round(random.uniform(0.5, 0.8), 3),
            "success_rate": round(random.uniform(0.8, 0.95), 2),
            "usage_count": random.randint(10, 500),
            "content_type": "solution",
            "search_type": search_type
        })
    
    # Sort by similarity_score (highest first)
    solutions.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    return solutions


@app.post("/retrieve/solutions", response_model=RetrievalResponse)
async def retrieve_solutions(
    request: RetrievalRequest,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Retrieve relevant solutions for a support ticket using RAG.
    
    This endpoint uses Retrieval-Augmented Generation (RAG) to find
    the most relevant solutions from the knowledge base based on
    semantic similarity and keyword matching.
    
    - **request**: Ticket information and search parameters
    """
    start_time = time.time()
    
    try:
        # Check if RAG pipeline is available
        if not manager.rag_initialized or not manager.rag_pipeline:
            # Fallback: Generate demo solutions based on the request
            logger.warning("RAG pipeline not available. Generating demo solutions...")
            
            # Create demo solutions based on the subject and description
            demo_solutions = generate_demo_solutions(
                subject=request.subject,
                description=request.description,
                k=request.k,
                search_type=request.search_type
            )
            
            return RetrievalResponse(
                query_summary=f"Demo search for: {request.subject}",
                solutions=demo_solutions,
                total_found=len(demo_solutions),
                search_type=request.search_type,
                processing_time=time.time() - start_time,
                status="success",
                version="1.0.0-demo",
                models_loaded=False,
                available_models=["demo"],
                mlflow_available=False,
                timestamp=datetime.now().isoformat()
            )
        
        # The following code runs only if RAG is available
        
        # Prepare ticket data for querying
        ticket_data = {
            "subject": request.subject,
            "description": request.description,
            "error_logs": request.error_logs or "",
            "stack_trace": request.stack_trace or "",
            "product": request.product or "unknown",
            "category": request.category
        }
        
        # Query solutions using RAG pipeline
        solutions = manager.rag_pipeline.query_solutions(
            ticket_data,
            k=request.k,
            search_type=request.search_type
        )
        
        # Convert to response format
        solution_results = []
        for solution in solutions:
            solution_result = SolutionResult(
                resolution_id=solution.get("resolution_id"),
                category=solution.get("category"),
                product=solution.get("product"),
                resolution=solution.get("resolution", "No resolution available"),
                resolution_steps=solution.get("resolution_steps"),
                similarity_score=solution.get("similarity_score", 0.0),
                semantic_score=solution.get("semantic_score"),
                keyword_score=solution.get("keyword_score"),
                success_rate=solution.get("success_rate"),
                usage_count=solution.get("usage_count"),
                content_type=solution.get("type"),
                search_type=solution.get("search_type", request.search_type)
            )
            solution_results.append(solution_result)
        
        processing_time = time.time() - start_time
        
        # Create query summary
        query_summary = request.subject
        if len(query_summary) > 50:
            query_summary = query_summary[:50] + "..."
        
        logger.info(
            f"Solution retrieval completed: found {len(solutions)} solutions "
            f"for query '{query_summary}' (time: {processing_time:.3f}s)"
        )
        
        return RetrievalResponse(
            query_summary=query_summary,
            solutions=solution_results,
            total_found=len(solutions),
            search_type=request.search_type,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Solution retrieval failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Solution retrieval failed: {str(e)}"
        )


@app.get("/models/info")
async def get_models_info(manager: ModelManager = Depends(get_model_manager)):
    """Get detailed information about loaded models."""
    info = {
        "models_loaded": manager.models_loaded,
        "available_models": manager.get_available_models(),
        "model_details": {}
    }
    
    if manager.xgb_classifier and manager.xgb_classifier.is_loaded:
        info["model_details"]["xgboost"] = manager.xgb_classifier.get_model_info()
    
    if manager.tf_classifier and manager.tf_classifier.is_loaded:
        info["model_details"]["tensorflow"] = manager.tf_classifier.get_model_info()
    
    return info


@app.post("/anomalies/detect", response_model=AnomalyDetectionResponse)
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Detect anomalies in support ticket data.
    
    This endpoint analyzes ticket data to identify:
    - Volume spikes in specific categories
    - Sentiment shifts from historical baselines
    - New/emerging issue patterns
    - Statistical outliers in ticket characteristics
    
    - **request**: Ticket data and detection parameters
    """
    start_time = time.time()
    
    try:
        # Check if anomaly detection is available
        if not manager.anomaly_initialized or not manager.anomaly_detector:
            raise HTTPException(
                status_code=503,
                detail="Anomaly detection not available. Please check server logs."
            )
        
        # Validate and convert detection types
        detection_types = None
        if request.detection_types:
            valid_types = ["volume_spike", "sentiment_shift", "new_issue", "outlier"]
            invalid_types = [t for t in request.detection_types if t not in valid_types]
            if invalid_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid detection types: {invalid_types}. Valid types: {valid_types}"
                )
            
            # Convert strings to AnomalyType enums
            type_mapping = {
                "volume_spike": AnomalyType.VOLUME_SPIKE,
                "sentiment_shift": AnomalyType.SENTIMENT_SHIFT,
                "new_issue": AnomalyType.NEW_ISSUE,
                "outlier": AnomalyType.OUTLIER
            }
            detection_types = [type_mapping[t] for t in request.detection_types]
        
        logger.info(
            f"Anomaly detection request: {len(request.tickets_data)} tickets, "
            f"types: {request.detection_types or 'all'}"
        )
        
        # Run anomaly detection
        result = manager.anomaly_detector.detect_all_anomalies(
            request.tickets_data,
            detection_types=detection_types
        )
        
        # Convert anomalies to serializable format
        anomalies_data = []
        for anomaly in result.anomalies:
            anomaly_dict = anomaly.dict()
            # Convert datetime objects to ISO strings
            anomaly_dict["timestamp"] = anomaly.timestamp.isoformat()
            anomalies_data.append(anomaly_dict)
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Anomaly detection completed: {result.total_anomalies} anomalies found "
            f"in {processing_time:.3f}s"
        )
        
        return AnomalyDetectionResponse(
            total_anomalies=result.total_anomalies,
            anomalies=anomalies_data,
            severity_breakdown=result.severity_breakdown,
            type_breakdown=result.type_breakdown,
            processing_time=processing_time,
            tickets_analyzed=result.tickets_analyzed,
            detection_period={
                "start": result.detection_period["start"],
                "end": result.detection_period["end"]
            },
            timestamp=datetime.now().isoformat(),
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Anomaly detection failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Anomaly detection failed: {str(e)}"
        )


@app.get("/anomalies/recent")
async def get_recent_anomalies(
    days: int = 7,
    severity: Optional[str] = None,
    anomaly_type: Optional[str] = None,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Get recent anomalies from detection history.
    
    - **days**: Number of days to look back (default: 7)
    - **severity**: Filter by severity (low, medium, high, critical)
    - **anomaly_type**: Filter by type (volume_spike, sentiment_shift, new_issue, outlier)
    """
    try:
        # Check if anomaly detection is available
        if not manager.anomaly_initialized or not manager.anomaly_detector:
            raise HTTPException(
                status_code=503,
                detail="Anomaly detection not available. Please check server logs."
            )
        
        # Get recent anomalies
        recent_anomalies = manager.anomaly_detector.get_recent_anomalies(days=days)
        
        # Apply filters
        filtered_anomalies = recent_anomalies
        
        if severity:
            if severity not in ["low", "medium", "high", "critical"]:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid severity. Must be one of: low, medium, high, critical"
                )
            filtered_anomalies = [a for a in filtered_anomalies if a.severity.value == severity]
        
        if anomaly_type:
            if anomaly_type not in ["volume_spike", "sentiment_shift", "new_issue", "outlier"]:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid anomaly_type. Must be one of: volume_spike, sentiment_shift, new_issue, outlier"
                )
            filtered_anomalies = [a for a in filtered_anomalies if a.type.value == anomaly_type]
        
        # Convert to serializable format
        anomalies_data = []
        for anomaly in filtered_anomalies:
            anomaly_dict = anomaly.dict()
            anomaly_dict["timestamp"] = anomaly.timestamp.isoformat()
            anomalies_data.append(anomaly_dict)
        
        # Calculate summary statistics
        severity_counts = {}
        type_counts = {}
        for anomaly in filtered_anomalies:
            severity_counts[anomaly.severity.value] = severity_counts.get(anomaly.severity.value, 0) + 1
            type_counts[anomaly.type.value] = type_counts.get(anomaly.type.value, 0) + 1
        
        logger.info(f"Retrieved {len(filtered_anomalies)} recent anomalies (filtered from {len(recent_anomalies)})")
        
        return {
            "total_anomalies": len(filtered_anomalies),
            "anomalies": anomalies_data,
            "severity_breakdown": severity_counts,
            "type_breakdown": type_counts,
            "lookback_days": days,
            "filters_applied": {
                "severity": severity,
                "anomaly_type": anomaly_type
            },
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve recent anomalies: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve recent anomalies: {str(e)}"
        )


# MLflow integration endpoints
@app.get("/experiments")
async def list_experiments():
    """List MLflow experiments."""
    if not MLFLOW_AVAILABLE:
        raise HTTPException(status_code=503, detail="MLflow not available")
    
    try:
        experiments = mlflow.search_experiments()
        return {
            "experiments": [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "lifecycle_stage": exp.lifecycle_stage
                }
                for exp in experiments
            ]
        }
    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list experiments: {str(e)}")


@app.get("/experiments/{experiment_id}/runs")
async def list_runs(experiment_id: str):
    """List runs for a specific experiment."""
    if not MLFLOW_AVAILABLE:
        raise HTTPException(status_code=503, detail="MLflow not available")
    
    try:
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        return {
            "runs": runs.to_dict(orient="records") if not runs.empty else []
        }
    except Exception as e:
        logger.error(f"Failed to list runs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list runs: {str(e)}")


# ================================
# MONITORING & DRIFT DETECTION ENDPOINTS
# ================================

class MonitoringStatusResponse(BaseModel):
    """Response model for monitoring status."""
    overall_status: str
    timestamp: str
    models: Dict[str, Any]
    alerts: Dict[str, Any]
    system_health: Dict[str, Any]


class DriftAnalysisRequest(BaseModel):
    """Request model for drift analysis."""
    tickets: List[TicketInput]
    model_name: str = "support_system"
    categorical_columns: List[str] = ["product", "priority", "category", "channel"]
    text_columns: List[str] = ["subject", "description"]


class AlertResponse(BaseModel):
    """Response model for alerts."""
    alerts: List[Dict[str, Any]]
    summary: Dict[str, Any]


@app.get("/monitoring/status", response_model=MonitoringStatusResponse)
def get_monitoring_status(model_manager: ModelManager = Depends(get_model_manager)):
    """Get comprehensive monitoring status."""
    try:
        # Handle case where monitoring is not initialized gracefully
        if not model_manager.monitoring_initialized:
            # Return mock/default status instead of raising an error
            return MonitoringStatusResponse(
                overall_status="monitoring_unavailable",
                timestamp=datetime.now().isoformat(),
                models={
                    "performance": {"status": "unavailable", "message": "Monitoring system not initialized"},
                    "drift": {"status": "unavailable", "message": "Drift detection not available"}
                },
                alerts={"total_active_alerts": 0, "message": "Alert system not available"},
                system_health={
                    "cpu_usage": 0,
                    "memory_usage": 0,
                    "disk_usage": 0,
                    "message": "System monitoring not available"
                }
            )
        
        # Get current system metrics
        try:
            import psutil
            system_metrics = {
                "cpu_usage": psutil.cpu_percent(interval=0.1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent if psutil.disk_usage('/') else 0
            }
        except ImportError:
            system_metrics = {
                "cpu_usage": 0,
                "memory_usage": 0,
                "disk_usage": 0
            }
        
        # Get model performance status
        models_status = {}
        if model_manager.performance_monitor:
            try:
                models_status = model_manager.performance_monitor.get_current_status()
            except Exception as e:
                logger.warning(f"Failed to get performance status: {e}")
                models_status = {"error": str(e)}
        
        # Get drift status
        drift_status = {}
        if model_manager.drift_detector:
            try:
                drift_status = model_manager.drift_detector.get_all_drift_status()
            except Exception as e:
                logger.warning(f"Failed to get drift status: {e}")
                drift_status = {"error": str(e)}
        
        # Get alerts summary
        alerts_summary = {}
        if model_manager.alert_manager:
            try:
                alerts_summary = model_manager.alert_manager.get_alert_summary()
            except Exception as e:
                logger.warning(f"Failed to get alerts summary: {e}")
                alerts_summary = {"error": str(e)}
        
        # Determine overall status
        overall_status = "healthy"
        if alerts_summary.get("total_active_alerts", 0) > 0:
            critical_alerts = alerts_summary.get("by_severity", {}).get("critical", 0)
            high_alerts = alerts_summary.get("by_severity", {}).get("high", 0)
            
            if critical_alerts > 0:
                overall_status = "critical"
            elif high_alerts > 0:
                overall_status = "warning"
            else:
                overall_status = "attention"
        
        return MonitoringStatusResponse(
            overall_status=overall_status,
            timestamp=datetime.now().isoformat(),
            models={
                "performance": models_status,
                "drift": drift_status
            },
            alerts=alerts_summary,
            system_health=system_metrics
        )
        
    except Exception as e:
        logger.error(f"Error getting monitoring status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring status: {str(e)}")


@app.post("/monitoring/drift/analyze")
def analyze_drift(
    request: DriftAnalysisRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Analyze data drift in the provided tickets."""
    try:
        if not model_manager.monitoring_initialized:
            raise HTTPException(
                status_code=503, 
                detail="Monitoring system not initialized"
            )
        
        if not model_manager.drift_detector:
            raise HTTPException(
                status_code=503, 
                detail="Drift detector not available"
            )
        
        # Convert tickets to DataFrame
        import pandas as pd
        
        tickets_data = []
        for ticket in request.tickets:
            ticket_dict = ticket.dict()
            tickets_data.append(ticket_dict)
        
        df = pd.DataFrame(tickets_data)
        
        # Perform drift detection
        drift_result = model_manager.drift_detector.detect_drift(
            current_data=df,
            categorical_columns=request.categorical_columns,
            text_columns=request.text_columns,
            model_name=request.model_name
        )
        
        # Log drift results
        if model_manager.metrics_logger:
            model_manager.metrics_logger.log_drift_metrics(
                request.model_name, drift_result
            )
        
        # Check for drift alerts
        if model_manager.alert_manager:
            model_manager.alert_manager.check_drift_alerts(
                request.model_name, drift_result
            )
        
        return {
            "drift_analysis": drift_result.to_dict(),
            "analyzed_tickets": len(request.tickets),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing drift: {e}")
        raise HTTPException(status_code=500, detail=f"Drift analysis failed: {str(e)}")


@app.get("/monitoring/alerts", response_model=AlertResponse)
def get_alerts(
    model_name: Optional[str] = None,
    severity: Optional[str] = None,
    hours: int = 24,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get active alerts and alert history."""
    try:
        if not model_manager.monitoring_initialized:
            raise HTTPException(
                status_code=503, 
                detail="Monitoring system not initialized"
            )
        
        if not model_manager.alert_manager:
            raise HTTPException(
                status_code=503, 
                detail="Alert manager not available"
            )
        
        # Convert severity string to enum if provided
        severity_enum = None
        if severity:
            try:
                from src.monitoring.alert_manager import AlertSeverity
                severity_enum = AlertSeverity(severity.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid severity: {severity}. Must be one of: low, medium, high, critical"
                )
        
        # Get alerts
        active_alerts = model_manager.alert_manager.get_active_alerts(
            model_name=model_name,
            severity=severity_enum
        )
        
        alert_history = model_manager.alert_manager.get_alert_history(
            hours=hours,
            model_name=model_name
        )
        
        summary = model_manager.alert_manager.get_alert_summary()
        
        return AlertResponse(
            alerts={
                "active": [alert.to_dict() for alert in active_alerts],
                "recent": [alert.to_dict() for alert in alert_history]
            },
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")


@app.post("/monitoring/alerts/{alert_id}/acknowledge")
def acknowledge_alert(
    alert_id: str,
    note: str = "",
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Acknowledge an alert."""
    try:
        if not model_manager.monitoring_initialized:
            raise HTTPException(
                status_code=503, 
                detail="Monitoring system not initialized"
            )
        
        if not model_manager.alert_manager:
            raise HTTPException(
                status_code=503, 
                detail="Alert manager not available"
            )
        
        success = model_manager.alert_manager.acknowledge_alert(alert_id, note)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Alert not found: {alert_id}")
        
        return {
            "status": "acknowledged",
            "alert_id": alert_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge alert: {str(e)}")


@app.post("/monitoring/alerts/{alert_id}/resolve")
def resolve_alert(
    alert_id: str,
    note: str = "",
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Resolve an alert."""
    try:
        if not model_manager.monitoring_initialized:
            raise HTTPException(
                status_code=503, 
                detail="Monitoring system not initialized"
            )
        
        if not model_manager.alert_manager:
            raise HTTPException(
                status_code=503, 
                detail="Alert manager not available"
            )
        
        success = model_manager.alert_manager.resolve_alert(alert_id, note)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Alert not found: {alert_id}")
        
        return {
            "status": "resolved",
            "alert_id": alert_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error resolving alert: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resolve alert: {str(e)}")


@app.get("/monitoring/performance/{model_name}")
def get_performance_history(
    model_name: str,
    hours: int = 24,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get performance metrics history for a model."""
    try:
        if not model_manager.monitoring_initialized:
            raise HTTPException(
                status_code=503, 
                detail="Monitoring system not initialized"
            )
        
        if not model_manager.metrics_logger:
            raise HTTPException(
                status_code=503, 
                detail="Metrics logger not available"
            )
        
        history = model_manager.metrics_logger.get_performance_history(
            model_name, hours
        )
        
        return {
            "model_name": model_name,
            "history": history,
            "time_range_hours": hours,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting performance history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance history: {str(e)}")


@app.get("/monitoring/drift/{model_name}")
def get_drift_history(
    model_name: str,
    hours: int = 24,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get drift detection history for a model."""
    try:
        if not model_manager.monitoring_initialized:
            raise HTTPException(
                status_code=503, 
                detail="Monitoring system not initialized"
            )
        
        if not model_manager.metrics_logger:
            raise HTTPException(
                status_code=503, 
                detail="Metrics logger not available"
            )
        
        history = model_manager.metrics_logger.get_drift_history(
            model_name, hours
        )
        
        return {
            "model_name": model_name,
            "history": history,
            "time_range_hours": hours,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting drift history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get drift history: {str(e)}")


@app.post("/monitoring/metrics/export")
def export_metrics(
    output_path: str,
    model_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Export metrics to file."""
    try:
        if not model_manager.monitoring_initialized:
            raise HTTPException(
                status_code=503, 
                detail="Monitoring system not initialized"
            )
        
        if not model_manager.metrics_logger:
            raise HTTPException(
                status_code=503, 
                detail="Metrics logger not available"
            )
        
        # Parse dates if provided
        start_dt = None
        end_dt = None
        
        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format")
        
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format")
        
        # Export metrics
        success = model_manager.metrics_logger.export_metrics(
            output_path, model_name, start_dt, end_dt
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Export failed")
        
        return {
            "status": "exported",
            "output_path": output_path,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error exporting metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export metrics: {str(e)}")


# =============================================================================
# FEEDBACK LOOP ENDPOINTS
# =============================================================================

@app.post("/feedback/correction")
async def record_agent_correction(
    request: AgentCorrectionRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Record an agent correction for model improvement."""
    if not FEEDBACK_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Feedback loop system not available"
        )
    
    if not model_manager.feedback_initialized:
        raise HTTPException(
            status_code=503, 
            detail="Feedback system not initialized"
        )
    
    try:
        correction_id = model_manager.feedback_manager.record_agent_correction(
            ticket_id=request.ticket_id,
            agent_id=request.agent_id,
            original_prediction=request.original_prediction,
            original_confidence=request.original_confidence,
            model_type=request.model_type,
            corrected_label=request.corrected_label,
            correction_reason=request.correction_reason,
            correction_notes=request.correction_notes,
            prediction_quality=PredictionQuality(request.prediction_quality),
            severity=FeedbackSeverity(request.severity),
            ticket_data=request.ticket_data,
            should_retrain=request.should_retrain,
            correction_confidence=request.correction_confidence
        )
        
        if not correction_id:
            raise HTTPException(status_code=500, detail="Failed to record correction")
        
        return {
            "status": "recorded",
            "correction_id": correction_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error recording agent correction: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record correction: {str(e)}")


@app.post("/feedback/customer")
async def record_customer_feedback(
    request: CustomerFeedbackRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Record customer feedback for quality assessment."""
    if not FEEDBACK_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Feedback loop system not available"
        )
    
    if not model_manager.feedback_initialized:
        raise HTTPException(
            status_code=503, 
            detail="Feedback system not initialized"
        )
    
    try:
        feedback_id = model_manager.feedback_manager.record_customer_feedback(
            ticket_id=request.ticket_id,
            rating=request.rating,
            customer_id=request.customer_id,
            comments=request.comments,
            feedback_type=request.feedback_type,
            resolution_helpful=request.resolution_helpful,
            resolution_accurate=request.resolution_accurate,
            would_recommend=request.would_recommend,
            ai_suggestions_used=request.ai_suggestions_used,
            ai_suggestions_helpful=request.ai_suggestions_helpful,
            ai_accuracy_rating=request.ai_accuracy_rating,
            resolution_time_hours=request.resolution_time_hours,
            agent_id=request.agent_id,
            needs_followup=request.needs_followup
        )
        
        if not feedback_id:
            raise HTTPException(status_code=500, detail="Failed to record feedback")
        
        return {
            "status": "recorded",
            "feedback_id": feedback_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error recording customer feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")


@app.get("/feedback/summary", response_model=FeedbackStatsResponse)
async def get_feedback_summary(
    days_back: int = 30,
    model_type: Optional[str] = None,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get feedback summary and analytics."""
    if not FEEDBACK_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Feedback loop system not available"
        )
    
    if not model_manager.feedback_initialized:
        raise HTTPException(
            status_code=503, 
            detail="Feedback system not initialized"
        )
    
    try:
        summary = model_manager.feedback_manager.generate_feedback_summary(
            days_back=days_back,
            model_type=model_type
        )
        
        return FeedbackStatsResponse(
            summary_id=summary.summary_id,
            period_start=summary.period_start,
            period_end=summary.period_end,
            total_corrections=summary.total_corrections,
            corrections_by_model=summary.corrections_by_model,
            corrections_by_severity=summary.corrections_by_severity,
            avg_original_confidence=summary.avg_original_confidence,
            total_customer_feedback=summary.total_customer_feedback,
            avg_customer_rating=summary.avg_customer_rating,
            satisfaction_breakdown=summary.satisfaction_breakdown,
            ai_usage_rate=summary.ai_usage_rate,
            events_by_type=summary.events_by_type,
            events_by_severity=summary.events_by_severity,
            quality_metrics=summary.quality_metrics,
            retraining_recommendations=summary.retraining_recommendations
        )
        
    except Exception as e:
        logger.error(f"Error generating feedback summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")


@app.get("/feedback/corrections")
async def get_agent_corrections(
    days_back: int = 30,
    ticket_id: Optional[str] = None,
    model_type: Optional[str] = None,
    agent_id: Optional[str] = None,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get agent corrections with filters."""
    if not FEEDBACK_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Feedback loop system not available"
        )
    
    if not model_manager.feedback_initialized:
        raise HTTPException(
            status_code=503, 
            detail="Feedback system not initialized"
        )
    
    try:
        corrections = model_manager.feedback_manager.get_corrections(
            days_back=days_back,
            ticket_id=ticket_id,
            model_type=model_type,
            agent_id=agent_id
        )
        
        return {
            "total": len(corrections),
            "corrections": [correction.to_dict() for correction in corrections],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving corrections: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve corrections: {str(e)}")


@app.get("/feedback/customer")
async def get_customer_feedback(
    days_back: int = 30,
    ticket_id: Optional[str] = None,
    rating_min: Optional[int] = None,
    customer_id: Optional[str] = None,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get customer feedback with filters."""
    if not FEEDBACK_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Feedback loop system not available"
        )
    
    if not model_manager.feedback_initialized:
        raise HTTPException(
            status_code=503, 
            detail="Feedback system not initialized"
        )
    
    try:
        feedback_list = model_manager.feedback_manager.get_customer_feedback(
            days_back=days_back,
            ticket_id=ticket_id,
            rating_min=rating_min,
            customer_id=customer_id
        )
        
        return {
            "total": len(feedback_list),
            "feedback": [feedback.to_dict() for feedback in feedback_list],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving customer feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve feedback: {str(e)}")


@app.get("/feedback/trends/{model_type}")
async def get_model_performance_trends(
    model_type: str,
    days_back: int = 30,
    granularity: str = "daily",
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get performance trends for a specific model."""
    if not FEEDBACK_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Feedback loop system not available"
        )
    
    if not model_manager.feedback_initialized:
        raise HTTPException(
            status_code=503, 
            detail="Feedback system not initialized"
        )
    
    if granularity not in ["hourly", "daily", "weekly"]:
        raise HTTPException(
            status_code=400, 
            detail="Invalid granularity. Must be: hourly, daily, or weekly"
        )
    
    try:
        trends = model_manager.feedback_manager.get_model_performance_trends(
            model_type=model_type,
            days_back=days_back,
            granularity=granularity
        )
        
        return {
            "model_type": model_type,
            "granularity": granularity,
            "trends": trends,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving performance trends: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve trends: {str(e)}")


@app.get("/feedback/agent/{agent_id}")
async def get_agent_performance_analysis(
    agent_id: str,
    days_back: int = 30,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get performance analysis for a specific agent."""
    if not FEEDBACK_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Feedback loop system not available"
        )
    
    if not model_manager.feedback_initialized:
        raise HTTPException(
            status_code=503, 
            detail="Feedback system not initialized"
        )
    
    try:
        analysis = model_manager.feedback_manager.get_agent_performance_analysis(
            agent_id=agent_id,
            days_back=days_back
        )
        
        return {
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing agent performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze performance: {str(e)}")


@app.get("/feedback/health")
async def get_feedback_health(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get feedback system health status."""
    if not FEEDBACK_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Feedback loop system not available"
        )
    
    if not model_manager.feedback_initialized:
        return {
            "status": "not_initialized",
            "feedback_available": FEEDBACK_AVAILABLE,
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        health = model_manager.feedback_manager.health_check()
        return health
        
    except Exception as e:
        logger.error(f"Error checking feedback health: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Graph-RAG Endpoints
@app.post("/retrieve/graph", response_model=GraphRetrievalResponse)
async def retrieve_graph_solutions(
    request: GraphRetrievalRequest,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Retrieve relevant solutions using Graph-RAG (hybrid semantic + graph search).
    
    This endpoint combines semantic similarity with knowledge graph relationships
    to find the most relevant solutions. It provides enhanced context through
    graph traversal and relationship discovery.
    
    - **request**: Enhanced ticket information with graph search parameters
    """
    start_time = time.time()
    
    try:
        # Check if Graph-RAG system is available
        if not GRAPH_RAG_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Graph-RAG system not available. Please check if Neo4j is running."
            )
        
        if not manager.graph_rag_initialized or not manager.hybrid_rag_pipeline:
            raise HTTPException(
                status_code=503,
                detail="Graph-RAG pipeline not initialized. Please check server logs."
            )
        
        # Prepare ticket data for querying
        ticket_data = {
            "subject": request.subject,
            "description": request.description,
            "error_logs": request.error_logs or "",
            "stack_trace": request.stack_trace or "",
            "product": request.product or "unknown",
            "category": request.category
        }
        
        # Configure search parameters
        search_params = {
            "k": request.k,
            "semantic_weight": request.semantic_weight,
            "graph_weight": request.graph_weight,
            "use_graph_expansion": request.use_graph_expansion,
            "max_depth": request.max_depth
        }
        
        # Query solutions using hybrid RAG pipeline
        result = manager.hybrid_rag_pipeline.hybrid_search(
            ticket_data,
            **search_params
        )
        
        # Convert solutions to GraphSolutionResult format
        graph_solutions = []
        for solution in result["solutions"]:
            graph_solution = GraphSolutionResult(
                resolution_id=solution.get("resolution_id"),
                category=solution.get("category"),
                product=solution.get("product"),
                resolution=solution.get("resolution", "No resolution available"),
                resolution_steps=solution.get("resolution_steps"),
                similarity_score=solution.get("similarity_score", 0.0),
                semantic_score=solution.get("semantic_score"),
                keyword_score=solution.get("keyword_score"),
                success_rate=solution.get("success_rate"),
                usage_count=solution.get("usage_count"),
                content_type=solution.get("type"),
                search_type=solution.get("search_type", "hybrid"),
                # Graph-specific fields
                graph_relevance_score=solution.get("graph_relevance_score"),
                relationship_path=solution.get("relationship_path"),
                connected_issues=solution.get("connected_issues"),
                kb_references=solution.get("kb_references"),
                hybrid_score=solution.get("hybrid_score")
            )
            graph_solutions.append(graph_solution)
        
        processing_time = time.time() - start_time
        
        # Create query summary
        query_summary = request.subject
        if len(query_summary) > 50:
            query_summary = query_summary[:50] + "..."
        
        logger.info(
            f"Graph-RAG retrieval completed: found {len(graph_solutions)} solutions "
            f"for query '{query_summary}' (time: {processing_time:.3f}s)"
        )
        
        return GraphRetrievalResponse(
            query_summary=query_summary,
            solutions=graph_solutions,
            total_found=len(graph_solutions),
            semantic_results=result.get("semantic_count", 0),
            graph_results=result.get("graph_count", 0),
            hybrid_ranking_applied=result.get("hybrid_ranking_applied", True),
            search_weights={
                "semantic": request.semantic_weight,
                "graph": request.graph_weight
            },
            processing_time=processing_time,
            graph_stats=result.get("graph_stats"),
            status="success",
            version="1.0.0",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Graph-RAG retrieval error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Graph-RAG retrieval failed: {str(e)}"
        )


@app.get("/retrieve/graph/stats", response_model=GraphStatsResponse)
async def get_graph_stats(
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Get knowledge graph statistics and health information.
    
    Returns information about nodes, relationships, and overall
    graph health for monitoring purposes.
    """
    try:
        # Check if Graph-RAG system is available
        if not GRAPH_RAG_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Graph-RAG system not available. Please check if Neo4j is running."
            )
        
        if not manager.graph_rag_initialized or not manager.graph_manager:
            raise HTTPException(
                status_code=503,
                detail="Graph manager not initialized. Please check server logs."
            )
        
        # Get graph statistics
        stats = manager.graph_manager.get_graph_stats()
        health = manager.graph_manager.health_check()
        
        return GraphStatsResponse(
            nodes=stats.get("nodes", {}),
            relationships=stats.get("relationships", {}),
            graph_health=health,
            last_updated=stats.get("last_updated"),
            connection_status="connected" if health.get("status") == "healthy" else "disconnected",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error retrieving graph stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve graph statistics: {str(e)}"
        )


@app.post("/retrieve/graph/query")
async def query_graph_directly(
    query: str,
    parameters: Optional[Dict[str, Any]] = None,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Execute a direct Cypher query against the knowledge graph.
    
    This endpoint allows advanced users to run custom Cypher queries
    for debugging and advanced analytics purposes.
    
    **Warning**: This endpoint exposes direct database access.
    Use with caution in production environments.
    """
    try:
        # Check if Graph-RAG system is available
        if not GRAPH_RAG_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Graph-RAG system not available. Please check if Neo4j is running."
            )
        
        if not manager.graph_rag_initialized or not manager.graph_manager:
            raise HTTPException(
                status_code=503,
                detail="Graph manager not initialized. Please check server logs."
            )
        
        # Validate query safety (basic check)
        unsafe_keywords = ["DELETE", "DROP", "CREATE INDEX", "REMOVE"]
        query_upper = query.upper()
        for keyword in unsafe_keywords:
            if keyword in query_upper:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsafe query operation '{keyword}' not allowed"
                )
        
        # Execute query
        start_time = time.time()
        results = manager.graph_manager.execute_query(query, parameters or {})
        execution_time = time.time() - start_time
        
        # Convert results to serializable format
        serializable_results = []
        for record in results:
            record_dict = {}
            for key, value in record.items():
                # Handle Neo4j node/relationship objects
                if hasattr(value, '_properties'):
                    record_dict[key] = dict(value._properties)
                else:
                    record_dict[key] = value
            serializable_results.append(record_dict)
        
        logger.info(f"Direct graph query executed: {len(serializable_results)} results (time: {execution_time:.3f}s)")
        
        return {
            "query": query,
            "parameters": parameters,
            "results": serializable_results,
            "result_count": len(serializable_results),
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error executing graph query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute graph query: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


class TicketBatchInput(BaseModel):
    """Input schema for batch ticket classification."""
    tickets: List[TicketInput] = Field(..., description="List of tickets to classify")


class CategoryPrediction(BaseModel):
    """Output schema for category prediction."""
    ticket_id: str
    predicted_category: str
    confidence: float
    top_3_predictions: Dict[str, float]
    model_type: str


class ModelComparison(BaseModel):
    """Output schema for model comparison."""
    ticket_id: str
    xgboost_result: Optional[Dict]
    tensorflow_result: Optional[Dict]
    recommendation: Optional[str]


@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "message": "InsightDesk AI API is running",
        "version": "1.0.0",
        "status": "healthy"
    }


@app.get("/health")
def health_check(manager: ModelManager = Depends(get_model_manager)):
    """Detailed health check with model status."""
    xgb_info = manager.xgb_classifier.get_model_info() if manager.xgb_classifier else {"status": "not_loaded"}
    tf_info = manager.tf_classifier.get_model_info() if manager.tf_classifier else {"status": "not_loaded"}
    
    return {
        "status": "healthy",
        "models": {
            "xgboost": xgb_info,
            "tensorflow": tf_info
        },
        "api_version": "1.0.0"
    }


@app.post("/tickets", response_model=CategoryPrediction)
def create_ticket(ticket: TicketInput):
    """Create and classify a ticket (legacy endpoint for backward compatibility)."""
    return classify_ticket_xgboost(ticket)


@app.post("/classify/xgboost", response_model=CategoryPrediction)
def classify_ticket_xgboost(ticket: TicketInput, manager: ModelManager = Depends(get_model_manager)):
    """Classify ticket using XGBoost model."""
    try:
        if not manager.xgb_classifier or not manager.xgb_classifier.is_loaded:
            raise HTTPException(
                status_code=503, 
                detail="XGBoost model not loaded. Please check server logs."
            )
        
        # Convert ticket to dict
        ticket_dict = ticket.dict()
        
        # Generate ticket_id if not provided
        if not ticket.ticket_id:
            ticket_dict["ticket_id"] = f"ticket_{int(time.time())}"
        
        # Get prediction using the model manager's classifier
        result = manager.xgb_classifier.predict(ticket_dict)
        
        return CategoryPrediction(
            ticket_id=ticket_dict["ticket_id"],
            predicted_category=result["predicted_category"],
            confidence=result["confidence"],
            top_3_predictions=result["top_3_predictions"],
            model_type=result["model_type"]
        )
        
    except Exception as e:
        logger.error(f"Error in XGBoost classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.post("/classify/tensorflow", response_model=CategoryPrediction)
def classify_ticket_tensorflow(ticket: TicketInput, manager: ModelManager = Depends(get_model_manager)):
    """Classify ticket using TensorFlow model."""
    try:
        if not manager.tf_classifier or not manager.tf_classifier.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="TensorFlow model not loaded. Please check server logs."
            )
        
        # Convert ticket to dict
        ticket_dict = ticket.dict()
        
        # Generate ticket_id if not provided
        if not ticket.ticket_id:
            ticket_dict["ticket_id"] = f"ticket_{int(time.time())}"
        
        # Get prediction using the model manager's classifier
        result = manager.tf_classifier.predict(ticket_dict)
        
        return CategoryPrediction(
            ticket_id=ticket.ticket_id,
            predicted_category=result["predicted_category"],
            confidence=result["confidence"],
            top_3_predictions=result["top_3_predictions"],
            model_type=result["model_type"]
        )
        
    except Exception as e:
        logger.error(f"Error in TensorFlow classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.post("/classify/compare", response_model=ModelComparison)
def compare_models(ticket: TicketInput, manager: ModelManager = Depends(get_model_manager)):
    """Compare predictions from both models."""
    xgb_result = None
    tf_result = None
    
    try:
        if manager.xgb_classifier and manager.xgb_classifier.is_loaded:
            ticket_dict = ticket.dict()
            # Generate ticket_id if not provided
            if not ticket.ticket_id:
                ticket_dict["ticket_id"] = f"ticket_{int(time.time())}"
            xgb_result = manager.xgb_classifier.predict(ticket_dict)
    except Exception as e:
        logger.warning(f"XGBoost prediction failed: {e}")
    
    try:
        if manager.tf_classifier and manager.tf_classifier.is_loaded:
            ticket_dict = ticket.dict()
            # Generate ticket_id if not provided
            if not ticket.ticket_id:
                ticket_dict["ticket_id"] = f"ticket_{int(time.time())}"
            tf_result = manager.tf_classifier.predict(ticket_dict)
    except Exception as e:
        logger.warning(f"TensorFlow prediction failed: {e}")
    
    # Generate recommendation
    recommendation = None
    if xgb_result and tf_result:
        if xgb_result["predicted_category"] == tf_result["predicted_category"]:
            recommendation = f"Both models agree: {xgb_result['predicted_category']}"
        else:
            if xgb_result["confidence"] > tf_result["confidence"]:
                recommendation = f"XGBoost more confident: {xgb_result['predicted_category']} ({xgb_result['confidence']:.3f})"
            else:
                recommendation = f"TensorFlow more confident: {tf_result['predicted_category']} ({tf_result['confidence']:.3f})"
    elif xgb_result:
        recommendation = f"Only XGBoost available: {xgb_result['predicted_category']}"
    elif tf_result:
        recommendation = f"Only TensorFlow available: {tf_result['predicted_category']}"
    else:
        recommendation = "No models available for prediction"
    
    return ModelComparison(
        ticket_id=ticket.ticket_id,
        xgboost_result=xgb_result,
        tensorflow_result=tf_result,
        recommendation=recommendation
    )


@app.post("/classify/batch/xgboost", response_model=List[CategoryPrediction])
def classify_tickets_batch_xgboost(batch: TicketBatchInput, manager: ModelManager = Depends(get_model_manager)):
    """Classify multiple tickets using XGBoost model."""
    try:
        if not manager.xgb_classifier or not manager.xgb_classifier.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="XGBoost model not loaded. Please check server logs."
            )
        
        # Convert tickets to dicts
        tickets_data = [ticket.dict() for ticket in batch.tickets]
        
        # Add ticket_ids if missing
        for i, ticket_dict in enumerate(tickets_data):
            if not ticket_dict.get("ticket_id"):
                ticket_dict["ticket_id"] = f"batch_ticket_{i}_{int(time.time())}"
        
        # Get predictions using the model manager's classifier
        results = manager.xgb_classifier.predict_batch(tickets_data)
        
        # Format response
        predictions = []
        for i, result in enumerate(results):
            predictions.append(CategoryPrediction(
                ticket_id=tickets_data[i]["ticket_id"],
                predicted_category=result.get("predicted_category", "unknown"),
                confidence=result.get("confidence", 0.0),
                top_3_predictions=result.get("top_3_predictions", {}),
                model_type=result.get("model_type", "xgboost")
            ))
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error in batch XGBoost classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch classification failed: {str(e)}")


@app.post("/classify/batch/tensorflow", response_model=List[CategoryPrediction])
def classify_tickets_batch_tensorflow(batch: TicketBatchInput, manager: ModelManager = Depends(get_model_manager)):
    """Classify multiple tickets using TensorFlow model."""
    try:
        if not manager.tf_classifier or not manager.tf_classifier.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="TensorFlow model not loaded. Please check server logs."
            )
        
        # Convert tickets to dicts
        tickets_data = [ticket.dict() for ticket in batch.tickets]
        
        # Add ticket_ids if missing
        for i, ticket_dict in enumerate(tickets_data):
            if not ticket_dict.get("ticket_id"):
                ticket_dict["ticket_id"] = f"batch_ticket_{i}_{int(time.time())}"
        
        # Get predictions using the model manager's classifier
        results = manager.tf_classifier.predict_batch(tickets_data)
        
        # Format response
        predictions = []
        for i, result in enumerate(results):
            predictions.append(CategoryPrediction(
                ticket_id=tickets_data[i]["ticket_id"],
                predicted_category=result.get("predicted_category", "unknown"),
                confidence=result.get("confidence", 0.0),
                top_3_predictions=result.get("top_3_predictions", {}),
                model_type=result.get("model_type", "tensorflow")
            ))
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error in batch TensorFlow classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch classification failed: {str(e)}")


@app.get("/models/info")
def get_models_info(manager: ModelManager = Depends(get_model_manager)):
    """Get information about loaded models."""
    return {
        "xgboost": manager.xgb_classifier.get_model_info() if manager.xgb_classifier else {"status": "not_loaded"},
        "tensorflow": manager.tf_classifier.get_model_info() if manager.tf_classifier else {"status": "not_loaded"}
    }


# Agentic AI Models
class AgentRequest(BaseModel):
    """Request model for agentic solution."""
    ticket_data: Dict[str, Any]
    max_steps: int = 5

class AgentResponse(BaseModel):
    """Response model for agentic solution."""
    result: Dict[str, Any]

@app.post("/agent/solve", response_model=AgentResponse)
async def solve_ticket_agent(request: AgentRequest):
    """
    Solve a ticket using the Agentic AI Orchestrator.
    
    This endpoint triggers the Plan-Act-Observe-Reflect loop.
    """
    if not AGENTIC_AVAILABLE:
        raise HTTPException(status_code=503, detail="Agentic components not available")
    
    try:
        orchestrator = AgentOrchestrator()
        result = orchestrator.run(request.ticket_data, max_steps=request.max_steps)
        return AgentResponse(result=result)
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
