# src/agentic/orchestrator_langgraph.py
"""
LangGraph-Based Agent Orchestrator

This module implements an enhanced version of the agent orchestrator using LangGraph,
which provides a graph-based approach to workflow orchestration with built-in state
management, conditional routing, and checkpointing capabilities.

Key Improvements over RuleBasedPlanner:
1. Graph-based workflow definition (more maintainable and visual)
2. Conditional routing based on intermediate results
3. Built-in state persistence and checkpointing
4. Support for cycles and iterative refinement
5. Easier to extend with new nodes and edges

Architecture:
    START → classify → [confidence check] → retrieve/clarify/escalate → END
    
The graph can handle:
- High confidence: Direct path to solution retrieval
- Medium confidence: Request clarification and retry classification
- Low confidence: Escalate to human agent
"""

from typing import Any, Dict, List, Annotated, TypedDict, Literal
import operator
import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agentic.tools.ml_tools import ClassificationTool
from src.agentic.tools.rag_tools import RAGSearchTool
from src.retrieval.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


# ============================================================================
# State Definition
# ============================================================================

class TicketState(TypedDict):
    """
    State schema for the ticket resolution workflow.
    
    LangGraph automatically manages state updates across nodes. Each node
    can read from and write to this state. The Annotated type with operator.add
    means that messages will be appended rather than replaced.
    
    Attributes:
        ticket_data: Original ticket information (id, subject, description, etc.)
        classification: Results from the classification tool
        solutions: Retrieved solutions from the knowledge base
        confidence: Classification confidence score (0.0 to 1.0)
        messages: Log of actions taken (for audit trail and debugging)
        retry_count: Number of classification retries (prevents infinite loops)
        status: Current workflow status (processing/success/escalated)
        error: Error message if something went wrong
    """
    ticket_data: dict
    classification: dict
    solutions: list
    confidence: float
    messages: Annotated[list, operator.add]  # Messages will accumulate
    retry_count: int
    status: str
    error: str | None


# ============================================================================
# Node Functions
# ============================================================================

def classify_ticket_node(state: TicketState) -> dict:
    """
    Classification node: Analyzes the ticket and predicts category/priority.
    
    This node wraps the existing ClassificationTool and extracts the confidence
    score to enable conditional routing in the graph.
    
    Args:
        state: Current ticket state
        
    Returns:
        dict: Updates to state including classification results and confidence
    """
    logger.info("🔍 Classifying ticket...")
    
    try:
        # Initialize and run classification tool
        classifier = ClassificationTool()
        result = classifier.run(state["ticket_data"])
        
        # Extract confidence score (default to 0.5 if not present)
        confidence = result.get("confidence", 0.5)
        
        # Log the classification result
        message = f"Classified as '{result.get('predicted_category')}' with {confidence:.2%} confidence"
        
        return {
            "classification": result,
            "confidence": confidence,
            "messages": [message],
            "status": "classified"
        }
        
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return {
            "classification": {},
            "confidence": 0.0,
            "messages": [f"Classification error: {str(e)}"],
            "status": "error",
            "error": str(e)
        }


def retrieve_solutions_node(state: TicketState) -> dict:
    """
    Solution retrieval node: Searches knowledge base for relevant solutions.
    
    This node uses the RAG pipeline to find similar past tickets and their
    solutions. It enhances the search query with the predicted category from
    classification.
    
    Args:
        state: Current ticket state with classification results
        
    Returns:
        dict: Updates to state including retrieved solutions
    """
    logger.info("📚 Retrieving solutions from knowledge base...")
    
    try:
        # Initialize RAG pipeline and tool
        rag_pipeline = RAGPipeline()
        retriever = RAGSearchTool(pipeline=rag_pipeline)
        
        # Enhance query with classification results
        query = state["ticket_data"].copy()
        if state.get("classification"):
            query["category"] = state["classification"].get("predicted_category")
        
        # Execute retrieval
        result = retriever.run(query)
        solutions = result.get("found_solutions", [])
        
        message = f"Found {len(solutions)} relevant solutions"
        
        return {
            "solutions": solutions,
            "messages": [message],
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Solution retrieval failed: {e}")
        return {
            "solutions": [],
            "messages": [f"Retrieval error: {str(e)}"],
            "status": "partial",
            "error": str(e)
        }


def request_clarification_node(state: TicketState) -> dict:
    """
    Clarification node: Handles medium-confidence classifications.
    
    In a full implementation, this would:
    1. Generate clarifying questions based on the ticket
    2. Send questions to the user
    3. Wait for user response
    4. Update ticket_data with additional context
    
    For now, this is a placeholder that logs the need for clarification.
    
    Args:
        state: Current ticket state
        
    Returns:
        dict: Updates to state with clarification request
    """
    logger.info("❓ Requesting clarification from user...")
    
    # In production, this would generate specific questions
    # For now, we'll just log and increment retry count
    message = "Classification confidence is medium. Would benefit from user clarification."
    
    # Increment retry count to prevent infinite loops
    retry_count = state.get("retry_count", 0) + 1
    
    return {
        "messages": [message],
        "retry_count": retry_count,
        "status": "needs_clarification"
    }


def escalate_to_human_node(state: TicketState) -> dict:
    """
    Escalation node: Handles low-confidence or failed classifications.
    
    This node is reached when:
    1. Classification confidence is too low
    2. Multiple retry attempts have failed
    3. An unrecoverable error occurred
    
    In production, this would:
    1. Create a task for a human agent
    2. Send notifications
    3. Update ticket status in the database
    
    Args:
        state: Current ticket state
        
    Returns:
        dict: Updates to state marking escalation
    """
    logger.info("🚨 Escalating to human agent...")
    
    reason = "Low classification confidence" if state.get("confidence", 0) < 0.4 else "Multiple retries failed"
    message = f"Escalated to human agent. Reason: {reason}"
    
    return {
        "messages": [message],
        "status": "escalated"
    }


# ============================================================================
# Conditional Routing Functions
# ============================================================================

def route_after_classification(state: TicketState) -> Literal["retrieve", "clarify", "escalate"]:
    """
    Conditional routing based on classification confidence.
    
    This function determines the next step in the workflow based on:
    1. Classification confidence score
    2. Number of retry attempts
    3. Error status
    
    Routing Logic:
    - High confidence (>0.7): Proceed directly to solution retrieval
    - Medium confidence (0.4-0.7): Request clarification (up to 2 retries)
    - Low confidence (<0.4): Escalate to human agent
    - Too many retries: Escalate to prevent infinite loops
    
    Args:
        state: Current ticket state
        
    Returns:
        str: Next node to execute ("retrieve", "clarify", or "escalate")
    """
    confidence = state.get("confidence", 0.0)
    retry_count = state.get("retry_count", 0)
    
    # Check for errors or too many retries
    if state.get("error") or retry_count >= 2:
        logger.info(f"Routing to escalate (error or max retries)")
        return "escalate"
    
    # Route based on confidence
    if confidence > 0.7:
        logger.info(f"Routing to retrieve (high confidence: {confidence:.2%})")
        return "retrieve"
    elif confidence > 0.4:
        logger.info(f"Routing to clarify (medium confidence: {confidence:.2%})")
        return "clarify"
    else:
        logger.info(f"Routing to escalate (low confidence: {confidence:.2%})")
        return "escalate"


# ============================================================================
# Graph Construction
# ============================================================================

def create_ticket_workflow() -> StateGraph:
    """
    Constructs the LangGraph workflow for ticket resolution.
    
    Graph Structure:
        START
          ↓
        classify_ticket
          ↓
        [confidence check]
          ├─ high (>0.7) → retrieve_solutions → END
          ├─ medium (0.4-0.7) → request_clarification → classify_ticket (loop)
          └─ low (<0.4) → escalate_to_human → END
    
    Features:
    - Conditional branching based on classification confidence
    - Iterative refinement through clarification loop
    - Automatic escalation for edge cases
    - Built-in state management and checkpointing
    
    Returns:
        StateGraph: Compiled workflow ready for execution
    """
    # Initialize the graph with our state schema
    workflow = StateGraph(TicketState)
    
    # Add nodes (processing steps)
    workflow.add_node("classify_ticket", classify_ticket_node)
    workflow.add_node("retrieve_solutions", retrieve_solutions_node)
    workflow.add_node("request_clarification", request_clarification_node)
    workflow.add_node("escalate_to_human", escalate_to_human_node)
    
    # Set the entry point
    workflow.set_entry_point("classify_ticket")
    
    # Add conditional edges from classification
    workflow.add_conditional_edges(
        "classify_ticket",
        route_after_classification,
        {
            "retrieve": "retrieve_solutions",
            "clarify": "request_clarification",
            "escalate": "escalate_to_human"
        }
    )
    
    # Add edges to END
    workflow.add_edge("retrieve_solutions", END)
    workflow.add_edge("escalate_to_human", END)
    
    # Add loop from clarification back to classification
    workflow.add_edge("request_clarification", "classify_ticket")
    
    return workflow


# ============================================================================
# Main Orchestrator Class
# ============================================================================

class LangGraphOrchestrator:
    """
    LangGraph-based orchestrator for ticket resolution.
    
    This class provides a high-level interface to the LangGraph workflow,
    maintaining API compatibility with the original AgentOrchestrator while
    leveraging LangGraph's advanced features.
    
    Features:
    - Graph-based workflow execution
    - Automatic state management
    - Checkpointing for resumable execution
    - Conditional routing and loops
    - Comprehensive logging and error handling
    
    Attributes:
        workflow: Compiled LangGraph workflow
        checkpointer: Memory-based checkpointer for state persistence
    
    Example:
        >>> orchestrator = LangGraphOrchestrator()
        >>> ticket = {"ticket_id": "T-123", "subject": "Login issue", ...}
        >>> result = orchestrator.run(ticket)
        >>> print(result["status"])  # "success" or "escalated"
    """
    
    def __init__(self):
        """
        Initialize the LangGraph orchestrator.
        
        Sets up the workflow graph and checkpointer for state persistence.
        """
        logger.info("Initializing LangGraph Orchestrator...")
        
        # Create and compile the workflow
        workflow = create_ticket_workflow()
        
        # Add checkpointer for state persistence
        # In production, use SQLite or Postgres checkpointer
        checkpointer = MemorySaver()
        self.workflow = workflow.compile(checkpointer=checkpointer)
        
        logger.info("✅ LangGraph Orchestrator initialized successfully")
    
    def run(self, ticket_data: Dict[str, Any], max_steps: int = 10) -> Dict[str, Any]:
        """
        Execute the ticket resolution workflow.
        
        This method runs the LangGraph workflow with the provided ticket data,
        managing state transitions and collecting results.
        
        Args:
            ticket_data: Ticket information (id, subject, description, etc.)
            max_steps: Maximum number of graph steps (prevents infinite loops)
            
        Returns:
            dict: Final result containing:
                - ticket_id: The processed ticket ID
                - agent_plan: List of actions taken
                - analysis: Classification and solutions
                - status: Workflow outcome (success/escalated/error)
                - messages: Execution log
        """
        logger.info(f"Starting LangGraph workflow for ticket: {ticket_data.get('ticket_id', 'unknown')}")
        
        # Initialize state
        initial_state: TicketState = {
            "ticket_data": ticket_data,
            "classification": {},
            "solutions": [],
            "confidence": 0.0,
            "messages": [],
            "retry_count": 0,
            "status": "processing",
            "error": None
        }
        
        try:
            # Execute the workflow
            # The graph will automatically manage state transitions
            final_state = self.workflow.invoke(
                initial_state,
                config={
                    "recursion_limit": max_steps,
                    "configurable": {"thread_id": ticket_data.get("ticket_id", "default")}
                }
            )
            
            # Construct final response in the same format as original orchestrator
            return self._construct_final_answer(final_state)
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "ticket_id": ticket_data.get("ticket_id"),
                "agent_plan": ["Workflow execution failed"],
                "analysis": {
                    "classification": {},
                    "suggested_solutions": []
                },
                "status": "error",
                "error": str(e),
                "messages": [f"Error: {str(e)}"]
            }
    
    def _construct_final_answer(self, state: TicketState) -> Dict[str, Any]:
        """
        Build the final response from the workflow state.
        
        Converts the LangGraph state into the expected response format,
        maintaining compatibility with the existing API.
        
        Args:
            state: Final state after workflow completion
            
        Returns:
            dict: Formatted response with classification, solutions, and metadata
        """
        return {
            "ticket_id": state["ticket_data"].get("ticket_id"),
            "agent_plan": state.get("messages", []),
            "analysis": {
                "classification": state.get("classification", {}),
                "suggested_solutions": state.get("solutions", [])
            },
            "status": state.get("status", "unknown"),
            "confidence": state.get("confidence", 0.0),
            "messages": state.get("messages", [])
        }
    
    def visualize(self, output_path: str = "workflow_graph.png"):
        """
        Generate a visual representation of the workflow graph.
        
        This is useful for documentation and debugging. Requires graphviz.
        
        Args:
            output_path: Path to save the graph visualization
        """
        try:
            from IPython.display import Image, display
            display(Image(self.workflow.get_graph().draw_mermaid_png()))
        except ImportError:
            logger.warning("IPython not available. Install it to visualize the graph.")
