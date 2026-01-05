# src/agentic/orchestrator.py
"""
Agent Orchestrator Module

This module implements the core orchestration logic for the agentic AI system.
It manages the Plan-Act-Observe-Reflect (PAOR) loop, which is the fundamental
execution pattern for autonomous agent behavior.

The orchestrator coordinates between:
- Planner: Decides what action to take next based on current state
- Tools: Execute specific tasks (classification, RAG search, etc.)
- Memory: Maintains context across the execution loop
- State: Tracks the agent's progress and accumulated knowledge

Key Concepts:
    - Plan: Determine the next action based on current state and history
    - Act: Execute the planned action using the appropriate tool
    - Observe: Capture the results and any errors from the action
    - Reflect: Update the agent's state based on observations
"""

from typing import Any, Dict, List
import logging
from src.agentic.core import AgentState, AgentAction, AgentObservation
from src.agentic.planner import RuleBasedPlanner
from src.agentic.tools.ml_tools import ClassificationTool
from src.agentic.tools.rag_tools import RAGSearchTool
from src.retrieval.rag_pipeline import RAGPipeline

# Configure logger for tracking agent execution flow
logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Orchestrates the autonomous agent execution loop.
    
    This class implements the Plan-Act-Observe-Reflect (PAOR) pattern, which enables
    the agent to autonomously process tickets through a series of intelligent steps:
    
    1. PLAN: The planner analyzes the current state and decides the next action
    2. ACT: The selected tool executes the planned action
    3. OBSERVE: Results (or errors) are captured and recorded
    4. REFLECT: The state is updated based on observations for the next iteration
    
    The orchestrator manages a registry of tools that the agent can use, including:
    - ticket_classifier: ML-based ticket categorization
    - solution_retriever: RAG-based solution search from knowledge base
    
    Attributes:
        planner (RuleBasedPlanner): The planning component that decides actions
        rag_pipeline (RAGPipeline): Shared RAG pipeline instance for retrieval
        tools (Dict[str, Any]): Registry mapping tool names to tool instances
    
    Example:
        >>> orchestrator = AgentOrchestrator()
        >>> ticket = {"ticket_id": "T-123", "subject": "Login issue", ...}
        >>> result = orchestrator.run(ticket, max_steps=5)
        >>> print(result["status"])  # "success" or "partial"
    """
    
    def __init__(self):
        """
        Initialize the orchestrator with planner and tools.
        
        Sets up the complete agent execution environment by:
        1. Creating a rule-based planner for action selection
        2. Initializing the RAG pipeline for knowledge retrieval
        3. Registering all available tools in the tool registry
        
        Note:
            In production, dependencies should be injected rather than
            instantiated here to enable better testing and flexibility.
        """
        # Initialize the planner that will decide what actions to take
        # Currently uses rule-based logic, but could be replaced with LLM-based planning
        self.planner = RuleBasedPlanner()
        
        # Initialize the RAG (Retrieval-Augmented Generation) pipeline
        # This is shared across tools to avoid redundant initialization
        # In production, this should be dependency-injected for better testability
        self.rag_pipeline = RAGPipeline() 
        
        # Tool registry: Maps tool names to their implementations
        # The planner will reference these names when deciding actions
        # Each tool must implement a run() method that accepts tool_input
        self.tools = {
            # Classifies tickets into categories (e.g., authentication, network, etc.)
            "ticket_classifier": ClassificationTool(),
            
            # Searches the knowledge base for relevant solutions using RAG
            "solution_retriever": RAGSearchTool(pipeline=self.rag_pipeline)
        }
        
    def run(self, ticket_data: Dict[str, Any], max_steps: int = 5) -> Dict[str, Any]:
        """
        Execute the agent loop for a given ticket.
        
        This is the main entry point for agent execution. It runs the PAOR loop
        for up to max_steps iterations, allowing the agent to:
        1. Classify the ticket
        2. Search for relevant solutions
        3. Gather additional context if needed
        4. Formulate a final response
        
        The loop terminates when either:
        - The planner decides no more actions are needed
        - The maximum number of steps is reached
        
        Args:
            ticket_data (Dict[str, Any]): The ticket to process, containing:
                - ticket_id: Unique identifier
                - subject: Brief description
                - description: Detailed problem description
                - priority: Urgency level (low/medium/high)
                - product: Which product/service this relates to
            max_steps (int, optional): Maximum iterations to prevent infinite loops.
                Defaults to 5. Typical tickets complete in 2-3 steps.
        
        Returns:
            Dict[str, Any]: Final agent response containing:
                - ticket_id: The processed ticket ID
                - agent_plan: List of actions taken (human-readable)
                - analysis: Classification results and suggested solutions
                - status: "success" if complete, "partial" if incomplete
        
        Raises:
            Exception: Any unhandled errors during execution are logged but
                not raised, ensuring graceful degradation.
        """
        logger.info(f"Starting agent loop for ticket: {ticket_data.get('ticket_id', 'unknown')}")
        
        # Initialize the agent state with the ticket data
        # State tracks the ticket, execution history, and accumulated knowledge
        state = AgentState(ticket_data=ticket_data)
        
        # Main execution loop - iterate until completion or max_steps reached
        step_count = 0
        while step_count < max_steps:
            step_count += 1
            logger.info(f"--- Step {step_count} ---")
            
            # ========== PHASE 1: PLAN ==========
            # Ask the planner to decide the next action based on current state
            # The planner examines the history to avoid redundant actions
            action = self.planner.plan(state)
            
            # If planner returns None, it has determined that no more actions are needed
            # This happens when all necessary information has been gathered
            if not action:
                logger.info("Planner decided to finish.")
                break
            
            # Log the planned action for debugging and audit trails
            logger.info(f"Action: {action.tool_name} | {action.log}")
            
            # Record the action in the state history before execution
            # This ensures we have a complete trace even if execution fails
            state.history.append(action)
            
            # ========== PHASE 2: ACT ==========
            # Retrieve the tool from the registry
            tool = self.tools.get(action.tool_name)
            
            if not tool:
                # Tool not found - this indicates a planner bug or misconfiguration
                error_msg = f"Tool {action.tool_name} not found"
                logger.error(error_msg)
                
                # Create an error observation to record the failure
                observation = AgentObservation(
                    tool_name=action.tool_name,
                    output=None,
                    error=error_msg
                )
            else:
                # Tool found - attempt to execute it
                try:
                    # Execute the tool with the provided input
                    # Each tool's run() method returns tool-specific output
                    output = tool.run(action.tool_input)
                    
                    # Create a success observation with the tool's output
                    observation = AgentObservation(
                        tool_name=action.tool_name,
                        output=output
                    )
                except Exception as e:
                    # Tool execution failed - capture the error
                    # This allows the agent to continue and potentially recover
                    logger.error(f"Tool execution failed: {e}")
                    
                    # Create an error observation to record the failure
                    observation = AgentObservation(
                        tool_name=action.tool_name,
                        output=None,
                        error=str(e)
                    )
            
            # ========== PHASE 3: OBSERVE ==========
            # Record the observation (success or failure) in the state history
            # This builds the complete action-observation trace
            state.history.append(observation)
            
            # ========== PHASE 4: REFLECT ==========
            # Allow the planner to update the state based on the observation
            # This can involve updating flags, extracting key information, etc.
            state = self.planner.reflect(state)
            
        # ========== FINALIZATION ==========
        # After the loop completes, construct the final response from the accumulated history
        final_response = self._construct_final_answer(state)
        
        # Store the final answer in the state for potential future reference
        state.final_answer = final_response
        
        return final_response

    def _construct_final_answer(self, state: AgentState) -> Dict[str, Any]:
        """
        Build the final response from the agent's execution history.
        
        This method extracts relevant information from the action-observation
        history to construct a comprehensive response. It safely handles cases
        where tools may have failed or returned partial results.
        
        The method scans through the history looking for:
        - Classification results from the ticket_classifier tool
        - Solution suggestions from the solution_retriever tool
        
        Args:
            state (AgentState): The final state after all agent steps,
                containing the complete history of actions and observations.
        
        Returns:
            Dict[str, Any]: Structured final answer containing:
                - ticket_id: The processed ticket identifier
                - agent_plan: Chronological list of actions taken (human-readable)
                - analysis: Dictionary with:
                    - classification: Category, confidence, and metadata
                    - suggested_solutions: List of relevant solutions from knowledge base
                - status: "success" if classification succeeded, "partial" otherwise
        
        Note:
            The status is "success" only if classification completed. This is because
            classification is considered the minimum viable output. Solutions are
            optional enhancements.
        """
        # Initialize variables to collect results from the history
        classification = None  # Will store ticket classification results
        solutions = []  # Will store retrieved solutions from knowledge base
        
        # Scan through the execution history to extract relevant observations
        for item in state.history:
            # Only process observations (skip actions)
            if isinstance(item, AgentObservation):
                # Check if this is a classification result
                if item.tool_name == "ticket_classifier" and item.output:
                    classification = item.output
                    
                # Check if this is a solution retrieval result
                elif item.tool_name == "solution_retriever" and item.output:
                    # Extract the solutions list from the tool output
                    solutions = item.output.get("found_solutions", [])
        
        # Construct the final response structure
        return {
            # Echo back the ticket ID for reference
            "ticket_id": state.ticket_data.get("ticket_id"),
            
            # Provide a human-readable summary of what the agent did
            # Extract the 'log' field from each action for readability
            "agent_plan": [a.log for a in state.history if isinstance(a, AgentAction)],
            
            # Package the analysis results
            "analysis": {
                "classification": classification,  # May be None if classification failed
                "suggested_solutions": solutions  # May be empty if retrieval failed/skipped
            },
            
            # Determine overall status based on whether we got classification
            # Classification is considered essential, solutions are optional
            "status": "success" if classification else "partial"
        }

