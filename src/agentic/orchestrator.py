# src/agentic/orchestrator.py
from typing import Any, Dict, List
import logging
from src.agentic.core import AgentState, AgentAction, AgentObservation
from src.agentic.planner import RuleBasedPlanner
from src.agentic.tools.ml_tools import ClassificationTool
from src.agentic.tools.rag_tools import RAGSearchTool
from src.retrieval.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """
    Manages the Plan-Act-Observe-Reflect loop.
    """
    def __init__(self):
        # Initialize components
        self.planner = RuleBasedPlanner()
        
        # Initialize tools
        # We share the RAG pipeline instance
        # In a real app, this should be dependency-injected
        self.rag_pipeline = RAGPipeline() 
        # Lazy init happens in tool if needed, or we can init here
        
        self.tools = {
            "ticket_classifier": ClassificationTool(),
            "solution_retriever": RAGSearchTool(pipeline=self.rag_pipeline)
        }
        
    def run(self, ticket_data: Dict[str, Any], max_steps: int = 5) -> Dict[str, Any]:
        """
        Executes the agent loop for a given ticket.
        """
        logger.info(f"Starting agent loop for ticket: {ticket_data.get('ticket_id', 'unknown')}")
        
        state = AgentState(ticket_data=ticket_data)
        
        step_count = 0
        while step_count < max_steps:
            step_count += 1
            logger.info(f"--- Step {step_count} ---")
            
            # 1. PLAN
            action = self.planner.plan(state)
            
            # If no action returned, we are done
            if not action:
                logger.info("Planner decided to finish.")
                break
                
            logger.info(f"Action: {action.tool_name} | {action.log}")
            state.history.append(action)
            
            # 2. ACT
            tool = self.tools.get(action.tool_name)
            if not tool:
                error_msg = f"Tool {action.tool_name} not found"
                logger.error(error_msg)
                observation = AgentObservation(
                    tool_name=action.tool_name,
                    output=None,
                    error=error_msg
                )
            else:
                try:
                    output = tool.run(action.tool_input)
                    observation = AgentObservation(
                        tool_name=action.tool_name,
                        output=output
                    )
                except Exception as e:
                    logger.error(f"Tool execution failed: {e}")
                    observation = AgentObservation(
                        tool_name=action.tool_name,
                        output=None,
                        error=str(e)
                    )
            
            # 3. OBSERVE
            state.history.append(observation)
            
            # 4. REFLECT (Optional update of state/memory)
            state = self.planner.reflect(state)
            
        # 5. Final Answer formulation
        # For this demo, we construct it from the history
        final_response = self._construct_final_answer(state)
        state.final_answer = final_response
        
        return final_response

    def _construct_final_answer(self, state: AgentState) -> Dict[str, Any]:
        """Constructs safe final answer from history."""
        classification = None
        solutions = []
        
        for item in state.history:
            if isinstance(item, AgentObservation):
                if item.tool_name == "ticket_classifier" and item.output:
                    classification = item.output
                elif item.tool_name == "solution_retriever" and item.output:
                    solutions = item.output.get("found_solutions", [])
                    
        return {
            "ticket_id": state.ticket_data.get("ticket_id"),
            "agent_plan": [a.log for a in state.history if isinstance(a, AgentAction)],
            "analysis": {
                "classification": classification,
                "suggested_solutions": solutions
            },
            "status": "success" if classification else "partial"
        }
