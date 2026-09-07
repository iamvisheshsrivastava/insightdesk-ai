# src/agentic/planner.py
import logging
from typing import List
from src.agentic.core import AgentState, AgentAction, BaseAgent

logger = logging.getLogger(__name__)

class RuleBasedPlanner(BaseAgent):
    """
    A simple rule-based planner that orchestrates the workflow:
    1. Classify Ticket
    2. Retrieve Solutions (using category)
    3. Finalize Answer
    
    In a full LLM version, this would prompt GPT-4 to generate a plan.
    """
    name = "rule_planner"

    def plan(self, state: AgentState) -> AgentAction:
        """Determines the next action based on state."""

        # Step 1: Check if we have classified the ticket
        classification = self._get_tool_output(state, "ticket_classifier")
        if not classification:
            # Don't keep retrying a tool that has already failed - that just
            # burns max_steps re-issuing an identical action that will fail
            # again. Give up (finish) so the failure surfaces in the final
            # answer instead of being retried silently until max_steps.
            if self._has_failed(state, "ticket_classifier"):
                return None
            return AgentAction(
                tool_name="ticket_classifier",
                tool_input=state.ticket_data,
                log="Deciding to classify ticket first to understand context."
            )

        # Step 2: Check if we have searched for solutions
        rag_results = self._get_tool_output(state, "solution_retriever")
        if not rag_results:
            if self._has_failed(state, "solution_retriever"):
                return None
            # enhance query with predicted category
            query = state.ticket_data.copy()
            query["category"] = classification.get("predicted_category")

            return AgentAction(
                tool_name="solution_retriever",
                tool_input=query,
                log=f"Searching solutions for category: {query['category']}"
            )

        # Step 3: If we have both, we are done
        return None # None implies task completion (or specific 'finish' action)

    def reflect(self, state: AgentState) -> AgentState:
        """
        Reflects on the current state. 
        For rule based, this just updates memory essentially.
        """
        return state

    def _get_tool_output(self, state: AgentState, tool_name: str):
        """Helper to find previous tool outputs."""
        for item in reversed(state.history):
            if hasattr(item, 'output') and item.tool_name == tool_name:
                return item.output
        return None

    def _has_failed(self, state: AgentState, tool_name: str) -> bool:
        """Checks whether the given tool already produced an error observation."""
        for item in reversed(state.history):
            if hasattr(item, 'output') and item.tool_name == tool_name:
                return getattr(item, 'error', None) is not None
        return False
