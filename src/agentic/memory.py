# src/agentic/memory.py
"""
Agent Memory Module

This module provides memory management capabilities for the agentic AI system.
It handles both short-term (session-based) and long-term (persistent vector) memory,
enabling agents to maintain context across interactions and learn from past experiences.

The memory system is crucial for the Plan-Act-Observe-Reflect loop, allowing agents
to build upon previous actions and observations.
"""

from typing import Any, Dict, List
from src.agentic.core import AgentAction, AgentObservation


class AgentMemory:
    """
    Manages agent memory across different time horizons.
    
    This class provides a dual-memory architecture:
    1. Short-term memory: Stores the current session's action-observation pairs
       for immediate context and decision-making within a single agent execution.
    2. Long-term memory: (Future) Will store historical patterns in a vector database
       for cross-session learning and retrieval of similar past scenarios.
    
    Attributes:
        short_term (List[Dict[str, Any]]): A chronological list of action-observation
            traces from the current agent execution session. Each entry contains:
            - action: The action taken by the agent (serialized)
            - observation: The result/feedback from executing that action
            - timestamp: When this trace was recorded (currently placeholder)
    
    Example:
        >>> memory = AgentMemory()
        >>> action = AgentAction(tool_name="classifier", tool_input={...})
        >>> observation = AgentObservation(tool_name="classifier", output={...})
        >>> memory.add_trace(action, observation)
        >>> context = memory.get_context()  # Retrieve full session history
    """
    
    def __init__(self):
        """
        Initialize the agent memory system.
        
        Creates an empty short-term memory buffer to store the current session's
        action-observation traces. This buffer is reset for each new agent execution.
        
        Future enhancements will include:
        - Long-term vector store initialization
        - Memory consolidation strategies
        - Automatic pruning of irrelevant traces
        """
        # Short-term memory: stores action-observation pairs for the current session
        # This is a simple list that grows during agent execution and is cleared
        # when a new session starts
        self.short_term: List[Dict[str, Any]] = []

    def add_trace(self, action: AgentAction, observation: AgentObservation):
        """
        Record an action-observation pair in short-term memory.
        
        This method captures a complete trace of what the agent did (action) and
        what happened as a result (observation). These traces form the agent's
        working memory and are used for:
        - Reflection: Understanding what worked and what didn't
        - Context building: Providing history for subsequent planning decisions
        - Debugging: Tracking the agent's decision-making process
        
        Args:
            action (AgentAction): The action that was executed, including:
                - tool_name: Which tool was invoked
                - tool_input: Parameters passed to the tool
                - log: Human-readable description of the action
            observation (AgentObservation): The result of executing the action:
                - tool_name: Which tool produced this observation
                - output: The actual result data (if successful)
                - error: Error message (if the action failed)
        
        Note:
            The timestamp field is currently a placeholder ("TODO") and should be
            replaced with actual datetime values in production.
        """
        # Serialize the action and observation objects to dictionaries for storage
        # This allows for easy JSON serialization and persistence if needed
        self.short_term.append({
            "action": action.dict(),  # Convert Pydantic model to dict
            "observation": observation.dict(),  # Convert Pydantic model to dict
            "timestamp": "TODO"  # TODO: Replace with datetime.now().isoformat()
        })
    
    def get_context(self) -> str:
        """
        Retrieve the complete session history as a string.
        
        This method provides a serialized view of all actions and observations
        from the current session. It's primarily used by the planner to understand
        what has already been attempted and what information is available.
        
        Returns:
            str: A string representation of the short-term memory buffer,
                containing all action-observation pairs in chronological order.
        
        Note:
            Current implementation uses simple string conversion. Future versions
            should implement smarter formatting:
            - Structured JSON output
            - Selective summarization for long histories
            - Semantic compression using LLMs
            - Relevance filtering based on current context
        """
        # Simple string conversion - adequate for debugging but should be enhanced
        # for production use with proper formatting and potentially LLM-based summarization
        return str(self.short_term)
