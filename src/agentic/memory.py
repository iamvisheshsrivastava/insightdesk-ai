# src/agentic/memory.py
from typing import Any, Dict, List
from src.agentic.core import AgentAction, AgentObservation

class AgentMemory:
    """
    Manages short-term (session) and long-term (vector) memory.
    Currently focuses on session context.
    """
    def __init__(self):
        self.short_term: List[Dict[str, Any]] = []

    def add_trace(self, action: AgentAction, observation: AgentObservation):
        self.short_term.append({
            "action": action.dict(),
            "observation": observation.dict(),
            "timestamp": "TODO"
        })
    
    def get_context(self) -> str:
        """Returns a string representation of the conversation so far."""
        return str(self.short_term)
