# src/agentic/core.py
from typing import Any, Dict, List, Optional, Protocol, Union
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

class AgentAction(BaseModel):
    """Represents an action taken by an agent."""
    tool_name: str
    tool_input: Dict[str, Any]
    log: str  # Thought process

class AgentObservation(BaseModel):
    """Represents the output of a tool."""
    tool_name: str
    output: Any
    error: Optional[str] = None

class AgentState(BaseModel):
    """Tracks the state of the agent execution loop."""
    ticket_data: Dict[str, Any]
    plan: List[str] = Field(default_factory=list)
    history: List[Union[AgentAction, AgentObservation]] = Field(default_factory=list)
    memory: Dict[str, Any] = Field(default_factory=dict)
    final_answer: Optional[Dict[str, Any]] = None
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)

class BaseTool(Protocol):
    """Interface for all tools."""
    name: str
    description: str

    def run(self, **kwargs) -> Any:
        ...

class BaseAgent(Protocol):
    """Interface for agents."""
    name: str

    def plan(self, state: AgentState) -> AgentAction:
        ...

    def reflect(self, state: AgentState) -> AgentState:
        ...
