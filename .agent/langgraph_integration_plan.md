# LangGraph Integration Plan for InsightDesk AI

## Overview
This document outlines the integration of **LangGraph** into the InsightDesk AI agentic system. LangGraph is a library for building stateful, multi-actor applications with LLMs, using a graph-based approach to orchestrate complex workflows.

## Why LangGraph?

### Current Architecture Limitations
1. **Rule-Based Planning**: The current `RuleBasedPlanner` uses hardcoded logic (classify → retrieve → finish)
2. **Limited Flexibility**: Cannot handle complex, multi-step reasoning or dynamic branching
3. **No Conditional Flows**: Cannot adapt workflow based on intermediate results
4. **Manual State Management**: State transitions are manually coded

### LangGraph Benefits
1. **Graph-Based Workflow**: Define agent workflows as directed graphs with nodes and edges
2. **Conditional Routing**: Dynamic branching based on state (e.g., if classification confidence is low, ask for more info)
3. **Built-in State Management**: Automatic state persistence and updates across nodes
4. **Checkpointing**: Save and resume agent execution at any point
5. **Human-in-the-Loop**: Easy integration of human approval steps
6. **Cycles & Loops**: Support for iterative refinement (e.g., retry with different parameters)
7. **Better LLM Integration**: Seamless integration with LangChain tools and LLMs

## Proposed Integration Points

### 1. **Replace RuleBasedPlanner with LangGraph Workflow** (Primary)
**Location**: `src/agentic/orchestrator_langgraph.py` (new file)

**Benefits**:
- Define the PAOR loop as a graph with nodes: Plan → Act → Observe → Reflect
- Add conditional edges (e.g., if classification fails, retry or escalate)
- Support complex workflows (e.g., multi-step troubleshooting)

**Graph Structure**:
```
START → classify_ticket → [confidence check]
                          ├─ high confidence → retrieve_solutions → END
                          ├─ medium confidence → ask_clarification → classify_ticket
                          └─ low confidence → escalate_to_human → END
```

### 2. **Add Conversational Memory with LangGraph** (Enhancement)
**Location**: `src/agentic/memory_langgraph.py` (new file)

**Benefits**:
- Track multi-turn conversations with users
- Maintain context across multiple ticket updates
- Support follow-up questions and clarifications

### 3. **Implement Multi-Agent Collaboration** (Advanced)
**Location**: `src/agentic/multi_agent_graph.py` (new file)

**Benefits**:
- Specialist agents for different ticket types (auth expert, network expert, etc.)
- Supervisor agent that routes to specialists
- Agents can collaborate and share findings

**Graph Structure**:
```
START → supervisor → [route by category]
                    ├─ auth_specialist → supervisor
                    ├─ network_specialist → supervisor
                    └─ general_specialist → supervisor
supervisor → synthesize_response → END
```

### 4. **Add Human-in-the-Loop Approval** (Production Feature)
**Location**: Integration in existing orchestrator

**Benefits**:
- Pause execution for human review before critical actions
- Allow humans to override agent decisions
- Audit trail of human interventions

## Implementation Roadmap

### Phase 1: Basic LangGraph Integration (Recommended Starting Point)
**Files to Create**:
1. `src/agentic/orchestrator_langgraph.py` - LangGraph-based orchestrator
2. `src/agentic/nodes/` - Directory for graph nodes
   - `classification_node.py`
   - `retrieval_node.py`
   - `reflection_node.py`
3. `scripts/test_langgraph_agent.py` - Test script for LangGraph version

**Features**:
- Convert current workflow to LangGraph
- Add conditional routing based on classification confidence
- Implement checkpointing for resumable execution
- Maintain backward compatibility with existing API

### Phase 2: Enhanced Workflows
**Features**:
- Multi-step troubleshooting workflows
- Iterative refinement (retry with different queries)
- Parallel tool execution (classify + sentiment analysis simultaneously)
- Human-in-the-loop for high-priority tickets

### Phase 3: Multi-Agent System
**Features**:
- Specialist agents for different domains
- Supervisor agent for routing and synthesis
- Agent collaboration and knowledge sharing

## Technical Requirements

### Dependencies to Add
```txt
langgraph>=0.0.20
langchain>=0.1.0
langchain-openai>=0.0.5  # If using OpenAI
langchain-community>=0.0.10
```

### Environment Variables
```bash
OPENAI_API_KEY=your_key_here  # If using OpenAI for LLM-based planning
LANGCHAIN_TRACING_V2=true  # For LangSmith debugging
LANGCHAIN_API_KEY=your_key_here  # Optional, for LangSmith
```

## Example: Basic LangGraph Workflow

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class TicketState(TypedDict):
    ticket_data: dict
    classification: dict
    solutions: list
    confidence: float
    messages: Annotated[list, operator.add]

def classify_node(state: TicketState):
    # Run classification tool
    result = classification_tool.run(state["ticket_data"])
    return {
        "classification": result,
        "confidence": result.get("confidence", 0.0)
    }

def should_retrieve(state: TicketState):
    if state["confidence"] > 0.7:
        return "retrieve"
    elif state["confidence"] > 0.4:
        return "clarify"
    else:
        return "escalate"

# Build graph
workflow = StateGraph(TicketState)
workflow.add_node("classify", classify_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("clarify", clarify_node)
workflow.add_node("escalate", escalate_node)

workflow.set_entry_point("classify")
workflow.add_conditional_edges(
    "classify",
    should_retrieve,
    {
        "retrieve": "retrieve",
        "clarify": "clarify",
        "escalate": "escalate"
    }
)
workflow.add_edge("retrieve", END)
workflow.add_edge("clarify", "classify")  # Loop back
workflow.add_edge("escalate", END)

app = workflow.compile()
```

## Migration Strategy

### Option A: Parallel Implementation (Recommended)
- Keep existing `orchestrator.py` unchanged
- Create new `orchestrator_langgraph.py`
- Add feature flag to switch between implementations
- Gradual migration with A/B testing

### Option B: Direct Replacement
- Replace `orchestrator.py` with LangGraph version
- Higher risk but cleaner codebase
- Requires comprehensive testing

## Success Metrics

1. **Workflow Flexibility**: Can handle at least 3 different ticket resolution paths
2. **Conditional Logic**: Successfully routes based on classification confidence
3. **State Persistence**: Can save and resume agent execution
4. **Performance**: No significant latency increase (<10% overhead)
5. **Maintainability**: Easier to add new workflow steps (measured by LOC required)

## Next Steps

1. **Install LangGraph**: Add to requirements.txt
2. **Create Basic Graph**: Implement Phase 1 with simple classify → retrieve workflow
3. **Add Conditional Routing**: Implement confidence-based branching
4. **Test & Compare**: Run side-by-side with existing orchestrator
5. **Iterate**: Add more complex workflows based on real ticket patterns

## Resources

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [LangGraph Tutorials](https://github.com/langchain-ai/langgraph/tree/main/examples)
- [Multi-Agent Systems with LangGraph](https://python.langchain.com/docs/langgraph/tutorials/multi_agent/)
