
import sys
import os
import json
# Add project root to path
sys.path.append(os.getcwd())

from src.agentic.orchestrator import AgentOrchestrator

def test_agent_loop():
    print("Initializing Orchestrator...")
    orchestrator = AgentOrchestrator()
    
    ticket = {
        "ticket_id": "TEST-001",
        "subject": "Login failed",
        "description": "I cannot login to the portal. It says invalid password even though I reset it.",
        "priority": "high"
    }
    
    print(f"Running agent for ticket: {ticket['subject']}")
    result = orchestrator.run(ticket)
    
    print("\n--- Agent Result ---")
    print(json.dumps(result, indent=2))
    
    if result.get("status") == "success":
        print("\n✅ Agent loop test PASSED")
    else:
        print("\n❌ Agent loop test FAILED or PARTIAL")

if __name__ == "__main__":
    test_agent_loop()
