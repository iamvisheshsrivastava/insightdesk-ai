"""
Simple test script for the Agentic AI Orchestrator.
This test verifies the Plan-Act-Observe-Reflect loop without requiring RAG dependencies.
"""
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_agent_basic():
    """Test basic agent functionality with classification only."""
    print("=" * 60)
    print("Testing Agentic AI Orchestrator")
    print("=" * 60)
    
    try:
        from src.agentic.orchestrator import AgentOrchestrator
        print("✅ Successfully imported AgentOrchestrator")
    except Exception as e:
        print(f"❌ Failed to import AgentOrchestrator: {e}")
        return False
    
    # Create test ticket
    ticket = {
        "ticket_id": "TEST-001",
        "subject": "Cannot login to application",
        "description": "User is unable to authenticate with correct credentials. Getting timeout error after password reset.",
        "priority": "high",
        "product": "web_application"
    }
    
    print(f"\n📋 Test Ticket:")
    print(f"   ID: {ticket['ticket_id']}")
    print(f"   Subject: {ticket['subject']}")
    print(f"   Priority: {ticket['priority']}")
    
    try:
        print("\n🚀 Initializing Orchestrator...")
        orchestrator = AgentOrchestrator()
        print("✅ Orchestrator initialized successfully")
        
        print("\n🔄 Running agent loop (max 5 steps)...")
        result = orchestrator.run(ticket, max_steps=5)
        
        print("\n" + "=" * 60)
        print("AGENT EXECUTION RESULT")
        print("=" * 60)
        print(json.dumps(result, indent=2))
        print("=" * 60)
        
        # Validate result structure
        if "ticket_id" in result:
            print(f"\n✅ Ticket ID present: {result['ticket_id']}")
        
        if "agent_plan" in result:
            print(f"✅ Agent plan present with {len(result['agent_plan'])} steps:")
            for i, step in enumerate(result['agent_plan'], 1):
                print(f"   {i}. {step}")
        
        if "analysis" in result:
            print("✅ Analysis present")
            if result["analysis"].get("classification"):
                classification = result["analysis"]["classification"]
                print(f"   - Category: {classification.get('predicted_category')}")
                print(f"   - Confidence: {classification.get('confidence')}")
        
        if result.get("status") == "success":
            print("\n🎉 Agent loop test PASSED - Status: SUCCESS")
            return True
        elif result.get("status") == "partial":
            print("\n⚠️  Agent loop test PARTIAL - Some components may not be initialized")
            return True
        else:
            print("\n❌ Agent loop test FAILED - Unexpected status")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during agent execution: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_agent_basic()
    sys.exit(0 if success else 1)
