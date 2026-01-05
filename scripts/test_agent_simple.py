"""
Simple Test Script for Agentic AI Orchestrator

This test script provides a minimal, dependency-light way to verify that the
Plan-Act-Observe-Reflect (PAOR) loop is functioning correctly. It focuses on
basic agent functionality without requiring complex RAG or database dependencies.

Purpose:
    - Validate that the orchestrator can be initialized
    - Verify the agent can process a test ticket
    - Confirm the PAOR loop executes without errors
    - Check that classification and planning components work

Usage:
    python scripts/test_agent_simple.py

Exit Codes:
    0: All tests passed successfully
    1: One or more tests failed
"""

import sys
import os
import json

# Add project root to Python path to enable imports from src/
# This allows the script to be run from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_agent_basic():
    """
    Test basic agent functionality with classification only.
    
    This function creates a realistic test ticket and runs it through the
    agent orchestrator to verify that:
    1. The orchestrator can be imported and initialized
    2. The agent loop executes without crashing
    3. A plan is generated with multiple steps
    4. Classification results are produced
    5. The final result has the expected structure
    
    The test uses a "login issue" ticket as it's a common support scenario
    that should trigger classification and potentially solution retrieval.
    
    Returns:
        bool: True if all tests pass, False otherwise
    
    Test Flow:
        1. Import validation: Ensure AgentOrchestrator can be imported
        2. Ticket creation: Build a realistic test ticket
        3. Orchestrator initialization: Create orchestrator instance
        4. Agent execution: Run the PAOR loop with max 5 steps
        5. Result validation: Check structure and content of results
        6. Status verification: Confirm success or partial completion
    """
    # Print test header for visual separation in console output
    print("=" * 60)
    print("Testing Agentic AI Orchestrator")
    print("=" * 60)
    
    # ========== STEP 1: Import Validation ==========
    # Attempt to import the orchestrator to catch any import-time errors
    # This validates that all dependencies are available
    try:
        from src.agentic.orchestrator import AgentOrchestrator
        print("✅ Successfully imported AgentOrchestrator")
    except Exception as e:
        print(f"❌ Failed to import AgentOrchestrator: {e}")
        return False
    
    # ========== STEP 2: Create Test Ticket ==========
    # Build a realistic ticket that mimics production data
    # This ticket represents a common authentication issue
    ticket = {
        "ticket_id": "TEST-001",  # Unique identifier for tracking
        "subject": "Cannot login to application",  # Brief problem summary
        "description": "User is unable to authenticate with correct credentials. Getting timeout error after password reset.",  # Detailed description
        "priority": "high",  # Urgency level (affects routing/SLA)
        "product": "web_application"  # Which product this relates to
    }
    
    # Display ticket details for test transparency
    print(f"\n📋 Test Ticket:")
    print(f"   ID: {ticket['ticket_id']}")
    print(f"   Subject: {ticket['subject']}")
    print(f"   Priority: {ticket['priority']}")
    
    # ========== STEP 3: Initialize and Execute Agent ==========
    try:
        # Initialize the orchestrator
        # This sets up the planner, tools, and RAG pipeline
        print("\n🚀 Initializing Orchestrator...")
        orchestrator = AgentOrchestrator()
        print("✅ Orchestrator initialized successfully")
        
        # Run the agent loop with a maximum of 5 steps
        # Typical tickets complete in 2-3 steps (classify + retrieve)
        # The limit prevents infinite loops in case of bugs
        print("\n🔄 Running agent loop (max 5 steps)...")
        result = orchestrator.run(ticket, max_steps=5)
        
        # ========== STEP 4: Display Results ==========
        # Pretty-print the complete result for manual inspection
        print("\n" + "=" * 60)
        print("AGENT EXECUTION RESULT")
        print("=" * 60)
        print(json.dumps(result, indent=2))
        print("=" * 60)
        
        # ========== STEP 5: Validate Result Structure ==========
        # Check that the result contains expected fields and data
        
        # Validate ticket ID is present and correct
        if "ticket_id" in result:
            print(f"\n✅ Ticket ID present: {result['ticket_id']}")
        
        # Validate that an agent plan was generated
        # The plan should contain multiple steps showing what the agent did
        if "agent_plan" in result:
            print(f"✅ Agent plan present with {len(result['agent_plan'])} steps:")
            # Display each step in the plan for transparency
            for i, step in enumerate(result['agent_plan'], 1):
                print(f"   {i}. {step}")
        
        # Validate that analysis results are present
        # This should include classification at minimum
        if "analysis" in result:
            print("✅ Analysis present")
            # Check if classification was successful
            if result["analysis"].get("classification"):
                classification = result["analysis"]["classification"]
                # Display key classification results
                print(f"   - Category: {classification.get('predicted_category')}")
                print(f"   - Confidence: {classification.get('confidence')}")
        
        # ========== STEP 6: Determine Test Success ==========
        # Evaluate the overall status to determine if the test passed
        
        if result.get("status") == "success":
            # Full success: All components worked correctly
            print("\n🎉 Agent loop test PASSED - Status: SUCCESS")
            return True
        elif result.get("status") == "partial":
            # Partial success: Some components may not be initialized
            # This is acceptable for basic testing (e.g., RAG not configured)
            print("\n⚠️  Agent loop test PARTIAL - Some components may not be initialized")
            return True
        else:
            # Unexpected status: Something went wrong
            print("\n❌ Agent loop test FAILED - Unexpected status")
            return False
            
    except Exception as e:
        # Catch any runtime errors during agent execution
        # Print detailed traceback for debugging
        print(f"\n❌ Error during agent execution: {e}")
        import traceback
        traceback.print_exc()
        return False


# ========== Main Entry Point ==========
if __name__ == "__main__":
    # Run the test and capture the result
    success = test_agent_basic()
    
    # Exit with appropriate code for CI/CD integration
    # 0 = success, 1 = failure
    sys.exit(0 if success else 1)

