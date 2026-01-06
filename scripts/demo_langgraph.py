"""
Quick Demo: LangGraph Integration

This script provides a quick demonstration of the LangGraph-based orchestrator
with a simple example ticket. Run this to see LangGraph in action!

Usage:
    python scripts/demo_langgraph.py
"""

import sys
import os
import json

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Run a quick demo of the LangGraph orchestrator."""
    
    print("=" * 70)
    print("🚀 LangGraph Orchestrator Demo")
    print("=" * 70)
    
    # Import the LangGraph orchestrator
    try:
        from src.agentic.orchestrator_langgraph import LangGraphOrchestrator
        print("✅ Successfully imported LangGraphOrchestrator\n")
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        print("\n💡 Install dependencies: pip install langgraph langchain langchain-community")
        return
    
    # Create a sample ticket
    ticket = {
        "ticket_id": "DEMO-001",
        "subject": "Cannot login to application",
        "description": "User is unable to authenticate with correct credentials. Getting timeout error after password reset.",
        "priority": "high",
        "product": "web_application"
    }
    
    print("📋 Sample Ticket:")
    print(f"   ID: {ticket['ticket_id']}")
    print(f"   Subject: {ticket['subject']}")
    print(f"   Priority: {ticket['priority']}")
    print()
    
    # Initialize orchestrator
    print("🔧 Initializing LangGraph Orchestrator...")
    orchestrator = LangGraphOrchestrator()
    print("✅ Orchestrator ready!\n")
    
    # Run the workflow
    print("🔄 Running workflow...")
    print("-" * 70)
    result = orchestrator.run(ticket, max_steps=10)
    print("-" * 70)
    
    # Display results
    print("\n📊 Results:")
    print(f"   Status: {result.get('status')}")
    print(f"   Confidence: {result.get('confidence', 0):.2%}")
    
    # Show execution log
    messages = result.get('messages', [])
    if messages:
        print(f"\n📝 Execution Log ({len(messages)} steps):")
        for i, msg in enumerate(messages, 1):
            print(f"   {i}. {msg}")
    
    # Show classification
    classification = result.get('analysis', {}).get('classification', {})
    if classification:
        print(f"\n🏷️  Classification:")
        print(f"   Category: {classification.get('predicted_category', 'N/A')}")
        print(f"   Confidence: {classification.get('confidence', 0):.2%}")
    
    # Show solutions
    solutions = result.get('analysis', {}).get('suggested_solutions', [])
    if solutions:
        print(f"\n💡 Found {len(solutions)} solutions")
    
    # Show full result (optional)
    print("\n" + "=" * 70)
    print("Full Result (JSON):")
    print("=" * 70)
    print(json.dumps(result, indent=2))
    print("=" * 70)
    
    print("\n✨ Demo complete!")
    print("\n📚 Learn more:")
    print("   - LANGGRAPH_README.md - Complete documentation")
    print("   - LANGGRAPH_INTEGRATION_SUMMARY.md - Quick overview")
    print("   - scripts/test_langgraph_agent.py - Full test suite")


if __name__ == "__main__":
    main()
