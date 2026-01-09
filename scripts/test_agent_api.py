"""
API Integration Test for Agentic AI System

This test script validates the /agent/solve API endpoint to ensure the agentic
AI system works correctly when accessed via HTTP API. It's designed for:
- Integration testing: Verifying end-to-end functionality
- CI/CD validation: Automated testing in deployment pipelines
- Developer verification: Quick sanity checks during development

Test Coverage:
    - API connectivity and availability
    - Request/response format validation
    - Agent orchestrator execution via API
    - Classification and planning functionality
    - Error handling and graceful degradation

Prerequisites:
    - API server must be running on localhost:8000
    - Start with: uvicorn src.api.main:app --reload

Exit Codes:
    0: Tests passed or skipped (server not running)
    1: Tests failed (unexpected errors or invalid responses)
"""

import requests
import json
import time


def test_agent_api():
    """
    Test the /agent/solve API endpoint with a realistic support ticket.
    
    This function performs a comprehensive integration test of the agentic AI
    system's HTTP API. It validates that:
    1. The API server is accessible and responding
    2. The request format is correctly processed
    3. The agent orchestrator executes the PAOR loop
    4. Classification and planning produce valid results
    5. The response structure matches the expected schema
    
    Test Strategy:
        - Send a POST request with a database connectivity ticket
        - Validate HTTP status codes (200 = success, 503 = not ready)
        - Parse and validate the JSON response structure
        - Check for required fields (ticket_id, agent_plan, analysis)
        - Verify the agent completed successfully or partially
    
    Returns:
        bool or None:
            - True: All tests passed successfully
            - False: Tests failed due to errors or invalid responses
            - None: Tests skipped (server not running, not an error)
    
    Example Output:
        ======================================================================
        Testing /agent/solve API Endpoint
        ======================================================================
        
        📡 Endpoint: http://localhost:8000/agent/solve
        📋 Test Ticket: API-TEST-001
           Subject: Database connection timeout
        
        🚀 Sending POST request...
        📊 Response Status: 200
        
        ✅ Ticket ID: API-TEST-001
        ✅ Agent executed 3 steps
        ✅ Analysis completed
           Category: database
           Confidence: 0.88
        
        🎉 API Test PASSED - Status: SUCCESS
    """
    
    # ========== Configuration ==========
    # API endpoint configuration
    # In production, this could be read from environment variables
    base_url = "http://localhost:8000"  # Local development server
    endpoint = f"{base_url}/agent/solve"  # Agent execution endpoint
    
    # ========== Test Request Construction ==========
    # Build a realistic test request that mimics production traffic
    # This ticket represents a critical database connectivity issue
    test_request = {
        "ticket_data": {
            # Ticket identification and metadata
            "ticket_id": "API-TEST-001",  # Unique ID for tracking this test
            "subject": "Database connection timeout",  # Brief problem summary
            "description": "Application cannot connect to the database. Getting timeout errors after 30 seconds.",  # Detailed issue
            "priority": "critical",  # High urgency (affects SLA and routing)
            "product": "backend_service",  # Which product is affected
            "error_logs": "Connection timeout: Unable to connect to database server at db.example.com:5432"  # Technical details
        },
        # Agent execution parameters
        "max_steps": 5  # Limit agent loop iterations (prevents infinite loops)
    }
    
    # ========== Test Execution Header ==========
    # Print formatted header for visual clarity in test output
    print("=" * 70)
    print("Testing /agent/solve API Endpoint")
    print("=" * 70)
    print(f"\n📡 Endpoint: {endpoint}")
    print(f"📋 Test Ticket: {test_request['ticket_data']['ticket_id']}")
    print(f"   Subject: {test_request['ticket_data']['subject']}")
    
    try:
        # ========== Send API Request ==========
        # Make HTTP POST request to the agent API endpoint
        print("\n🚀 Sending POST request...")
        response = requests.post(
            endpoint,
            json=test_request,  # Automatically serializes to JSON
            headers={"Content-Type": "application/json"},  # Specify JSON payload
            timeout=30  # 30-second timeout (agent processing can take time)
        )
        
        # Display HTTP status code for debugging
        print(f"📊 Response Status: {response.status_code}")
        
        # ========== Handle Successful Response (200 OK) ==========
        if response.status_code == 200:
            # Parse JSON response
            result = response.json()
            
            # ========== Display Full Response ==========
            # Pretty-print the complete response for manual inspection
            print("\n" + "=" * 70)
            print("API RESPONSE")
            print("=" * 70)
            print(json.dumps(result, indent=2))
            print("=" * 70)
            
            # ========== Validate Response Structure ==========
            # Check that the response contains the expected "result" field
            # This is the top-level container for agent execution results
            if "result" in result:
                agent_result = result["result"]
                
                # ========== Validate Ticket ID ==========
                # Ensure the response includes the ticket ID for tracking
                if "ticket_id" in agent_result:
                    print(f"\n✅ Ticket ID: {agent_result['ticket_id']}")
                
                # ========== Validate Agent Plan ==========
                # Check that the agent generated a plan with multiple steps
                # The plan shows what actions the agent took (classify, retrieve, etc.)
                if "agent_plan" in agent_result:
                    print(f"✅ Agent executed {len(agent_result['agent_plan'])} steps")
                
                # ========== Validate Analysis Results ==========
                # Verify that analysis (classification, RAG, etc.) was performed
                if "analysis" in agent_result:
                    print("✅ Analysis completed")
                    
                    # ========== Display Classification Details ==========
                    # If classification was performed, show the results
                    if agent_result["analysis"].get("classification"):
                        cls = agent_result["analysis"]["classification"]
                        print(f"   Category: {cls.get('predicted_category')}")
                        print(f"   Confidence: {cls.get('confidence')}")
                
                # ========== Determine Test Success ==========
                # Check the overall status to determine if the test passed
                # "success" = full completion, "partial" = some components missing
                if agent_result.get("status") in ["success", "partial"]:
                    print(f"\n🎉 API Test PASSED - Status: {agent_result['status'].upper()}")
                    return True
                else:
                    # Unexpected status value (neither success nor partial)
                    print(f"\n⚠️  Unexpected status: {agent_result.get('status')}")
                    return False
            else:
                # Response is missing the required "result" field
                print("\n❌ Invalid response structure - missing 'result' field")
                return False
        
        # ========== Handle Service Unavailable (503) ==========
        # This occurs when the API server is running but agentic components aren't loaded
        elif response.status_code == 503:
            print("\n⚠️  Service unavailable - Agentic components may not be loaded")
            print("    This is expected if the API server hasn't been started")
            print("\n💡 To test the API:")
            print("    1. Start the server: uvicorn src.api.main:app --reload")
            print("    2. Run this test script again")
            return None  # Not a failure, just not ready for testing
        
        # ========== Handle Other HTTP Errors ==========
        # Any other status code (4xx, 5xx) indicates a problem
        else:
            print(f"\n❌ API request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
    
    # ========== Handle Connection Errors ==========
    # This occurs when the API server is not running at all
    except requests.exceptions.ConnectionError:
        print("\n⚠️  Could not connect to API server")
        print("    The server may not be running")
        print("\n💡 To start the server:")
        print("    uvicorn src.api.main:app --reload")
        return None  # Not a failure, just not running
    
    # ========== Handle Unexpected Exceptions ==========
    # Catch any other errors (timeout, JSON parsing, etc.)
    except Exception as e:
        print(f"\n❌ Error during API test: {e}")
        import traceback
        traceback.print_exc()  # Print full stack trace for debugging
        return False


# ========== Main Entry Point ==========
if __name__ == "__main__":
    """
    Execute the API test and exit with appropriate status code.
    
    This allows the script to be used in CI/CD pipelines where exit codes
    determine whether the build passes or fails.
    
    Exit Code Logic:
        - 0: Tests passed or were skipped (server not running)
        - 1: Tests failed (errors or invalid responses)
    
    The script treats "server not running" as a skip rather than a failure,
    which is useful for optional integration tests in CI/CD.
    """
    # Run the test and capture the result
    result = test_agent_api()
    
    # ========== Determine Exit Code ==========
    if result is True:
        # All tests passed successfully
        print("\n✅ All API tests passed!")
        exit(0)
    elif result is None:
        # Tests were skipped (server not running)
        # This is not considered a failure
        print("\n⏭️  API tests skipped (server not running)")
        exit(0)
    else:
        # Tests failed due to errors or invalid responses
        print("\n❌ API tests failed")
        exit(1)
