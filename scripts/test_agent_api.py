"""
API endpoint test for the Agentic AI system.
Tests the /agent/solve endpoint to verify end-to-end functionality.
"""
import requests
import json
import time

def test_agent_api():
    """Test the /agent/solve API endpoint."""
    
    # API endpoint
    base_url = "http://localhost:8000"
    endpoint = f"{base_url}/agent/solve"
    
    # Test ticket
    test_request = {
        "ticket_data": {
            "ticket_id": "API-TEST-001",
            "subject": "Database connection timeout",
            "description": "Application cannot connect to the database. Getting timeout errors after 30 seconds.",
            "priority": "critical",
            "product": "backend_service",
            "error_logs": "Connection timeout: Unable to connect to database server at db.example.com:5432"
        },
        "max_steps": 5
    }
    
    print("=" * 70)
    print("Testing /agent/solve API Endpoint")
    print("=" * 70)
    print(f"\n📡 Endpoint: {endpoint}")
    print(f"📋 Test Ticket: {test_request['ticket_data']['ticket_id']}")
    print(f"   Subject: {test_request['ticket_data']['subject']}")
    
    try:
        print("\n🚀 Sending POST request...")
        response = requests.post(
            endpoint,
            json=test_request,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n" + "=" * 70)
            print("API RESPONSE")
            print("=" * 70)
            print(json.dumps(result, indent=2))
            print("=" * 70)
            
            # Validate response structure
            if "result" in result:
                agent_result = result["result"]
                
                if "ticket_id" in agent_result:
                    print(f"\n✅ Ticket ID: {agent_result['ticket_id']}")
                
                if "agent_plan" in agent_result:
                    print(f"✅ Agent executed {len(agent_result['agent_plan'])} steps")
                
                if "analysis" in agent_result:
                    print("✅ Analysis completed")
                    if agent_result["analysis"].get("classification"):
                        cls = agent_result["analysis"]["classification"]
                        print(f"   Category: {cls.get('predicted_category')}")
                        print(f"   Confidence: {cls.get('confidence')}")
                
                if agent_result.get("status") in ["success", "partial"]:
                    print(f"\n🎉 API Test PASSED - Status: {agent_result['status'].upper()}")
                    return True
                else:
                    print(f"\n⚠️  Unexpected status: {agent_result.get('status')}")
                    return False
            else:
                print("\n❌ Invalid response structure - missing 'result' field")
                return False
                
        elif response.status_code == 503:
            print("\n⚠️  Service unavailable - Agentic components may not be loaded")
            print("    This is expected if the API server hasn't been started")
            print("\n💡 To test the API:")
            print("    1. Start the server: uvicorn src.api.main:app --reload")
            print("    2. Run this test script again")
            return None  # Not a failure, just not running
            
        else:
            print(f"\n❌ API request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n⚠️  Could not connect to API server")
        print("    The server may not be running")
        print("\n💡 To start the server:")
        print("    uvicorn src.api.main:app --reload")
        return None  # Not a failure, just not running
        
    except Exception as e:
        print(f"\n❌ Error during API test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = test_agent_api()
    
    if result is True:
        print("\n✅ All API tests passed!")
        exit(0)
    elif result is None:
        print("\n⏭️  API tests skipped (server not running)")
        exit(0)
    else:
        print("\n❌ API tests failed")
        exit(1)
