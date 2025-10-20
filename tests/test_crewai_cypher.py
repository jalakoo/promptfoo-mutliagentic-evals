#!/usr/bin/env python3
"""
Test script to verify CrewAI Cypher query tracking
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_crewai_cypher_tracking():
    """Test that CrewAI captures and returns Cypher queries"""
    print("🧪 Testing CrewAI Cypher Query Tracking")
    print("=" * 50)
    
    test_prompt = "How many nodes are in the database?"
    
    try:
        print(f"\n🔍 Testing with prompt: {test_prompt}")
        import crewai_manager
        
        # Test the call_api function directly
        options = {
            "config": {
                "model_name": "ollama/llama3.2"
            }
        }
        context = {
            "vars": {"question": test_prompt},
            "prompt": {"raw": "{{question}}", "label": "{{question}}"},
            "filters": {}
        }
        
        result = crewai_manager.call_api(test_prompt, options, context)
        
        print(f"\n📊 Result:")
        print(f"Output: {result.get('output', 'No output')}")
        
        if "Cypher used:" in result.get('output', ''):
            print("✅ SUCCESS: Cypher query captured and returned!")
        else:
            print("❌ FAILED: No Cypher query found in result")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_crewai_cypher_tracking()
