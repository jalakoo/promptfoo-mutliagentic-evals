#!/usr/bin/env python3
"""
Test script for LangGraph implementation
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_langgraph_manager():
    """Test the LangGraph manager implementation"""
    try:
        from langgraph_manager import run
        
        # Test with a simple prompt
        prompt = "Describe the data from the database"
        model_name = "openai/gpt-4o-mini"
        
        print(f"Testing LangGraph with prompt: {prompt}")
        print(f"Model: {model_name}")
        
        result = run(prompt, model_name)
        
        print(f"Result: {result}")
        print("✅ LangGraph manager test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing LangGraph manager: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_langgraph_direct():
    """Test the LangGraph direct API implementation"""
    try:
        from langgraph._direct import call_api
        
        # Test with promptfoo-compatible parameters
        prompt = "What is the schema of the database?"
        options = {
            "config": {
                "model_name": "openai/gpt-4o-mini"
            }
        }
        context = {
            "user": prompt,
            "system": "You are a helpful assistant."
        }
        
        print(f"Testing LangGraph direct API with prompt: {prompt}")
        
        result = call_api(prompt, options, context)
        
        print(f"Result: {result}")
        print("✅ LangGraph direct API test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing LangGraph direct API: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Testing LangGraph Implementation")
    print("=" * 50)
    
    # Test both implementations
    manager_success = test_langgraph_manager()
    print()
    direct_success = test_langgraph_direct()
    
    print()
    print("=" * 50)
    if manager_success and direct_success:
        print("🎉 All tests passed! LangGraph implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        sys.exit(1)
