#!/usr/bin/env python3
"""
Test script for LlamaIndex implementation
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_llamaindex_manager():
    """Test the LlamaIndex manager implementation"""
    try:
        from llamaindex_manager import run
        
        # Test with a simple prompt
        prompt = "Describe the data from the database"
        model_name = "openai/gpt-4o-mini"
        
        print(f"Testing LlamaIndex with prompt: {prompt}")
        print(f"Model: {model_name}")
        
        result = run(prompt, model_name)
        
        print(f"Result: {result}")
        print("✅ LlamaIndex manager test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing LlamaIndex manager: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llamaindex_direct():
    """Test the LlamaIndex direct API implementation"""
    try:
        from llamaindex_manager import call_api
        
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
        
        print(f"Testing LlamaIndex direct API with prompt: {prompt}")
        
        result = call_api(prompt, options, context)
        
        print(f"Result: {result}")
        print("✅ LlamaIndex direct API test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing LlamaIndex direct API: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Testing LlamaIndex Implementation")
    print("=" * 50)
    
    # Test both implementations
    manager_success = test_llamaindex_manager()
    print()
    direct_success = test_llamaindex_direct()
    
    print()
    print("=" * 50)
    if manager_success and direct_success:
        print("🎉 All tests passed! LlamaIndex implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        sys.exit(1)
