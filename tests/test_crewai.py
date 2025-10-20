#!/usr/bin/env python3
"""
Test script for CrewAI implementation
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_crewai_manager():
    """Test the CrewAI manager implementation"""
    try:
        from crewai_manager import run
        
        # Test with a simple prompt
        prompt = "Describe the data from the database"
        model_name = "openai/gpt-4o-mini"
        
        print(f"Testing CrewAI with prompt: {prompt}")
        print(f"Model: {model_name}")
        
        result = run(prompt, model_name)
        
        print(f"Result: {result}")
        print("✅ CrewAI manager test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing CrewAI manager: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_crewai_direct():
    """Test the CrewAI direct API implementation"""
    try:
        from crewai_manager import call_api
        
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
        
        print(f"Testing CrewAI direct API with prompt: {prompt}")
        
        result = call_api(prompt, options, context)
        
        print(f"Result: {result}")
        print("✅ CrewAI direct API test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing CrewAI direct API: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_crewai_ollama():
    """Test the CrewAI implementation with Ollama model"""
    try:
        from crewai_manager import run
        
        # Test with Ollama model
        prompt = "List all nodes in the database"
        model_name = "ollama/qwen3"
        
        print(f"Testing CrewAI with Ollama model: {model_name}")
        print(f"Prompt: {prompt}")
        
        result = run(prompt, model_name)
        
        print(f"Result: {result}")
        print("✅ CrewAI Ollama test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing CrewAI with Ollama: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_crewai_llm_by_name():
    """Test the llm_by_name function"""
    try:
        from crewai_manager import llm_by_name
        
        # Test different model configurations
        test_models = [
            "openai/gpt-4o-mini",
            "ollama/qwen3",
            "sambanova/Meta-Llama-3.1-8B-Instruct"
        ]
        
        print("Testing llm_by_name function with different models:")
        
        for model in test_models:
            print(f"  Testing model: {model}")
            llm = llm_by_name(model)
            print(f"    ✅ LLM created successfully for {model}")
        
        print("✅ llm_by_name function test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing llm_by_name function: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Testing CrewAI Implementation")
    print("=" * 50)
    
    # Test all implementations
    manager_success = test_crewai_manager()
    print()
    direct_success = test_crewai_direct()
    print()
    ollama_success = test_crewai_ollama()
    print()
    llm_function_success = test_crewai_llm_by_name()
    
    print()
    print("=" * 50)
    if manager_success and direct_success and ollama_success and llm_function_success:
        print("🎉 All tests passed! CrewAI implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        sys.exit(1)
