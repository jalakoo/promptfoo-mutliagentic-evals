#!/usr/bin/env python3
"""
Test script to verify Cypher query tracking in all three managers
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_cypher_tracking():
    """Test that all managers capture and return Cypher queries"""
    print("🧪 Testing Cypher Query Tracking")
    print("=" * 50)
    
    test_prompt = "How many nodes are in the database?"
    
    # Test CrewAI
    try:
        print("\n1️⃣ Testing CrewAI Manager...")
        import crewai_manager
        result = crewai_manager.run(test_prompt, "openai/gpt-4o-mini")
        
        if "Cypher used:" in result:
            print("✅ CrewAI: Cypher query captured and returned")
            print(f"   Result preview: {result[:100]}...")
        else:
            print("❌ CrewAI: No Cypher query found in result")
            print(f"   Result: {result}")
            
    except Exception as e:
        print(f"❌ CrewAI test failed: {e}")
    
    # Test LangGraph
    try:
        print("\n2️⃣ Testing LangGraph Manager...")
        import langgraph_manager
        result = langgraph_manager.run(test_prompt, "openai/gpt-4o-mini")
        
        if "Cypher used:" in result:
            print("✅ LangGraph: Cypher query captured and returned")
            print(f"   Result preview: {result[:100]}...")
        else:
            print("❌ LangGraph: No Cypher query found in result")
            print(f"   Result: {result}")
            
    except Exception as e:
        print(f"❌ LangGraph test failed: {e}")
    
    # Test LlamaIndex
    try:
        print("\n3️⃣ Testing LlamaIndex Manager...")
        import llamaindex_manager
        import asyncio
        result = asyncio.run(llamaindex_manager.run(test_prompt, "openai/gpt-4o-mini"))
        
        if "Cypher used:" in result:
            print("✅ LlamaIndex: Cypher query captured and returned")
            print(f"   Result preview: {result[:100]}...")
        else:
            print("❌ LlamaIndex: No Cypher query found in result")
            print(f"   Result: {result}")
            
    except Exception as e:
        print(f"❌ LlamaIndex test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Cypher Query Tracking Test Complete")
    print("📝 All managers should now append 'Cypher used: [query]' to their outputs")


if __name__ == "__main__":
    test_cypher_tracking()
