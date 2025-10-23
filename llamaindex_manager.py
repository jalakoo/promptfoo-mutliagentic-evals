import logging
import os
import asyncio
from typing import Dict, Any, List
from dotenv import load_dotenv
from shared_utils import get_model_name_from_config

# LlamaIndex imports
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.groq import Groq
from llama_index.llms.sambanovasystems import SambaNovaCloud
from llama_index.core import Settings
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

# Configure logging to write to error.log
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="./logs/debug_llamaindex_manager.log",
    filemode="w",  # overwrite the file on each run
)
logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()

# Global variable to store the last Cypher query used
_last_cypher_query = ""


def wrap_tools_with_cypher_tracking(tools):
    """Wrap MCP tools to capture Cypher queries"""
    global _last_cypher_query
    
    wrapped_tools = []
    for tool in tools:
        if hasattr(tool, 'metadata') and tool.metadata.name and 'cypher' in tool.metadata.name.lower():
            logger.info(f"🔧 Wrapping tool: {tool.metadata.name}")
            
            # Wrap multiple possible call methods
            methods_to_wrap = ['call', '_call', 'ainvoke', '_ainvoke', 'run', '_run']
            
            for method_name in methods_to_wrap:
                if hasattr(tool, method_name):
                    original_method = getattr(tool, method_name)
                    
                    def create_wrapper(orig_method, method_name):
                        def cypher_wrapper(*args, **kwargs):
                            global _last_cypher_query
                            
                            logger.info(f"🔍 Tool {tool.metadata.name} called via {method_name} with args: {args}, kwargs: {kwargs}")
                            
                            # Extract query from various argument patterns
                            query_found = False
                            
                            # Check kwargs
                            if 'query' in kwargs:
                                _last_cypher_query = kwargs['query']
                                logger.info(f"🔍 Captured Cypher query from kwargs: {_last_cypher_query}")
                                query_found = True
                            
                            # Check first argument if it's a dict
                            elif args and len(args) > 0 and isinstance(args[0], dict) and 'query' in args[0]:
                                _last_cypher_query = args[0]['query']
                                logger.info(f"🔍 Captured Cypher query from args[0]: {_last_cypher_query}")
                                query_found = True
                            
                            # Check all args for dict with query
                            elif args:
                                for i, arg in enumerate(args):
                                    if isinstance(arg, dict) and 'query' in arg:
                                        _last_cypher_query = arg['query']
                                        logger.info(f"🔍 Captured Cypher query from args[{i}]: {_last_cypher_query}")
                                        query_found = True
                                        break
                            
                            if not query_found:
                                logger.debug(f"🔍 No query found in {method_name} call for {tool.metadata.name}")
                            
                            # Call the original method
                            return orig_method(*args, **kwargs)
                        
                        return cypher_wrapper
                    
                    # Replace the method
                    setattr(tool, method_name, create_wrapper(original_method, method_name))
                    logger.info(f"✅ Wrapped {method_name} for {tool.metadata.name}")
        
        wrapped_tools.append(tool)
    
    return wrapped_tools


def llm_by_name(name: str = "openai/gpt-4o-mini"):
    """Create LLM instance based on model name with explicit validation"""
    logger.info(f"🔧 Creating LLM for model: {name}")
    
    # Extract prefix and model name
    if "/" in name:
        prefix, model_name = name.split("/", 1)
    else:
        prefix = "openai"  # default prefix
        model_name = name
    
    logger.info(f"   Parsed prefix: {prefix}, model: {model_name}")
    
    # Validate supported model providers
    if prefix == "ollama":
        logger.info(f"🦙 Using Ollama model: {model_name}")
        
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        timeout = os.getenv("OLLAMA_TIMEOUT", 120)
        request_timeout = os.getenv("OLLAMA_REQUEST_TIMEOUT", 180)

        logger.info(f"   Base URL: {base_url}")
        logger.info(f"   Timeout: {timeout} seconds")
        logger.info(f"   Request Timeout: {request_timeout} seconds")
        
        try:
            llm = Ollama(
                model=model_name,
                temperature=0,
                base_url=base_url,
                timeout=timeout,
                request_timeout=request_timeout
            )
            logger.info(f"   ✅ Ollama LLM created successfully")
            return llm
        except Exception as e:
            logger.error(f"   ❌ Failed to create Ollama LLM: {e}")
            raise
            
    elif prefix == "openai":
        logger.info(f"🤖 Using OpenAI model: {model_name}")
        api_key = os.getenv("OPENAI_API_KEY")
        logger.info(f"   API Key present: {'Yes' if api_key else 'No'}")
        
        try:
            llm = OpenAI(
                model=model_name,
                temperature=0,
                api_key=api_key
            )
            logger.info(f"   ✅ OpenAI LLM created successfully")
            return llm
        except Exception as e:
            logger.error(f"   ❌ Failed to create OpenAI LLM: {e}")
            raise
            
    elif prefix == "anthropic":
        logger.info(f"🧠 Using Anthropic model: {model_name}")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        logger.info(f"   API Key present: {'Yes' if api_key else 'No'}")
        
        try:
            llm = Anthropic(
                model=model_name,
                temperature=0,
                api_key=api_key
            )
            logger.info(f"   ✅ Anthropic LLM created successfully")
            return llm
        except Exception as e:
            logger.error(f"   ❌ Failed to create Anthropic LLM: {e}")
            raise
            
    elif prefix == "groq":
        logger.info(f"🤖 Using Groq model: {model_name}")
        api_key = os.getenv("GROQ_API_KEY")
        logger.info(f"   API Key present: {'Yes' if api_key else 'No'}")
        
        try:
            llm = Groq(model=model_name, api_key=api_key)
            logger.info(f"   ✅ Groq LLM created successfully")
            return llm
        except Exception as e:
            logger.error(f"   ❌ Failed to create Groq LLM: {e}")
            raise
    elif prefix == "sambanova":
        logger.info(f"   Creating SambaNova LLM with model: {model_name}")
        
        llm = SambaNovaCloud(
            model=model_name,
            context_window=100000,
            max_tokens=1024,
            temperature=0.7,
            top_k=1,
            top_p=0.01,
        )
        logger.info(f"   ✅ SambaNova LLM created successfully")
        return llm
    else:
        logger.error(f"Unsupported model: {name}")
        raise ValueError(f"Unsupported model: {name}. Supported providers: ollama, openai, anthropic")


def create_llamaindex_agent(tools: Any, llm_name: str, llm=None):
    """Create a LlamaIndex FunctionAgent with Neo4j MCP tools"""
    logger.info(f"Creating LlamaIndex agent for model: {llm_name}")
    logger.info(f"   Tools available: {len(tools)}")
    logger.info(f"   Tool names: {[tool.metadata.name for tool in tools]}")
    
    try:
        # Use provided LLM or create new one
        if llm is None:
            llm = llm_by_name(llm_name)
            logger.info(f"✅ LLM created: {type(llm).__name__}")
        else:
            logger.info(f"✅ Using provided LLM: {type(llm).__name__}")
        
        logger.info(f"   Model: {llm.model if hasattr(llm, 'model') else 'unknown'}")

        # Switch statement to choose agent type based on LLM provider
        if "sambanova/" in llm_name:
            # Use ReActAgent for SambaNova Cloud (doesn't work with FunctionAgent)
            # Anthropic can also use this agent type
            logger.info("   Using LlamaIndex ReActAgent")
            agent = ReActAgent(
                streaming=False,
                tools=tools,
                llm=llm,
                verbose=os.getenv("LLAMAINDEX_VERBOSE", "0") in ("1", "true", "True"),
                system_prompt="""You are an expert data analyst proficient in Cypher and Neo4j.

You have access to these tools:
1. get_neo4j_schema: Get the database schema to understand node types and relationships
2. read_neo4j_cypher: Execute read-only Cypher queries to retrieve data
3. write_neo4j_cypher: Execute write Cypher queries to modify data (use with caution)

For questions about data:
- First use get_neo4j_schema to understand the database structure
- Then use read_neo4j_cypher with appropriate Cypher queries to answer the question
- For "how many" questions, use: MATCH (n) RETURN count(n) as node_count
- Always provide the final answer clearly

Return ONLY the final answer for the user, not internal reasoning.""",
            )
            logger.info("   ✅ ReActAgent created successfully")
        else:
            # Use FunctionAgent for other LLMs (requires FunctionCallingLLM)
            logger.info("   Using LlamaIndex FunctionAgent")
            agent = FunctionAgent(
                tools=tools,
                llm=llm,
                verbose=os.getenv("LLAMAINDEX_VERBOSE", "0") in ("1", "true", "True"),
                system_prompt="""You are an expert data analyst proficient in Cypher and Neo4j.

You have access to these tools:
1. get_neo4j_schema: Get the database schema to understand node types and relationships
2. read_neo4j_cypher: Execute read-only Cypher queries to retrieve data
3. write_neo4j_cypher: Execute write Cypher queries to modify data (use with caution)

For questions about data:
- First use get_neo4j_schema to understand the database structure
- Then use read_neo4j_cypher with appropriate Cypher queries to answer the question
- For "how many" questions, use: MATCH (n) RETURN count(n) as node_count
- Always provide the final answer clearly

Return ONLY the final answer for the user, not internal reasoning.""",
            )
            logger.info("   ✅ FunctionAgent created successfully")
        
        # Add callback to monitor tool usage
        if hasattr(agent, 'callback_manager'):
            logger.info("   Adding callback manager for tool monitoring")
            # The callback manager will help us track tool calls
        else:
            logger.info("   Agent doesn't have callback_manager, using wrapper approach")

        logger.info(f"✅ Agent created: {type(agent).__name__}")
        return agent
        
    except Exception as e:
        logger.error(f"❌ Failed to create agent: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        logger.error(f"   Model: {llm_name}")
        logger.error(f"   Tools count: {len(tools) if tools else 'None'}")
        raise


async def run(prompt: str, full_model_name: str):
    """Run the LlamaIndex agent with the given prompt and model"""
    global _last_cypher_query
    
    # Reset the Cypher query tracker
    _last_cypher_query = ""
    
    try:
        # Step 1: Create MCP client with enhanced logging
        logger.info("Step 1: Creating MCP client...")
        client = BasicMCPClient("uvx", args=["mcp-neo4j-cypher@0.3.0"], env=os.environ)
        
        # Add logging to the client's call method if it exists
        if hasattr(client, 'call_tool'):
            original_call_tool = client.call_tool
            
            def logged_call_tool(tool_name, arguments):
                global _last_cypher_query
                
                logger.info(f"🔍 MCP Client calling tool: {tool_name} with args: {arguments}")
                
                # Extract Cypher query if this is a Cypher tool
                if 'cypher' in tool_name.lower() and isinstance(arguments, dict) and 'query' in arguments:
                    _last_cypher_query = arguments['query']
                    logger.info(f"🔍 Captured Cypher query from MCP client: {_last_cypher_query}")
                
                return original_call_tool(tool_name, arguments)
            
            client.call_tool = logged_call_tool
            logger.info("✅ Added logging to MCP client call_tool method")
        
        logger.info("✅ Created MCP client successfully")

        # Step 2: Create MCP tool spec
        logger.info("Step 2: Creating MCP tool spec...")
        mcp_tool_spec = McpToolSpec(client=client)
        logger.info("✅ Created MCP tool spec successfully")

        # Step 3: Get tools and wrap them for Cypher tracking
        logger.info("Step 3: Loading MCP tools...")
        tools = await mcp_tool_spec.to_tool_list_async()
        tools = wrap_tools_with_cypher_tracking(tools)
        logger.info(f"✅ Loaded {len(tools)} MCP tools: {[tool.metadata.name for tool in tools]}")
        
        # Step 4: Create LLM instance
        logger.info("Step 4: Creating LLM instance...")
        llm = llm_by_name(full_model_name)
        logger.info(f"✅ Created LLM: {type(llm).__name__}")
        logger.info(f"   Model: {llm.model if hasattr(llm, 'model') else 'unknown'}")
        logger.info(f"   Base URL: {getattr(llm, 'base_url', 'N/A')}")
        logger.info(f"   Timeout: {getattr(llm, 'timeout', 'N/A')}")
        
        # Step 5: Test LLM connectivity (for Ollama)
        if "ollama" in full_model_name.lower():
            logger.info("Step 5: Testing Ollama connectivity...")
            try:
                # Test a simple completion to verify Ollama is working
                # Use a very short prompt to minimize load time
                test_response = await llm.acomplete("Hi")
                logger.info(f"✅ Ollama test successful: {str(test_response)[:100]}...")
            except Exception as ollama_error:
                logger.error(f"❌ Ollama connectivity test failed: {ollama_error}")
                logger.error(f"   This might indicate Ollama server is not running or model not available")
                logger.error(f"   Try: ollama pull {full_model_name.split('/')[1]}")
                logger.error(f"   Or check: ollama list")
                return f"Ollama connectivity error: {ollama_error}"

        # Step 6: Create agent (pass the already-created LLM)
        logger.info("Step 6: Creating agent...")
        agent = create_llamaindex_agent(tools, full_model_name, llm)
        logger.info(f"✅ Created agent: {type(agent).__name__}")
        
        # Step 7: Run agent
        logger.info("Step 7: Running agent with prompt...")
        logger.info(f"   Prompt: {prompt}")
        logger.info(f"   Agent type: {type(agent).__name__}")
        logger.info(f"   Tools available: {len(tools)}")
        
        response = await agent.run(prompt)
        logger.info(f"✅ Agent response received: {str(response)[:200]}...")
        return str(response)

    except Exception as e:
        logger.error(f"❌ Error in run: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        logger.error(f"   Model: {full_model_name}")
        
        # Additional error context
        if "ollama" in full_model_name.lower():
            logger.error("   This appears to be an Ollama-related error")
            logger.error("   Common Ollama issues:")
            logger.error("   - Ollama server not running (check: ollama list)")
            logger.error("   - Model not available (check: ollama list)")
            logger.error("   - Model not pulled (run: ollama pull <model>)")
            logger.error("   - Port 11434 not accessible")
            logger.error("   - Model loading timeout (try smaller model or increase timeout)")
            logger.error("   - Hardware limitations (CPU/GPU performance)")
            
            # Specific timeout error guidance
            if "ReadTimeout" in str(type(e)):
                logger.error("   TIMEOUT-SPECIFIC SOLUTIONS:")
                logger.error("   - Model is too large for your hardware")
                logger.error("   - Try: ollama pull llama3.2:1b (smaller model)")
                logger.error("   - Increase system resources (RAM/CPU)")
                logger.error("   - Check: ollama ps (model memory usage)")
        
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return f"LLM error: {e}"


# Required by promptfoo
def call_api(
    prompt: str,
    options: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, str]:
    """Call the LlamaIndex agent with the given prompt and options.
    Args:
        prompt (str): The prompt to send to the LlamaIndex agent.
        options (Dict[str, Any]): Options for the API call, from the 'config' arg.
        context (Dict[str, Any]): Prompt data for the API call - user & system prompts.
    Returns:
        Dict[str, str]: The response from the LlamaIndex agent.
    """

    logger.info(f"\n\n================================================")
    logger.info(f"Running LlamaIndex with prompt: {prompt} and context: {context}")
    logger.info(f"================================================\n\n")
    logger.debug(f"call_api: prompt: {prompt}")
    logger.debug(f"call_api: options: {options}")
    logger.debug(f"call_api: context: {context}")

    try:
        # Get model name using shared utility
        model_name = get_model_name_from_config(options, context)
        
        logger.info(f"🔧 Running LlamaIndex with model: {model_name}")
        result = asyncio.run(run(prompt, model_name))
        
        # Append Cypher query if one was captured
        result_text = str(result)
        if _last_cypher_query:
            result_text += f"\n\nCypher used: {_last_cypher_query}"
        
        # Normalize to Promptfoo provider response shape
        return {"output": result_text}
    except Exception as e:
        # Can't use print here when promptfoo running
        # Uncertain what promptfoos own logger name is, if even using
        error_msg = f"Error in call_api: {str(e)}"
        logging.error(error_msg, exc_info=True)  # This will log the full traceback
        return {"output": f"Error: {str(e)}"}


# For running as a script
if __name__ == "__main__":
    # Test with a simple prompt
    # llm_name = "openai/gpt-4o-mini"
    llm_name = "ollama/llama3.2"
    read_command = "Describe the data from the database"
    result = asyncio.run(run(read_command, llm_name))

    print(
        f"""
        Query completed!
        result: {result}
    """
    )