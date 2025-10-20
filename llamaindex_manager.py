import logging
import os
import asyncio
from typing import Dict, Any, List
from dotenv import load_dotenv

# LlamaIndex imports
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow import ReActAgent

from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.sambanovasystems import SambaNovaCloud
from llama_index.core import Settings
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

# Configure logging to write to error.log
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="./logs/debug_llamaindex_manager.log",
    filemode="w",  # append to the file if it exists
)
logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()


def llm_by_name(name: str = "openai/gpt-4o-mini"):
    """Create LLM instance based on model name"""
    logger.info(f"Creating LLM for name: {name}")
    
    # Extract prefix and model name
    if "/" in name:
        prefix, model_name = name.split("/", 1)
    else:
        prefix = "openai"  # default prefix
        model_name = name
    
    logger.info(f"   Parsed prefix: {prefix}, model: {model_name}")
    
    try:
        if prefix == "ollama":

            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            timeout = os.getenv("OLLAMA_TIMEOUT", 120)
            request_timeout = os.getenv("OLLAMA_REQUEST_TIMEOUT", 180)

            logger.info(f"   Creating Ollama LLM with model: {model_name}")
            logger.info(f"   Base URL: {base_url}")
            logger.info(f"   Timeout: {timeout} seconds")
            logger.info(f"   Request Timeout: {request_timeout} seconds")
            
            llm = Ollama(
                model=model_name,
                temperature=0,
                base_url=base_url,
                timeout=timeout,
                request_timeout=request_timeout
            )
            logger.info(f"   ✅ Ollama LLM created successfully")
            return llm
            
        elif prefix == "anthropic":
            logger.info(f"   Creating Anthropic LLM with model: {model_name}")
            api_key = os.getenv("ANTHROPIC_API_KEY")
            logger.info(f"   API Key present: {'Yes' if api_key else 'No'}")
            
            llm = Anthropic(
                model=model_name,
                temperature=0,
                api_key=api_key
            )
            logger.info(f"   ✅ Anthropic LLM created successfully")
            return llm
            
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
            # Default to OpenAI - many LLMs conform to this interface
            logger.info(f"   Creating OpenAI LLM with model: {model_name}")
            api_key = os.getenv("OPENAI_API_KEY")
            logger.info(f"   API Key present: {'Yes' if api_key else 'No'}")
            
            llm = OpenAI(
                model=model_name,
                temperature=0,
                api_key=api_key
            )
            logger.info(f"   ✅ OpenAI LLM created successfully")
            return llm
            
    except Exception as e:
        logger.error(f"   ❌ Failed to create LLM: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        logger.error(f"   Prefix: {prefix}, Model: {model_name}")
        raise


def create_llamaindex_agent(tools: Any, llm_name: str):
    """Create a LlamaIndex FunctionAgent with Neo4j MCP tools"""
    logger.info(f"Creating LlamaIndex agent for model: {llm_name}")
    logger.info(f"   Tools available: {len(tools)}")
    logger.info(f"   Tool names: {[tool.metadata.name for tool in tools]}")
    
    try:
        llm = llm_by_name(llm_name)
        logger.info(f"✅ LLM created: {type(llm).__name__}")
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
    logger.info(f"Running LlamaIndex with model: {full_model_name} and prompt: {prompt}")
    
    try:
        # Step 1: Create MCP client
        logger.info("Step 1: Creating MCP client...")
        client = BasicMCPClient("uvx", args=["mcp-neo4j-cypher@0.3.0"], env=os.environ)
        logger.info("✅ Created MCP client successfully")

        # Step 2: Create MCP tool spec
        logger.info("Step 2: Creating MCP tool spec...")
        mcp_tool_spec = McpToolSpec(client=client)
        logger.info("✅ Created MCP tool spec successfully")

        # Step 3: Get tools
        logger.info("Step 3: Loading MCP tools...")
        tools = await mcp_tool_spec.to_tool_list_async()
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

        # Step 6: Create agent
        logger.info("Step 6: Creating agent...")
        agent = create_llamaindex_agent(tools, full_model_name)
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

    logger.debug(f"call_api: prompt: {prompt}")
    logger.debug(f"call_api: options: {options}")
    logger.debug(f"call_api: context: {context}")

    try:
        model_name = options["config"]["model_name"]
        result = asyncio.run(run(prompt, model_name))
        # Normalize to Promptfoo provider response shape
        return {"output": result}
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