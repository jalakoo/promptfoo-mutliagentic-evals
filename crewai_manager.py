# OPTIONAL -------------------------------
# Using ollama - by default OpenAI is used
# Remove / comment this block if using OpenAI
# from crewai import LLM
# from langchain_community.chat_models import ChatOpenAI

# llm = ChatOpenAI(
#     model="ollama/mixtral:latest",
#     base_url="http://localhost:11434",
#     streaming=True
# )
# ----------------------------------------

from crewai import Agent, Task, Crew, LLM, Process
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
from dotenv import load_dotenv
import os
import atexit
import threading

import logging
from typing import Dict, Any
from shared_utils import get_model_name_from_config

# Configure logging to write to error.log
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="./logs/debug_crewai_manager.log",
    filemode="w",  # append to the file if it exists
)
logger = logging.getLogger(__name__)


# Load .env file
load_dotenv()

# Store generated crews by model name
crews = {}

# Global variables to track Cypher queries and responses
_last_cypher_query = ""
_last_raw_response = ""

# Create a StdioServerParameters object
server_params = [
    StdioServerParameters(
        command="uvx",
        args=["mcp-neo4j-cypher@0.3.0"],
        env=os.environ,
    ),
    # StdioServerParameters(
    #     command="uvx",
    #     args=["mcp-neo4j-data-modeling"],
    #     env=os.environ,
    # )
]

# Singleton MCP adapter/tools
_mcp_adapter = None
_tools = None
_tools_lock = threading.Lock()


def wrap_tools_with_cypher_tracking(tools):
    """Wrap MCP tools to capture Cypher queries"""
    global _last_cypher_query
    
    wrapped_tools = []
    for tool in tools:
        if hasattr(tool, 'name') and 'cypher' in tool.name.lower():
            logger.info(f"🔧 Wrapping Cypher tool: {tool.name}")
            
            # Wrap the tool's call method
            original_call = getattr(tool, 'call', None)
            if original_call:
                def create_wrapper(orig_call, tool_name):
                    def cypher_wrapper(*args, **kwargs):
                        global _last_cypher_query
                        
                        logger.info(f"🔍 Tool {tool_name} called with args: {args}, kwargs: {kwargs}")
                        
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
                            logger.debug(f"🔍 No query found in call for {tool_name}")
                        
                        # Call the original method
                        return orig_call(*args, **kwargs)
                    
                    return cypher_wrapper
                
                # Replace the call method
                tool.call = create_wrapper(original_call, tool.name)
                logger.info(f"✅ Wrapped call method for {tool.name}")
        
        wrapped_tools.append(tool)
    
    return wrapped_tools


def _ensure_tools():
    global _mcp_adapter, _tools
    if _tools is None:
        with _tools_lock:
            if _tools is None:
                # Persist the adapter across calls
                _mcp_adapter = MCPServerAdapter(server_params)
                tools = _mcp_adapter.__enter__()
                # Wrap tools to capture Cypher queries
                tools = wrap_tools_with_cypher_tracking(tools)
                # Ensure clean shutdown on process exit
                atexit.register(
                    lambda: _mcp_adapter and _mcp_adapter.__exit__(None, None, None)
                )
                _tools = tools
    return _tools


# Enhanced logging callbacks with Cypher query tracking
def log_step_callback(output):
    global _last_cypher_query, _last_raw_response
    
    logger.info(f"🔍 Step completed - Output type: {type(output)}")
    logger.debug(f"Step output details: {output.__dict__}")
    
    # Try to extract Cypher queries and responses from step output
    try:
        if hasattr(output, 'raw') and output.raw:
            logger.info(f"🔍 Raw step response: {str(output.raw)[:500]}...")
            _last_raw_response = str(output.raw)
        
        if hasattr(output, 'content') and output.content:
            content = str(output.content)
            logger.info(f"🔍 Step content: {content[:500]}...")
            
            # Look for Cypher queries in the content
            if any(keyword in content.upper() for keyword in ['MATCH', 'CREATE', 'MERGE', 'DELETE', 'SET', 'RETURN']):
                logger.info(f"🔍 Potential Cypher query found in step content: {content}")
                _last_cypher_query = content
        
        # Check for tool calls in the output
        if hasattr(output, 'tool_calls') and output.tool_calls:
            for tool_call in output.tool_calls:
                if hasattr(tool_call, 'name') and 'cypher' in tool_call.name.lower():
                    logger.info(f"🔍 Cypher tool call detected: {tool_call.name}")
                    if hasattr(tool_call, 'args') and 'query' in tool_call.args:
                        _last_cypher_query = tool_call.args['query']
                        logger.info(f"🔍 Captured Cypher query from tool call: {_last_cypher_query}")
        
        # Check for CrewAI ToolResult objects (this is where the actual tool calls happen)
        if hasattr(output, 'tool') and hasattr(output, 'tool_input'):
            tool_name = output.tool
            tool_input = output.tool_input
            logger.info(f"🔍 CrewAI tool call detected: {tool_name}")
            logger.info(f"🔍 Tool input: {tool_input}")
            
            # Check if this is a Cypher tool call
            if 'cypher' in tool_name.lower():
                # Try to extract query from tool_input
                if isinstance(tool_input, str):
                    try:
                        import json
                        tool_input_dict = json.loads(tool_input)
                        if 'query' in tool_input_dict:
                            _last_cypher_query = tool_input_dict['query']
                            logger.info(f"🔍 Captured Cypher query from CrewAI tool call: {_last_cypher_query}")
                    except:
                        # If not JSON, check if it contains Cypher keywords
                        if any(keyword in tool_input.upper() for keyword in ['MATCH', 'CREATE', 'MERGE', 'DELETE', 'SET', 'RETURN']):
                            _last_cypher_query = tool_input
                            logger.info(f"🔍 Captured Cypher query from CrewAI tool input: {_last_cypher_query}")
                elif isinstance(tool_input, dict) and 'query' in tool_input:
                    _last_cypher_query = tool_input['query']
                    logger.info(f"🔍 Captured Cypher query from CrewAI tool dict: {_last_cypher_query}")
        
        # Also check the text field for any Cypher queries that might be mentioned
        if hasattr(output, 'text') and output.text:
            text_content = str(output.text)
            logger.info(f"🔍 Step text content: {text_content[:500]}...")
            
            # Look for Cypher queries in the text (LLM reasoning)
            if any(keyword in text_content.upper() for keyword in ['MATCH', 'CREATE', 'MERGE', 'DELETE', 'SET', 'RETURN']):
                logger.info(f"🔍 Potential Cypher query found in step text: {text_content}")
                # Extract just the Cypher query part if possible
                lines = text_content.split('\n')
                for line in lines:
                    if any(keyword in line.upper() for keyword in ['MATCH', 'CREATE', 'MERGE', 'DELETE', 'SET', 'RETURN']):
                        _last_cypher_query = line.strip()
                        logger.info(f"🔍 Captured Cypher query from step text: {_last_cypher_query}")
                        break
                        
    except Exception as e:
        logger.debug(f"Error extracting Cypher info from step: {e}")


def log_task_callback(output):
    global _last_cypher_query, _last_raw_response
    
    logger.info(f"✅ Task completed - Output type: {type(output)}")
    logger.debug(f"Task output details: {output.__dict__}")
    
    # Try to extract Cypher queries and responses from task output
    try:
        if hasattr(output, 'raw') and output.raw:
            logger.info(f"🔍 Raw task response: {str(output.raw)[:500]}...")
            _last_raw_response = str(output.raw)
        
        if hasattr(output, 'content') and output.content:
            content = str(output.content)
            logger.info(f"🔍 Task content: {content[:500]}...")
            
            # Look for Cypher queries in the content
            if any(keyword in content.upper() for keyword in ['MATCH', 'CREATE', 'MERGE', 'DELETE', 'SET', 'RETURN']):
                logger.info(f"🔍 Potential Cypher query found in task content: {content}")
                _last_cypher_query = content
        
        # Check for tool calls in the output
        if hasattr(output, 'tool_calls') and output.tool_calls:
            for tool_call in output.tool_calls:
                if hasattr(tool_call, 'name') and 'cypher' in tool_call.name.lower():
                    logger.info(f"🔍 Cypher tool call detected: {tool_call.name}")
                    if hasattr(tool_call, 'args') and 'query' in tool_call.args:
                        _last_cypher_query = tool_call.args['query']
                        logger.info(f"🔍 Captured Cypher query from tool call: {_last_cypher_query}")
        
        # Check for CrewAI ToolResult objects (this is where the actual tool calls happen)
        if hasattr(output, 'tool') and hasattr(output, 'tool_input'):
            tool_name = output.tool
            tool_input = output.tool_input
            logger.info(f"🔍 CrewAI tool call detected: {tool_name}")
            logger.info(f"🔍 Tool input: {tool_input}")
            
            # Check if this is a Cypher tool call
            if 'cypher' in tool_name.lower():
                # Try to extract query from tool_input
                if isinstance(tool_input, str):
                    try:
                        import json
                        tool_input_dict = json.loads(tool_input)
                        if 'query' in tool_input_dict:
                            _last_cypher_query = tool_input_dict['query']
                            logger.info(f"🔍 Captured Cypher query from CrewAI tool call: {_last_cypher_query}")
                    except:
                        # If not JSON, check if it contains Cypher keywords
                        if any(keyword in tool_input.upper() for keyword in ['MATCH', 'CREATE', 'MERGE', 'DELETE', 'SET', 'RETURN']):
                            _last_cypher_query = tool_input
                            logger.info(f"🔍 Captured Cypher query from CrewAI tool input: {_last_cypher_query}")
                elif isinstance(tool_input, dict) and 'query' in tool_input:
                    _last_cypher_query = tool_input['query']
                    logger.info(f"🔍 Captured Cypher query from CrewAI tool dict: {_last_cypher_query}")
                        
    except Exception as e:
        logger.debug(f"Error extracting Cypher info from task: {e}")


def llm_by_name(name: str = "sambanova/Meta-Llama-3.1-8B-Instruct"):
    # Local ollama models require the base url be defined
    if "ollama/" in name:
        return LLM(model=name, temperature=0, base_url="http://localhost:11434")
    if "gpt-5" in name:
        return LLM(
            model=name, drop_params=True, additional_drop_params=["stop", "temperature"]
        )
    else:
        return LLM(model=name, temperature=0)


def mcp_crew(tools, llm_name: str):

    llm = llm_by_name(llm_name)

    # Create an agent with access to tools
    cypher_agent = Agent(
        role="Cypher MCP Tool User",
        goal=(
            "Accurately answer user questions by querying a Neo4j database "
            "using the available MCP tools when needed. Prefer precision over verbosity."
        ),
        backstory=("""
You are an expert data analyst proficient in Cypher and Neo4j.

You have access to these tools:
1. get_neo4j_schema: Get the database schema to understand node types and relationships
2. read_neo4j_cypher: Execute read-only Cypher queries to retrieve data
3. write_neo4j_cypher: Execute write Cypher queries to modify data (use with caution)

For questions about data:
- First use get_neo4j_schema to understand the database structure
- Then use read_neo4j_cypher with appropriate Cypher queries to answer the question
- For "how many" questions, use: MATCH (n) RETURN count(n) as node_count
- Always provide the final answer clearly

Return ONLY the final answer for the user, not internal reasoning."""
        ),
        max_iter=3,
        tools=tools,
        reasoning=False,  # False for better compatibility
        step_callback=log_step_callback,  # Optional
        llm=llm,  # Optional - Remove if using OpenAI
    )

    cypher_task = Task(
        description=(
            "Given the user's request: '{prompt}', decide if tools are required.\n"
            "If needed, first get the schema, then compose precise Cypher.\n"
            "Return ONLY the final answer for the user, not internal reasoning."
        ),
        expected_output=(
            "1-3 concise sentences answering the question. Include specific figures when asked."
        ),
        agent=cypher_agent,
        callback=log_task_callback,  # Optional
    )

    # Create the crew
    return Crew(
        agents=[cypher_agent],
        tasks=[cypher_task],
        verbose=os.getenv("CREWAI_VERBOSE", "0") in ("1", "true", "True"),
        memory=False,
    )


def run(prompt: str, full_model_name: str):
    global _last_cypher_query, _last_raw_response
    
    # Reset tracking variables for this run
    _last_cypher_query = ""
    _last_raw_response = ""
    
    # Load the MCP Tools once and reuse across calls
    tools = _ensure_tools()

    print(f"\nRunning Crew with model: {full_model_name} and prompt: {prompt}")
    print(f"Available tools from MCP server(s): {[tool.name for tool in tools]}")

    if full_model_name not in crews:
        crews[full_model_name] = mcp_crew(tools, full_model_name)

    crew = crews[full_model_name]

    # Run the crew w/ the user prompt (with a simple retry for transient LLM errors)
    last_err = None
    for attempt in range(2):
        try:
            result = crew.kickoff(inputs={"prompt": prompt})
            break
        except Exception as e:
            last_err = e
            import time, logging

            logging.exception("Error during crew.kickoff (attempt %d)", attempt + 1)
            time.sleep(0.5)
    else:
        # All attempts failed
        result = f"LLM error: {last_err}"

    # Log the captured information
    if _last_cypher_query:
        logger.info(f"🔍 Final Cypher query captured: {_last_cypher_query}")
    
    if _last_raw_response:
        logger.info(f"🔍 Raw response captured: {_last_raw_response[:200]}...")
        # Don't append raw response to user-facing result, just log it
    
    # Return a plain string for provider compatibility
    return str(result)


# Required by promptfoo
def call_api(
    prompt: str,
    options: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, str]:
    """Call the crew manager API with the given prompt and options.
    Args:
        prompt (str): The prompt to send to the crew manager.
        options (Dict[str, Any]): Options for the API call, from the 'config' arg.
        context (Dict[str, Any]): Prompt data for the API call - user & system prompts.
    Returns:
        Dict[str, str]: The response from the crew manager API.
    """

    logger.info(f"\n\n================================================")
    logger.info(f"Running CrewAI with prompt: {prompt} and context: {context}")
    logger.info(f"================================================\n\n")
    logger.debug(f"call_api: prompt: {prompt}")
    logger.debug(f"call_api: options: {options}")
    logger.debug(f"call_api: context: {context}")

    try:
        # Get model name using shared utility
        model_name = get_model_name_from_config(options, context)
        
        logger.info(f"🔧 Running CrewAI with model: {model_name}")
        result = run(prompt, model_name)
        
        # Append Cypher query if one was captured
        if _last_cypher_query:
            result += f"\n\nCypher used: {_last_cypher_query}"
        
        # Normalize to Promptfoo provider response shape
        return {"output": result}
    except Exception as e:
        # Can't user print here when promptfoo running
        # Uncertain what promptfoos own logger name is, if even using
        error_msg = f"Error in call_api: {str(e)}"
        logging.error(error_msg, exc_info=True)  # This will log the full traceback
        return {"output": f"Error: {str(e)}"}


# For running as a script
if __name__ == "__main__":

    # llm_name = "sambanova/Meta-Llama-3.1-8B-Instruct"
    llm_name = "ollama/qwen3"
    # write_command = "Create a database record for a company named 'Acme Inc'"
    read_command = "Describe the data from the database"
    result = run(read_command, llm_name)

    print(
        f"""
        Query completed!
        result: {result}
    """
    )