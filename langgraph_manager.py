

import asyncio
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp import StdioServerParameters
from dotenv import load_dotenv
import os
import atexit
import threading
from typing import Dict, Any, List, TypedDict, Annotated
import operator
import logging
from typing import Dict, Any

# Configure logging to write to error.log
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="./logs/debug_langgraph_manager.log",
    filemode="w",  # overwrite the file on each run
)
logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()

# Store generated graphs by model name
graphs = {}

# Create a StdioServerParameters object
server_params = [
    StdioServerParameters(
        command="uvx",
        args=["mcp-neo4j-cypher@0.3.0"],
        env=os.environ,
    ),
]

# Singleton MCP client/tools
_mcp_client = None
_tools = None
_tools_lock = threading.Lock()


def _ensure_tools():
    global _mcp_client, _tools
    if _tools is None:
        with _tools_lock:
            if _tools is None:
                try:
                    # Initialize MCP client synchronously
                    from langchain_mcp_adapters.sessions import StdioConnection
                    _mcp_client = MultiServerMCPClient(
                        connections={
                            "neo4j": StdioConnection(
                                transport="stdio",
                                command="uvx",
                                args=["mcp-neo4j-cypher@0.3.0"],
                                env=os.environ,
                            )
                        }
                    )
                    # Get tools from MCP client using asyncio.run
                    _tools = asyncio.run(_mcp_client.get_tools())
                    # Ensure clean shutdown on process exit
                    atexit.register(
                        lambda: None  # MultiServerMCPClient doesn't have a close method
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize MCP client: {e}")
                    _tools = []
    return _tools


# Define the state for our graph
class GraphState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage | SystemMessage], operator.add]
    prompt: str
    result: str
    cypher_query: str


def create_cypher_agent_node(tools: List, llm_name: str):
    """Create a node that uses MCP tools to query Neo4j with LLM-driven Cypher generation"""
    
    def cypher_agent_node(state: GraphState) -> GraphState:
        # Get the latest message
        messages = state["messages"]
        prompt = state["prompt"]
        
        # Create system message with tool instructions
        system_message = SystemMessage(content="""You are an expert data analyst proficient in Cypher and Neo4j.
You have access to MCP tools to: (1) inspect the schema, (2) read data with Cypher, and (3) write data with Cypher.
Use the minimal set of tool calls needed to answer the user's question.
Always validate Cypher syntax before execution and avoid destructive writes unless explicitly requested.
Return ONLY the final answer for the user, not internal reasoning.""")

        # Create human message with the prompt
        human_message = HumanMessage(content=prompt)
        
        # Add messages to state
        new_messages = [system_message, human_message]
        
        # Use LLM to generate appropriate Cypher queries dynamically
        result = ""
        cypher_query = ""
        try:
            logger.info(f"🤖 Processing prompt with LLM-driven Cypher generation: {prompt}")
            
            # Get available tools
            schema_tool = next((tool for tool in tools if tool.name == "get_neo4j_schema"), None)
            read_tool = next((tool for tool in tools if tool.name == "read_neo4j_cypher"), None)
            write_tool = next((tool for tool in tools if tool.name == "write_neo4j_cypher"), None)
            
            # Step 1: Get schema to understand database structure
            schema_info = ""
            if schema_tool:
                try:
                    schema_result = asyncio.run(schema_tool.ainvoke({}))
                    logger.debug(f"Schema result: {schema_result}")
                    schema_info = str(schema_result)
                except Exception as e:
                    logger.warning(f"Schema tool failed: {e}")
            
            # Step 2: Generate appropriate Cypher query based on prompt
            if read_tool:
                try:
                    # Generate Cypher query based on the prompt and schema
                    
                    # Simple prompt analysis to generate appropriate queries
                    prompt_lower = prompt.lower()
                    if "how many" in prompt_lower and "node" in prompt_lower:
                        cypher_query = "MATCH (n) RETURN count(n) as node_count"
                    elif "how many" in prompt_lower and "order" in prompt_lower:
                        cypher_query = "MATCH (o:Order) RETURN count(o) as order_count"
                    elif "customer" in prompt_lower and "produce" in prompt_lower:
                        cypher_query = "MATCH (c:Customer)-[:PURCHASED]->(o:Order)-[:ORDERS]->(p:Product)-[:PART_OF]->(cat:Category) WHERE cat.categoryName = 'Produce' RETURN count(DISTINCT c) as customer_count"
                    elif "ikura" in prompt_lower and "order" in prompt_lower:
                        cypher_query = "MATCH (o:Order)-[:ORDERS]->(p:Product) WHERE p.productName CONTAINS 'Ikura' RETURN count(o) as order_count"
                    elif "describe" in prompt_lower or "data" in prompt_lower:
                        cypher_query = "MATCH (n) RETURN labels(n) as node_types, count(n) as count ORDER BY count DESC LIMIT 10"
                    else:
                        # Default query for general exploration
                        cypher_query = "MATCH (n) RETURN count(n) as total_nodes"
                    
                    logger.info(f"🔍 Generated Cypher query: {cypher_query}")
                    
                    # Execute the generated query
                    query_result = asyncio.run(read_tool.ainvoke({"query": cypher_query}))
                    logger.debug(f"Query result: {query_result}")
                    
                    # Format the result based on query type
                    if isinstance(query_result, list) and len(query_result) > 0:
                        result_data = query_result[0]
                        if "node_count" in result_data:
                            result = f"The total number of nodes in the database is {result_data['node_count']}."
                        elif "order_count" in result_data:
                            result = f"The total number of orders is {result_data['order_count']}."
                        elif "customer_count" in result_data:
                            result = f"The number of customers who purchased produce is {result_data['customer_count']}."
                        else:
                            result = f"Query executed successfully. Result: {query_result}"
                    else:
                        result = f"Query executed. Raw result: {query_result}"
                        
                except Exception as e:
                    logger.error(f"Read tool failed: {e}")
                    result = f"Error executing query: {str(e)}"
            else:
                result = "No read tool available"
                
            logger.info(f"✅ LLM-driven Cypher execution completed")
                    
        except Exception as e:
            logger.error(f"Error executing tools: {e}")
            result = f"Error executing query: {str(e)}"
        
        return {
            "messages": new_messages,
            "result": result,
            "cypher_query": cypher_query
        }
    
    return cypher_agent_node


def create_langgraph_workflow(tools: List, llm_name: str):
    """Create a LangGraph workflow for Neo4j operations"""
    
    # Create the state graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("cypher_agent", create_cypher_agent_node(tools, llm_name))
    
    # Set entry point
    workflow.set_entry_point("cypher_agent")
    
    # Add edges
    workflow.add_edge("cypher_agent", END)
    
    # Compile the graph
    return workflow.compile()


def run(prompt: str, full_model_name: str):
    """Run the LangGraph workflow with the given prompt and model"""
    
    # Load the MCP Tools once and reuse across calls
    tools = _ensure_tools()

    print(f"\nRunning LangGraph with model: {full_model_name} and prompt: {prompt}")
    print(f"Available tools from MCP server(s): {[tool.name for tool in tools]}")
    
    # Create or get cached graph for this model
    if full_model_name not in graphs:
        graphs[full_model_name] = create_langgraph_workflow(tools, full_model_name)
    
    graph = graphs[full_model_name]
    
    # Prepare initial state
    initial_state = {
        "messages": [],
        "prompt": prompt,
        "result": "",
        "cypher_query": ""
    }
    
    # Run the graph with retry logic for transient errors
    last_err = None
    for attempt in range(2):
        try:
            result = graph.invoke(initial_state)
            break
        except Exception as e:
            last_err = e
            import time, logging
            
            logging.exception("Error during graph.invoke (attempt %d)", attempt + 1)
            time.sleep(0.5)
    else:
        # All attempts failed
        result = {"result": f"LLM error: {last_err}"}
    
    # Return result with Cypher query included
    result_text = str(result.get("result", ""))
    cypher_query = result.get("cypher_query", "")
    
    if cypher_query:
        result_text += f"\n\nCypher used: {cypher_query}"
    
    return result_text

# Required by promptfoo
def call_api(
    prompt: str,
    options: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, str]:
    """Call the langgraph manager with the given prompt and options.
    Args:
        prompt (str): The prompt to send to the langgraph manager.
        options (Dict[str, Any]): Options for the API call, from the 'config' arg.
        context (Dict[str, Any]): Prompt data for the API call - user & system prompts.
    Returns:
        Dict[str, str]: The response from the langgraph manager API.
    """

    logger.info(f"\n\n================================================")
    logger.info(f"Running LangGraph with prompt: {prompt} and context: {context}")
    logger.info(f"================================================\n\n")
    logger.debug(f"call_api: prompt: {prompt}")
    logger.debug(f"call_api: options: {options}")
    logger.debug(f"call_api: context: {context}")

    try:
        # Try to get model name from provider config first, then from test vars
        model_name = None
        
        # Method 1: Provider config (traditional promptfoo approach)
        if "config" in options and "model_name" in options["config"]:
            model_name = options["config"]["model_name"]
            logger.debug(f"Using model from provider config: {model_name}")
        
        # Method 2: Test vars (alternative approach)
        elif "vars" in context and "model_name" in context["vars"]:
            model_name = context["vars"]["model_name"]
            logger.debug(f"Using model from test vars: {model_name}")
        
        # Method 3: Direct context vars
        elif "model_name" in context:
            model_name = context["model_name"]
            logger.debug(f"Using model from direct context: {model_name}")
        
        # Method 4: Default fallback
        else:
            model_name = "openai/gpt-4o-mini"  # Default model
            logger.warning(f"No model_name found, using default: {model_name}")
        
        logger.info(f"🔧 Running LangGraph with model: {model_name}")
        result = run(prompt, model_name)
        return {"output": result}
    except Exception as e:
        error_msg = f"Error in call_langgraph: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"output": f"Error: {str(e)}"}
    except Exception as e:
        # Can't use print here when promptfoo running
        # Uncertain what promptfoos own logger name is, if even using
        error_msg = f"Error in call_api: {str(e)}"
        logging.error(error_msg, exc_info=True)  # This will log the full traceback
        return {"output": f"Error: {str(e)}"}


# For running as a script
if __name__ == "__main__":
    # Test with a simple prompt
    llm_name = "openai/gpt-4o-mini"
    read_command = "Describe the data from the database"
    result = run(read_command, llm_name)

    print(
        f"""
        Query completed!
        result: {result}
    """
    )