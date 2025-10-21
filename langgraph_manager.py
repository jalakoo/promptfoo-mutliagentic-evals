import asyncio
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_anthropic import ChatAnthropic
# from langchain_community.llms import SambaNova
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
from shared_utils import get_model_name_from_config

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

You have access to these tools:
1. get_neo4j_schema: Get the database schema to understand node types and relationships
2. read_neo4j_cypher: Execute read-only Cypher queries to retrieve data
3. write_neo4j_cypher: Execute write Cypher queries to modify data (use with caution)

For questions about data:
- First use get_neo4j_schema to understand the database structure
- Then use read_neo4j_cypher with appropriate Cypher queries to answer the question
- For "how many" questions, use: MATCH (n) RETURN count(n) as node_count
- Always provide the final answer clearly

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
            
            # Step 1: Get schema to understand database structure
            schema_info = ""
            if schema_tool:
                try:
                    schema_result = asyncio.run(schema_tool.ainvoke({}))
                    logger.debug(f"Schema result: {schema_result}")
                    schema_info = str(schema_result)
                except Exception as e:
                    logger.warning(f"Schema tool failed: {e}")
            
            # Step 2: Use LLM to generate Cypher query based on prompt and schema
            if read_tool:
                try:
                    
                    # Create LLM instance based on model name
                    logger.info(f"🔧 Creating LLM for model: {llm_name}")
                    
                    if "ollama/" in llm_name:
                        model_name = llm_name.split("/")[1]
                        logger.info(f"🦙 Using Ollama model: {model_name}")
                        llm = Ollama(model=model_name, base_url="http://localhost:11434")
                    elif "openai/" in llm_name:
                        model_name = llm_name.split("/")[1]
                        logger.info(f"🤖 Using OpenAI model: {model_name}")
                        llm = ChatOpenAI(model=model_name)
                    elif "anthropic/" in llm_name:
                        model_name = llm_name.split("/")[1]
                        logger.info(f"🧠 Using Anthropic model: {model_name}")
                        llm = ChatAnthropic(model=model_name, temperature=0)
                    elif "sambanova/" in llm_name:
                        logger.error(f"SambaNova is not supported for the current version of langchain-core")
                        raise ValueError("SambaNova is not supported for the current version of langchain-core")
                    else:
                        logger.error(f"Unsupported model: {llm_name}")
                        raise ValueError(f"Unsupported model: {llm_name}")

                    
                    # Generate Cypher query using LLM
                    query_generation_prompt = f"""Based on the user's question: "{prompt}"

And the database schema: {schema_info}

Generate a single, precise Cypher query to answer the question. Return ONLY the Cypher query, nothing else.

Examples:
- "How many nodes are in the database?" → MATCH (n) RETURN count(n) as node_count
- "How many orders are there?" → MATCH (o:Order) RETURN count(o) as order_count
- "Describe the data" → MATCH (n) RETURN labels(n) as node_types, count(n) as count ORDER BY count DESC LIMIT 10

Cypher query:"""
                    
                    logger.info(f"🧠 Generating Cypher query with LLM for prompt: {prompt}")
                    llm_response = llm.invoke(query_generation_prompt)
                    logger.debug(f"Raw LLM response: {llm_response}")
                    
                    # Extract the actual content from the LLM response
                    if hasattr(llm_response, 'content'):
                        cypher_query = llm_response.content.strip()
                        logger.debug(f"Extracted content from response: {cypher_query}")
                    else:
                        cypher_query = str(llm_response).strip()
                        logger.debug(f"Using string representation: {cypher_query}")
                    
                    # Clean up the query (remove any extra text and metadata)
                    # Handle the specific case from the error: content='QUERY' additional_kwargs=...
                    if "content='" in cypher_query and "' additional_kwargs" in cypher_query:
                        start = cypher_query.find("content='") + 9
                        end = cypher_query.find("' additional_kwargs")
                        cypher_query = cypher_query[start:end]
                    elif 'content="' in cypher_query and '" additional_kwargs' in cypher_query:
                        start = cypher_query.find('content="') + 9
                        end = cypher_query.find('" additional_kwargs')
                        cypher_query = cypher_query[start:end]
                    
                    # Remove common prefixes/suffixes that LLMs might add
                    cypher_query = cypher_query.replace("Cypher query:", "").strip()
                    cypher_query = cypher_query.replace("Query:", "").strip()
                    cypher_query = cypher_query.replace("```cypher", "").strip()
                    cypher_query = cypher_query.replace("```", "").strip()
                    
                    # Take only the first line if multiple lines
                    if "\n" in cypher_query:
                        cypher_query = cypher_query.split("\n")[0].strip()
                    
                    # Remove any remaining quotes or extra characters
                    cypher_query = cypher_query.strip('"').strip("'").strip()
                    
                    logger.info(f"🔍 Generated Cypher query: {cypher_query}")
                    
                    # Validate that we have a proper Cypher query
                    if not cypher_query or len(cypher_query.strip()) == 0:
                        logger.error("❌ Generated Cypher query is empty")
                        result = "Error: Could not generate a valid Cypher query"
                        cypher_query = ""
                    elif not any(keyword in cypher_query.upper() for keyword in ['MATCH', 'CREATE', 'MERGE', 'DELETE', 'SET', 'RETURN']):
                        logger.error(f"❌ Generated query doesn't appear to be valid Cypher: {cypher_query}")
                        result = f"Error: Generated query doesn't appear to be valid Cypher: {cypher_query}"
                    else:
                        # Execute the generated query
                        logger.info(f"🚀 Executing Cypher query: {cypher_query}")
                        query_result = asyncio.run(read_tool.ainvoke({"query": cypher_query}))
                        logger.debug(f"Query result: {query_result}")
                        
                        # Format the result
                        if isinstance(query_result, list) and len(query_result) > 0:
                            result_data = query_result[0]
                            if isinstance(result_data, dict):
                                # Format based on the actual data returned
                                if len(result_data) == 1:
                                    key, value = next(iter(result_data.items()))
                                    result = f"The {key.replace('_', ' ')} is {value}."
                                else:
                                    result = f"Query executed successfully. Result: {query_result}"
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
            logger.info(f"🔄 Running graph.invoke (attempt {attempt + 1})")
            result = graph.invoke(initial_state)
            logger.info(f"✅ Graph execution completed successfully")
            break
        except Exception as e:
            last_err = e
            import time, logging
            
            logger.error(f"❌ Error during graph.invoke (attempt {attempt + 1}): {e}")
            logging.exception("Error during graph.invoke (attempt %d)", attempt + 1)
            time.sleep(0.5)
    else:
        # All attempts failed
        logger.error(f"❌ All graph execution attempts failed: {last_err}")
        result = {"result": f"LLM error: {last_err}", "cypher_query": ""}
    
    # Return result with Cypher query included
    result_text = str(result.get("result", ""))
    cypher_query = result.get("cypher_query", "")
    
    logger.info(f"📊 Final result: {result_text[:100]}...")
    logger.info(f"🔍 Cypher query: {cypher_query}")
    
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
        # Get model name using shared utility
        model_name = get_model_name_from_config(options, context)
        
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