# Multi-Agentic LLM Evaluations with Promptfoo

A comprehensive evaluation framework for testing multiple AI agent implementations (CrewAI, LangGraph, LlamaIndex) with Neo4j MCP (Model Context Protocol) server integration.

## Overview

This project provides a unified testing environment for comparing different multi-agent AI frameworks:
- **CrewAI**: Multi-agent collaboration with specialized roles
- **LangGraph**: Stateful workflow management with graph-based execution
- **LlamaIndex**: ReAct agents with tool integration

All implementations connect to Neo4j databases via MCP servers for consistent evaluation.

## Requirements

- [Astral's UV](https://docs.astral.sh/uv/#installation) - Python package manager
- [Promptfoo](https://www.promptfoo.dev/docs/installation/) - LLM evaluation framework
- Neo4j database (local or cloud instance)
- OpenAI API key or Ollama setup for local models

## Setup

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Configure environment:**
   ```bash
   cp sample.env .env
   # Edit .env with your API keys and Neo4j connection details
   ```

3. **Start Neo4j MCP server:**
   ```bash
   uvx mcp-neo4j-cypher@0.3.0
   ```

## Running Evaluations

### Full Evaluation Suite
```bash
uv run promptfoo eval -o ./outputs/output.html --no-cache
```

### Individual Framework Tests
```bash
# Test CrewAI implementation
uv run python test_crewai.py

# Test LangGraph implementation  
uv run python test_langgraph.py

# Test LlamaIndex implementation
uv run python test_llamaindex.py
```

## Viewing Results

**Option 1: Static HTML Report**
- Open `./outputs/output.html` in your browser

**Option 2: Live Webview**
```bash
npx promptfoo@latest view
```


## Framework Implementations

### CrewAI Implementation

**Files:**
- `crewai_manager.py` - Core CrewAI workflow management with Neo4j MCP Server integration

**Features:**
- **Multi-Agent Collaboration**: Specialized agents with defined roles and goals
- **MCP Integration**: Connects to Neo4j Cypher MCP server using `crewai_tools`
- **Tool Management**: Singleton pattern for efficient MCP client management
- **Multi-LLM Support**: Supports OpenAI, Ollama, and other model providers
- **Error Handling**: Retry logic and comprehensive error handling

**Usage:**
```python
from crewai_manager import run

# Run a CrewAI workflow
result = run("Describe the database schema", "openai/gpt-4o-mini")
print(result)
```

**Testing:**
```bash
uv run python test_crewai.py
```

### LangGraph Implementation

**Files:**
- `langgraph_manager.py` - Core LangGraph workflow management with Neo4j MCP server integration

**Features:**
- **Stateful Workflows**: Uses LangGraph's StateGraph for managing conversation state
- **MCP Integration**: Connects to Neo4j Cypher MCP server using `langchain-mcp-adapters`
- **Tool Management**: Singleton pattern for efficient MCP client management
- **Error Handling**: Retry logic and comprehensive error handling
- **Caching**: Graph caching by model name for performance

**Usage:**
```python
from langgraph_manager import run

# Run a LangGraph workflow
result = run("Describe the database schema", "openai/gpt-4o-mini")
print(result)
```

**Testing:**
```bash
uv run python test_langgraph.py
```

### LlamaIndex Implementation

**Files:**
- `llamaindex_manager.py` - Combined LlamaIndex agent management with Neo4j MCP server integration

**Features:**
- **ReAct Agent**: Uses LlamaIndex's ReActAgent for reasoning and tool usage
- **MCP Integration**: Direct integration with Neo4j Cypher MCP server using `mcp.client.stdio`
- **Tool Conversion**: Converts MCP tools to LlamaIndex FunctionTool format
- **Agent Caching**: Agent caching by model name for performance
- **Multi-LLM Support**: Supports both OpenAI and Ollama models
- **Error Handling**: Comprehensive error handling and retry logic

**Usage:**
```python
from llamaindex_manager import run, call_api

# Direct usage
result = run("Describe the database schema", "openai/gpt-4o-mini")

# Promptfoo integration
result = call_api(prompt, options, context)
```

**Testing:**
```bash
uv run python test_llamaindex.py
```

## Testing Framework

### Test Structure
Each framework includes comprehensive test coverage:

- **Manager Tests**: Test the core `run()` function with various prompts
- **API Tests**: Test the `call_api()` function for promptfoo integration
- **Model Tests**: Test different LLM providers (OpenAI, Ollama, etc.)
- **Error Handling**: Verify graceful error handling and retry logic

### Running All Tests
```bash
# Run all individual framework tests
uv run python test_crewai.py
uv run python test_langgraph.py  
uv run python test_llamaindex.py

# Run full promptfoo evaluation suite
uv run promptfoo eval -o ./outputs/output.html --no-cache
```

### Test Output
Tests provide detailed output including:
- ✅ Success indicators for each test
- ❌ Error messages with full tracebacks
- 📊 Performance metrics and response times
- 🔧 Tool availability and MCP server status

## Troubleshooting

### Common Issues
1. **MCP Server Connection**: Ensure Neo4j MCP server is running
2. **API Keys**: Verify OpenAI API key or Ollama setup
3. **Dependencies**: Run `uv sync` to ensure all packages are installed
4. **Environment**: Check `.env` file configuration

### Cache Management
```bash
# Clear promptfoo cache
promptfoo cache clear

# Clear Python cache
find . -name "__pycache__" -type d -exec rm -rf {} +
```

## License
MIT