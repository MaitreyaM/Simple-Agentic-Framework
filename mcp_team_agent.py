# --- START OF team_agent.py (Using MCP) ---

import os
import asyncio
from dotenv import load_dotenv
from pydantic import HttpUrl

# Team/Agent imports
from agentic_patterns.multiagent_pattern.team import Team
from agentic_patterns.multiagent_pattern.agent import Agent
# Local Tool imports
from agentic_patterns.tools.web_search import duckduckgo_search 
from agentic_patterns.tool_pattern.tool import tool 
# MCP Client imports
from agentic_patterns.mcp_client.client import MCPClientManager, SseServerConfig

load_dotenv()

# Example: Define a local tool for the summarizer if needed
@tool
def count_words(text: str) -> int:
    """Counts the number of words in a given text."""
    return len(text.split())

# Define the async function to run the team
async def run_agile_team_with_mcp():
    """Defines and runs the Agile benefits team, potentially using MCP tools."""

    
    
    mcp_server_configs = {
        "adder_server": SseServerConfig(url=HttpUrl("http://localhost:8000"))
       
    }
    mcp_manager = MCPClientManager(mcp_server_configs)
    

    topic = "the benefits of Agile methodology in software development"

    try: # Use try/finally to ensure manager disconnects
        with Team() as team:
            researcher = Agent(
                name="Web_Researcher",
                backstory="You are an expert web researcher. You can use local search tools or potentially remote MCP tools.",
                task_description=f"Search the web for information on '{topic}'. Also calculate 5 + 3 using the 'add' tool.", # Modified task to use 'add'
                task_expected_output="Raw search results AND the result of 5 + 3.",
                # Local tools for this agent
                tools=[duckduckgo_search],
                # MCP configuration for this agent
                mcp_manager=mcp_manager, # Pass the shared manager
                mcp_server_names=["adder_server"] # Tell it which server(s) to use
            )

            summarizer = Agent(
                name="Content_Summarizer",
                backstory="You are an expert analyst. You can count words locally.",
                task_description="Analyze the provided context (search results and addition result) and extract the main benefits. Also count the words in the addition result.",
                task_expected_output="A concise bullet-point list summarizing key benefits, and the word count.",
                # Provide the local word count tool to this agent
                tools=[count_words],
                # This agent doesn't need MCP tools, so we don't pass mcp_manager/names
            )

            reporter = Agent(
                name="Report_Writer",
                backstory="You are a skilled writer.",
                task_description="Take the summarized key points and word count, and write a short paragraph.",
                task_expected_output="A single paragraph summarizing the benefits and mentioning the word count."
                # No tools needed for this agent
            )

            researcher >> summarizer >> reporter

            # Run the team asynchronously
            await team.run()

    finally:
        # Ensure MCP connections are closed when done
        print("--- Cleaning up MCP connections ---")
        await mcp_manager.disconnect_all()

# --- Main Execution ---
if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
         print("Warning: GROQ_API_KEY not found in .env")
         # Decide if exit is needed based on LLM used

    # Make sure the minimal_mcp_server.py is running in another terminal!
    print("Ensure the minimal_mcp_server.py is running on port 8000...")
    asyncio.run(run_agile_team_with_mcp())

# --- END OF team_agent.py (Using MCP) ---