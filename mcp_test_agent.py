# --- START OF test_tool_agent_mcp.py ---

import os
import asyncio
from dotenv import load_dotenv
from pydantic import HttpUrl

# Import ToolAgent and MCP client components
from agentic_patterns.tool_pattern.tool_agent import ToolAgent
from agentic_patterns.mcp_client.client import MCPClientManager, SseServerConfig
# We are not defining local tools in this example
# from agentic_patterns.tool_pattern.tool import tool

load_dotenv()

async def main():
    # 1. Configure the MCP Server
    server_name = "adder_server"
    server_configs = {
        server_name: SseServerConfig(url=HttpUrl("http://localhost:8000"))
    }

    # 2. Create the MCPClientManager
    manager = MCPClientManager(server_configs)

    # 3. Create the ToolAgent, passing the manager
    #    Note: We pass an empty list for local tools 'tools=[]'
    agent = ToolAgent(
        tools=[], # No local tools for this example
        mcp_manager=manager,
        mcp_server_names=[server_name], # Specify which server to use
        model="llama-3.3-70b-versatile" # Or your preferred model
    )

    # 4. Define a query that requires the remote 'add' tool
    user_query = "What is 123 plus 456?"
    print(f"Query: {user_query}")

    # 5. Run the agent and print the response
    try:
        response = await agent.run(user_msg=user_query)
        print(f"\nFinal Response:\n{response}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # 6. Clean up MCP connections
        print("\n--- Cleaning up MCP connections ---")
        await manager.disconnect_all()

# --- Script Entry Point ---
if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
         print("Warning: GROQ_API_KEY not found in .env")

    print("Ensure the minimal_mcp_server.py is running on port 8000...")
    asyncio.run(main())

# --- END OF test_tool_agent_mcp.py ---