# --- START OF test_react_google_multi_tool.py ---

import asyncio
import os
from dotenv import load_dotenv
from pydantic import HttpUrl

from agentic_patterns.react_pattern.react_agent import ReactAgent
from agentic_patterns.llm_services.google_openai_compat_service import GoogleOpenAICompatService
from agentic_patterns.mcp_client.client import MCPClientManager, SseServerConfig
from agentic_patterns.tool_pattern.tool import tool

load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies integer a by integer b."""
    print(f"[Local Tool] Multiplying: {a} * {b}")
    return a * b

async def main():
    

    
    server_configs = {
        "adder_server": SseServerConfig(url=HttpUrl("http://localhost:8000")),
        "subtract_server": SseServerConfig(url=HttpUrl("http://localhost:8001"))
    }
    manager = MCPClientManager(server_configs)

    # 3. Instantiate the Google LLM Service
    google_llm_service = GoogleOpenAICompatService()
    gemini_model = "gemini-2.0-flash" # Or your preferred compatible model

    # 4. Create the ReactAgent
    agent = ReactAgent(
        llm_service=google_llm_service, # Use Google Service
        model=gemini_model,
        tools=[multiply], # Pass the local tool
        mcp_manager=manager, # Pass the MCP manager
        mcp_server_names=["adder_server","subtract_server"] # Specify the remote server to use
    )

    # 5. Define the user query requiring both tools
    user_query = "Calculate (10 + 5) * 3"
  


    # 6. Run the agent
    response = "Agent run failed."
    try:
        response = await agent.run(user_msg=user_query)
        print(f"\n--- Agent Final Response ---")
        print(response)
    except Exception as e:
        print(f"\n--- Agent Run Error ---")
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n--- Cleaning up MCP connections ---")
        await manager.disconnect_all()

# --- Script Entry Point ---
if __name__ == "__main__":
    # Ensure openai library is installed
    try:
        import openai
    except ImportError:
        print("Error: openai library not found. Please install it: pip install openai")
        exit(1)

    print("Ensure the minimal_mcp_server.py (adder) is running on port 8000...")
    asyncio.run(main())

# --- END OF test_react_google_multi_tool.py ---