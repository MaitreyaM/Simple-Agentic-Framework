
import asyncio
import os
from dotenv import load_dotenv
from pydantic import HttpUrl 
from agentic_patterns.llm_services import groq_service
from agentic_patterns.react_pattern.react_agent import ReactAgent
from agentic_patterns.mcp_client.client import MCPClientManager, SseServerConfig
from agentic_patterns.tool_pattern.tool import tool
from agentic_patterns.llm_services.groq_service import GroqService

@tool
def multiply(a: int, b: int) -> int:
    """Subtracts integer b from integer a."""
    print(f"[Local Tool] Multiplying: {a} * {b}")
    return a * b

async def main():
    load_dotenv() 
    groq_llm_service = GroqService()

    server_configs = {
        "adder_server": SseServerConfig(url=HttpUrl("http://localhost:8000")),
        "subtract_server": SseServerConfig(url=HttpUrl("http://localhost:8001"))
    }
    manager = MCPClientManager(server_configs)

    agent = ReactAgent(
        llm_service=groq_llm_service,
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        tools=[multiply], 
        mcp_manager=manager,
        mcp_server_names=["adder_server","subtract_server"] 
    )

    user_query = "Calculate ((15 + 7) - 5) * 2"
    response = await agent.run(user_msg=user_query)
    print(response)
    await manager.disconnect_all()

asyncio.run(main())