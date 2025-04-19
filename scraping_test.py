# --- START OF test_crawler_agent.py ---

import asyncio
import os
from dotenv import load_dotenv
import crawl4ai
from agentic_patterns.tool_pattern.tool_agent import ToolAgent
from agentic_patterns.tools import scrape_url, extract_text_by_query

load_dotenv()

async def main():
    # Instantiate agent with the web crawler tools
    # Note: ToolAgent uses AsyncGroq by default now
    agent = ToolAgent(
        tools=[scrape_url, extract_text_by_query], # Provide the relevant tools
        model="llama-3.3-70b-versatile" # Or your preferred model
    )

    # --- Test Case 1: Simple Scrape ---
    query1 = "Can you scrape the content of https://docs.agno.com/introduction for me?"
    
    response1 = await agent.run(user_msg=query1)
    
    print(response1[:500] + "...")
    print("-" * 30)

    await asyncio.sleep(1) # Small delay between independent agent runs

    # --- Test Case 2: Extract specific text ---
    query2 = "Look for the term 'library' on the page https://docs.agno.com/introduction and show me the surrounding text."
    response2 = await agent.run(user_msg=query2)
    print(f"\nResponse 2:")
    print(response2)
    print("-" * 30)

    # Note: No MCPClientManager needed or passed here as these are local tools



try:
    asyncio.run(main())
except Exception as e:
    print(f"\nAn error occurred: {e}")
    import traceback
    traceback.print_exc()

