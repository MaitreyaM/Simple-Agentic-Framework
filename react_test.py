import os
from dotenv import load_dotenv
from agentic_patterns.react_pattern.react_agent import ReactAgent 
from agentic_patterns.tools import duckduckgo_search
import asyncio 

load_dotenv()

agent = ReactAgent(tools=duckduckgo_search, model="meta-llama/llama-4-maverick-17b-128e-instruct")

async def main():
    user_query = "tell me the working of engine mechanics of bmw m5"
    response = await agent.run(user_msg=user_query) 
    print(response)

asyncio.run(main())
   
