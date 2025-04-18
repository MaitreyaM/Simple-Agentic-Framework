import os
from dotenv import load_dotenv
from agentic_patterns.tool_pattern.tool_agent import ToolAgent
from agentic_patterns.tools import duckduckgo_search 

load_dotenv()

agent = ToolAgent(tools=duckduckgo_search, model="llama-3.3-70b-versatile")

user_query = "Search the web for recent news about AI advancements."

print(f"Query: {user_query}")
response = agent.run(user_msg=user_query)
print(f"Response:\n{response}")

