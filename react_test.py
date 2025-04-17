import os
from dotenv import load_dotenv

# Import the specific agent and the tool
from agentic_patterns.react_pattern.react_agent import ReactAgent 
from agentic_patterns.tools import duckduckgo_search

load_dotenv()



# --- Main Execution ---
if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY") or not os.getenv("BRAVE_API_KEY"):
        print("Error: Ensure GROQ_API_KEY and BRAVE_API_KEY are in .env")
        exit(1)


    agent = ReactAgent(tools=duckduckgo_search, model="llama-3.3-70b-versatile")

    # Define user query
    user_query = "what do react agents do think about it in multiple steps and tell me"

   
    response = agent.run(user_msg=user_query) 
    print("THE RESPONSE IS : \n\n")
    print(response)
   
# --- END OF FILE run_react_agent_example.py ---