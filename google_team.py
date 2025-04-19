# --- START OF test_team_google.py ---

import asyncio
import os
from dotenv import load_dotenv

# Import Team/Agent structure
from agentic_patterns.multiagent_pattern.team import Team
from agentic_patterns.multiagent_pattern.agent import Agent
from agentic_patterns.llm_services.google_openai_compat_service import GoogleOpenAICompatService

load_dotenv()

async def run_google_team():
    google_llm_service = GoogleOpenAICompatService()
    gemini_model = "gemini-2.5-pro-exp-03-25" 
    topic = "the benefits of using asynchronous programming in Python"
    with Team() as team:
        planner = Agent(
            name="Topic_Planner",
            backstory="Expert in outlining content.",
            task_description=f"Create a short, 3-bullet point outline for explaining '{topic}'.",
            task_expected_output="A 3-item bullet list.",
            llm_service=google_llm_service,
            model=gemini_model,
            
        )

        writer = Agent(
            name="Content_Writer",
            backstory="Skilled technical writer.",
            task_description="Take the outline provided in the context and write a concise paragraph explaining the topic.",
            task_expected_output="One paragraph based on the outline.",
            # --- Pass the Google service and model ---
            llm_service=google_llm_service,
            model=gemini_model,
            # --- End Service Passing ---
            # No tools needed for this agent
        )

        # Define dependency
        planner >> writer

        
        await team.run()

# --- Script Entry Point ---
if __name__ == "__main__":
    # Check for necessary keys
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in environment variables or .env file.")
    else:
        # Ensure openai library is installed for the compatibility service
        try:
            import openai
        except ImportError:
            print("Error: openai library not found. Please install it: pip install openai")
            exit(1)

        try:
            asyncio.run(run_google_team())
        except Exception as e:
            print(f"\n--- Team Run Error ---")
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()

# --- END OF test_team_google.py ---