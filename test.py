import asyncio
import os
from dotenv import load_dotenv

# Framework imports
from agentic_patterns.react_pattern.react_agent import ReactAgent
from agentic_patterns.llm_services.groq_service import GroqService
from agentic_patterns.llm_services.google_openai_compat_service import GoogleOpenAICompatService
from agentic_patterns.tool_pattern.tool import tool

load_dotenv()

# --- Define Tool ---
@tool
def simple_math(expression: str) -> str:
    """Evaluates a simple math expression string. Use for simple arithmetic."""
    print(f"[Local Tool] Evaluating: {expression}")
    try:
        # WARNING: eval() is unsafe with untrusted input. OK for this demo.
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

# --- Main Test Function ---
async def main():
    # --- LLM Service Setup ---
    # Check for keys first
    groq_key = os.getenv("GROQ_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    services_to_test = {}
    if groq_key:
        services_to_test["Groq"] = (GroqService(), "llama-3.3-70b-versatile")
    else:
        print("Skipping Groq test: GROQ_API_KEY not found.")

    if google_key:
        try:
            import openai # Check if installed for compat layer
            services_to_test["Google"] = (GoogleOpenAICompatService(), "gemini-1.5-flash")
        except ImportError:
            print("Skipping Google test: openai library not installed.")
    else:
        print("Skipping Google test: GOOGLE_API_KEY not found.")

    if not services_to_test:
        print("No LLM services configured. Exiting.")
        return
    # --- End LLM Service Setup ---

    user_query = "What is (4 * 8) + 10? Use the math tool."

    for service_name, (llm_service, model_name) in services_to_test.items():
        print(f"\n{'='*20} TESTING WITH {service_name} ({model_name}) {'='*20}")

        # --- Agent Setup ---
        agent = ReactAgent(
            llm_service=llm_service,
            model=model_name,
            tools=[simple_math], # Only local tool
            # No MCP needed for this specific test
        )
        # --- End Agent Setup ---

        print(f"\n--- Running Agent for Query: '{user_query}' ---")
        response = f"Agent run failed for {service_name}."
        try:
            response = await agent.run(user_msg=user_query)
            print(f"\n--- {service_name} Agent Final Response ---")
            print(response)
        except Exception as e:
            print(f"\n--- {service_name} Agent Run Error ---")
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()

        print(f"\n{'='*20} FINISHED {service_name} TEST {'='*20}")
        await asyncio.sleep(1) # Small delay between tests if needed

if __name__ == "__main__":
    # Ensure openai library is installed if testing Google
    if os.getenv("GOOGLE_API_KEY"):
        try:
            import openai
        except ImportError:
            print("Warning: openai library needed for Google test not installed (pip install openai)")

    asyncio.run(main())