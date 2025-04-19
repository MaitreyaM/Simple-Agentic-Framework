# --- START OF test_email_agent.py ---
import asyncio
import os
from dotenv import load_dotenv
from agentic_patterns.tool_pattern.tool_agent import ToolAgent
from agentic_patterns.tools import send_email, fetch_recent_emails

load_dotenv()

async def main():
    # Check for email credentials BEFORE creating the agent
    if not os.getenv("SMTP_USERNAME") or not os.getenv("SMTP_PASSWORD"):
        print("\nERROR: SMTP_USERNAME or SMTP_PASSWORD not set in .env file. Cannot run email tests.")
        return

    # Instantiate agent with the email tools
    agent = ToolAgent(
        tools=[send_email, fetch_recent_emails], # Provide the relevant tools
        model="llama-3.3-70b-versatile" # Or your preferred model
    )

    # --- Test Case 1: Fetch recent emails ---
    query1 = "Check my INBOX and show me the subject of the last 2 emails."
    print(f"\nQuery 1: {query1}")
    print("WARNING: This will access your actual Gmail INBOX.")
    response1 = await agent.run(user_msg=query1)
    print(f"\nResponse 1:")
    print(response1)
    print("-" * 30)

    await asyncio.sleep(1) # Small delay

    # --- Test Case 2: Send an email ---
    test_recipient = "maitreyamishra04@gmail.com" # <-- !!! CHANGE THIS !!!
    if test_recipient == "your_test_recipient@example.com":
        print("\nSKIPPING email sending test: Please update 'test_recipient' in the script.")
    else:
        query2 = f"Send an email to {test_recipient}. Subject should be 'Agent Test' and body should be 'Hello from the framework!'"
        print(f"\nQuery 2: {query2}")
        print(f"WARNING: This will send a REAL email to {test_recipient}.")
        response2 = await agent.run(user_msg=query2)
        print(f"\nResponse 2:")
        print(response2)
        print("-" * 30)

    # Note: No MCPClientManager needed here

if __name__ == "__main__":
    # Check for LLM API key
    if not os.getenv("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY not found in .env")

    # Check for email credentials
    if not os.getenv("SMTP_USERNAME") or not os.getenv("SMTP_PASSWORD"):
        print("Warning: Email credentials (SMTP_USERNAME, SMTP_PASSWORD) not found in .env")

    # Check required libraries
    try:
        import requests # requests is used by email tool helpers
    except ImportError:
        print("Error: requests library not found. Please install it: pip install requests")
        exit(1)

    print("\n--- Running Email Agent Tests ---")
    print("Requires email credentials in .env and potentially Gmail 'App Password'.")
    print("Modify the test_recipient variable before running the send test.")

    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

# --- END OF test_email_agent.py ---