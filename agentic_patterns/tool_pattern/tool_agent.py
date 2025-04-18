# --- START OF REFACTORED tool_agent.py ---

import json
from typing import List, Dict, Any, Optional

from colorama import Fore
from dotenv import load_dotenv
from groq import Groq

from agentic_patterns.tool_pattern.tool import Tool
# No longer need validate_arguments here if Tool.run handles it
from agentic_patterns.utils.completions import build_prompt_structure
from agentic_patterns.utils.completions import ChatHistory
from agentic_patterns.utils.completions import completions_create
from agentic_patterns.utils.completions import update_chat_history
# No longer need extract_tag_content
# from agentic_patterns.utils.extraction import extract_tag_content

load_dotenv()

# Simple system prompt for direct tool use
NATIVE_TOOL_SYSTEM_PROMPT = """
You are a helpful assistant. Use the available tools if necessary to answer the user's request.
If you use a tool, you will be given the results, and then you should provide the final response to the user.
"""

class ToolAgent:
    """
    A simple agent that uses native tool calling to answer user queries.
    It makes one attempt to call tools if needed, processes the results,
    and then generates a final response.
    """

    def __init__(
        self,
        tools: Tool | list[Tool],
        model: str = "llama-3.3-70b-versatile", # Changed default model
        system_prompt: str = NATIVE_TOOL_SYSTEM_PROMPT,
    ) -> None:
        self.client = Groq()
        self.model = model
        self.system_prompt = system_prompt # Use the provided or default system prompt
        self.tools = tools if isinstance(tools, list) else [tools]
        self.tools_dict = {tool.name: tool for tool in self.tools}
        self.tool_schemas = [tool.fn_schema for tool in self.tools] # Store schemas

    # Removed add_tool_signatures method

    def process_tool_calls(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """
        Processes tool calls requested by the LLM, executes the tools,
        and collects results formatted as 'tool' role messages for the chat history.

        Args:
            tool_calls (list): List of tool call objects from the LLM API response.

        Returns:
            List[Dict[str, Any]]: A list of message dictionaries, one for each observation,
                                   ready to be added to chat history.
        """
        observation_messages = []
        if not isinstance(tool_calls, list):
             print(Fore.RED + f"Error: Expected a list of tool_calls, got {type(tool_calls)}")
             return observation_messages

        for tool_call in tool_calls:
            if not hasattr(tool_call, 'id') or not hasattr(tool_call, 'function'):
                 print(Fore.YELLOW + f"Warning: Skipping invalid tool call object: {tool_call}")
                 continue

            tool_call_id = tool_call.id
            function_call = tool_call.function
            tool_name = function_call.name
            result_str = f"Error: Tool '{tool_name}' processing failed." # Default error message

            try:
                arguments_str = function_call.arguments
                arguments = json.loads(arguments_str)

                if tool_name not in self.tools_dict:
                    print(Fore.RED + f"Error: Tool '{tool_name}' not found.")
                    result_str = f"Error: Tool '{tool_name}' is not available."
                else:
                    tool = self.tools_dict[tool_name]
                    print(Fore.GREEN + f"\nUsing Tool: {tool_name}")
                    print(Fore.GREEN + f"Tool call ID: {tool_call_id}")
                    print(Fore.GREEN + f"Arguments: {arguments}")

                    # Execute the tool using its run method (which includes validation)
                    try:
                        result = tool.run(**arguments)
                        # Ensure result is serializable
                        if not isinstance(result, (str, int, float, bool, list, dict, type(None))):
                            result_str = str(result)
                        else:
                            try:
                                result_str = json.dumps(result) # Prefer JSON string representation
                            except TypeError:
                                result_str = str(result) # Fallback to string
                        print(Fore.GREEN + f"\nTool result: \n{result_str}")

                    except Exception as e:
                        print(Fore.RED + f"Error running tool {tool_name}: {e}")
                        result_str = f"Error executing tool {tool_name}: {e}"

            except json.JSONDecodeError:
                print(Fore.RED + f"Error: Could not decode arguments for tool {tool_name}: {arguments_str}")
                result_str = f"Error: Invalid arguments JSON provided for {tool_name}"
            except Exception as e:
                print(Fore.RED + f"Error processing tool call arguments for {tool_name}: {e}")
                # Keep default error message

            # Create the message dictionary for this observation
            observation_messages.append(
                build_prompt_structure(role="tool", content=result_str, tool_call_id=tool_call_id)
            )

        return observation_messages

    def run(
        self,
        user_msg: str,
    ) -> str:
        """
        Handles the interaction: user message -> LLM (tool decision) -> execute tools -> LLM (final response).

        Args:
            user_msg (str): The user's message.

        Returns:
            str: The final response from the model.
        """
        initial_user_message = build_prompt_structure(role="user", content=user_msg)

        chat_history = ChatHistory(
            [
                build_prompt_structure(role="system", content=self.system_prompt),
                initial_user_message,
            ]
        )

        print(Fore.CYAN + "\n--- Calling LLM for Tool Decision ---")
        # First call to LLM to decide if tools are needed
        assistant_message_1 = completions_create(
            self.client,
            messages=list(chat_history),
            model=self.model,
            tools=self.tool_schemas, # Provide tool schemas
            tool_choice="auto"       # Let model decide
        )

        # Add assistant's first response (which might contain tool calls) to history
        update_chat_history(chat_history, assistant_message_1)

        final_response = "Agent encountered an issue." # Default

        # Check if tools were called
        if hasattr(assistant_message_1, 'tool_calls') and assistant_message_1.tool_calls:
            print(Fore.YELLOW + "\nAssistant requests tool calls:")
            # Process tool calls and get observation messages
            observation_messages = self.process_tool_calls(assistant_message_1.tool_calls)
            print(Fore.BLUE + f"\nObservations prepared for LLM: {observation_messages}") # Show observations being sent back

            # Add observation messages to history
            for obs_msg in observation_messages:
                update_chat_history(chat_history, obs_msg)

            print(Fore.CYAN + "\n--- Calling LLM for Final Response ---")
            # Second call to LLM for the final response based on observations
            assistant_message_2 = completions_create(
                self.client,
                messages=list(chat_history),
                model=self.model,
                # No tools needed for the final response generation
                # tools=None,
                # tool_choice="none" # Optional: explicitly prevent tool use here
            )
            final_response = str(assistant_message_2.content) if assistant_message_2.content else "Agent did not provide a final response after using tools."

        elif assistant_message_1.content is not None:
            # If no tool calls were made, the first response is the final response
            print(Fore.CYAN + "\nAssistant provided direct response (no tools used):")
            final_response = assistant_message_1.content
        else:
            # Handle unexpected case: no tool calls and no content
            print(Fore.RED + "Error: Assistant message has neither content nor tool calls.")
            final_response = "Error: Received an unexpected empty response from the assistant."

        print(Fore.GREEN + f"\nFinal Response:\n{final_response}")
        return final_response

# --- END OF REFACTORED tool_agent.py ---