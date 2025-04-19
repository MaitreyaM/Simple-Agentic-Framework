# --- START OF ASYNC MODIFIED tool_agent.py ---

import json
from typing import List, Dict, Any, Optional
import asyncio # Import asyncio

from colorama import Fore
from dotenv import load_dotenv
# Import AsyncGroq instead of Groq
from groq import AsyncGroq

from agentic_patterns.tool_pattern.tool import Tool
from agentic_patterns.utils.completions import build_prompt_structure
from agentic_patterns.utils.completions import ChatHistory
# Import the async version of completions_create
from agentic_patterns.utils.completions import completions_create
from agentic_patterns.utils.completions import update_chat_history

load_dotenv()

# Simple system prompt for direct tool use
NATIVE_TOOL_SYSTEM_PROMPT = """
You are a helpful assistant. Use the available tools if necessary to answer the user's request.
If you use a tool, you will be given the results, and then you should provide the final response to the user.
"""

class ToolAgent:
    """
    A simple agent that uses native tool calling asynchronously to answer user queries.
    It makes one attempt to call tools if needed, processes the results,
    and then generates a final response.
    """

    def __init__(
        self,
        tools: Tool | list[Tool],
        model: str = "llama-3.3-70b-versatile",
        system_prompt: str = NATIVE_TOOL_SYSTEM_PROMPT,
    ) -> None:
        # Use AsyncGroq for asynchronous operations
        self.client = AsyncGroq()
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools if isinstance(tools, list) else [tools]
        self.tools_dict = {tool.name: tool for tool in self.tools}
        self.tool_schemas = [tool.fn_schema for tool in self.tools]

    # Changed to async def
    async def process_tool_calls(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """
        Processes tool calls requested by the LLM asynchronously, executes the tools,
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

        # Consider using asyncio.gather here if concurrent tool execution is desired
        for tool_call in tool_calls:
            tool_call_id = "error_no_id"
            tool_name = "error_unknown_name"
            result_str = f"Error: Tool '{tool_name}' processing failed."

            try:
                if not hasattr(tool_call, 'id') or not hasattr(tool_call, 'function'):
                     print(Fore.YELLOW + f"Warning: Skipping invalid tool call object: {tool_call}")
                     continue

                tool_call_id = tool_call.id
                function_call = tool_call.function
                tool_name = function_call.name
                arguments_str = function_call.arguments
                arguments = json.loads(arguments_str) # JSON parsing remains sync

                if tool_name not in self.tools_dict:
                    print(Fore.RED + f"Error: Tool '{tool_name}' not found.")
                    result_str = f"Error: Tool '{tool_name}' is not available."
                else:
                    tool = self.tools_dict[tool_name]
                    print(Fore.GREEN + f"\nUsing Tool: {tool_name}")
                    print(Fore.GREEN + f"Tool call ID: {tool_call_id}")
                    print(Fore.GREEN + f"Arguments: {arguments}")

                    # Execute the tool using its async run method
                    try:
                        # Use await as Tool.run is now async
                        result = await tool.run(**arguments)
                        # Ensure result is serializable (sync processing)
                        if not isinstance(result, (str, int, float, bool, list, dict, type(None))):
                            result_str = str(result)
                        else:
                            try:
                                result_str = json.dumps(result)
                            except TypeError:
                                result_str = str(result)
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

            observation_messages.append(
                build_prompt_structure(role="tool", content=result_str, tool_call_id=tool_call_id)
            )

        return observation_messages

    # Changed to async def
    async def run(
        self,
        user_msg: str,
    ) -> str:
        """
        Handles the asynchronous interaction: user message -> LLM (tool decision) ->
        execute tools -> LLM (final response).

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
        # Use await for the async completions_create call
        assistant_message_1 = await completions_create(
            self.client,
            messages=list(chat_history),
            model=self.model,
            tools=self.tool_schemas,
            tool_choice="auto"
        )

        # Synchronous update
        update_chat_history(chat_history, assistant_message_1)

        final_response = "Agent encountered an issue."

        # Check if tools were called
        if hasattr(assistant_message_1, 'tool_calls') and assistant_message_1.tool_calls:
            print(Fore.YELLOW + "\nAssistant requests tool calls:")
            # Use await for the async process_tool_calls
            observation_messages = await self.process_tool_calls(assistant_message_1.tool_calls)
            print(Fore.BLUE + f"\nObservations prepared for LLM: {observation_messages}")

            # Synchronous loop and update
            for obs_msg in observation_messages:
                update_chat_history(chat_history, obs_msg)

            print(Fore.CYAN + "\n--- Calling LLM for Final Response ---")
            # Use await for the async completions_create call
            assistant_message_2 = await completions_create(
                self.client,
                messages=list(chat_history),
                model=self.model,
            )
            final_response = str(assistant_message_2.content) if assistant_message_2.content else "Agent did not provide a final response after using tools."

        elif assistant_message_1.content is not None:
            print(Fore.CYAN + "\nAssistant provided direct response (no tools used):")
            final_response = assistant_message_1.content
        else:
            print(Fore.RED + "Error: Assistant message has neither content nor tool calls.")
            final_response = "Error: Received an unexpected empty response from the assistant."

        print(Fore.GREEN + f"\nFinal Response:\n{final_response}")
        return final_response

# --- END OF ASYNC MODIFIED tool_agent.py ---