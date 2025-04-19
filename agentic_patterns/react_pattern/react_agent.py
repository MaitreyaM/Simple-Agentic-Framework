# --- START OF ASYNC CORRECTED react_agent.py ---

import json
import re
from typing import List, Dict, Any, Optional
import asyncio

from colorama import Fore
from dotenv import load_dotenv
# Import AsyncGroq instead of Groq
from groq import AsyncGroq

from agentic_patterns.tool_pattern.tool import Tool
from agentic_patterns.utils.completions import build_prompt_structure
from agentic_patterns.utils.completions import ChatHistory
# Ensure this function uses await client.chat.completions.create
from agentic_patterns.utils.completions import completions_create
from agentic_patterns.utils.completions import update_chat_history

load_dotenv()

# CORE_SYSTEM_PROMPT using "Thought:" and "Final Response:" prefixes
CORE_SYSTEM_PROMPT = """
You are an AI assistant that uses the ReAct (**Reason**->**Act**) process to answer questions and perform tasks using available tools.

**Your Interaction Loop:**
1.  **Thought:** You MUST first analyze the query/situation and formulate a plan. Start your response **only** with your thought process, prefixed with "**Thought:**" on a new line.
2.  **Action Decision:** Based on your thought, decide if a tool is needed.
3.  **Observation:** If a tool is called, the system will provide the result. Analyze this in your next Thought.
4.  **Final Response:** When you have enough information, provide the final answer. Start this **only** with "**Final Response:**" on a new line, following your final thought.

**Output Syntax:**

*   **For Tool Use:**
    Thought: [Your reasoning and plan to use a tool]
    *(System executes tool based on your thought's intent)*

*   **After Observation:**
    Thought: [Your analysis of the observation and next step]
    *(Either signal another tool use implicitly or provide final response)*

*   **For Final Answer:**
    Thought: [Your final reasoning]
    Final Response: [Your final answer to the user]

---

**Constraint:** Always begin your response content with "Thought:". If providing the final answer, include "Final Response:" after the final thought. Do not add any other text before "Thought:" or "Final Response:" on their respective lines.
"""

class ReactAgent:
    """
    A class that represents an agent using the ReAct logic with native tool calling.
    It interacts with tools via structured API calls, processes user inputs, makes decisions,
    and executes tool calls asynchronously.

    Attributes:
        client (AsyncGroq): The AsyncGroq client used for async model completions.
        model (str): The name of the model used for generating responses.
        tools (list[Tool]): A list of Tool instances available for execution.
        tools_dict (dict): A dictionary mapping tool names to their corresponding Tool instances.
        tool_schemas (list[dict]): A list of JSON schemas for the available tools.
        system_prompt (str): Base system prompt for the agent.
    """

    def __init__(
        self,
        tools: Optional[Tool | list[Tool]] = None,
        model: str = "llama-3.3-70b-versatile",
        system_prompt: str = "",
    ) -> None:
        self.client = AsyncGroq()
        self.model = model
        # Combine provided system prompt with the modified core instructions
        self.system_prompt = (system_prompt + "\n\n" + CORE_SYSTEM_PROMPT).strip() # Uses the updated CORE_SYSTEM_PROMPT

        if tools is None:
            self.tools = []
        else:
            self.tools = tools if isinstance(tools, list) else [tools]

        self.tools_dict = {tool.name: tool for tool in self.tools}
        self.tool_schemas = [tool.fn_schema for tool in self.tools]

    async def process_tool_calls(self, tool_calls: List[Any]) -> dict:
        """
        Processes tool calls requested by the LLM asynchronously, executes the tools,
        and collects results.

        Args:
            tool_calls (list): List of tool call objects from the LLM API response.

        Returns:
            dict: A dictionary where keys are tool call IDs and values are the results from the tools.
        """
        observations = {}
        if not isinstance(tool_calls, list):
             print(Fore.RED + f"Error: Expected a list of tool_calls, got {type(tool_calls)}")
             return observations

        for tool_call in tool_calls:
            tool_call_id = "error_no_id" # Default in case parsing fails early
            tool_name = "error_unknown_name"
            result_str = "Error: Tool processing failed before execution."

            try:
                if not hasattr(tool_call, 'id') or not hasattr(tool_call, 'function'):
                     print(Fore.YELLOW + f"Warning: Skipping invalid tool call object: {tool_call}")
                     continue

                tool_call_id = tool_call.id
                function_call = tool_call.function
                tool_name = function_call.name
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

                    # Await the tool run, which now handles sync/async internally
                    result = await tool.run(**arguments)

                    if not isinstance(result, (str, int, float, bool, list, dict, type(None))):
                        result_str = str(result)
                    else:
                        try:
                            result_str = json.dumps(result)
                        except TypeError:
                            result_str = str(result)

                    print(Fore.GREEN + f"Tool result: {result_str}")

            except json.JSONDecodeError:
                print(Fore.RED + f"Error: Could not decode arguments for tool {tool_name}: {arguments_str}")
                result_str = f"Error: Invalid arguments JSON provided for {tool_name}"
            except Exception as e:
                 print(Fore.RED + f"Error processing or running tool {tool_name} (id: {tool_call_id}): {e}")
                 result_str = f"Error executing tool {tool_name}: {e}"

            observations[tool_call_id] = result_str

        return observations

    async def run(
        self,
        user_msg: str,
        max_rounds: int = 5,
    ) -> str:
        """
        Executes an asynchronous user interaction session using native tool calling.

        Args:
            user_msg (str): The user's input message.
            max_rounds (int): Maximum number of LLM call rounds.

        Returns:
            str: The final response generated by the agent.
        """
        initial_user_message = build_prompt_structure(role="user", content=user_msg)

        chat_history = ChatHistory(
            [
                build_prompt_structure(role="system", content=self.system_prompt),
                initial_user_message,
            ]
        )

        final_response = "Agent failed to produce a response."

        for round_num in range(max_rounds):
            print(Fore.CYAN + f"\n--- Round {round_num + 1} ---")
            current_tools = self.tool_schemas if self.tools else None
            current_tool_choice = "auto" if self.tools else "none"

            # Use await for the async completions_create call
            assistant_message = await completions_create(
                self.client,
                messages=list(chat_history),
                model=self.model,
                tools=current_tools,
                tool_choice=current_tool_choice
            )

            assistant_content = None
            extracted_thought = None
            potential_final_response = None

            if hasattr(assistant_message, 'content') and assistant_message.content is not None:
                 assistant_content = assistant_message.content
                 lines = assistant_content.strip().split('\n')
                 thought_lines = []
                 response_lines = []
                 in_thought = False
                 in_response = False

                 for line in lines:
                    stripped_line = line.strip()
                    if stripped_line.startswith("Thought:"):
                        in_thought = True
                        in_response = False
                        thought_content = stripped_line[len("Thought:"):].strip()
                        if thought_content:
                             thought_lines.append(thought_content)
                    elif stripped_line.startswith("Final Response:"):
                         in_response = True
                         in_thought = False
                         response_content = stripped_line[len("Final Response:"):].strip()
                         if response_content:
                              response_lines.append(response_content)
                    elif in_thought:
                         thought_lines.append(line)
                    elif in_response:
                         response_lines.append(line)

                 if thought_lines:
                     extracted_thought = "\n".join(thought_lines).strip()
                     print(Fore.MAGENTA + f"\nThought: {extracted_thought}")

                 if response_lines:
                      potential_final_response = "\n".join(response_lines).strip()

            update_chat_history(chat_history, assistant_message)

            has_tool_calls = hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls

            if has_tool_calls:
                print(Fore.YELLOW + "\nAssistant requests tool calls:")
                # Use await for the async process_tool_calls
                observations = await self.process_tool_calls(assistant_message.tool_calls)
                print(Fore.BLUE + f"\nObservations: {observations}")

                for tool_call in assistant_message.tool_calls:
                     tool_call_id = tool_call.id
                     result = observations.get(tool_call_id, "Error: Observation not found.")
                     tool_message = build_prompt_structure(role="tool", content=str(result), tool_call_id=tool_call_id)
                     update_chat_history(chat_history, tool_message)

            elif potential_final_response is not None:
                print(Fore.CYAN + "\nAssistant provides final response:")
                final_response = potential_final_response
                print(Fore.GREEN + final_response)
                return final_response

            elif assistant_content is not None and not has_tool_calls:
                 print(Fore.YELLOW + "\nAssistant provided content without 'Final Response:' prefix and no tool calls.")
                 final_response = assistant_content
                 print(Fore.GREEN + final_response)
                 return final_response


            elif not has_tool_calls and assistant_content is None:
                 print(Fore.RED + "Error: Assistant message has neither content nor tool calls.")
                 final_response = "Error: Received an unexpected empty or invalid response from the assistant."
                 return final_response


        print(Fore.YELLOW + f"\nMaximum rounds ({max_rounds}) reached.")
        if potential_final_response and not has_tool_calls:
            final_response = potential_final_response
            print(Fore.GREEN + f"(Last response from agent): {final_response}")
        elif assistant_content and not has_tool_calls:
             final_response = assistant_content
             print(Fore.GREEN + f"(Last raw content from agent): {final_response}")
        else:
            final_response = "Agent stopped after maximum rounds without reaching a final answer."
            print(Fore.YELLOW + final_response)

        return final_response

# --- END OF ASYNC CORRECTED react_agent.py ---