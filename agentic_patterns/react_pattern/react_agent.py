# --- START OF MODIFIED react_agent.py ---

import json
import re
from typing import List, Dict, Any, Optional

from colorama import Fore
from dotenv import load_dotenv
from groq import Groq

from agentic_patterns.tool_pattern.tool import Tool
from agentic_patterns.utils.completions import build_prompt_structure
from agentic_patterns.utils.completions import ChatHistory
from agentic_patterns.utils.completions import completions_create
from agentic_patterns.utils.completions import update_chat_history
# Remove extract_tag_content if no longer needed for other tags
# from agentic_patterns.utils.extraction import extract_tag_content

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
    and executes tool calls.

    Attributes:
        client (Groq): The Groq client used to handle model-based completions.
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
        self.client = Groq()
        self.model = model
        # Combine provided system prompt with the modified core instructions
        self.system_prompt = (system_prompt + "\n\n" + CORE_SYSTEM_PROMPT).strip() # Uses the updated CORE_SYSTEM_PROMPT

        if tools is None:
            self.tools = []
        else:
            self.tools = tools if isinstance(tools, list) else [tools]

        self.tools_dict = {tool.name: tool for tool in self.tools}
        self.tool_schemas = [tool.fn_schema for tool in self.tools]

    def process_tool_calls(self, tool_calls: List[Any]) -> dict:
        observations = {}
        if not isinstance(tool_calls, list):
             print(Fore.RED + f"Error: Expected a list of tool_calls, got {type(tool_calls)}")
             return observations

        for tool_call in tool_calls:
            if not hasattr(tool_call, 'id') or not hasattr(tool_call, 'function'):
                 print(Fore.YELLOW + f"Warning: Skipping invalid tool call object: {tool_call}")
                 continue

            tool_call_id = tool_call.id
            function_call = tool_call.function
            tool_name = function_call.name
            try:
                arguments_str = function_call.arguments
                arguments = json.loads(arguments_str)
            except json.JSONDecodeError:
                print(Fore.RED + f"Error: Could not decode arguments for tool {tool_name}: {arguments_str}")
                observations[tool_call_id] = f"Error: Invalid arguments JSON provided for {tool_name}"
                continue
            except Exception as e:
                print(Fore.RED + f"Error processing arguments for tool {tool_name}: {e}")
                observations[tool_call_id] = f"Error: Could not process arguments for {tool_name}"
                continue

            if tool_name not in self.tools_dict:
                print(Fore.RED + f"Error: Tool '{tool_name}' not found.")
                observations[tool_call_id] = f"Error: Tool '{tool_name}' is not available."
                continue

            tool = self.tools_dict[tool_name]
            print(Fore.GREEN + f"\nUsing Tool: {tool_name}")
            print(Fore.GREEN + f"Tool call ID: {tool_call_id}")
            print(Fore.GREEN + f"Arguments: {arguments}")

            try:
                 result = tool.run(**arguments)
                 if not isinstance(result, (str, int, float, bool, list, dict, type(None))):
                     result_str = str(result)
                 else:
                     try:
                          result_str = json.dumps(result)
                     except TypeError:
                          result_str = str(result)
            except Exception as e:
                 print(Fore.RED + f"Error running tool {tool_name}: {e}")
                 result_str = f"Error executing tool {tool_name}: {e}"

            print(Fore.GREEN + f"Tool result: {result_str}")
            observations[tool_call_id] = result_str

        return observations

    def run(
        self,
        user_msg: str,
        max_rounds: int = 5,
    ) -> str:
        initial_user_message = build_prompt_structure(role="user", content=user_msg)

        chat_history = ChatHistory(
            [
                build_prompt_structure(role="system", content=self.system_prompt), # Ensure self.system_prompt uses the new CORE_SYSTEM_PROMPT
                initial_user_message,
            ]
        )

        final_response = "Agent failed to produce a response." # Default value

        for round_num in range(max_rounds):
            print(Fore.CYAN + f"\n--- Round {round_num + 1} ---")
            current_tools = self.tool_schemas if self.tools else None
            current_tool_choice = "auto" if self.tools else "none"

            assistant_message = completions_create(
                self.client,
                messages=list(chat_history),
                model=self.model,
                tools=current_tools,
                tool_choice=current_tool_choice
            )

            # --- Thought Extraction and Printing (Prefix Method) ---
            assistant_content = None
            extracted_thought = None
            # Variable to store content potentially identified as final response
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
                        in_response = False # Cannot be in both
                        # Capture the rest of the line after "Thought:"
                        thought_content = stripped_line[len("Thought:"):].strip()
                        if thought_content: # Add if not empty
                             thought_lines.append(thought_content)
                    elif stripped_line.startswith("Final Response:"):
                         in_response = True
                         in_thought = False # Cannot be in both
                         # Capture the rest of the line after "Final Response:"
                         response_content = stripped_line[len("Final Response:"):].strip()
                         if response_content: # Add if not empty
                              response_lines.append(response_content)
                    elif in_thought:
                         # Continue capturing multi-line thought
                         thought_lines.append(line) # Keep original indentation/spacing
                    elif in_response:
                         # Continue capturing multi-line response
                         response_lines.append(line) # Keep original indentation/spacing
                    # Lines before the first marker or between markers might be ignored or handled differently if needed

                 if thought_lines:
                     extracted_thought = "\n".join(thought_lines).strip()
                     print(Fore.MAGENTA + f"\nThought: {extracted_thought}")

                 if response_lines:
                      potential_final_response = "\n".join(response_lines).strip()


            # --- Update History ---
            # Add the original assistant message (potentially including prefixes) to history
            update_chat_history(chat_history, assistant_message)

            # --- Process Tool Calls or Identify Final Response ---
            has_tool_calls = hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls

            if has_tool_calls:
                # If thought was extracted, it's already printed. LLM decided on tool call.
                print(Fore.YELLOW + "\nAssistant requests tool calls:")
                observations = self.process_tool_calls(assistant_message.tool_calls)
                print(Fore.BLUE + f"\nObservations: {observations}")

                for tool_call in assistant_message.tool_calls:
                     tool_call_id = tool_call.id
                     result = observations.get(tool_call_id, "Error: Observation not found.")
                     tool_message = build_prompt_structure(role="tool", content=str(result), tool_call_id=tool_call_id)
                     update_chat_history(chat_history, tool_message)

            # Check for final response: If the content had "Final Response:" AND no tool calls were made
            elif potential_final_response is not None:
                print(Fore.CYAN + "\nAssistant provides final response:")
                final_response = potential_final_response
                print(Fore.GREEN + final_response)
                return final_response # Exit loop and return the final response

            # Handle case where there was content, but no "Final Response:" prefix and no tool call
            # Could be an error, or maybe the LLM just output text without the prefix.
            elif assistant_content is not None and not has_tool_calls:
                 print(Fore.YELLOW + "\nAssistant provided content without 'Final Response:' prefix and no tool calls.")
                 # Decide how to handle this: return the raw content, or consider it an error?
                 # Returning raw content might be safer initially.
                 final_response = assistant_content # Return the raw content
                 print(Fore.GREEN + final_response)
                 return final_response


            elif not has_tool_calls and assistant_content is None:
                 # Handles cases like API errors returning minimal message objects
                 print(Fore.RED + "Error: Assistant message has neither content nor tool calls.")
                 final_response = "Error: Received an unexpected empty or invalid response from the assistant."
                 return final_response # Exit loop on error


        print(Fore.YELLOW + f"\nMaximum rounds ({max_rounds}) reached.")
        # If loop finishes, check if the last assistant message could be interpreted as a final response
        if potential_final_response and not has_tool_calls:
            final_response = potential_final_response
            print(Fore.GREEN + f"(Last response from agent): {final_response}")
        elif assistant_content and not has_tool_calls: # Fallback to last content if no prefix found
             final_response = assistant_content
             print(Fore.GREEN + f"(Last raw content from agent): {final_response}")
        else:
            final_response = "Agent stopped after maximum rounds without reaching a final answer."
            print(Fore.YELLOW + final_response)

        return final_response

# --- END OF MODIFIED react_agent.py ---