# --- START OF MODIFIED agent.py (Circular Import Fix) ---

import asyncio
import json # Import json for create_prompt formatting
from textwrap import dedent
from typing import Any, List, Optional

# REMOVE this top-level import:
# from agentic_patterns.multiagent_pattern.team import Team

# Keep these imports
from agentic_patterns.react_pattern.react_agent import ReactAgent
from agentic_patterns.tool_pattern.tool import Tool


class Agent:
    """
    Represents an AI agent that can work as part of a team to complete tasks.
    ... (rest of docstring) ...
    """

    def __init__(
        self,
        name: str,
        backstory: str,
        task_description: str,
        task_expected_output: str = "",
        tools: Optional[List[Tool]] = None,
        llm: str = "llama-3.3-70b-versatile",
    ):
        self.name = name
        self.backstory = backstory
        self.task_description = task_description
        self.task_expected_output = task_expected_output
        self.react_agent = ReactAgent(
            model=llm, system_prompt=self.backstory, tools=tools # Pass tools correctly
        )

        # Use string literal for Agent type hint to avoid needing Agent import at top level in team.py
        self.dependencies: List['Agent'] = []
        self.dependents: List['Agent'] = []

        self.received_context: dict[str, Any] = {}

        # Import Team *inside* the method, right before use
        from agentic_patterns.multiagent_pattern.team import Team
        Team.register_agent(self)

    def __repr__(self):
        return f"{self.name}"

    # Type hint 'Agent' as string literal
    def __rshift__(self, other: 'Agent') -> 'Agent':
        self.add_dependent(other)
        return other

    # Type hint 'Agent' as string literal
    def __lshift__(self, other: 'Agent') -> 'Agent':
        self.add_dependency(other)
        return other

    # Type hint 'Agent' as string literal
    def __rrshift__(self, other: List['Agent'] | 'Agent'):
        self.add_dependency(other)
        return self

    # Type hint 'Agent' as string literal
    def __rlshift__(self, other: List['Agent'] | 'Agent'):
        self.add_dependent(other)
        return self

    # Type hint 'Agent' as string literal
    def add_dependency(self, other: 'Agent' | List['Agent']):
        # Check type without importing Agent directly at top level
        AgentClass = type(self)
        if isinstance(other, AgentClass):
            if other not in self.dependencies:
                self.dependencies.append(other)
            if self not in other.dependents:
                other.dependents.append(self)
        elif isinstance(other, list) and all(isinstance(item, AgentClass) for item in other):
            for item in other:
                 if item not in self.dependencies:
                     self.dependencies.append(item)
                 if self not in item.dependents:
                     item.dependents.append(self)
        else:
            raise TypeError("The dependency must be an instance or list of Agent.")

    # Type hint 'Agent' as string literal
    def add_dependent(self, other: 'Agent' | List['Agent']):
        AgentClass = type(self)
        if isinstance(other, AgentClass):
            if self not in other.dependencies:
                other.dependencies.append(self)
            if other not in self.dependents:
                self.dependents.append(other)
        elif isinstance(other, list) and all(isinstance(item, AgentClass) for item in other):
            for item in other:
                if self not in item.dependencies:
                     item.dependencies.append(self)
                if item not in self.dependents:
                     self.dependents.append(item)
        else:
            raise TypeError("The dependent must be an instance or list of Agent.")

    def receive_context(self, sender_name: str, input_data: Any):
        """
        Receives and stores structured context information from a specific dependency agent.

        Args:
            sender_name (str): The name of the agent sending the context.
            input_data (Any): The context information (e.g., a dictionary) to be added.
        """
        self.received_context[sender_name] = input_data

    def create_prompt(self) -> str:
        """
        Creates a prompt for the agent based on its task description, expected output,
        and formatted context from dependencies.

        Returns:
            str: The formatted prompt string.
        """
        context_str = "\n---\n".join(
            f"Context from {name}:\n{json.dumps(data, indent=2) if isinstance(data, dict) else str(data)}"
            for name, data in self.received_context.items()
        )
        if not context_str:
            context_str = "No context received from other agents."

        prompt = dedent(
            f"""
        You are an AI agent named {self.name}. Your backstory: {self.backstory}
        You are part of a team of agents working together to complete a task.
        Your immediate task is described below. Use the provided context from other agents if relevant.

        <task_description>
        {self.task_description}
        </task_description>

        <task_expected_output>
        {self.task_expected_output or 'Produce a meaningful response to complete the task.'}
        </task_expected_output>

        <context>
        {context_str}
        </context>

        Now, execute your task based on the description, context, and expected output. Your response:
        """
        ).strip()

        return prompt

    async def run(self) -> dict[str, Any]:
        """
        Runs the agent's task asynchronously and generates the output as a dictionary.

        This method creates a prompt, runs it through the ReactAgent asynchronously,
        wraps the output in a dictionary, and passes this dictionary to all dependent agents.

        Returns:
            dict[str, Any]: A dictionary containing the agent's output (e.g., {'output': 'result text'}).
        """
        msg = self.create_prompt()
        raw_output = await self.react_agent.run(user_msg=msg)

        output_data = {"output": raw_output}

        for dependent in self.dependents:
            dependent.receive_context(self.name, output_data)

        return output_data

# --- END OF MODIFIED agent.py (Circular Import Fix) ---