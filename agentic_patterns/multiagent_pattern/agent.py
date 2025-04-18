# --- START OF MODIFIED agent.py ---

from textwrap import dedent
from typing import Any # Import Any

from agentic_patterns.multiagent_pattern.team import Team
from agentic_patterns.react_pattern.react_agent import ReactAgent
from agentic_patterns.tool_pattern.tool import Tool


class Agent:
    """
    Represents an AI agent that can work as part of a team to complete tasks.

    This class implements an agent with dependencies, context handling, and task execution capabilities.
    It can be used in a multi-agent system where agents collaborate to solve complex problems.

    Attributes:
        name (str): The name of the agent.
        backstory (str): The backstory or background of the agent.
        task_description (str): A description of the task assigned to the agent.
        task_expected_output (str): The expected format or content of the task output.
        react_agent (ReactAgent): An instance of ReactAgent used for generating responses.
        dependencies (list[Agent]): A list of Agent instances that this agent depends on.
        dependents (list[Agent]): A list of Agent instances that depend on this agent.
        received_context (dict[str, Any]): Accumulated structured context from dependency agents.

    Args:
        name (str): The name of the agent.
        backstory (str): The backstory or background of the agent.
        task_description (str): A description of the task assigned to the agent.
        task_expected_output (str, optional): The expected format or content of the task output. Defaults to "".
        tools (list[Tool] | None, optional): A list of Tool instances available to the agent. Defaults to None.
        llm (str, optional): The name of the language model to use. Defaults to "llama-3.3-70b-versatile".
    """

    def __init__(
        self,
        name: str,
        backstory: str,
        task_description: str,
        task_expected_output: str = "",
        tools: list[Tool] | None = None,
        llm: str = "llama-3.3-70b-versatile",
    ):
        self.name = name
        self.backstory = backstory
        self.task_description = task_description
        self.task_expected_output = task_expected_output
        self.react_agent = ReactAgent(
            model=llm, system_prompt=self.backstory, tools=tools or []
        )

        self.dependencies: list[Agent] = []
        self.dependents: list[Agent] = []

        # Changed from self.context = "" to a dictionary
        self.received_context: dict[str, Any] = {}

        Team.register_agent(self)

    def __repr__(self):
        return f"{self.name}"

    def __rshift__(self, other):
        self.add_dependent(other)
        return other

    def __lshift__(self, other):
        self.add_dependency(other)
        return other

    def __rrshift__(self, other):
        self.add_dependency(other)
        return self

    def __rlshift__(self, other):
        self.add_dependent(other)
        return self

    def add_dependency(self, other):
        if isinstance(other, Agent):
            self.dependencies.append(other)
            other.dependents.append(self)
        elif isinstance(other, list) and all(isinstance(item, Agent) for item in other):
            for item in other:
                self.dependencies.append(item)
                item.dependents.append(self)
        else:
            raise TypeError("The dependency must be an instance or list of Agent.")

    def add_dependent(self, other):
        if isinstance(other, Agent):
            other.dependencies.append(self)
            self.dependents.append(other)
        elif isinstance(other, list) and all(isinstance(item, Agent) for item in other):
            for item in other:
                item.dependencies.append(self)
                self.dependents.append(item)
        else:
            raise TypeError("The dependent must be an instance or list of Agent.")

    # Modified to accept structured data and store it keyed by the sender agent's name
    def receive_context(self, sender_name: str, input_data: Any):
        """
        Receives and stores structured context information from a specific dependency agent.

        Args:
            sender_name (str): The name of the agent sending the context.
            input_data (Any): The context information (e.g., a dictionary) to be added.
        """
        self.received_context[sender_name] = input_data


    # Modified to format the structured context
    def create_prompt(self):
        """
        Creates a prompt for the agent based on its task description, expected output,
        and formatted context from dependencies.

        Returns:
            str: The formatted prompt string.
        """
        # Format the received context dictionary into a string
        context_str = "\n".join(
            f"Context from {name}:\n{str(data)}\n---"
            for name, data in self.received_context.items()
        )
        if not context_str:
            context_str = "No context received from other agents."


        prompt = dedent(
            f"""
        You are an AI agent. You are part of a team of agents working together to complete a task.
        I'm going to give you the task description enclosed in <task_description></task_description> tags. I'll also give
        you the available context from the other agents in <context></context> tags. If the context
        is not available, the <context></context> tags will be empty or indicate no context was received. You'll also receive the task
        expected output enclosed in <task_expected_output></task_expected_output> tags. With all this information
        you need to create the best possible response, always respecting the format as describe in
        <task_expected_output></task_expected_output> tags. If expected output is not available, just create
        a meaningful response to complete the task.

        <task_description>
        {self.task_description}
        </task_description>

        <task_expected_output>
        {self.task_expected_output}
        </task_expected_output>

        <context>
        {context_str}
        </context>

        Your response:
        """
        ).strip()

        return prompt

    # Modified to return a dictionary and pass structured context
    def run(self) -> dict[str, Any]:
        """
        Runs the agent's task and generates the output as a dictionary.

        This method creates a prompt, runs it through the ReactAgent, wraps the output
        in a dictionary, and passes this dictionary to all dependent agents.

        Returns:
            dict[str, Any]: A dictionary containing the agent's output (e.g., {'output': 'result text'}).
        """
        msg = self.create_prompt()
        raw_output = self.react_agent.run(user_msg=msg)

        # Wrap the output in a standard dictionary structure
        output_data = {"output": raw_output}

        # Pass the structured output to all dependents
        for dependent in self.dependents:
            # Pass the sender's name along with the data
            dependent.receive_context(self.name, output_data)

        return output_data

# --- END OF MODIFIED agent.py ---