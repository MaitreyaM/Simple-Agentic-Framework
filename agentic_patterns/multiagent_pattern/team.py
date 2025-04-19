# --- START OF ASYNC MODIFIED team.py ---

import asyncio # Import asyncio
from collections import deque
from typing import Any, List # Import List

from colorama import Fore
from graphviz import Digraph # type: ignore

from agentic_patterns.utils.logging import fancy_print
# Assuming Agent class is imported and its run method is now async
from agentic_patterns.multiagent_pattern.agent import Agent


class Team:
    """
    A class representing a team of agents working together asynchronously.

    This class manages a group of agents, their dependencies, and provides methods
    for running the agents in a topologically sorted order using async/await.

    Attributes:
        current_team (Team | None): Class-level variable to track the active Team context. None if no team context is active.
        agents (list[Agent]): A list of agents in the team.
    """

    current_team = None

    def __init__(self):
        self.agents: List[Agent] = [] # Use List for type hint
        # Stores the final results of each agent run
        self.results: dict[str, Any] = {}

    def __enter__(self):
        Team.current_team = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Team.current_team = None

    # Type hint for agent
    def add_agent(self, agent: Agent):
        if agent not in self.agents:
            self.agents.append(agent)

    # Type hint for agent
    @staticmethod
    def register_agent(agent: Agent):
        if Team.current_team is not None:
            Team.current_team.add_agent(agent)

    # topological_sort remains synchronous as it only deals with graph structure
    def topological_sort(self) -> List[Agent]:
        in_degree: Dict[Agent, int] = {agent: 0 for agent in self.agents} # Type hint
        adj: Dict[Agent, List[Agent]] = {agent: [] for agent in self.agents} # Type hint
        agent_map: Dict[str, Agent] = {agent.name: agent for agent in self.agents} # Type hint

        for agent in self.agents:
            valid_dependencies = [dep for dep in agent.dependencies if dep in self.agents]
            agent.dependencies = valid_dependencies

            for dependency in agent.dependencies:
                if dependency in agent_map.values():
                    adj[dependency].append(agent)
                    in_degree[agent] += 1

        queue: deque[Agent] = deque([agent for agent in self.agents if in_degree[agent] == 0]) # Type hint
        sorted_agents: List[Agent] = [] # Type hint


        while queue:
            current_agent = queue.popleft()
            sorted_agents.append(current_agent)

            # Find dependents (agents that have current_agent as a dependency)
            for potential_dependent in self.agents:
                if current_agent in potential_dependent.dependencies:
                    in_degree[potential_dependent] -= 1
                    if in_degree[potential_dependent] == 0:
                        queue.append(potential_dependent)


        if len(sorted_agents) != len(self.agents):
            detected_agents = {agent.name for agent in sorted_agents}
            missing_agents = {agent.name for agent in self.agents} - detected_agents
            remaining_degrees = {agent.name: in_degree[agent] for agent in self.agents if agent not in sorted_agents}

            raise ValueError(
                "Circular dependencies detected. Cannot perform topological sort. "
                f"Agents processed: {list(detected_agents)}. "
                f"Agents potentially in cycle (or dependent on cycle): {list(missing_agents)}. "
                 f"Remaining in-degrees: {remaining_degrees}"
            )

        return sorted_agents

    def plot(self):
        dot = Digraph(format="png")
        for agent in self.agents:
            dot.node(agent.name)
            for dependent in agent.dependents:
                 if dependent in self.agents:
                    dot.edge(agent.name, dependent.name)
        return dot


    # Changed function definition to async def
    async def run(self):
        """
        Runs all agents in the team asynchronously in topologically sorted order.

        This method awaits each agent's async run method, stores the structured result,
        and prints the agent's name and its result.
        """
        try:
            sorted_agents = self.topological_sort()
        except ValueError as e:
            print(Fore.RED + f"Error during team setup: {e}")
            return

        self.results = {}

        for agent in sorted_agents:
            fancy_print(f"RUNNING AGENT: {agent.name}")
            try:
                 # Use await to call the async agent.run() method
                agent_result = await agent.run()
                self.results[agent.name] = agent_result

                if isinstance(agent_result, dict) and 'output' in agent_result:
                    print(Fore.GREEN + f"Agent {agent.name} Result:\n{agent_result['output']}")
                else:
                    print(Fore.YELLOW + f"Agent {agent.name} Result (raw):\n{str(agent_result)}")

            except Exception as e:
                # Consider logging the traceback for better debugging
                # import traceback
                # print(Fore.RED + f"Error running agent {agent.name}: {e}\n{traceback.format_exc()}")
                print(Fore.RED + f"Error running agent {agent.name}: {e}")
                # Continue running other agents unless critical error handling is needed


# --- END OF ASYNC MODIFIED team.py ---