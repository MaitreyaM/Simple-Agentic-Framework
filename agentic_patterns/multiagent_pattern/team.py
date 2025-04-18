# --- START OF MODIFIED team.py ---

from collections import deque
from typing import Any # Import Any

from colorama import Fore
from graphviz import Digraph # type: ignore

from agentic_patterns.utils.logging import fancy_print
# Assuming Agent is imported correctly, e.g.:
# from .agent import Agent


class Team:
    """
    A class representing a team of agents working together.

    This class manages a group of agents, their dependencies, and provides methods
    for running the agents in a topologically sorted order.

    Attributes:
        current_team (Team | None): Class-level variable to track the active Team context. None if no team context is active.
        agents (list['Agent']): A list of agents in the team.
    """

    current_team = None

    def __init__(self):
        self.agents = []
        # Stores the final results of each agent run
        self.results: dict[str, Any] = {}

    def __enter__(self):
        Team.current_team = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Team.current_team = None

    def add_agent(self, agent):
        if agent not in self.agents:
            self.agents.append(agent)

    @staticmethod
    def register_agent(agent):
        if Team.current_team is not None:
            Team.current_team.add_agent(agent)

    def topological_sort(self):
        in_degree = {agent: 0 for agent in self.agents}
        adj = {agent: [] for agent in self.agents}
        agent_map = {agent.name: agent for agent in self.agents}

        # Build adjacency list and calculate in-degrees based on dependencies
        for agent in self.agents:
             # Ensure all dependencies are actually in the team's agent list
            valid_dependencies = [dep for dep in agent.dependencies if dep in self.agents]
            agent.dependencies = valid_dependencies # Update agent's dependencies

            for dependency in agent.dependencies:
                 # Check if dependency is registered before adding edge
                if dependency in agent_map.values():
                    adj[dependency].append(agent)
                    in_degree[agent] += 1
                # else: # Optional: Add warning for unregistered dependency
                #     print(f"Warning: Dependency '{dependency.name}' for agent '{agent.name}' not found in team.")


        # Initialize queue with agents having an in-degree of 0
        queue = deque([agent for agent in self.agents if in_degree[agent] == 0])
        sorted_agents = []


        while queue:
            current_agent = queue.popleft()
            sorted_agents.append(current_agent)

            # Process dependents (agents that depend on current_agent)
            # Need to find dependents correctly. Let's iterate through all agents.
            for potential_dependent in self.agents:
                if current_agent in potential_dependent.dependencies:
                    in_degree[potential_dependent] -= 1
                    if in_degree[potential_dependent] == 0:
                        queue.append(potential_dependent)


        if len(sorted_agents) != len(self.agents):
             # Provide more context for circular dependency error
            detected_agents = {agent.name for agent in sorted_agents}
            missing_agents = {agent.name for agent in self.agents} - detected_agents
            # Calculate remaining in-degrees to pinpoint cycle
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
            # Use agent.dependents for correct edge direction in plot
            for dependent in agent.dependents:
                 # Ensure dependent is also in the team list before drawing edge
                 if dependent in self.agents:
                    dot.edge(agent.name, dependent.name) # Edge from dependency to dependent
        return dot


    # Modified to handle dictionary output from agent.run()
    def run(self):
        """
        Runs all agents in the team in topologically sorted order.

        This method executes each agent's run method, stores the structured result,
        and prints the agent's name and its result.
        """
        try:
            sorted_agents = self.topological_sort()
        except ValueError as e:
            print(Fore.RED + f"Error during team setup: {e}")
            return # Stop execution if sorting fails

        self.results = {} # Clear previous results if any

        for agent in sorted_agents:
            fancy_print(f"RUNNING AGENT: {agent.name}")
            try:
                 # Agent.run now returns a dictionary
                agent_result = agent.run()
                self.results[agent.name] = agent_result # Store the structured result

                 # Print the result (assuming it's in a dict like {'output': ...})
                if isinstance(agent_result, dict) and 'output' in agent_result:
                    print(Fore.GREEN + f"Agent {agent.name} Result:\n{agent_result['output']}")
                else:
                     # Fallback for unexpected result format
                    print(Fore.YELLOW + f"Agent {agent.name} Result (raw):\n{str(agent_result)}")

            except Exception as e:
                print(Fore.RED + f"Error running agent {agent.name}: {e}")
                 # Decide if the team run should stop on error or continue
                 # For now, let's print the error and continue
                 # To stop on error, uncomment the line below
                 # raise e # or return


# --- END OF MODIFIED team.py ---