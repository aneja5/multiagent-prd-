"""Orchestrator for coordinating multiple agents.

This module manages the execution flow of all agents and ensures
they work together to generate the PRD.
"""

from typing import List, Optional

from openai import OpenAI

from app.config import get_config
from app.logger import get_logger
from app.state import State, save_state
from agents.base_agent import BaseAgent


class Orchestrator:
    """Coordinates the execution of multiple agents to generate a PRD.

    The orchestrator manages the agent execution loop, determines which
    agent should run next, and handles the overall workflow.

    Attributes:
        client: OpenAI client for agent use
        agents: List of registered agents
        config: Application configuration
        logger: Logger instance
    """

    def __init__(self, client: OpenAI) -> None:
        """Initialize the orchestrator.

        Args:
            client: Configured OpenAI client instance
        """
        self.client = client
        self.agents: List[BaseAgent] = []
        self.config = get_config()
        self.logger = get_logger(__name__)

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator.

        Args:
            agent: The agent to register
        """
        self.agents.append(agent)
        self.logger.info(f"Registered agent: {agent.name}")

    def run(self, state: State, max_iterations: int = 50) -> State:
        """Execute the orchestration loop.

        This is the main execution loop that coordinates all agents.
        Currently a skeleton implementation that will be expanded.

        Args:
            state: The current state
            max_iterations: Maximum number of iterations to prevent infinite loops

        Returns:
            The final state after orchestration

        TODO:
            - Implement agent selection logic
            - Add workflow coordination
            - Handle blocking and completion conditions
        """
        self.logger.info(f"Starting orchestration for run {state.run_id}")
        self.logger.info(f"Registered agents: {[a.name for a in self.agents]}")

        iteration = 0

        while iteration < max_iterations and state.status == "running":
            iteration += 1
            self.logger.info(f"Orchestration iteration {iteration}/{max_iterations}")

            # Select next agent to run
            next_agent = self.select_next_agent(state)

            if next_agent is None:
                self.logger.info("No more agents to run, marking as done")
                state.status = "done"
                break

            # Execute the selected agent
            self.logger.info(f"Running agent: {next_agent.name}")
            try:
                state = next_agent.run(state)
                save_state(state)
            except Exception as e:
                self.logger.error(f"Agent {next_agent.name} failed: {e}")
                state.status = "blocked"
                save_state(state)
                break

        if iteration >= max_iterations:
            self.logger.warning(
                f"Orchestration stopped: reached max iterations ({max_iterations})"
            )
            state.status = "blocked"

        self.logger.info(f"Orchestration completed with status: {state.status}")

        return state

    def select_next_agent(self, state: State) -> Optional[BaseAgent]:
        """Determine which agent should run next based on state.

        Current simple logic:
        1. If clarification hasn't been run yet (no domain set), run clarification agent
        2. Otherwise, no more agents (for now)

        TODO: Implement more sophisticated agent selection logic.

        Args:
            state: The current state

        Returns:
            The next agent to run, or None if orchestration should stop
        """
        # Simple workflow for now: just run clarification once
        # Check if clarification has been run (domain is set)
        if not state.metadata.domain:
            # Clarification hasn't run yet
            for agent in self.agents:
                if agent.name == "clarification":
                    return agent

        # No more agents to run (or clarification already completed)
        return None
