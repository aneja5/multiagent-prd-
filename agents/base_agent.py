"""Base agent class implementing the ReAct framework.

This module provides the abstract base class that all agents inherit from.
It implements the Think-Act-Observe-Update-Reflect loop.
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI, OpenAIError

from app.config import get_config
from app.logger import get_logger
from app.state import AgentTraceEntry, State


class BaseAgent(ABC):
    """Abstract base class for all agents in the system.

    Implements the ReAct (Reason + Act) framework:
    1. Think: Analyze the current state
    2. Act: Call LLM with appropriate prompt
    3. Observe: Parse the response
    4. Update: Modify the state based on observations
    5. Reflect: Log actions for transparency

    Attributes:
        name: The agent's identifier
        client: OpenAI client for API calls
        config: Application configuration
        logger: Logger instance for this agent
        turn_counter: Tracks execution turns for this agent
    """

    def __init__(self, name: str, client: OpenAI) -> None:
        """Initialize the base agent.

        Args:
            name: The agent's identifier (e.g., "clarification", "research_planner")
            client: Configured OpenAI client instance
        """
        self.name = name
        self.client = client
        self.config = get_config()
        self.logger = get_logger(f"agent.{name}")
        self.turn_counter = 0

    @abstractmethod
    def run(self, state: State) -> State:
        """Execute the agent's main logic using the ReAct loop.

        This method should be implemented by each specific agent.
        It follows the pattern:
        1. Think: Analyze state and determine if action is needed
        2. Act: Call LLM with appropriate prompts
        3. Observe: Parse LLM response
        4. Update: Modify state based on observations
        5. Reflect: Log actions taken

        Args:
            state: The current shared state

        Returns:
            The updated state

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement run()")

    def _load_prompt(self, prompt_name: Optional[str] = None) -> str:
        """Load a prompt template from file.

        Args:
            prompt_name: Optional name of the prompt file (defaults to agent name)

        Returns:
            The prompt template as a string

        Raises:
            FileNotFoundError: If the prompt file doesn't exist
        """
        if prompt_name is None:
            prompt_name = self.name

        prompt_path = Path(f"agents/prompts/{prompt_name}.txt")

        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_path}. "
                f"Please create a prompt for the {self.name} agent."
            )

        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Failed to load prompt {prompt_path}: {e}")
            raise

    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Call the OpenAI API with error handling and retry logic.

        Args:
            messages: List of message dictionaries for the chat API
            response_format: Optional response format specification (for JSON mode)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens in response

        Returns:
            The parsed response from the API

        Raises:
            OpenAIError: If all retry attempts fail
        """
        retries = 0
        last_exception = None

        while retries < self.config.max_retries:
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.config.openai_model,
                    "messages": messages,
                    "temperature": temperature,
                }

                if response_format:
                    kwargs["response_format"] = response_format

                if max_tokens:
                    kwargs["max_tokens"] = max_tokens

                self.logger.debug(f"Calling OpenAI API (attempt {retries + 1})")

                response = self.client.chat.completions.create(**kwargs)

                # Extract the response content
                result = {
                    "content": response.choices[0].message.content,
                    "role": response.choices[0].message.role,
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0,
                    }
                }

                self.logger.debug(
                    f"API call successful. Tokens used: {result['usage']['total_tokens']}"
                )

                return result

            except OpenAIError as e:
                last_exception = e
                retries += 1

                self.logger.warning(
                    f"OpenAI API error (attempt {retries}/{self.config.max_retries}): {e}"
                )

                if retries < self.config.max_retries:
                    sleep_time = self.config.retry_delay * retries
                    self.logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)

        # All retries exhausted
        self.logger.error(f"All retry attempts failed for {self.name} agent")
        raise OpenAIError(
            f"Failed to call OpenAI API after {self.config.max_retries} attempts: {last_exception}"
        )

    def _log_action(
        self,
        state: State,
        action: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an entry to the agent execution trace.

        Args:
            state: The current state
            action: Description of the action taken
            details: Optional additional details about the action
        """
        self.turn_counter += 1

        trace_entry = AgentTraceEntry(
            agent=self.name,
            turn=self.turn_counter,
            action=action,
            details=details
        )

        state.agent_trace.append(trace_entry)

        self.logger.info(f"[Turn {self.turn_counter}] {action}")

    def _think(self, state: State) -> Dict[str, Any]:
        """Analyze the current state and determine what needs to be done.

        This is a helper method that agents can override to implement
        their thinking phase.

        Args:
            state: The current state

        Returns:
            A dictionary with the agent's analysis
        """
        return {
            "should_act": True,
            "reasoning": "Default thinking - agent should act"
        }

    def _observe(self, llm_response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate the LLM response.

        This is a helper method that agents can override to implement
        custom response parsing.

        Args:
            llm_response: The raw response from _call_llm

        Returns:
            Parsed and validated observations
        """
        return {
            "raw_content": llm_response.get("content", ""),
            "valid": True
        }

    def _update_state(self, state: State, observations: Dict[str, Any]) -> State:
        """Update the state based on observations.

        This is a helper method that agents must override to implement
        their state update logic.

        Args:
            state: The current state
            observations: Parsed observations from _observe

        Returns:
            The updated state
        """
        self.logger.warning(f"{self.name} agent using default _update_state")
        return state
