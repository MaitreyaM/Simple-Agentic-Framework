# --- START OF agentic_patterns/llm_services/base.py ---

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# --- Standardized Response ---
# We need a consistent way to represent the LLM's response,
# regardless of the specific API used (Groq, Google GenAI, etc.).
# This includes the main text content and any tool calls requested.

@dataclass
class LLMToolCall:
    """Represents a tool call requested by the LLM."""
    id: str # The unique ID for this specific tool call instance
    function_name: str
    # Arguments are often returned as a JSON string by APIs
    function_arguments_json_str: str

@dataclass
class StandardizedLLMResponse:
    """A consistent format for LLM responses passed back to the agent."""
    # The main text content, if any. Can be None if only tool calls are made.
    text_content: Optional[str] = None
    # A list of tool calls requested by the LLM in this turn. Empty if none.
    tool_calls: List[LLMToolCall] = field(default_factory=list)
    # Optional: Add other common fields if needed later, like finish_reason

# --- LLM Service Interface (Abstract Base Class) ---

class LLMServiceInterface(abc.ABC):
    """
    Abstract Base Class defining the interface for interacting with different LLM backends.
    Concrete implementations (e.g., for Groq, Google GenAI) will inherit from this.
    """

    @abc.abstractmethod
    async def get_llm_response(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        # Optional: Add other common configuration parameters if needed later
        # temperature: Optional[float] = None,
        # max_tokens: Optional[int] = None,
    ) -> StandardizedLLMResponse:
        """
        Sends messages to the configured LLM backend and returns a standardized response.

        Args:
            model: The specific model identifier for the backend.
            messages: The chat history in a list of dictionaries format
                      (e.g., [{'role': 'user', 'content': 'Hello'}]).
                      Implementations will need to translate this if their
                      native API uses a different format.
            tools: A list of tool schemas available for the LLM to use,
                   formatted according to the OpenAI/Groq standard
                   `{"type": "function", "function": {...}}`.
                   Implementations will need to translate this if their
                   native API uses a different format.
            tool_choice: How the LLM should use tools (e.g., "auto", "none").

        Returns:
            A StandardizedLLMResponse object containing the text content and/or
            tool calls requested by the LLM.

        Raises:
            Exception: Can raise exceptions if the API call fails.
        """
        pass

    # Optional: Add other common methods if needed, e.g., for embedding generation
    # @abc.abstractmethod
    # async def get_embedding(self, text: str, model: str) -> List[float]:
    #     pass

# --- END OF agentic_patterns/llm_services/base.py ---