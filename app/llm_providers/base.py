from abc import ABC, abstractmethod
from typing import List, Dict, Optional # For potential future chat methods

class LLMStrategy(ABC):
    @abstractmethod
    async def generate_analysis(self, prompt: str) -> str:
        """
        Generates analysis text based on the given prompt.
        Should handle API calls and error responses for the specific LLM provider.
        """
        pass

    # Optional: If you plan for chat-like interactions in the future
    # @abstractmethod
    # async def generate_chat_completion(self, messages: List[Dict[str, str]]) -> str:
    #     pass

    async def close_clients(self):
        """Optional: Close any persistent clients if necessary."""
        pass