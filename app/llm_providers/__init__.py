from ..core.config import Settings # Import your global settings
from .base import LLMStrategy
from .gemini_provider import GeminiLLMStrategy
# Import other providers here when you add them, e.g.:
# from .openai_provider import OpenAILLMStrategy

def get_llm_strategy(settings_instance: Settings) -> LLMStrategy:
    """
    Factory function to get the configured LLM strategy.
    """
    provider = settings_instance.ACTIVE_LLM_PROVIDER.lower()

    if provider == "gemini":
        if not settings_instance.GEMINI_API_KEY:
            raise ValueError("ACTIVE_LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is missing in settings.")
        return GeminiLLMStrategy(
            api_key=settings_instance.GEMINI_API_KEY,
            model_name=settings_instance.GEMINI_MODEL_NAME
            # You can pass other Gemini-specific configs from settings if needed
        )
    # elif provider == "openai":
    #     if not settings_instance.OPENAI_API_KEY:
    #         raise ValueError("ACTIVE_LLM_PROVIDER is 'openai' but OPENAI_API_KEY is missing in settings.")
    #     return OpenAILLMStrategy(
    #         api_key=settings_instance.OPENAI_API_KEY,
    #         model_name=settings_instance.OPENAI_MODEL_NAME
    #     )
    # Add more providers here with "elif provider == 'provider_name':"
    else:
        # Default to Gemini if only Gemini is configured, or raise error if provider is unrecognized
        print(f"Warning: ACTIVE_LLM_PROVIDER '{provider}' not explicitly handled, defaulting to Gemini if configured, or raising error.")
        if settings_instance.GEMINI_API_KEY:
             print("Defaulting to Gemini as it's configured.")
             return GeminiLLMStrategy(
                api_key=settings_instance.GEMINI_API_KEY,
                model_name=settings_instance.GEMINI_MODEL_NAME
            )
        raise ValueError(f"Unsupported or unconfigured LLM provider: {settings_instance.ACTIVE_LLM_PROVIDER}")