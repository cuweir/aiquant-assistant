import google.generativeai as genai
from httpx import AsyncClient # Still useful for other async HTTP if needed
from .base import LLMStrategy
# No direct import of settings here; configuration will be passed during instantiation

class GeminiLLMStrategy(LLMStrategy):
    def __init__(self, api_key: str, model_name: str, generation_config: dict = None, safety_settings: list = None):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set for GeminiLLMStrategy.")

        genai.configure(api_key=api_key)

        self.model_name = model_name
        self.generation_config = generation_config or {
            "temperature": 0.5,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048, # Increased slightly for comprehensive analysis
        }
        self.safety_settings = safety_settings or [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        try:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            print(f"Gemini LLM Strategy initialized with model: {self.model_name}")
        except Exception as e:
            print(f"Error initializing Gemini model '{self.model_name}': {e}")
            raise

        self._http_client = AsyncClient() # For other potential direct HTTP calls

    async def generate_analysis(self, prompt: str) -> str:
        try:
            response = await self.model.generate_content_async(prompt)

            if not response.candidates:
                error_message = "Gemini response has no candidates"
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    error_message += f". Reason: {response.prompt_feedback.block_reason}"
                    if response.prompt_feedback.block_reason_message:
                        error_message += f" - {response.prompt_feedback.block_reason_message}"
                print(error_message)
                return f"Error: {error_message}"

            # Ensure text part exists
            if response.candidates[0].content and response.candidates[0].content.parts and response.candidates[0].content.parts[0].text:
                return response.candidates[0].content.parts[0].text.strip()
            else:
                # Log the problematic response for debugging
                print(f"Unexpected Gemini response structure or empty content. Response: {response}")
                return "Error: Gemini response format not as expected or content is empty."

        except Exception as e:
            print(f"Error calling Gemini API with model '{self.model_name}': {e}")
            import traceback
            traceback.print_exc()
            return f"Error: Could not get analysis from Gemini LLM. Details: {str(e)}"

    async def close_clients(self):
        # The google-generativeai library manages its own connections for the model.
        # Closing the httpx client is for any other direct use of it.
        await self._http_client.aclose()
        print("GeminiLLMStrategy resources (like its httpx client) potentially closed.")