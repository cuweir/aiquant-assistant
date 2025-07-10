# import google.generativeai as genai
# from httpx import AsyncClient  # Still useful if you ever need direct async HTTP
# from .config import settings
#
#
# class LLMClient:
#     def __init__(self, api_key: str = settings.GEMINI_API_KEY, model_name: str = settings.GEMINI_MODEL_NAME):
#         self.api_key = api_key
#         self.model_name = model_name
#
#         if not self.api_key:
#             raise ValueError("GEMINI_API_KEY is not set.")
#
#         genai.configure(api_key=self.api_key)
#
#         # For safety configurations, generation configurations, etc.
#         # These are examples, adjust as needed.
#         self.generation_config = {
#             "temperature": 0.5,  # Adjust for creativity vs. determinism
#             "top_p": 1,
#             "top_k": 1,
#             "max_output_tokens": 500,  # Adjust based on expected output length
#         }
#         self.safety_settings = [  # Example safety settings
#             {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#             {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#             {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#             {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#         ]
#
#         try:
#             self.model = genai.GenerativeModel(
#                 model_name=self.model_name,
#                 generation_config=self.generation_config,
#                 safety_settings=self.safety_settings
#             )
#             print(f"Gemini LLM Client initialized with model: {self.model_name}")
#         except Exception as e:
#             print(f"Error initializing Gemini model '{self.model_name}': {e}")
#             raise  # Re-raise the exception to halt if model initialization fails
#
#         self.http_client = AsyncClient()  # Keep for potential future direct HTTP calls
#
#     async def generate_analysis(self, prompt: str) -> str:
#         try:
#             # For Gemini, the content structure might be slightly different
#             # For simple text prompts, just passing the string usually works.
#             # If you were doing chat, you'd pass a list of Content objects.
#             response = await self.model.generate_content_async(prompt)  # Use generate_content_async
#
#             # Gemini's response structure might require checking for blocked prompts or errors
#             if not response.candidates:  # Check if there are any candidates
#                 if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
#                     error_message = f"Prompt blocked by Gemini. Reason: {response.prompt_feedback.block_reason}"
#                     if response.prompt_feedback.block_reason_message:
#                         error_message += f" - {response.prompt_feedback.block_reason_message}"
#                     print(error_message)
#                     return f"Error: {error_message}"
#                 else:
#                     print("Gemini response has no candidates and no clear block reason.")
#                     return "Error: Gemini returned no candidates for the prompt."
#
#             # Assuming the first candidate has the text part.
#             # You might need to handle cases where parts are not just text.
#             if response.candidates[0].content.parts and response.candidates[0].content.parts[0].text:
#                 return response.candidates[0].content.parts[0].text.strip()
#             else:
#                 return "Error: Gemini response format not as expected or content is empty."
#
#         except Exception as e:
#             print(f"Error calling Gemini API with model '{self.model_name}': {e}")
#             # In a real app, you'd have more robust error handling & logging
#             return f"Error: Could not get analysis from Gemini LLM. Details: {str(e)}"
#
#     async def close_http_client(self):
#         # The google-generativeai library manages its own connections,
#         # so closing the httpx client is only for other potential uses.
#         await self.http_client.aclose()
#
#
# # Global instance (can be managed with FastAPI dependencies later)
# # Ensure settings are loaded before this instance is created.
# llm_client_instance = LLMClient()