"""Unified LLM provider interface.

Wraps Google Gemini via LangChain's ChatGoogleGenerativeAI behind a factory
function `get_llm()`. This abstraction makes it easy to switch models or
providers in one place without modifying agent code.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from core.config import get_settings


def get_llm(
    model_name: str | None = None,
    temperature: float = 0.0,
    streaming: bool = False,
) -> BaseChatModel:
    """Factory function that returns a configured Chat LLM instance.

    Args:
        model_name: Name of the Gemini model (e.g. 'gemini-2.5-flash').
                    If None, uses GEMINI_MODEL from .env.
        temperature: Sampling temperature between 0.0 (deterministic) and 1.0 (creative).
                     Agents usually use 0.0 to prevent hallucinations.
        streaming: Whether to stream tokens incrementally.

    Returns:
        A LangChain-compatible BaseChatModel instance.

    Raises:
        ValueError: If GEMINI_API_KEY is not configured in .env.
    """
    settings = get_settings()

    api_key = settings.gemini_api_key
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is missing or contains the default placeholder.\n"
            "Please open .env and set your valid Google AI Studio key:\n"
            "GEMINI_API_KEY=AIzaSy..."
        )

    model = model_name or settings.gemini_model

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        streaming=streaming,
    )


def check_llm_connection() -> str:
    """Quick diagnostic function to verify that the Gemini API is reachable.

    Returns:
        The text response from the model.
    """
    llm = get_llm()
    response = llm.invoke("Respond with: 'LLM connection successful!'")
    return str(response.content)
