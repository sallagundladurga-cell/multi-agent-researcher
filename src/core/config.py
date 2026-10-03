"""Application configuration management.

Uses Pydantic Settings to validate and load environment variables from .env.
This ensures type safety and centralized configuration across all agents.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration schema for the multi-agent system."""

    # Gemini LLM Settings
    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API key from AI Studio",
        alias="GEMINI_API_KEY",
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Default Gemini model to use for all agents",
        alias="GEMINI_MODEL",
    )

    # Vector Database (Qdrant)
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="URL for Qdrant vector database",
        alias="QDRANT_URL",
    )
    qdrant_collection_name: str = Field(
        default="research_documents",
        description="Collection name for embedded documents",
        alias="QDRANT_COLLECTION_NAME",
    )

    # Memory & Cache (Redis)
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for state and caching",
        alias="REDIS_URL",
    )

    # Observability (Langfuse)
    langfuse_public_key: str = Field(
        default="",
        description="Langfuse public API key",
        alias="LANGFUSE_PUBLIC_KEY",
    )
    langfuse_secret_key: str = Field(
        default="",
        description="Langfuse secret API key",
        alias="LANGFUSE_SECRET_KEY",
    )
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse host URL",
        alias="LANGFUSE_HOST",
    )

    # Search Tool (Tavily)
    tavily_api_key: str = Field(
        default="",
        description="Tavily Search API key",
        alias="TAVILY_API_KEY",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings.
    
    Using @lru_cache prevents re-parsing the .env file on every import.
    """
    return Settings()
