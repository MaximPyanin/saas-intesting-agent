"""ORACLE configuration loaded from environment variables.

Single source of truth for runtime config. Add a key here, then read it via
`get_settings()` from anywhere. Keys correspond 1:1 to .env.example.

Most keys default to empty strings — modules that need a key check it before
use and fail with a clear message if it's missing.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration sourced from .env / OS env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ----- LLM (Step 9+) -----
    # Two backends supported, auto-detected by oracle.observability:
    #   1. OpenAI direct  → set openai_api_key only
    #   2. Azure OpenAI / Azure AI Foundry → set azure_openai_endpoint
    #      + azure_openai_api_key + deployment names below
    # When azure_openai_endpoint is set, the Azure client is used and
    # `openai_model_heavy` / `openai_model_light` are interpreted as
    # Azure DEPLOYMENT NAMES (not model family names).
    openai_api_key: str = ""
    # When Azure is used, these are DEPLOYMENT names in Azure AI Foundry.
    # Defaults match Maksim's Azure deployments (gpt-5.4 + gpt-5.4-mini).
    # Override per environment via .env if your deployments are named differently.
    openai_model_heavy: str = "gpt-5.4"
    openai_model_light: str = "gpt-5.4-mini"

    # ----- Azure OpenAI / Azure AI Foundry (optional alternative to OpenAI) -----
    # If set, observability uses AsyncAzureOpenAI instead of AsyncOpenAI.
    # The model_heavy / model_light values become Azure deployment names.
    azure_openai_endpoint: str = ""           # https://YOUR_RESOURCE.openai.azure.com
    azure_openai_api_key: str = ""            # Azure OpenAI key (NOT openai_api_key)
    azure_openai_api_version: str = "2024-10-21"

    # ----- Telegram bot (Step 3) -----
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ----- Telegram reader (Step 7) -----
    telegram_api_id: str = ""
    telegram_api_hash: str = ""
    telegram_session_name: str = "oracle_reader"

    # ----- Market data (Step 4) -----
    coingecko_api_key: str = ""
    fred_api_key: str = ""
    producthunt_token: str = ""

    # ----- Reddit (Step 6) -----
    reddit_client_id: str = ""
    reddit_client_secret: str = ""

    # ----- YouTube (Step 7) -----
    youtube_api_key: str = ""

    # ----- Observability (Step 17) -----
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # ----- Azure (Step 19) -----
    azure_storage_connection_string: str = ""

    # ----- Operational settings -----
    timezone: str = "Europe/Warsaw"
    min_idea_score: int = Field(default=70, ge=0, le=100)
    min_investment_signal: int = Field(default=6, ge=1, le=10)
    max_ideas_per_digest: int = Field(default=3, ge=1, le=10)
    max_investment_signals: int = Field(default=3, ge=1, le=10)
    reflexion_max_rounds: int = Field(default=3, ge=1, le=5)
    idea_focus_weight: float = Field(default=0.70, ge=0.0, le=1.0)


@lru_cache
def get_settings() -> Settings:
    """Cached settings factory — call from anywhere."""
    return Settings()
