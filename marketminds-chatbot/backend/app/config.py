"""
Application configuration for MarketMinds Chatbot.

This module handles all configuration settings required by the backend.
Settings are read from environment variables (with .env support via
python-dotenv) so nothing sensitive is hardcoded.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (marketminds-chatbot/.env or parent .env)
_project_root = Path(__file__).resolve().parents[2]
load_dotenv(_project_root / ".env")
load_dotenv(_project_root.parent / ".env")   # also check repo root


# ---------------------------------------------------------------------------
# LLM provider selection
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
# allowed: "ollama", "gemini"

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# ---------------------------------------------------------------------------
# Centralised application config
# ---------------------------------------------------------------------------

class AppConfig:
    BASE_DIR: Path = Path(__file__).resolve().parents[2]

    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    VECTOR_STORE_DIR: Path = DATA_DIR / "vector_store"

    APP_NAME: str = "MarketMinds"
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    DEFAULT_MODEL_NAME: str = os.getenv("DEFAULT_MODEL", "mistral")
    USE_LLM_ROUTING: bool = os.getenv("USE_LLM_ROUTING", "false").lower() == "true"


config = AppConfig()