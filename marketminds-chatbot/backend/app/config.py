import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path Resolution (Handles PyInstaller and Local Dev)
# ---------------------------------------------------------------------------
if getattr(sys, 'frozen', False):
    # PyInstaller creates a temp folder and stores path in _MEIPASS
    BASE_DIR = Path(sys._MEIPASS).resolve()
    # Data must be persistent! Store in AppData\Local\MarketMinds
    appdata = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
    USER_DATA_DIR = Path(appdata) / "MarketMinds"
else:
    BASE_DIR = Path(__file__).resolve().parents[2]
    USER_DATA_DIR = BASE_DIR / "data"

USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load .env: prioritize user data dir, then bundle/project root, then repo root
load_dotenv(USER_DATA_DIR / ".env")
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")  # also check repo root (d:\MarketMinds\.env)


# ---------------------------------------------------------------------------
# LLM provider selection
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
# allowed: "gemini"

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# ---------------------------------------------------------------------------
# Centralised application config
# ---------------------------------------------------------------------------

class AppConfig:
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = USER_DATA_DIR
    RAW_DATA_DIR: Path = USER_DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = USER_DATA_DIR / "processed"
    VECTOR_STORE_DIR: Path = USER_DATA_DIR / "vector_store"

    APP_NAME: str = "MarketMinds"
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    DEFAULT_MODEL_NAME: str = os.getenv("DEFAULT_MODEL", "mistral")
    USE_LLM_ROUTING: bool = os.getenv("USE_LLM_ROUTING", "false").lower() == "true"


config = AppConfig()