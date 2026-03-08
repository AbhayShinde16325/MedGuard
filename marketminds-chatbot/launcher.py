"""
MarketMinds — Application Launcher

This is the single entry point for the packaged .exe.
It handles:
  - PyInstaller path resolution
  - First-run user environment setup
  - Embedding model path configuration
  - Auto-opening the browser
  - Starting the FastAPI server
"""

import sys
import os

# Fix for PyInstaller --windowed apps where stdout/stderr are None
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import webbrowser
import multiprocessing
import threading
import time
from pathlib import Path

# Required for PyInstaller on Windows
multiprocessing.freeze_support()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_base_path() -> Path:
    """Return the root directory of the application.

    When running from source: the project folder.
    When packaged: PyInstaller's temp extraction folder (_MEIPASS).
    """
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS).resolve()
    return Path(__file__).resolve().parent


def get_user_data_dir() -> Path:
    """Return the persistent user data directory.

    On Windows: %LOCALAPPDATA%\\MarketMinds
    Fallback  : ~/MarketMinds
    """
    appdata = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    return Path(appdata) / "MarketMinds"


# ---------------------------------------------------------------------------
# First-run setup
# ---------------------------------------------------------------------------

def setup_user_environment(user_data_dir: Path) -> None:
    """Create the user data directory and a starter .env on first run."""
    user_data_dir.mkdir(parents=True, exist_ok=True)

    # Create sub-directories the app expects
    (user_data_dir / "raw").mkdir(exist_ok=True)
    (user_data_dir / "processed").mkdir(exist_ok=True)
    (user_data_dir / "vector_store").mkdir(exist_ok=True)

    env_file = user_data_dir / ".env"

    if not env_file.exists():
        print(f"\n[SETUP] Creating configuration at: {env_file}")

        env_content = """\
# MarketMinds Configuration
# ========================
# This file is stored in your user profile and is NOT shared.

# LLM Provider: 'gemini' (cloud)
LLM_PROVIDER=gemini

# Gemini API Key (required if LLM_PROVIDER=gemini)
# Get yours at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=

# Model settings
GEMINI_MODEL=gemini-2.5-flash
DEFAULT_MODEL=mistral

# Advanced: use LLM for query routing (true/false)
USE_LLM_ROUTING=false
"""
        env_file.write_text(env_content, encoding="utf-8")

        print()
        print("=" * 60)
        print("  WELCOME TO MARKETMINDS!")
        print("=" * 60)
        print("  A configuration file has been created at:")
        print(f"    {env_file}")
        print()
        print("  To use Gemini, edit that file and paste your API key.")
        print("=" * 60)
        print()


# ---------------------------------------------------------------------------
# Embedding model path setup
# ---------------------------------------------------------------------------

def setup_model_path(base_path: Path) -> None:
    """Placeholder for model path setup. No local embedding model is used."""
    # no-op; embeddings are generated via Gemini API
    pass


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def start_server() -> None:
    """Start the FastAPI backend via uvicorn."""
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
        workers=1,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def setup_ssl_certs(base_path: Path) -> None:
    """Fix SSL certificate path for frozen exe (yfinance, requests, curl).

    PyInstaller bundles certifi but path resolution breaks; curl error 77 occurs
    unless we explicitly set SSL_CERT_FILE / REQUESTS_CA_BUNDLE.
    """
    if not getattr(sys, "frozen", False):
        return

    cert_path = base_path / "certifi" / "cacert.pem"
    if not cert_path.exists():
        try:
            import certifi
            cert_path = Path(certifi.where())
        except Exception:
            return

    if cert_path.exists():
        sp = str(cert_path.resolve())
        os.environ["SSL_CERT_FILE"] = sp
        os.environ["REQUESTS_CA_BUNDLE"] = sp
        os.environ["CURL_CA_BUNDLE"] = sp  # some HTTP stacks use curl


def main() -> None:
    print("Starting MarketMinds...")

    base_path = get_base_path()
    user_data_dir = get_user_data_dir()

    # 0. Fix SSL cert path for frozen exe (must run before any HTTP requests)
    setup_ssl_certs(base_path)

    # 1. Ensure sys.path includes the base so 'backend' is importable
    sys.path.insert(0, str(base_path))

    # 2. Setup persistent storage and .env
    setup_user_environment(user_data_dir)

    # 3. Point to bundled embedding model
    setup_model_path(base_path)

    # 4. Open browser after a short delay
    def open_browser():
        time.sleep(3)
        webbrowser.open("http://127.0.0.1:8000")

    threading.Thread(target=open_browser, daemon=True).start()

    # 5. Start the server (blocking)
    try:
        start_server()
    except KeyboardInterrupt:
        print("\nShutting down MarketMinds...")
    except Exception as exc:
        print(f"\n[ERROR] Server failed to start: {exc}")
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()
