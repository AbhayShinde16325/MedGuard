"""
LLM Client for MarketMinds

This module defines a strict interface for interacting with
Large Language Models (LLMs), with concrete implementations for
Ollama (local, via HTTP API) and Gemini (cloud).

Supports both synchronous generation and streaming (SSE).
"""

import json
import logging
from typing import Generator, Optional

import requests
import google.generativeai as genai

from backend.app.config import (
    LLM_PROVIDER,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    config,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Base LLM client abstraction.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
    ) -> str:
        """Generate a response from the LLM."""
        raise NotImplementedError(
            "LLMClient.generate() must be implemented"
        )

    def stream(
        self,
        prompt: str,
        context: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream response tokens one at a time. Yields partial strings."""
        # Default fallback: yield the entire response at once
        yield self.generate(prompt, context)


# ---------------------------------------------------------------------------
# Ollama (local inference via HTTP API)
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaLLMClient(LLMClient):
    """
    LLM client that uses Ollama's HTTP API for local inference.
    Supports both batch and streaming generation.
    """

    def __init__(self, model_name: str = "mistral", base_url: str = OLLAMA_BASE_URL) -> None:
        super().__init__(model_name)
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        """Send a prompt to the Ollama /api/generate endpoint."""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()

        except requests.ConnectionError:
            logger.error(
                "Cannot connect to Ollama at %s. Is `ollama serve` running?",
                self.base_url,
            )
            return (
                "⚠️ Could not connect to the Ollama server. "
                "Please make sure Ollama is running (`ollama serve`)."
            )
        except requests.Timeout:
            logger.error("Ollama request timed out after 120 s.")
            return "⚠️ The LLM request timed out. Please try a shorter question."
        except Exception as exc:
            logger.error("Ollama request failed: %s", exc)
            return f"⚠️ LLM request failed: {exc}"

    def stream(self, prompt: str, context: Optional[str] = None) -> Generator[str, None, None]:
        """Stream tokens from Ollama using its native streaming API."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
        }

        try:
            response = requests.post(url, json=payload, timeout=120, stream=True)
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        except requests.ConnectionError:
            yield "⚠️ Could not connect to Ollama. Is `ollama serve` running?"
        except requests.Timeout:
            yield "⚠️ The LLM request timed out."
        except Exception as exc:
            yield f"⚠️ Streaming failed: {exc}"

    def is_available(self) -> bool:
        """Quick health check — is the Ollama server reachable?"""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Gemini (cloud inference)
# ---------------------------------------------------------------------------

class GeminiLLMClient(LLMClient):
    """
    LLM client that uses Google Gemini for cloud-based inference.
    Supports streaming via Gemini's stream API.
    """

    def __init__(self) -> None:
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Set it in your .env file or as an environment variable."
            )

        super().__init__(model_name=GEMINI_MODEL)
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL)

    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as exc:
            error_msg = str(exc)
            logger.error("Gemini request failed: %s", error_msg)

            if "API_KEY_INVALID" in error_msg or "API key not valid" in error_msg:
                return (
                    "⚠️ The Gemini API key is invalid or expired. "
                    "Please update `GEMINI_API_KEY` in your `.env` file."
                )
            if "quota" in error_msg.lower() or "rate" in error_msg.lower():
                return "⚠️ Gemini API quota exceeded. Please try again later."

            return "⚠️ The AI model encountered an error. Please try again."

    def stream(self, prompt: str, context: Optional[str] = None) -> Generator[str, None, None]:
        """Stream tokens from Gemini using its stream=True parameter."""
        try:
            response = self.model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            error_msg = str(exc)
            logger.error("Gemini streaming failed: %s", error_msg)

            if "API_KEY_INVALID" in error_msg or "API key not valid" in error_msg:
                yield (
                    "⚠️ The Gemini API key is invalid or expired. "
                    "Please update `GEMINI_API_KEY` in your `.env` file."
                )
            else:
                yield "⚠️ The AI model encountered a streaming error. Please try again."


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_llm_client() -> LLMClient:
    """Return the correct LLM client based on the LLM_PROVIDER config."""
    provider = LLM_PROVIDER.lower()

    if provider == "gemini":
        logger.info("Using Gemini LLM client (model=%s).", GEMINI_MODEL)
        return GeminiLLMClient()
    else:
        model = config.DEFAULT_MODEL_NAME
        logger.info("Using Ollama LLM client (model=%s).", model)
        return OllamaLLMClient(model_name=model)
