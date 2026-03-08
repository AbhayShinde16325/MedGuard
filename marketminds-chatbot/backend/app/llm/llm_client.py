"""
LLM Client for MarketMinds

This module defines a strict interface for interacting with
Large Language Models (LLMs), with implementation for
Gemini (cloud).

Supports both synchronous generation and streaming (SSE).
"""

import logging
from typing import Generator, Optional

from backend.app.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
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

        # Lazy import so that users don't see warnings unless they enable Gemini.
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Gemini support requires the 'google-generativeai' package. "
                "Install it in your development environment to use Gemini."
            ) from exc

        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL)

    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as exc:
            error_msg = str(exc)
            logger.error("Gemini request failed: %s", error_msg)

            if "API_KEY_INVALID" in error_msg or "API key not valid" in error_msg or "leaked" in error_msg:
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

            if "API_KEY_INVALID" in error_msg or "API key not valid" in error_msg or "leaked" in error_msg:
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
    """Return the Gemini LLM client."""
    logger.info("Using Gemini LLM client (model=%s).", GEMINI_MODEL)
    return GeminiLLMClient()
