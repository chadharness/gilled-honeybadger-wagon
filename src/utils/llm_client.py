"""LLM client — OpenAI SDK configured for Portkey gateway.

Uses @slug/model format from models.yaml portkey_model field.
Single shared client instance.
"""

from __future__ import annotations

import os

from openai import OpenAI


_client: OpenAI | None = None


def get_llm_client() -> OpenAI:
    """Return a shared OpenAI client configured for Portkey."""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.environ.get("PORTKEY_GATEWAY_URL", "https://api.portkey.ai/v1"),
            api_key=os.environ.get("PORTKEY_API_KEY", ""),
        )
    return _client


def reset_client() -> None:
    """Reset the shared client (useful for testing)."""
    global _client
    _client = None
