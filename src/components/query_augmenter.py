"""QueryAugmenter — LLM-based query clarification and expansion.

Injects effective date, temporal context, and resolves ambiguity.
Uses Portkey gateway via llm_client.
"""

from pathlib import Path

from src.utils.date_provider import get_today
from src.utils.llm_client import get_llm_client
from src.utils.model_loader import get_model_config


_PROMPT_PATH = Path(__file__).resolve().parent.parent / "config" / "prompts" / "augmenter_system.txt"


class QueryAugmenter:
    """Clarifies, rewrites, and expands queries to reduce ambiguity."""

    def __init__(self) -> None:
        self._model_config = get_model_config("gpt_4o_mini")
        self._system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        if _PROMPT_PATH.exists() and _PROMPT_PATH.stat().st_size > 0:
            return _PROMPT_PATH.read_text()
        return ""

    def augment(self, query: str) -> str:
        """Augment a query with temporal context and clarification.

        Args:
            query: Raw user query.

        Returns:
            Augmented query string.
        """
        today = get_today()
        if not self._system_prompt:
            return query

        client = get_llm_client()
        response = client.chat.completions.create(
            model=self._model_config["portkey_model"],
            messages=[
                {"role": "system", "content": self._system_prompt.format(today=today.isoformat())},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content or query
