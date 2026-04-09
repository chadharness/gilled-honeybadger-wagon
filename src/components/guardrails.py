"""GuardrailChecker — PII regex + LLM safety check.

Applied by the Manager Agent as post-processing after worker agent output.
PII patterns are externalized in src/config/pii_patterns.yaml for
translation boundary compliance.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.utils.llm_client import get_llm_client
from src.utils.model_loader import get_model_config


_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_PROMPT_PATH = _CONFIG_DIR / "prompts" / "guardrails_safety.txt"
_PII_PATTERNS_PATH = _CONFIG_DIR / "pii_patterns.yaml"


def _load_pii_patterns() -> dict[str, re.Pattern]:
    """Load PII regex patterns from external config."""
    if _PII_PATTERNS_PATH.exists():
        with open(_PII_PATTERNS_PATH) as f:
            data = yaml.safe_load(f) or {}
        patterns = {}
        for name, cfg in data.get("patterns", {}).items():
            patterns[name] = re.compile(cfg["regex"])
        return patterns
    # Fallback if config missing
    return {
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "phone": re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "account_number": re.compile(r"\b\d{8,17}\b"),
    }


_PII_PATTERNS = _load_pii_patterns()


@dataclass
class GuardrailResult:
    is_safe: bool
    pii_detected: list[str] = field(default_factory=list)
    redacted_text: str = ""
    safety_issues: list[str] = field(default_factory=list)
    redaction_count: int = 0


class GuardrailChecker:
    """Two-layer guardrail: deterministic PII regex + LLM safety check."""

    def __init__(self) -> None:
        self._model_config = get_model_config("gpt_4o_mini")
        self._safety_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        if _PROMPT_PATH.exists() and _PROMPT_PATH.stat().st_size > 0:
            return _PROMPT_PATH.read_text()
        return ""

    def check_pii(self, text: str) -> tuple[list[str], str, int]:
        """Regex-based PII detection and redaction.

        Returns:
            Tuple of (list of PII types found, redacted text, total redaction count).
        """
        found = []
        redacted = text
        total_redactions = 0
        for pii_type, pattern in _PII_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                found.append(pii_type)
                total_redactions += len(matches)
                redacted = pattern.sub(f"[REDACTED-{pii_type.upper()}]", redacted)
        return found, redacted, total_redactions

    def check_safety(self, text: str) -> list[str]:
        """LLM-based content safety check.

        Returns:
            List of safety issues found. Empty list means safe.
        """
        if not self._safety_prompt:
            return []

        client = get_llm_client()
        response = client.chat.completions.create(
            model=self._model_config["portkey_model"],
            messages=[
                {"role": "system", "content": self._safety_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
        )
        result = response.choices[0].message.content or ""
        if result.strip().upper() == "SAFE":
            return []
        return [result.strip()]

    def check(self, text: str) -> GuardrailResult:
        """Run both PII and safety checks.

        Args:
            text: Text to check (typically worker agent output).

        Returns:
            GuardrailResult with safety status and redacted text.
        """
        pii_found, redacted, redaction_count = self.check_pii(text)
        safety_issues = self.check_safety(redacted)

        return GuardrailResult(
            is_safe=len(pii_found) == 0 and len(safety_issues) == 0,
            pii_detected=pii_found,
            redacted_text=redacted,
            safety_issues=safety_issues,
            redaction_count=redaction_count,
        )
