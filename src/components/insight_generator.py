"""InsightGenerator — generates analytical narratives from data.

Uses Claude Opus via Portkey with per-category prompt templates to produce
analytical narratives. Each category (trend, benchmark, anomaly) has its
own distinct prompt. Hybrid queries merge required elements into a single
integrated narrative using the primary category's prompt as base.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from src.utils.llm_client import get_llm_client
from src.utils.model_loader import get_model_config
from src.utils.sanitize_output import sanitize


_PROMPT_DIR = Path(__file__).resolve().parent.parent / "config" / "prompts"
_PROMPT_PATHS = {
    "trend": _PROMPT_DIR / "insight_generation_trend.txt",
    "benchmark": _PROMPT_DIR / "insight_generation_benchmark.txt",
    "anomaly": _PROMPT_DIR / "insight_generation_anomaly.txt",
}


@runtime_checkable
class InsightGeneratorProtocol(Protocol):
    def generate(self, query: str, categories: list[str], execution_results: dict[str, Any], effective_date: str = "2024-10-01") -> str: ...


class InsightGenerator:
    """LLM-based insight generator using Claude Opus via Portkey."""

    def __init__(self) -> None:
        self._prompts: dict[str, str] = {}
        for category, path in _PROMPT_PATHS.items():
            if path.exists():
                self._prompts[category] = path.read_text().strip()
        self._model_config = get_model_config("claude_opus")

    def generate(self, query: str, categories: list[str], execution_results: dict[str, Any], effective_date: str = "2024-10-01") -> str:
        """Generate an analytical narrative from execution results.

        Args:
            query: The original user query.
            categories: Classified analysis categories (primary first).
            execution_results: Dict of step_id -> result from executor.
            effective_date: Effective date string (YYYY-MM-DD).

        Returns:
            Plain-text analytical narrative (sanitized).
        """
        # Handle rejections
        if "rejection" in execution_results:
            return self._format_rejection(execution_results["rejection"])

        # Handle NL2SQL results
        if "nl2sql" in execution_results:
            nl2sql = execution_results["nl2sql"]
            if not nl2sql.get("success"):
                return (
                    f"I attempted to generate custom SQL but encountered an issue: "
                    f"{nl2sql.get('error', 'Unknown error')}. Please try rephrasing your question."
                )

        # Check for empty results
        if self._is_empty_result(execution_results):
            return "No data was found matching your criteria. Please try adjusting your query or time range."

        # Select prompt based on primary category
        primary = categories[0] if categories else "benchmark"
        prompt_template = self._prompts.get(primary, self._prompts.get("benchmark", ""))

        # Format data context
        data_text = self._format_data(execution_results)
        categories_str = json.dumps(categories)

        prompt = (
            prompt_template
            .replace("{query}", query)
            .replace("{categories}", categories_str)
            .replace("{effective_date}", effective_date)
            .replace("{data}", data_text)
        )

        # For hybrid queries, append secondary category requirements
        if len(categories) > 1:
            prompt += self._hybrid_addendum(categories)

        client = get_llm_client()
        response = client.chat.completions.create(
            model=self._model_config["portkey_model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2000,
        )

        raw = response.choices[0].message.content.strip()
        return sanitize(raw)

    def get_prompt_template(self, category: str) -> str:
        """Return the prompt template for a given category (for testing)."""
        return self._prompts.get(category, "")

    def _is_empty_result(self, execution_results: dict[str, Any]) -> bool:
        """Check if all step results have empty data."""
        for step_id, result in execution_results.items():
            if isinstance(result, dict):
                data = result.get("data", [])
                if data:
                    return False
                if result.get("row_count", 0) > 0:
                    return False
            elif result:
                return False
        return True

    def _format_data(self, execution_results: dict[str, Any]) -> str:
        """Format execution results as readable data context."""
        parts = []
        for step_id, result in execution_results.items():
            if isinstance(result, dict):
                if "data" in result:
                    rows = result["data"]
                    if rows:
                        parts.append(f"Step {step_id}: {json.dumps(rows, default=str)}")
                    else:
                        parts.append(f"Step {step_id}: (no data)")
                elif "value" in result:
                    parts.append(f"Step {step_id}: {json.dumps(result, default=str)}")
                else:
                    parts.append(f"Step {step_id}: {json.dumps(result, default=str)}")
            else:
                parts.append(f"Step {step_id}: {result}")
        return "\n".join(parts)

    def _format_rejection(self, rejection: str) -> str:
        """Format a planner rejection as a user-friendly response."""
        if rejection.startswith("OUT:"):
            reason = rejection[4:].strip()
            return (
                f"I can only help with questions about deposit activity, transaction flows, "
                f"and related banking analytics. {reason}"
            )
        if rejection.startswith("FUTURE:"):
            reason = rejection[7:].strip()
            return (
                f"This question requires data that is not yet available in the system. {reason}"
            )
        return f"I'm unable to answer this query. {rejection}"

    def _hybrid_addendum(self, categories: list[str]) -> str:
        """Append secondary category requirements for hybrid queries."""
        secondary = categories[1:]
        lines = [
            "\n\nADDITIONAL REQUIREMENTS (hybrid query: integrate into a single narrative, do NOT produce separate sections):"
        ]
        for cat in secondary:
            if cat == "anomaly":
                lines.append(
                    "Also include anomaly elements: baseline description, observed anomaly, deviation magnitude. "
                    "NEVER use: fraud, suspicious, illegal, criminal, money laundering. "
                    "ALWAYS use: unusual activity, worth investigating, warrants review, atypical."
                )
            elif cat == "benchmark":
                lines.append(
                    "Also include benchmark elements: entity vs group average, top/bottom performers, "
                    "relative framing, peer group definition."
                )
            elif cat == "trend":
                lines.append(
                    "Also include trend elements: direction, magnitude, time period comparison, reversal flag."
                )
        return "\n".join(lines)
