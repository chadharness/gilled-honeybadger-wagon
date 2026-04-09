"""DataPresenter — generates natural language from execution results.

Uses Claude Sonnet via Portkey to produce plain-text responses from
query + execution result data. Applies output sanitization to strip
any residual markdown/LaTeX from LLM output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from src.utils.llm_client import get_llm_client
from src.utils.model_loader import get_model_config
from src.utils.sanitize_output import sanitize


_PROMPT_PATH = Path(__file__).resolve().parent.parent / "config" / "prompts" / "data_presenter_generation.txt"


@runtime_checkable
class DataPresenterProtocol(Protocol):
    def present(self, query: str, execution_results: dict[str, Any]) -> str: ...


class DataPresenter:
    """LLM-based presenter using Claude Sonnet via Portkey."""

    def __init__(self) -> None:
        self._prompt_template = ""
        if _PROMPT_PATH.exists():
            self._prompt_template = _PROMPT_PATH.read_text().strip()
        self._model_config = get_model_config("claude_sonnet")

    def present(self, query: str, execution_results: dict[str, Any]) -> str:
        """Generate a natural-language response from execution results.

        Args:
            query: The original user query.
            execution_results: Dict of step_id -> result from executor,
                or dict with 'rejection', or dict with 'nl2sql' key.

        Returns:
            Plain-text response string (sanitized).
        """
        # Handle rejections passed through from planner
        if "rejection" in execution_results:
            return self._format_rejection(execution_results["rejection"])

        # Handle NL2SQL pipeline results
        if "nl2sql" in execution_results:
            return self._format_nl2sql(query, execution_results["nl2sql"])

        # Handle escalation signals (before NL2SQL pipeline runs)
        for step_id, result in execution_results.items():
            if isinstance(result, dict) and result.get("escalation"):
                return self._format_escalation(result)

        # Check for all-empty results
        if self._is_empty_result(execution_results):
            return "No data was found matching your criteria. Please try adjusting your query or time range."

        data_text = self._format_data(execution_results)
        prompt = self._prompt_template.replace("{query}", query).replace("{data}", data_text)

        client = get_llm_client()
        response = client.chat.completions.create(
            model=self._model_config["portkey_model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1500,
        )

        raw = response.choices[0].message.content.strip()
        return sanitize(raw)

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
        """Format execution results as readable data context for the LLM."""
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

    def _format_escalation(self, result: dict[str, Any]) -> str:
        """Format an escalation result as a user-friendly response."""
        return (
            f"This query requires advanced SQL generation. "
            f"Reason: {result.get('reason', 'structured tools insufficient')}. "
            f"The query will be processed through the NL2SQL pipeline."
        )

    def _format_nl2sql(self, query: str, nl2sql_result: dict[str, Any]) -> str:
        """Format NL2SQL pipeline result into a response."""
        if not nl2sql_result.get("success"):
            error = nl2sql_result.get("error", "Unknown error")
            return (
                f"I attempted to generate custom SQL for your query but encountered an issue: {error}. "
                f"Please try rephrasing your question or contact support for complex queries."
            )

        data = nl2sql_result.get("data", [])
        uncertainty = nl2sql_result.get("uncertainty_signals", [])

        if not data:
            return "No data was found matching your criteria. Please try adjusting your query or time range."

        # Build data context and generate via LLM
        data_text = f"NL2SQL result: {json.dumps(data, default=str)}"
        if uncertainty:
            data_text += f"\nNote: {'; '.join(uncertainty)}"

        prompt = self._prompt_template.replace("{query}", query).replace("{data}", data_text)

        client = get_llm_client()
        response = client.chat.completions.create(
            model=self._model_config["portkey_model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1500,
        )

        raw = response.choices[0].message.content.strip()
        result_text = sanitize(raw)

        # Append uncertainty signals if present
        if uncertainty:
            result_text += "\n\nNote: " + "; ".join(uncertainty)

        return result_text
