"""DataWorkflowPlanner — decomposes queries into executable tool chains.

Uses Claude Opus via Portkey to parse natural-language queries into
structured ExecutionPlans with tool steps, or rejection strings for
out-of-scope queries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from src.utils.llm_client import get_llm_client
from src.utils.model_loader import get_model_config


_PROMPT_PATH = Path(__file__).resolve().parent.parent / "config" / "prompts" / "data_presenter_planner_system.txt"


@dataclass
class ExecutionStep:
    step_id: str
    tool: str
    parameters: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)


@dataclass
class ExecutionPlan:
    steps: list[ExecutionStep] = field(default_factory=list)
    rejection: str | None = None  # "OUT: ..." or "FUTURE: ..."


@runtime_checkable
class DataWorkflowPlannerProtocol(Protocol):
    def plan(self, query: str, effective_date: str) -> ExecutionPlan: ...


class DataWorkflowPlanner:
    """LLM-based planner using Claude Opus via Portkey."""

    def __init__(self) -> None:
        self._system_prompt = ""
        if _PROMPT_PATH.exists():
            self._system_prompt = _PROMPT_PATH.read_text().strip()
        self._model_config = get_model_config("claude_opus")

    def plan(self, query: str, effective_date: str) -> ExecutionPlan:
        """Decompose a query into an execution plan.

        Retries once on JSON parse failure, appending the error to the prompt.

        Args:
            query: Augmented query string.
            effective_date: Effective date (YYYY-MM-DD).

        Returns:
            ExecutionPlan with steps or rejection.
        """
        prompt = self._system_prompt.replace("{effective_date}", effective_date)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ]

        client = get_llm_client()
        response = client.chat.completions.create(
            model=self._model_config["portkey_model"],
            messages=messages,
            temperature=0,
            max_tokens=2000,
        )

        raw = response.choices[0].message.content.strip()
        plan, parse_error = self._try_parse(raw)
        if plan is not None:
            return plan

        # Retry once: append the failed response and error to messages
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": (
                f"Your response could not be parsed as JSON. Error: {parse_error}\n"
                "Please respond with ONLY a valid JSON object. No prose, no explanation."
            ),
        })

        retry_response = client.chat.completions.create(
            model=self._model_config["portkey_model"],
            messages=messages,
            temperature=0,
            max_tokens=2000,
        )

        retry_raw = retry_response.choices[0].message.content.strip()
        plan, _ = self._try_parse(retry_raw)
        if plan is not None:
            return plan

        return ExecutionPlan(rejection="OUT: Failed to parse planner response after retry")

    def _try_parse(self, raw: str) -> tuple[ExecutionPlan | None, str | None]:
        """Attempt to parse LLM response. Returns (plan, None) or (None, error_msg)."""
        text = raw
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            # Try to extract JSON from mixed output
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end])
                except json.JSONDecodeError as e2:
                    return None, str(e2)
            else:
                return None, str(e)

        if "rejection" in data:
            return ExecutionPlan(rejection=data["rejection"]), None

        steps = []
        for s in data.get("steps", []):
            step = ExecutionStep(
                step_id=s.get("step_id", "s1"),
                tool=s["tool"],
                parameters=s.get("parameters", {}),
                depends_on=s.get("depends_on", []),
            )
            steps.append(step)

        return ExecutionPlan(steps=steps), None
