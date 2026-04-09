"""InsightWorkflowPlanner — plans insight generation queries.

Uses Claude Opus via Portkey to classify queries into analysis categories
(trend, benchmark, anomaly) and construct resolution-path-style execution
plans with over-fetch supporting retrievals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from src.components.data_workflow_planner import ExecutionStep, ExecutionPlan
from src.utils.llm_client import get_llm_client
from src.utils.model_loader import get_model_config


_PROMPT_PATH = Path(__file__).resolve().parent.parent / "config" / "prompts" / "insight_planner_system.txt"


@dataclass
class InsightPlan:
    primary_category: str = ""  # trend, benchmark, anomaly
    analysis_categories: list[str] = field(default_factory=list)
    data_retrieval_plan: ExecutionPlan = field(default_factory=ExecutionPlan)
    rejection: str | None = None


@runtime_checkable
class InsightWorkflowPlannerProtocol(Protocol):
    def plan(self, query: str, effective_date: str) -> InsightPlan: ...


class InsightWorkflowPlanner:
    """LLM-based planner using Claude Opus via Portkey."""

    def __init__(self) -> None:
        self._system_prompt = ""
        if _PROMPT_PATH.exists():
            self._system_prompt = _PROMPT_PATH.read_text().strip()
        self._model_config = get_model_config("claude_opus")

    def plan(self, query: str, effective_date: str) -> InsightPlan:
        """Classify query and construct an execution plan.

        Retries once on JSON parse failure.

        Args:
            query: Augmented query string.
            effective_date: Effective date (YYYY-MM-DD).

        Returns:
            InsightPlan with categories, steps, or rejection.
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
            max_tokens=3000,
        )

        raw = response.choices[0].message.content.strip()
        plan, parse_error = self._try_parse(raw)
        if plan is not None:
            return plan

        # Retry once
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
            max_tokens=3000,
        )

        retry_raw = retry_response.choices[0].message.content.strip()
        plan, _ = self._try_parse(retry_raw)
        if plan is not None:
            return plan

        return InsightPlan(rejection="OUT: Failed to parse planner response after retry")

    def _try_parse(self, raw: str) -> tuple[InsightPlan | None, str | None]:
        """Attempt to parse LLM response into an InsightPlan."""
        text = raw
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end])
                except json.JSONDecodeError as e2:
                    return None, str(e2)
            else:
                return None, str(e)

        # Rejection
        if "rejection" in data:
            return InsightPlan(rejection=data["rejection"]), None

        # Classification
        classification = data.get("classification", [])
        if isinstance(classification, str):
            classification = [classification]
        primary_category = classification[0] if classification else ""

        # Steps
        steps = []
        for s in data.get("steps", []):
            step = ExecutionStep(
                step_id=s.get("step_id", "s1"),
                tool=s["tool"],
                parameters=s.get("parameters", {}),
                depends_on=s.get("depends_on", []),
            )
            steps.append(step)

        execution_plan = ExecutionPlan(steps=steps)

        return InsightPlan(
            primary_category=primary_category,
            analysis_categories=classification,
            data_retrieval_plan=execution_plan,
        ), None
