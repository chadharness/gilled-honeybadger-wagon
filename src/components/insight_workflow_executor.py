"""InsightWorkflowExecutor — executes insight retrieval plans.

Same pattern as DataWorkflowExecutor. All tool invocation via MCP.
Handles analytical tools (benchmark_comparison, anomaly_detection,
trend_extraction) in addition to data retrieval and computation tools.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd

from src.components.insight_workflow_planner import InsightPlan
from src.tools.mcp_server import STEP_REF_PARAMS, call_tool, execute_computation


@runtime_checkable
class InsightWorkflowExecutorProtocol(Protocol):
    def execute(self, plan: InsightPlan) -> dict[str, Any]: ...


class InsightWorkflowExecutor:
    """Executes insight plans via MCP, resolving step references at runtime."""

    def execute(self, plan: InsightPlan) -> dict[str, Any]:
        """Execute an insight plan.

        Args:
            plan: InsightPlan with categories and execution steps.

        Returns:
            Dict with step results keyed by step_id.
        """
        if plan.rejection:
            return {"rejection": plan.rejection}

        results: dict[str, Any] = {}

        for step in plan.data_retrieval_plan.steps:
            result = self._execute_step(step, results)
            results[step.step_id] = result

        return results

    def _execute_step(self, step: Any, prior_results: dict[str, Any]) -> Any:
        """Execute a single step via MCP call_tool()."""
        tool_name = step.tool
        params = step.parameters

        result = call_tool(tool_name, params)

        # Computation/analytical tools return routing dicts needing step resolution
        if isinstance(result, dict) and result.get("requires_step_resolution"):
            return self._resolve_and_execute(result["tool"], result["params"], prior_results)

        return result

    def _resolve_and_execute(
        self, tool_name: str, params: dict[str, Any], prior_results: dict[str, Any]
    ) -> Any:
        """Resolve step_id references to DataFrames, then execute via MCP."""
        ref_map = STEP_REF_PARAMS.get(tool_name, {})
        resolved_params = {}

        for param_name, value in params.items():
            if param_name in ref_map:
                step_result = prior_results.get(value, {})
                df = self._to_dataframe(step_result)
                resolved_params[ref_map[param_name]] = df
            else:
                resolved_params[param_name] = value

        result = execute_computation(tool_name, resolved_params)

        if isinstance(result, pd.DataFrame):
            return {"data": result.to_dict(orient="records"), "row_count": len(result)}

        return result

    def _to_dataframe(self, step_result: Any) -> pd.DataFrame:
        """Convert a step result to a DataFrame."""
        if isinstance(step_result, pd.DataFrame):
            return step_result
        if isinstance(step_result, dict):
            if "data" in step_result:
                return pd.DataFrame(step_result["data"])
        return pd.DataFrame()
