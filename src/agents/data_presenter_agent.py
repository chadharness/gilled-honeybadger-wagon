"""Data Presenter worker agent — plan, execute, present.

Orchestrates the three-stage pipeline: DataWorkflowPlanner decomposes the
query into an ExecutionPlan, DataWorkflowExecutor runs the plan against
real data, and DataPresenter generates a natural-language response.

When the planner escalates to NL2SQL, the agent invokes the NL2SQL pipeline
and passes results to the presenter.

Each stage produces a trace span for observability.
"""

from __future__ import annotations

from typing import Any

from src.components.data_workflow_planner import DataWorkflowPlanner
from src.components.data_workflow_executor import DataWorkflowExecutor
from src.components.data_presenter import DataPresenter
from src.components.nl2sql_pipeline import NL2SQLPipeline
from src.utils.model_loader import get_model_config
from src.utils.tracing import TraceContext


def run_data_presenter(
    query: str,
    effective_date: str,
    trace: TraceContext | None = None,
) -> dict[str, Any]:
    """Execute the data presenter workflow.

    Steps: plan -> execute -> (optional NL2SQL) -> present

    Args:
        query: Augmented query string.
        effective_date: Effective date string (YYYY-MM-DD).
        trace: Optional TraceContext for span creation.

    Returns:
        Dict with response text and execution metadata.
    """
    planner = DataWorkflowPlanner()
    executor = DataWorkflowExecutor()
    presenter = DataPresenter()

    # Stage 1: Plan
    planner_config = get_model_config("claude_opus")
    planner_span = trace.create_span("planner", model_id=planner_config["portkey_model"]) if trace else None

    plan = planner.plan(query, effective_date)

    if planner_span:
        planner_span.prompt = query
        planner_span.metadata["rejection"] = plan.rejection
        planner_span.metadata["step_count"] = len(plan.steps) if plan.steps else 0
        planner_span.metadata["tools"] = [s.tool for s in plan.steps] if plan.steps else []
        planner_span.finish()

    # Stage 2: Execute
    executor_span = trace.create_span("executor") if trace else None

    execution_results = executor.execute(plan)

    if executor_span:
        executor_span.metadata["step_count"] = len(execution_results)
        executor_span.metadata["has_rejection"] = "rejection" in execution_results
        executor_span.finish()

    # Stage 2b: NL2SQL escalation (if needed)
    nl2sql_result = None
    for step_id, result in execution_results.items():
        if isinstance(result, dict) and result.get("escalation"):
            nl2sql_span = trace.create_span(
                "nl2sql_pipeline",
                model_id=get_model_config("claude_sonnet")["portkey_model"],
            ) if trace else None

            pipeline = NL2SQLPipeline()
            nl2sql_raw = pipeline.execute(
                query=result.get("query", query),
                reason=result.get("reason", "structured tools insufficient"),
                effective_date=effective_date,
            )
            nl2sql_result = {
                "success": nl2sql_raw.success,
                "sql": nl2sql_raw.sql,
                "data": nl2sql_raw.data.to_dict(orient="records") if not nl2sql_raw.data.empty else [],
                "uncertainty_signals": nl2sql_raw.uncertainty_signals,
                "error": nl2sql_raw.error,
                "trace_metadata": nl2sql_raw.trace_metadata,
            }

            if nl2sql_span:
                nl2sql_span.prompt = result.get("query", query)
                nl2sql_span.response = nl2sql_raw.sql
                nl2sql_span.metadata["success"] = nl2sql_raw.success
                nl2sql_span.metadata["uncertainty_signals"] = nl2sql_raw.uncertainty_signals
                nl2sql_span.metadata["attempts"] = nl2sql_raw.trace_metadata.get("attempts", [])
                nl2sql_span.finish()

            # Add NL2SQL result to execution_results for presenter
            execution_results["nl2sql"] = nl2sql_result
            break

    # Stage 3: Present
    presenter_config = get_model_config("claude_sonnet")
    presenter_span = trace.create_span("presenter", model_id=presenter_config["portkey_model"]) if trace else None

    response = presenter.present(query, execution_results)

    if presenter_span:
        presenter_span.prompt = query
        presenter_span.response = response
        presenter_span.finish()

    plan_dict = {
        "steps": [
            {"step_id": s.step_id, "tool": s.tool, "parameters": s.parameters}
            for s in plan.steps
        ] if plan.steps else [],
        "rejection": plan.rejection,
    }

    result: dict[str, Any] = {
        "response": response,
        "plan": plan_dict,
        "execution_results": execution_results,
    }

    if nl2sql_result:
        result["nl2sql"] = nl2sql_result

    return result
