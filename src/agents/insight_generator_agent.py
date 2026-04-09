"""Insight Generator worker agent — plan, execute, generate.

Orchestrates the three-stage pipeline: InsightWorkflowPlanner classifies
the query and constructs a resolution-path execution plan,
InsightWorkflowExecutor runs the plan via MCP tools, and InsightGenerator
produces an analytical narrative using per-category prompt templates.

When the planner escalates to NL2SQL, the agent invokes the NL2SQL pipeline
and passes results to the generator.

Each stage produces a trace span for observability.
"""

from __future__ import annotations

from typing import Any

from src.components.insight_workflow_planner import InsightWorkflowPlanner
from src.components.insight_workflow_executor import InsightWorkflowExecutor
from src.components.insight_generator import InsightGenerator
from src.components.nl2sql_pipeline import NL2SQLPipeline
from src.utils.model_loader import get_model_config
from src.utils.tracing import TraceContext


def run_insight_generator(
    query: str,
    effective_date: str,
    trace: TraceContext | None = None,
) -> dict[str, Any]:
    """Execute the insight generator workflow.

    Steps: classify + plan -> execute -> (optional NL2SQL) -> generate

    Args:
        query: Augmented query string.
        effective_date: Effective date string (YYYY-MM-DD).
        trace: Optional TraceContext for span creation.

    Returns:
        Dict with response text, categories, and execution metadata.
    """
    planner = InsightWorkflowPlanner()
    executor = InsightWorkflowExecutor()
    generator = InsightGenerator()

    # Stage 1: Classify + Plan
    planner_config = get_model_config("claude_opus")
    planner_span = trace.create_span(
        "insight_planner", model_id=planner_config["portkey_model"]
    ) if trace else None

    plan = planner.plan(query, effective_date)

    if planner_span:
        planner_span.prompt = query
        planner_span.metadata["classification"] = plan.analysis_categories
        planner_span.metadata["primary_category"] = plan.primary_category
        planner_span.metadata["rejection"] = plan.rejection
        planner_span.metadata["step_count"] = len(plan.data_retrieval_plan.steps) if plan.data_retrieval_plan.steps else 0
        planner_span.metadata["tools"] = [s.tool for s in plan.data_retrieval_plan.steps] if plan.data_retrieval_plan.steps else []
        planner_span.finish()

    # Stage 2: Execute
    executor_span = trace.create_span("insight_executor") if trace else None

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
                nl2sql_span.finish()

            execution_results["nl2sql"] = nl2sql_result
            break

    # Stage 3: Generate
    generator_config = get_model_config("claude_opus")
    generator_span = trace.create_span(
        "insight_generator", model_id=generator_config["portkey_model"]
    ) if trace else None

    categories = plan.analysis_categories or []
    response = generator.generate(query, categories, execution_results, effective_date=effective_date)

    if generator_span:
        generator_span.prompt = query
        generator_span.response = response
        generator_span.metadata["categories"] = categories
        generator_span.metadata["primary_category"] = plan.primary_category
        generator_span.finish()

    plan_dict = {
        "steps": [
            {"step_id": s.step_id, "tool": s.tool, "parameters": s.parameters}
            for s in plan.data_retrieval_plan.steps
        ] if plan.data_retrieval_plan.steps else [],
        "rejection": plan.rejection,
        "classification": plan.analysis_categories,
    }

    result: dict[str, Any] = {
        "response": response,
        "plan": plan_dict,
        "execution_results": execution_results,
        "analysis_categories": categories,
        "primary_category": plan.primary_category,
    }

    if nl2sql_result:
        result["nl2sql"] = nl2sql_result

    return result
