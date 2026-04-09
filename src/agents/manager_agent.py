"""Manager Agent — LangGraph StateGraph orchestrating the query pipeline.

Graph structure:
  embed → [ood ∥ router] → intake_gate →
    (if OOD) → reject_ood → END
    (if in-domain) → augmenter → dispatch_gate →
      (data_presenter) → run_data_presenter → guardrails → END
      (insight_generator) → run_insight_generator → guardrails → END
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.data_presenter_agent import run_data_presenter
from src.agents.insight_generator_agent import run_insight_generator
from src.components.agent_router import AgentRouter
from src.components.guardrails import GuardrailChecker
from src.components.embedder import SentenceEmbedder
from src.components.ood_detector import OODDetector
from src.components.query_augmenter import QueryAugmenter
from src.utils.model_loader import get_model_config
from src.utils.tracing import TraceContext


# State schema: the contract between all graph nodes.
# Each node reads what it needs and writes its results back.
class InsightAgentState(TypedDict, total=False):
    query: str
    effective_date: str
    embedding: Any  # np.ndarray
    ood_result: dict[str, Any]
    route_decision: dict[str, Any]
    augmented_query: str
    execution_plan: dict[str, Any]
    execution_results: dict[str, Any]
    analysis_categories: list[str]
    primary_category: str
    response: str
    trace: TraceContext


# Lazy-loaded singletons. Model loading is expensive (~2s for embedder,
# ~1s each for OOD and router). Load once on first query, reuse thereafter.
_embedder: SentenceEmbedder | None = None
_ood_detector: OODDetector | None = None
_agent_router: AgentRouter | None = None
_augmenter: QueryAugmenter | None = None


def _get_embedder() -> SentenceEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = SentenceEmbedder()
    return _embedder


def _get_ood_detector() -> OODDetector:
    global _ood_detector
    if _ood_detector is None:
        # Pass embedder for self-calibration of threshold
        _ood_detector = OODDetector(embedder=_get_embedder())
    return _ood_detector


def _get_agent_router() -> AgentRouter:
    global _agent_router
    if _agent_router is None:
        _agent_router = AgentRouter()
    return _agent_router


def _get_augmenter() -> QueryAugmenter:
    global _augmenter
    if _augmenter is None:
        _augmenter = QueryAugmenter()
    return _augmenter


def embed_node(state: InsightAgentState) -> dict[str, Any]:
    """Produce embedding for the query."""
    trace: TraceContext = state["trace"]
    embedder_config = get_model_config("embedder")
    span = trace.create_span("embedder", model_id=embedder_config["model_id"])

    embedder = _get_embedder()
    embedding = embedder.embed(state["query"])

    span.finish()
    return {"embedding": embedding}


def ood_node(state: InsightAgentState) -> dict[str, Any]:
    """Check if query is out-of-distribution."""
    trace: TraceContext = state["trace"]
    ood_config = get_model_config("ood_detector")
    span = trace.create_span("ood_detector", model_id=ood_config["model_id"])

    detector = _get_ood_detector()
    result = detector.detect(state["embedding"])

    span.metadata["reconstruction_error"] = result.reconstruction_error
    span.metadata["threshold"] = result.threshold
    span.metadata["is_ood"] = result.is_ood
    span.finish()

    return {
        "ood_result": {
            "is_ood": result.is_ood,
            "reconstruction_error": result.reconstruction_error,
            "threshold": result.threshold,
        }
    }


def router_node(state: InsightAgentState) -> dict[str, Any]:
    """Route query to appropriate worker agent."""
    trace: TraceContext = state["trace"]
    router_config = get_model_config("agent_router")
    span = trace.create_span("agent_router", model_id=router_config["model_id"])

    router = _get_agent_router()
    decision = router.route(state["query"], embedding=state.get("embedding"))

    span.metadata["agent"] = decision.agent
    span.metadata["confidence"] = decision.confidence
    span.finish()

    return {
        "route_decision": {
            "agent": decision.agent,
            "confidence": decision.confidence,
        }
    }


def reject_ood_node(state: InsightAgentState) -> dict[str, Any]:
    """Generate rejection response for out-of-domain queries."""
    trace: TraceContext = state["trace"]
    span = trace.create_span("reject_ood")

    response = (
        "I can only help with questions about deposit activity, transaction flows, "
        "and related banking analytics. Your question appears to be outside this scope. "
        "Please rephrase your question to focus on deposit-related topics."
    )

    span.response = response
    span.finish()

    return {"response": response}


def augmenter_node(state: InsightAgentState) -> dict[str, Any]:
    """Augment query with temporal context."""
    trace: TraceContext = state["trace"]
    augmenter_config = get_model_config("gpt_4o_mini")
    span = trace.create_span("query_augmenter", model_id=augmenter_config["portkey_model"])

    augmenter = _get_augmenter()
    augmented = augmenter.augment(state["query"])

    span.prompt = state["query"]
    span.response = augmented
    span.finish()

    return {"augmented_query": augmented}


def run_data_presenter_node(state: InsightAgentState) -> dict[str, Any]:
    """Execute data presenter workflow: plan → execute → present."""
    trace: TraceContext = state["trace"]

    query = state.get("augmented_query", state.get("query", ""))
    effective_date = state.get("effective_date", "2024-10-01")

    result = run_data_presenter(query, effective_date, trace=trace)

    return {
        "response": result["response"],
        "execution_plan": result.get("plan", {}),
        "execution_results": result.get("execution_results", {}),
    }


def run_insight_generator_node(state: InsightAgentState) -> dict[str, Any]:
    """Execute insight generator workflow: classify + plan → execute → generate."""
    trace: TraceContext = state["trace"]

    query = state.get("augmented_query", state.get("query", ""))
    effective_date = state.get("effective_date", "2024-10-01")

    result = run_insight_generator(query, effective_date, trace=trace)

    return {
        "response": result["response"],
        "execution_plan": result.get("plan", {}),
        "execution_results": result.get("execution_results", {}),
        "analysis_categories": result.get("analysis_categories", []),
        "primary_category": result.get("primary_category", ""),
    }


def guardrails_node(state: InsightAgentState) -> dict[str, Any]:
    """Apply PII and safety guardrails to response."""
    trace: TraceContext = state["trace"]
    response = state.get("response", "")

    if not response:
        span = trace.create_span("guardrails")
        span.finish()
        return {}

    checker = GuardrailChecker()

    # Layer 1: Deterministic PII redaction
    pii_span = trace.create_span("guardrail_pii")
    pii_found, redacted, redaction_count = checker.check_pii(response)
    pii_span.metadata["pii_types"] = pii_found
    pii_span.metadata["redaction_count"] = redaction_count
    pii_span.finish()

    # Layer 2: LLM safety check
    guardrails_config = get_model_config("gpt_4o_mini")
    safety_span = trace.create_span("guardrail_safety", model_id=guardrails_config["portkey_model"])
    safety_issues = checker.check_safety(redacted)
    safety_span.metadata["issues"] = safety_issues
    if safety_issues:
        safety_span.metadata["original_response"] = redacted
    safety_span.finish()

    if safety_issues:
        issue_summary = "; ".join(safety_issues)
        return {
            "response": (
                f"I'm unable to provide this response as it may contain content that "
                f"doesn't meet safety guidelines: {issue_summary}. "
                f"Please rephrase your question."
            )
        }

    if pii_found:
        return {"response": redacted}

    return {}


def intake_gate(state: InsightAgentState) -> str:
    """Conditional edge after OOD + Router merge.

    If OOD → reject_ood. Otherwise → augmenter.
    """
    ood = state.get("ood_result", {})
    if ood.get("is_ood", False):
        return "reject_ood"
    return "augmenter"


def dispatch_gate(state: InsightAgentState) -> str:
    """Conditional edge after augmenter.

    Routes to run_data_presenter or run_insight_generator based on router result.
    """
    route = state.get("route_decision", {})
    agent = route.get("agent", "data_presenter")
    if agent == "insight_generator":
        return "run_insight_generator"
    return "run_data_presenter"


def build_graph() -> StateGraph:
    """Build the manager agent StateGraph.

    Returns:
        Compiled StateGraph ready for invocation.
    """
    graph = StateGraph(InsightAgentState)

    # Nodes
    graph.add_node("embed", embed_node)
    graph.add_node("ood", ood_node)
    graph.add_node("router", router_node)
    graph.add_node("reject_ood", reject_ood_node)
    graph.add_node("augmenter", augmenter_node)
    graph.add_node("run_data_presenter", run_data_presenter_node)
    graph.add_node("run_insight_generator", run_insight_generator_node)
    graph.add_node("guardrails", guardrails_node)

    # Entry
    graph.set_entry_point("embed")

    # Parallel fanout: embed → [ood, router]
    graph.add_edge("embed", "ood")
    graph.add_edge("embed", "router")

    # After OOD + Router merge → intake_gate
    graph.add_conditional_edges(
        "ood",
        intake_gate,
        {"reject_ood": "reject_ood", "augmenter": "augmenter"},
    )

    # reject_ood → END
    graph.add_edge("reject_ood", END)

    # augmenter → dispatch_gate
    graph.add_conditional_edges(
        "augmenter",
        dispatch_gate,
        {
            "run_data_presenter": "run_data_presenter",
            "run_insight_generator": "run_insight_generator",
        },
    )

    # Workers → guardrails → END
    graph.add_edge("run_data_presenter", "guardrails")
    graph.add_edge("run_insight_generator", "guardrails")
    graph.add_edge("guardrails", END)

    return graph.compile()
