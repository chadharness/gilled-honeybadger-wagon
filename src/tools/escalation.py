"""Escalation tool — routes to NL2SQL pipeline when structured tools can't handle the query.

This is a thin signal tool. It does NOT execute SQL. The executor routes
the result to the NL2SQL pipeline module.
"""


def escalate_to_nl2sql(query: str, reason: str) -> dict:
    """Signal that this query needs NL2SQL escalation.

    ONLY use when structured tools (get_aggregate, get_time_series, get_top_n,
    compute_delta, filter_by_threshold, rank_by_metric, compute_statistics,
    regroup, join_results, benchmark_comparison, anomaly_detection,
    trend_extraction) cannot satisfy the query.

    Args:
        query: The original natural-language query to translate to SQL.
        reason: Why structured tools are insufficient.

    Returns:
        Signal dict with query and reason for the executor to route.
    """
    return {
        "escalation": True,
        "query": query,
        "reason": reason,
        "meta": {
            "insightagents/risk_level": "elevated",
            "insightagents/requires_uncertainty_signal": True,
        },
    }
