"""MCP server module — tool registration and embedded server.

Embedded server: runs in-process, planner/executor communicate via MCP client.
Structured for extractability: extracting to standalone service is a deployment
change, not a code change.

Public interface for executor:
    call_tool(name, arguments): invoke any registered tool
    execute_computation(name, resolved_params): invoke computation/analytical
        tools with resolved DataFrame parameters
    STEP_REF_PARAMS: maps step_id parameter names to DataFrame parameter names
"""

from __future__ import annotations

from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False

from src.tools.data_retrieval import get_aggregate as _get_aggregate
from src.tools.data_retrieval import get_time_series as _get_time_series
from src.tools.data_retrieval import get_top_n as _get_top_n
from src.tools.escalation import escalate_to_nl2sql as _escalate_to_nl2sql
from src.tools import computation as _computation
from src.tools import analytical as _analytical

# Create the embedded MCP server (when mcp package is available)
mcp = FastMCP("insight-agents-tools") if _HAS_MCP else None


# ---------------------------------------------------------------------------
# Step reference parameter mapping. Executor uses this to resolve step_id
# parameters to DataFrame parameters before calling execute_computation().
# ---------------------------------------------------------------------------

STEP_REF_PARAMS: dict[str, dict[str, str]] = {
    "compute_delta": {"current": "current_df", "baseline": "baseline_df"},
    "filter_by_threshold": {"input": "input_df"},
    "rank_by_metric": {"input": "input_df"},
    "compute_statistics": {"input": "input_df"},
    "regroup": {"input": "input_df"},
    "join_results": {"left": "left_df", "right": "right_df"},
    "benchmark_comparison": {"input": "input_df"},
    "anomaly_detection": {"current": "current_df", "baseline": "baseline_df"},
    "trend_extraction": {"input": "input_df"},
}


# ---------------------------------------------------------------------------
# Tool dispatch tables (always available, regardless of MCP package)
# ---------------------------------------------------------------------------

_RETRIEVAL_DISPATCH: dict[str, Any] = {
    "get_aggregate": _get_aggregate,
    "get_time_series": _get_time_series,
    "get_top_n": _get_top_n,
}

_ESCALATION_DISPATCH: dict[str, Any] = {
    "escalate_to_nl2sql": _escalate_to_nl2sql,
}

_COMPUTATION_DISPATCH: dict[str, Any] = {
    "compute_delta": _computation.compute_delta,
    "filter_by_threshold": _computation.filter_by_threshold,
    "rank_by_metric": _computation.rank_by_metric,
    "compute_statistics": _computation.compute_statistics,
    "regroup": _computation.regroup,
    "join_results": _computation.join_results,
    "benchmark_comparison": _analytical.benchmark_comparison,
    "anomaly_detection": _analytical.anomaly_detection,
    "trend_extraction": _analytical.trend_extraction,
}

# Computation/analytical tools return routing dicts from MCP layer
_ROUTING_TOOLS: set[str] = set(STEP_REF_PARAMS.keys())


def _make_routing_dict(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Build a routing dict for computation/analytical tools."""
    return {
        "tool": tool_name,
        "params": params,
        "requires_step_resolution": True,
    }


# ---------------------------------------------------------------------------
# Public interface for executor
# ---------------------------------------------------------------------------

def call_tool(name: str, arguments: dict[str, Any]) -> Any:
    """Invoke a registered tool by name (embedded MCP pattern).

    For retrieval/escalation tools: executes and returns results directly.
    For computation/analytical tools: returns a routing dict with
    requires_step_resolution=True. The executor resolves step references
    to DataFrames, then calls execute_computation().

    Args:
        name: Tool name (e.g. "get_aggregate", "compute_delta").
        arguments: Tool parameters as a dict.

    Returns:
        Tool result (retrieval/escalation) or routing dict (computation).

    Raises:
        ValueError: If tool name is not registered.
    """
    if name in _RETRIEVAL_DISPATCH:
        return _RETRIEVAL_DISPATCH[name](**arguments)

    if name in _ESCALATION_DISPATCH:
        return _ESCALATION_DISPATCH[name](**arguments)

    if name in _ROUTING_TOOLS:
        return _make_routing_dict(name, arguments)

    raise ValueError(f"Unknown tool: {name}")


def execute_computation(name: str, resolved_params: dict[str, Any]) -> Any:
    """Execute a computation/analytical tool with resolved DataFrame parameters.

    Called by the executor AFTER resolving step_id references to DataFrames
    using STEP_REF_PARAMS.

    Args:
        name: Tool name (e.g. "compute_delta", "benchmark_comparison").
        resolved_params: Parameters with DataFrames substituted for step_ids.

    Returns:
        Tool result (typically a DataFrame).

    Raises:
        ValueError: If tool name is not a computation/analytical tool.
    """
    fn = _COMPUTATION_DISPATCH.get(name)
    if fn is None:
        raise ValueError(f"Unknown computation tool: {name}")
    return fn(**resolved_params)


# ---------------------------------------------------------------------------
# MCP tool registrations (only when mcp package is available)
# ---------------------------------------------------------------------------

if mcp is not None:

    @mcp.tool()
    def get_aggregate(
        metric: str,
        aggregation: str,
        time_range: dict,
        group_by: list[str] | None = None,
        filter: dict | None = None,
        order_by: dict | None = None,
        limit: int | None = None,
    ) -> dict:
        """Retrieve an aggregated metric for a time range, optionally grouped by one or more dimensions.

        Args:
            metric: Column to aggregate (INFLOW_AMOUNT, OUTFLOW_AMOUNT, NET_AMOUNT, GROSS_AMOUNT, INFLOW_TRANSACTION_VOLUME, OUTFLOW_TRANSACTION_VOLUME, TRANSACTION_VOLUME, etc.)
            aggregation: Aggregation function (SUM, AVG, COUNT, MIN, MAX)
            time_range: Date range with 'start' and 'end' keys (YYYY-MM-DD)
            group_by: Entity dimensions to group by (optional)
            filter: Filter on entity dimensions e.g. {'PAYMENT_INSTRUMENT_TYPE': 'Wire'} (optional)
            order_by: Sort with 'column' and 'direction' keys (optional)
            limit: Maximum rows to return (optional)
        """
        return _get_aggregate(
            metric=metric, aggregation=aggregation, time_range=time_range,
            group_by=group_by, filter=filter, order_by=order_by, limit=limit,
        )

    @mcp.tool()
    def get_time_series(
        metric: str,
        aggregation: str,
        granularity: str,
        time_range: dict,
        group_by: list[str] | None = None,
        filter: dict | None = None,
    ) -> dict:
        """Retrieve a metric over time periods at a specified granularity, optionally grouped.

        Args:
            metric: Column to aggregate (INFLOW_AMOUNT, OUTFLOW_AMOUNT, NET_AMOUNT, etc.)
            aggregation: Aggregation function (SUM, AVG, COUNT)
            granularity: Time grain (DAY, WEEK, MONTH, QUARTER, YEAR)
            time_range: Date range with 'start' and 'end' keys (YYYY-MM-DD)
            group_by: Entity dimensions to group by (optional)
            filter: Filter on entity dimensions (optional)
        """
        return _get_time_series(
            metric=metric, aggregation=aggregation, granularity=granularity,
            time_range=time_range, group_by=group_by, filter=filter,
        )

    @mcp.tool()
    def get_top_n(
        metric: str,
        aggregation: str,
        dimension: str,
        n: int,
        direction: str,
        time_range: dict | None = None,
        filter: dict | None = None,
    ) -> dict:
        """Retrieve the top or bottom N entities ranked by a raw metric from the warehouse.

        Args:
            metric: Column to aggregate (INFLOW_AMOUNT, NET_AMOUNT, etc.)
            aggregation: Aggregation function (SUM, AVG, COUNT)
            dimension: Entity dimension to rank (WCIS_GUP_NAME, LINE_OF_BUSINESS_LEVEL_2_NAME, etc.)
            n: Number of results to return
            direction: DESC for top N, ASC for bottom N
            time_range: Date range with 'start' and 'end' keys (optional)
            filter: Filter on entity dimensions (optional)
        """
        return _get_top_n(
            metric=metric, aggregation=aggregation, dimension=dimension,
            n=n, direction=direction, time_range=time_range, filter=filter,
        )

    @mcp.tool()
    def compute_delta(current: str, baseline: str, join_on: list[str], metrics: list[str]) -> dict:
        """Compute the change between two datasets (absolute difference and percentage change per entity).

        Args:
            current: Step ID of the current/later period data
            baseline: Step ID of the baseline/earlier period data
            join_on: Column(s) to join the two datasets on
            metrics: Column(s) to compute deltas for
        """
        return _make_routing_dict("compute_delta", {"current": current, "baseline": baseline, "join_on": join_on, "metrics": metrics})

    @mcp.tool()
    def filter_by_threshold(input: str, metric: str, threshold: float, operator: str) -> dict:
        """Filter rows where a metric meets a threshold condition.

        Args:
            input: Step ID of the data to filter
            metric: Column name to threshold on (can be a computed column like pct_change)
            threshold: Threshold value
            operator: Comparison operator (gt, gte, lt, lte, eq)
        """
        return _make_routing_dict("filter_by_threshold", {"input": input, "metric": metric, "threshold": threshold, "operator": operator})

    @mcp.tool()
    def rank_by_metric(input: str, metric: str, direction: str, n: int | None = None) -> dict:
        """Rank entities by any column (including computed columns) and optionally take top/bottom N.

        Args:
            input: Step ID of the data to rank
            metric: Column to rank by (can be computed e.g. pct_change, absolute_change)
            direction: DESC for highest first, ASC for lowest first
            n: If specified, return only top/bottom N
        """
        return _make_routing_dict("rank_by_metric", {"input": input, "metric": metric, "direction": direction, "n": n})

    @mcp.tool()
    def compute_statistics(input: str, group_by: list[str], metric: str, statistics: list[str]) -> dict:
        """Compute statistical measures per entity or group (mean, std_dev, variance, etc.).

        Args:
            input: Step ID, typically a time_series result with multiple observations per entity
            group_by: Column(s) defining the entities to compute stats for
            metric: Column to compute statistics on
            statistics: List of stats to compute (mean, std_dev, variance, min, max, median, percentile_25, percentile_75, count)
        """
        return _make_routing_dict("compute_statistics", {"input": input, "group_by": group_by, "metric": metric, "statistics": statistics})

    @mcp.tool()
    def regroup(input: str, group_by: list[str], metrics: list[str], aggregation: str) -> dict:
        """Re-aggregate a result by a new dimension.

        Args:
            input: Step ID of data to re-aggregate
            group_by: New dimension(s) to group by
            metrics: Columns to aggregate in the new grouping
            aggregation: Aggregation function (SUM, AVG, COUNT, MIN, MAX)
        """
        return _make_routing_dict("regroup", {"input": input, "group_by": group_by, "metrics": metrics, "aggregation": aggregation})

    @mcp.tool()
    def join_results(left: str, right: str, on: list[str], how: str) -> dict:
        """Join two result sets from prior steps on shared key columns.

        Args:
            left: Step ID of the left dataset
            right: Step ID of the right dataset
            on: Column(s) to join on
            how: Join type (inner, left, outer)
        """
        return _make_routing_dict("join_results", {"left": left, "right": right, "on": on, "how": how})

    @mcp.tool()
    def benchmark_comparison(input: str, entity_column: str, metric: str, peer_group_filter: dict | None = None) -> dict:
        """Compare entity performance against peer group average.

        Args:
            input: Step ID of entity-level aggregated data
            entity_column: Column identifying the entities being compared
            metric: Column containing the metric to compare
            peer_group_filter: Optional filter defining the peer group
        """
        return _make_routing_dict("benchmark_comparison", {"input": input, "entity_column": entity_column, "metric": metric, "peer_group_filter": peer_group_filter})

    @mcp.tool()
    def anomaly_detection(current: str, baseline: str, entity_column: str, metric: str, threshold_std_devs: float = 2.0) -> dict:
        """Identify entities deviating significantly from their baseline.

        Args:
            current: Step ID of current period data
            baseline: Step ID of historical baseline data
            entity_column: Column identifying entities to check
            metric: Column to check for anomalies
            threshold_std_devs: Number of standard deviations to flag as anomalous (default: 2.0)
        """
        return _make_routing_dict("anomaly_detection", {"current": current, "baseline": baseline, "entity_column": entity_column, "metric": metric, "threshold_std_devs": threshold_std_devs})

    @mcp.tool()
    def trend_extraction(input: str, metric: str, segment_column: str | None = None) -> dict:
        """Identify directional trends and reversals in time-series data.

        Args:
            input: Step ID of time-series data (from get_time_series)
            metric: Column to analyze for trends
            segment_column: Optional dimension column
        """
        return _make_routing_dict("trend_extraction", {"input": input, "metric": metric, "segment_column": segment_column})

    @mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
    def escalate_to_nl2sql(query: str, reason: str) -> dict:
        """ONLY use when the user's data question CANNOT be answered by the structured tools. Escalates to a natural-language-to-SQL pipeline.

        Args:
            query: The original natural-language query to translate to SQL
            reason: Why structured tools are insufficient
        """
        return _escalate_to_nl2sql(query=query, reason=reason)


# ---------------------------------------------------------------------------
# Server access
# ---------------------------------------------------------------------------

def get_server():
    """Return the FastMCP server instance for embedding (None if mcp not installed)."""
    return mcp


def list_tools() -> list[str]:
    """Return the names of all registered tools (for testing/introspection)."""
    return [
        "get_aggregate",
        "get_time_series",
        "get_top_n",
        "compute_delta",
        "filter_by_threshold",
        "rank_by_metric",
        "compute_statistics",
        "regroup",
        "join_results",
        "benchmark_comparison",
        "anomaly_detection",
        "trend_extraction",
        "escalate_to_nl2sql",
    ]


# For standalone deployment (future):
# if __name__ == "__main__":
#     mcp.run(transport="streamable-http")
