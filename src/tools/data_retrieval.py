"""Data retrieval tools — generate validated SQL, execute via data_client.

Tools: get_aggregate, get_time_series, get_top_n.
These tools generate SQL internally from constrained parameters.
The LLM never touches raw SQL on this path.
"""

from __future__ import annotations

import pandas as pd

from src.components.sql_validator import validate_sql
from src.tools.sql_builder import (
    build_aggregate_sql,
    build_time_series_sql,
    build_top_n_sql,
)
from src.utils.data_client import execute_sql


def _validate_and_execute(sql: str) -> pd.DataFrame:
    """Validate SQL for safety, then execute."""
    result = validate_sql(sql)
    if not result.is_valid:
        raise ValueError(f"SQL validation failed: {'; '.join(result.errors)}")
    return execute_sql(sql)


def _df_to_serializable(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to a JSON-serializable list of dicts."""
    return df.to_dict(orient="records")


def get_aggregate(
    metric: str,
    aggregation: str,
    time_range: dict,
    group_by: list[str] | None = None,
    filter: dict | None = None,
    order_by: dict | None = None,
    limit: int | None = None,
) -> dict:
    """Retrieve an aggregated metric for a time range, optionally grouped."""
    sql = build_aggregate_sql(
        metric=metric,
        aggregation=aggregation,
        time_range=time_range,
        group_by=group_by,
        filter_dict=filter,
        order_by=order_by,
        limit=limit,
    )
    df = _validate_and_execute(sql)
    return {"data": _df_to_serializable(df), "sql": sql, "row_count": len(df)}


def get_time_series(
    metric: str,
    aggregation: str,
    granularity: str,
    time_range: dict,
    group_by: list[str] | None = None,
    filter: dict | None = None,
) -> dict:
    """Retrieve a metric over time at specified granularity."""
    sql = build_time_series_sql(
        metric=metric,
        aggregation=aggregation,
        granularity=granularity,
        time_range=time_range,
        group_by=group_by,
        filter_dict=filter,
    )
    df = _validate_and_execute(sql)
    return {"data": _df_to_serializable(df), "sql": sql, "row_count": len(df)}


def get_top_n(
    metric: str,
    aggregation: str,
    dimension: str,
    n: int,
    direction: str,
    time_range: dict | None = None,
    filter: dict | None = None,
) -> dict:
    """Retrieve the top/bottom N entities ranked by a raw metric."""
    sql = build_top_n_sql(
        metric=metric,
        aggregation=aggregation,
        dimension=dimension,
        n=n,
        direction=direction,
        time_range=time_range,
        filter_dict=filter,
    )
    df = _validate_and_execute(sql)
    return {"data": _df_to_serializable(df), "sql": sql, "row_count": len(df)}
