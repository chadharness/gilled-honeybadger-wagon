"""SQL builder — generates SQL from constrained tool parameters.

Data retrieval tools call these functions to build SQL. The LLM never
touches raw SQL on the structured tool path. All generated SQL uses the
table name from config, not hardcoded.
"""

from __future__ import annotations

from src.config.settings import WAREHOUSE_TABLE


def _quote_identifier(name: str) -> str:
    """Quote a column/table identifier to prevent injection."""
    # Only allow alphanumeric and underscore
    if not all(c.isalnum() or c == "_" for c in name):
        raise ValueError(f"Invalid identifier: {name}")
    return f'"{name}"'


def _build_where_clause(time_range: dict, filter_dict: dict | None) -> str:
    """Build WHERE clause from time_range and optional filter."""
    conditions = []

    start = time_range["start"]
    end = time_range["end"]
    conditions.append(f"SETTLEMENT_DATE >= DATE '{start}'")
    conditions.append(f"SETTLEMENT_DATE <= DATE '{end}'")

    if filter_dict:
        for col, val in filter_dict.items():
            col_id = _quote_identifier(col)
            # Escape single quotes in value
            safe_val = str(val).replace("'", "''")
            conditions.append(f"{col_id} = '{safe_val}'")

    return " AND ".join(conditions)


def build_aggregate_sql(
    metric: str,
    aggregation: str,
    time_range: dict,
    group_by: list[str] | None = None,
    filter_dict: dict | None = None,
    order_by: dict | None = None,
    limit: int | None = None,
) -> str:
    """Build SQL for get_aggregate tool."""
    metric_id = _quote_identifier(metric)
    agg = aggregation.upper()
    if agg not in ("SUM", "AVG", "COUNT", "MIN", "MAX"):
        raise ValueError(f"Invalid aggregation: {agg}")

    table = _quote_identifier(WAREHOUSE_TABLE)
    where = _build_where_clause(time_range, filter_dict)

    if group_by:
        group_cols = [_quote_identifier(c) for c in group_by]
        group_str = ", ".join(group_cols)
        select = f"{group_str}, {agg}({metric_id}) AS metric_value"
        sql = f"SELECT {select} FROM {table} WHERE {where} GROUP BY {group_str}"
    else:
        sql = f"SELECT {agg}({metric_id}) AS metric_value FROM {table} WHERE {where}"

    if order_by:
        order_col = order_by.get("column", "metric_value")
        order_dir = order_by.get("direction", "DESC").upper()
        if order_dir not in ("ASC", "DESC"):
            order_dir = "DESC"
        # If ordering by metric_value alias, use directly; otherwise quote
        if order_col == "metric_value":
            sql += f" ORDER BY metric_value {order_dir}"
        else:
            sql += f" ORDER BY {_quote_identifier(order_col)} {order_dir}"

    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    return sql


def build_time_series_sql(
    metric: str,
    aggregation: str,
    granularity: str,
    time_range: dict,
    group_by: list[str] | None = None,
    filter_dict: dict | None = None,
) -> str:
    """Build SQL for get_time_series tool."""
    metric_id = _quote_identifier(metric)
    agg = aggregation.upper()
    if agg not in ("SUM", "AVG", "COUNT"):
        raise ValueError(f"Invalid aggregation: {agg}")

    gran = granularity.upper()
    if gran not in ("DAY", "WEEK", "MONTH", "QUARTER", "YEAR"):
        raise ValueError(f"Invalid granularity: {gran}")

    table = _quote_identifier(WAREHOUSE_TABLE)
    where = _build_where_clause(time_range, filter_dict)

    # Use DATE_TRUNC for time bucketing
    time_col = f"DATE_TRUNC('{gran}', SETTLEMENT_DATE) AS time_period"

    group_parts = ["time_period"]
    select_parts = [time_col, f"{agg}({metric_id}) AS metric_value"]

    if group_by:
        for col in group_by:
            col_id = _quote_identifier(col)
            select_parts.insert(1, col_id)
            group_parts.append(col_id)

    select_str = ", ".join(select_parts)
    group_str = ", ".join(group_parts)

    sql = f"SELECT {select_str} FROM {table} WHERE {where} GROUP BY {group_str} ORDER BY time_period ASC"

    return sql


def build_top_n_sql(
    metric: str,
    aggregation: str,
    dimension: str,
    n: int,
    direction: str,
    time_range: dict | None = None,
    filter_dict: dict | None = None,
) -> str:
    """Build SQL for get_top_n tool."""
    metric_id = _quote_identifier(metric)
    dim_id = _quote_identifier(dimension)
    agg = aggregation.upper()
    if agg not in ("SUM", "AVG", "COUNT"):
        raise ValueError(f"Invalid aggregation: {agg}")

    dir_str = direction.upper()
    if dir_str not in ("ASC", "DESC"):
        dir_str = "DESC"

    table = _quote_identifier(WAREHOUSE_TABLE)

    if time_range:
        where = _build_where_clause(time_range, filter_dict)
    else:
        # No time range — use all data, but still apply filters
        conditions = []
        if filter_dict:
            for col, val in filter_dict.items():
                col_id = _quote_identifier(col)
                safe_val = str(val).replace("'", "''")
                conditions.append(f"{col_id} = '{safe_val}'")
        where = " AND ".join(conditions) if conditions else "1=1"

    sql = (
        f"SELECT {dim_id}, {agg}({metric_id}) AS metric_value "
        f"FROM {table} WHERE {where} "
        f"GROUP BY {dim_id} ORDER BY metric_value {dir_str} LIMIT {int(n)}"
    )

    return sql
