"""Computation tools — operate on DataFrames from prior execution steps.

Tools: compute_delta, filter_by_threshold, rank_by_metric,
       compute_statistics, regroup, join_results.

These tools do NOT query the warehouse directly. They receive DataFrames
from prior step results (resolved by the executor at runtime).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _resolve_metric_column(df: pd.DataFrame, metric: str) -> str:
    """Resolve metric column name, handling aggregation aliases.

    DuckDB aggregation queries (SUM, AVG, etc.) return results as 'metric_value'
    rather than the original column name. This helper maps the requested metric
    to the actual column in the DataFrame.
    """
    if metric in df.columns:
        return metric
    if "metric_value" in df.columns:
        return "metric_value"
    # Try common aggregation suffixes
    for suffix in ["_agg", "_sum", "_avg", "_count"]:
        candidate = f"{metric}{suffix}"
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Column '{metric}' not found in input. Available: {list(df.columns)}")


def compute_delta(
    current_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    join_on: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    """Compute change between two datasets (absolute and percentage per entity).

    Returns DataFrame with columns: join_on keys + for each metric:
    {metric}_current, {metric}_baseline, absolute_change, pct_change, direction.
    """
    # Validate join columns exist in both DataFrames
    for col in join_on:
        if col not in current_df.columns:
            raise ValueError(f"Join column '{col}' not found in current data. Available: {list(current_df.columns)}")
        if col not in baseline_df.columns:
            raise ValueError(f"Join column '{col}' not found in baseline data. Available: {list(baseline_df.columns)}")

    merged = current_df.merge(
        baseline_df,
        on=join_on,
        how="outer",
        suffixes=("_current", "_baseline"),
    )

    for metric in metrics:
        cur_col = f"{metric}_current"
        base_col = f"{metric}_baseline"

        # If the original column is 'metric_value' (from aggregation), handle alias
        if cur_col not in merged.columns and "metric_value_current" in merged.columns:
            cur_col = "metric_value_current"
            base_col = "metric_value_baseline"

        merged[cur_col] = merged[cur_col].fillna(0)
        merged[base_col] = merged[base_col].fillna(0)

        merged["absolute_change"] = merged[cur_col] - merged[base_col]
        merged["pct_change"] = np.where(
            merged[base_col] != 0,
            (merged["absolute_change"] / merged[base_col].abs()) * 100,
            np.where(merged[cur_col] != 0, np.inf, 0.0),
        )
        merged["direction"] = np.where(
            merged["absolute_change"] > 0,
            "up",
            np.where(merged["absolute_change"] < 0, "down", "flat"),
        )

    return merged


def filter_by_threshold(
    input_df: pd.DataFrame,
    metric: str,
    threshold: float,
    operator: str,
) -> pd.DataFrame:
    """Filter rows where metric meets threshold condition."""
    ops = {
        "gt": lambda s, t: s > t,
        "gte": lambda s, t: s >= t,
        "lt": lambda s, t: s < t,
        "lte": lambda s, t: s <= t,
        "eq": lambda s, t: s == t,
    }
    if operator not in ops:
        raise ValueError(f"Invalid operator: {operator}. Must be one of {list(ops.keys())}")
    metric = _resolve_metric_column(input_df, metric)

    mask = ops[operator](input_df[metric], threshold)
    return input_df[mask].reset_index(drop=True)


def rank_by_metric(
    input_df: pd.DataFrame,
    metric: str,
    direction: str,
    n: int | None = None,
) -> pd.DataFrame:
    """Rank entities by any column, optionally take top/bottom N."""
    metric = _resolve_metric_column(input_df, metric)

    ascending = direction.upper() == "ASC"
    result = input_df.sort_values(by=metric, ascending=ascending).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)

    if n is not None:
        result = result.head(n)

    return result


def compute_statistics(
    input_df: pd.DataFrame,
    group_by: list[str],
    metric: str,
    statistics: list[str],
) -> pd.DataFrame:
    """Compute statistical measures per entity/group."""
    metric = _resolve_metric_column(input_df, metric)

    stat_funcs = {
        "mean": "mean",
        "std_dev": "std",
        "variance": "var",
        "min": "min",
        "max": "max",
        "median": "median",
        "count": "count",
    }

    # Handle empty group_by (portfolio-level statistics)
    if not group_by:
        result_parts = {}
        series = input_df[metric]
        for stat in statistics:
            if stat == "percentile_25":
                result_parts["percentile_25"] = series.quantile(0.25)
            elif stat == "percentile_75":
                result_parts["percentile_75"] = series.quantile(0.75)
            elif stat in stat_funcs:
                result_parts[stat] = getattr(series, stat_funcs[stat])()
            else:
                raise ValueError(f"Unknown statistic: {stat}")
        return pd.DataFrame([result_parts])

    results = input_df[group_by].drop_duplicates().reset_index(drop=True)
    for stat in statistics:
        if stat == "percentile_25":
            agg = input_df.groupby(group_by)[metric].quantile(0.25).reset_index()
            agg = agg.rename(columns={metric: "percentile_25"})
            results = results.merge(agg, on=group_by, how="left")
        elif stat == "percentile_75":
            agg = input_df.groupby(group_by)[metric].quantile(0.75).reset_index()
            agg = agg.rename(columns={metric: "percentile_75"})
            results = results.merge(agg, on=group_by, how="left")
        elif stat in stat_funcs:
            agg = input_df.groupby(group_by, as_index=False)[metric].agg(stat_funcs[stat])
            agg = agg.rename(columns={metric: stat})
            results = results.merge(agg, on=group_by, how="left")
        else:
            raise ValueError(f"Unknown statistic: {stat}")

    return results


def regroup(
    input_df: pd.DataFrame,
    group_by: list[str],
    metrics: list[str],
    aggregation: str,
) -> pd.DataFrame:
    """Re-aggregate data by a new dimension."""
    agg = aggregation.upper()
    agg_map = {"SUM": "sum", "AVG": "mean", "COUNT": "count", "MIN": "min", "MAX": "max"}
    if agg not in agg_map:
        raise ValueError(f"Invalid aggregation: {agg}")

    # Validate group_by columns exist
    for col in group_by:
        if col not in input_df.columns:
            raise ValueError(f"Group column '{col}' not found in input. Available: {list(input_df.columns)}")

    # Resolve metric columns (handle aggregation aliases)
    resolved_metrics = [_resolve_metric_column(input_df, m) for m in metrics]

    result = input_df.groupby(group_by, as_index=False)[resolved_metrics].agg(agg_map[agg])
    return result


def join_results(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    on: list[str],
    how: str,
) -> pd.DataFrame:
    """Join two result sets on shared key columns."""
    if how not in ("inner", "left", "outer"):
        raise ValueError(f"Invalid join type: {how}. Must be inner, left, or outer.")

    return left_df.merge(right_df, on=on, how=how)
