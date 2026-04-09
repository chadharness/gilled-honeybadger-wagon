"""Analytical tools — domain-aware analytics on DataFrames.

Tools: benchmark_comparison, anomaly_detection, trend_extraction.
These bundle domain knowledge with computation and produce domain-aware outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.tools.computation import _resolve_metric_column


def benchmark_comparison(
    input_df: pd.DataFrame,
    entity_column: str,
    metric: str,
    peer_group_filter: dict | None = None,
) -> pd.DataFrame:
    """Compare entity performance against peer group average.

    Returns per-entity: metric_value, group_average, absolute_delta,
    pct_delta, rank. Summary rows for top/bottom and peer group size.
    """
    if entity_column not in input_df.columns:
        raise ValueError(f"Column '{entity_column}' not found. Available: {list(input_df.columns)}")
    metric = _resolve_metric_column(input_df, metric)

    df = input_df.copy()

    # Apply peer group filter if specified
    if peer_group_filter:
        for col, val in peer_group_filter.items():
            if col in df.columns:
                df = df[df[col] == val]

    group_average = df[metric].mean()
    peer_group_size = len(df)

    df["group_average"] = group_average
    df["absolute_delta"] = df[metric] - group_average
    df["pct_delta"] = np.where(
        group_average != 0,
        (df["absolute_delta"] / abs(group_average)) * 100,
        0.0,
    )
    df["rank"] = df[metric].rank(ascending=False, method="min").astype(int)
    df["peer_group_size"] = peer_group_size

    # Sort by rank
    df = df.sort_values("rank").reset_index(drop=True)

    return df


def anomaly_detection(
    current_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    entity_column: str,
    metric: str,
    threshold_std_devs: float = 2.0,
) -> pd.DataFrame:
    """Flag entities deviating from baseline by N standard deviations.

    Returns per-entity: metric_current, metric_baseline_mean,
    baseline_std_dev, deviation_std_devs, is_anomaly.
    """
    if entity_column not in current_df.columns:
        raise ValueError(f"Column '{entity_column}' not in current data. Available: {list(current_df.columns)}")
    if entity_column not in baseline_df.columns:
        raise ValueError(f"Column '{entity_column}' not in baseline data. Available: {list(baseline_df.columns)}")

    # Determine the metric column name (may be 'metric_value' from SQL agg)
    cur_metric = metric if metric in current_df.columns else "metric_value"
    base_metric = metric if metric in baseline_df.columns else "metric_value"

    # Compute baseline stats per entity
    baseline_stats = baseline_df.groupby(entity_column, as_index=False).agg(
        metric_baseline_mean=(base_metric, "mean"),
        baseline_std_dev=(base_metric, "std"),
        baseline_count=(base_metric, "count"),
    )
    # Fill NaN std_dev (single observation) with 0
    baseline_stats["baseline_std_dev"] = baseline_stats["baseline_std_dev"].fillna(0)

    # Get current values per entity
    current_agg = current_df.groupby(entity_column, as_index=False).agg(
        metric_current=(cur_metric, "sum"),
    )

    # Merge
    result = current_agg.merge(baseline_stats, on=entity_column, how="left")

    # Compute deviation
    result["deviation_std_devs"] = np.where(
        result["baseline_std_dev"] > 0,
        (result["metric_current"] - result["metric_baseline_mean"]).abs() / result["baseline_std_dev"],
        np.where(
            result["metric_current"] != result["metric_baseline_mean"],
            np.inf,
            0.0,
        ),
    )

    result["is_anomaly"] = result["deviation_std_devs"] >= threshold_std_devs

    # Sort anomalies first, by deviation magnitude
    result = result.sort_values("deviation_std_devs", ascending=False).reset_index(drop=True)

    return result


def trend_extraction(
    input_df: pd.DataFrame,
    metric: str,
    segment_column: str | None = None,
) -> pd.DataFrame:
    """Identify directional trends and reversals in time-series data.

    Returns per-period: metric_value, absolute_change, pct_change,
    direction, is_reversal. Plus overall_direction summary.
    """
    # Determine the metric column
    metric_col = metric if metric in input_df.columns else "metric_value"
    if metric_col not in input_df.columns:
        raise ValueError(f"Column '{metric_col}' not found. Available: {list(input_df.columns)}")

    # Identify the time column (time_period from get_time_series)
    time_col = None
    for candidate in ("time_period", "SETTLEMENT_DATE", "FRB_BUSINESS_DATE"):
        if candidate in input_df.columns:
            time_col = candidate
            break
    if time_col is None:
        raise ValueError(f"No time column found. Available: {list(input_df.columns)}")

    if segment_column and segment_column not in input_df.columns:
        raise ValueError(f"Segment column '{segment_column}' not found. Available: {list(input_df.columns)}")

    if segment_column:
        # Per-segment trend analysis
        results = []
        for segment_val, group in input_df.groupby(segment_column):
            segment_result = _compute_trend(group, time_col, metric_col)
            segment_result[segment_column] = segment_val
            results.append(segment_result)
        result = pd.concat(results, ignore_index=True)
    else:
        # Aggregate trend: group by time period if duplicates exist
        agg_df = input_df.groupby(time_col, as_index=False)[metric_col].sum()
        result = _compute_trend(agg_df, time_col, metric_col)

    return result


def _compute_trend(df: pd.DataFrame, time_col: str, metric_col: str) -> pd.DataFrame:
    """Compute trend metrics for a single series."""
    df = df.sort_values(time_col).reset_index(drop=True)
    result = df[[time_col, metric_col]].copy()
    result.columns = [time_col, "metric_value"]

    result["absolute_change"] = result["metric_value"].diff()
    result["pct_change"] = result["metric_value"].pct_change() * 100

    result["direction"] = np.where(
        result["absolute_change"] > 0,
        "up",
        np.where(result["absolute_change"] < 0, "down", "flat"),
    )

    # Reversal: direction changed from prior period
    result["prev_direction"] = result["direction"].shift(1)
    result["is_reversal"] = (
        (result["direction"] != result["prev_direction"])
        & result["prev_direction"].notna()
        & (result["direction"] != "flat")
        & (result["prev_direction"] != "flat")
    )
    result = result.drop(columns=["prev_direction"])

    # Overall direction: based on first vs last value
    if len(result) >= 2:
        first_val = result["metric_value"].iloc[0]
        last_val = result["metric_value"].iloc[-1]
        if last_val > first_val:
            result["overall_direction"] = "up"
        elif last_val < first_val:
            result["overall_direction"] = "down"
        else:
            result["overall_direction"] = "flat"
        result["num_reversals"] = int(result["is_reversal"].sum())

    return result
