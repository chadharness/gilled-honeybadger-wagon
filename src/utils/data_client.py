"""Data client — execute SQL against DuckDB warehouse.

Translation boundary: execute_sql interface. DuckDB now, Dremio/Snowflake later.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DB_PATH = str(_PROJECT_ROOT / "data" / "warehouse.duckdb")

_connection: duckdb.DuckDBPyConnection | None = None


def _get_connection() -> duckdb.DuckDBPyConnection:
    global _connection
    if _connection is None:
        db_path = os.environ.get("DUCKDB_PATH", _DEFAULT_DB_PATH)
        _connection = duckdb.connect(db_path, read_only=True)
    return _connection


def execute_sql(sql: str) -> pd.DataFrame:
    """Execute a SQL query against the warehouse and return a DataFrame.

    Args:
        sql: SQL query string. Table name: deposits_data_expanded.

    Returns:
        Query results as a pandas DataFrame.
    """
    conn = _get_connection()
    return conn.execute(sql).fetchdf()


def reset_connection() -> None:
    """Close and reset the connection (useful for testing)."""
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None
