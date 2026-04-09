"""Application settings — centralized configuration constants.

All host, port, table name, and model references live here.
Translation boundary: change these values when moving to client infrastructure.
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Database
WAREHOUSE_TABLE = os.environ.get("WAREHOUSE_TABLE", "deposits_data_expanded")
DUCKDB_PATH = os.environ.get("DUCKDB_PATH", str(PROJECT_ROOT / "data" / "warehouse.duckdb"))

# Schema reference
SCHEMA_CSV_PATH = str(PROJECT_ROOT / "data" / "schema_expanded.csv")

# Domain knowledge
DOMAIN_KNOWLEDGE_PATH = str(PROJECT_ROOT / "src" / "config" / "domain_knowledge.yaml")
