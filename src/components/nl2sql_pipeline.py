"""NL2SQLPipeline — escalation path for queries that exceed tool library.

Generates SQL from natural language, validates it, executes against the
warehouse, and returns results with uncertainty signals. Retries once on
failure.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from src.components.sql_validator import validate_sql
from src.utils.data_client import execute_sql
from src.utils.llm_client import get_llm_client
from src.utils.model_loader import get_model_config


_PROMPT_PATH = Path(__file__).resolve().parent.parent / "config" / "prompts" / "nl2sql_generation.txt"
_SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "schema_expanded.csv"


@dataclass
class NL2SQLResult:
    sql: str = ""
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    uncertainty_signals: list[str] = field(default_factory=list)
    success: bool = False
    error: str = ""
    trace_metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class NL2SQLPipelineProtocol(Protocol):
    def execute(self, query: str, reason: str, effective_date: str) -> NL2SQLResult: ...


class NL2SQLPipeline:
    """Plain Python NL2SQL pipeline — no LangGraph dependency."""

    def __init__(self) -> None:
        self._prompt_template = ""
        if _PROMPT_PATH.exists():
            self._prompt_template = _PROMPT_PATH.read_text().strip()
        self._model_config = get_model_config("claude_sonnet")
        self._schema_text = self._load_schema()

    def _load_schema(self) -> str:
        """Load schema CSV as text for the prompt."""
        if not _SCHEMA_PATH.exists():
            return ""
        rows = []
        with open(_SCHEMA_PATH) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(f"{row['COLUMN_NAME']} ({row['DATA_TYPE']})")
        return "\n".join(rows)

    def execute(self, query: str, reason: str, effective_date: str = "2024-10-01") -> NL2SQLResult:
        """Execute the NL2SQL pipeline: generate -> validate -> execute.

        Retries once on validation or execution failure.

        Args:
            query: Original natural-language query.
            reason: Why structured tools couldn't handle it.
            effective_date: Effective date for temporal resolution.

        Returns:
            NL2SQLResult with data, SQL, uncertainty signals, and metadata.
        """
        trace: dict[str, Any] = {"attempts": []}

        # First attempt
        sql, gen_error = self._generate_sql(query, reason, effective_date)
        if gen_error:
            return NL2SQLResult(error=gen_error, trace_metadata=trace)

        trace["attempts"].append({"sql": sql})

        result = self._validate_and_execute(sql, query)
        if result.success:
            result.trace_metadata = trace
            return result

        # Retry once with error feedback
        first_error = result.error
        trace["attempts"][0]["error"] = first_error

        sql_retry, gen_error = self._generate_sql(
            query, reason, effective_date,
            prior_attempt=sql, error=first_error,
        )
        if gen_error:
            return NL2SQLResult(error=gen_error, trace_metadata=trace)

        trace["attempts"].append({"sql": sql_retry})

        result = self._validate_and_execute(sql_retry, query)
        result.trace_metadata = trace
        if not result.success:
            trace["attempts"][-1]["error"] = result.error
        return result

    def _generate_sql(
        self,
        query: str,
        reason: str,
        effective_date: str,
        prior_attempt: str | None = None,
        error: str | None = None,
    ) -> tuple[str, str | None]:
        """Generate SQL via LLM. Returns (sql, error_or_none)."""
        prompt = (
            self._prompt_template
            .replace("{effective_date}", effective_date)
            .replace("{reason}", reason)
            .replace("{schema}", self._schema_text)
            .replace("{query}", query)
        )

        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]

        if prior_attempt and error:
            messages.append({"role": "assistant", "content": prior_attempt})
            messages.append({
                "role": "user",
                "content": (
                    f"The SQL above failed with error: {error}\n"
                    "Please fix the SQL and respond with ONLY the corrected query."
                ),
            })

        try:
            client = get_llm_client()
            response = client.chat.completions.create(
                model=self._model_config["portkey_model"],
                messages=messages,
                temperature=0,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content.strip()
            return self._extract_sql(raw), None
        except Exception as e:
            return "", f"SQL generation failed: {e}"

    def _extract_sql(self, raw: str) -> str:
        """Extract SQL from LLM response, stripping markdown fences."""
        text = raw
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            text = "\n".join(lines)
        return text.strip()

    def _validate_and_execute(self, sql: str, query: str) -> NL2SQLResult:
        """Validate SQL and execute. Returns NL2SQLResult."""
        validation = validate_sql(sql)
        if not validation.is_valid:
            return NL2SQLResult(
                sql=sql,
                error=f"SQL validation failed: {'; '.join(validation.errors)}",
            )

        try:
            df = execute_sql(sql)
        except Exception as e:
            return NL2SQLResult(sql=sql, error=f"SQL execution failed: {e}")

        uncertainty = self._check_column_uncertainty(sql, query)

        return NL2SQLResult(
            sql=sql,
            data=df,
            uncertainty_signals=uncertainty,
            success=True,
        )

    def _check_column_uncertainty(self, sql: str, user_query: str) -> list[str]:
        """Check for non-obvious column mappings between query terms and SQL columns."""
        signals = []

        # Known mappings where user term != column name
        _MAPPINGS = {
            "customer": "WCIS_GUP_NAME",
            "client": "WCIS_GUP_NAME",
            "counterparty": "CPRTY_WCIS_GUP_NAME",
            "branch": "LINE_OF_BUSINESS_LEVEL_2_NAME",
            "lob": "LINE_OF_BUSINESS_LEVEL_2_NAME",
            "line of business": "LINE_OF_BUSINESS_LEVEL_2_NAME",
            "product": "PRODUCT_LEVEL_1_NAME",
            "industry": "NAICS_NAME",
            "sector": "NAICS_NAME",
            "state": "ACCOUNT_DOMICILE_STATE",
            "region": "ACCOUNT_DOMICILE_STATE",
            "customer group name": "WCIS_GUP_NAME",
            "customer name": "WCIS_GUP_NAME",
        }

        query_lower = user_query.lower()
        sql_upper = sql.upper()

        for term, column in _MAPPINGS.items():
            if term in query_lower and column in sql_upper:
                col_words = column.lower().replace("_", " ")
                if term not in col_words:
                    signals.append(f"I interpreted '{term}' as {column} — is that correct?")

        return signals
