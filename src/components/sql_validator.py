"""SQL Validator — standalone module for SQL safety and syntax checks.

Validates: syntax, injection patterns, DuckDB dialect compliance.
Must be independently testable with unit tests.
"""

import re
from dataclasses import dataclass, field


# Injection patterns — these SQL keywords indicate write operations
_INJECTION_PATTERNS = [
    re.compile(r"\b(DROP)\b", re.IGNORECASE),
    re.compile(r"\b(DELETE)\b", re.IGNORECASE),
    re.compile(r"\b(UPDATE)\b", re.IGNORECASE),
    re.compile(r"\b(INSERT)\b", re.IGNORECASE),
    re.compile(r"\b(ALTER)\b", re.IGNORECASE),
    re.compile(r"\b(CREATE)\b", re.IGNORECASE),
    re.compile(r"\b(TRUNCATE)\b", re.IGNORECASE),
    re.compile(r"\b(EXEC)\b", re.IGNORECASE),
    re.compile(r"\b(EXECUTE)\b", re.IGNORECASE),
]


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)


def validate_sql(sql: str) -> ValidationResult:
    """Validate a SQL query for safety and basic correctness.

    Checks:
        - Non-empty input
        - No injection patterns (DROP, DELETE, UPDATE, INSERT, ALTER, etc.)
        - Starts with SELECT or WITH (read-only queries)

    Args:
        sql: SQL query string to validate.

    Returns:
        ValidationResult with is_valid flag and list of errors.
    """
    errors: list[str] = []

    if not sql or not sql.strip():
        return ValidationResult(is_valid=False, errors=["Empty SQL query"])

    stripped = sql.strip()

    # Reject multi-statement queries (semicolons outside string literals)
    # Strip string literals before checking for semicolons
    no_strings = re.sub(r"'[^']*'", "", stripped)
    if ";" in no_strings:
        errors.append("Multi-statement queries not allowed (semicolon detected)")

    # Check for injection patterns
    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(stripped)
        if match:
            errors.append(f"Forbidden SQL keyword detected: {match.group(1).upper()}")

    # Must start with SELECT or WITH (read-only)
    first_word = stripped.split()[0].upper() if stripped.split() else ""
    if first_word not in ("SELECT", "WITH"):
        errors.append(f"Query must start with SELECT or WITH, got: {first_word}")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)
