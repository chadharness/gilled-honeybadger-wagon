"""Strip markdown and LaTeX formatting from LLM output.

Output must be plain text suitable for direct display.
"""

import re


def _convert_markdown_table(text: str) -> str:
    """Convert markdown tables to aligned plain text."""
    lines = text.split("\n")
    result = []
    table_rows = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        if "|" in stripped and stripped.startswith("|"):
            # Skip separator rows (|---|---|)
            if re.match(r"^\|[\s\-:|]+\|$", stripped):
                continue
            # Parse table row
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            table_rows.append(cells)
            in_table = True
        else:
            if in_table and table_rows:
                # Flush table as aligned text
                result.extend(_format_table_rows(table_rows))
                table_rows = []
                in_table = False
            result.append(line)

    # Flush any remaining table
    if table_rows:
        result.extend(_format_table_rows(table_rows))

    return "\n".join(result)


def _format_table_rows(rows: list[list[str]]) -> list[str]:
    """Format table rows as aligned plain text."""
    if not rows:
        return []
    # Find max width per column
    n_cols = max(len(r) for r in rows)
    widths = [0] * n_cols
    for row in rows:
        for i, cell in enumerate(row):
            if i < n_cols:
                widths[i] = max(widths[i], len(cell))

    formatted = []
    for row in rows:
        parts = []
        for i in range(n_cols):
            cell = row[i] if i < len(row) else ""
            parts.append(cell.ljust(widths[i]))
        formatted.append("  ".join(parts))
    return formatted


def sanitize(text: str) -> str:
    """Remove markdown and LaTeX formatting tokens from text.

    Strips: bold/italic markers, headers, code fences, bullet markers,
    LaTeX $...$ notation, \\frac{}{} and similar constructs.
    Converts markdown tables to aligned plain text.
    """
    # Convert markdown tables before stripping other formatting
    text = _convert_markdown_table(text)
    # Remove code fences
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    # Remove LaTeX inline math (but preserve currency like $4.2M)
    text = re.sub(r"\$(?!\d)([^$]+)\$", r"\1", text)
    # Remove LaTeX commands like \frac{a}{b}, \text{x}
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}\{([^}]*)\}", r"\1/\2", text)
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    # Remove standalone LaTeX commands (\times, \%, \cdot, etc.)
    text = text.replace("\\times", "x")
    text = text.replace("\\%", "%")
    text = text.replace("\\cdot", ".")
    text = text.replace("\\approx", "~")
    text = text.replace("\\leq", "<=")
    text = text.replace("\\geq", ">=")
    text = text.replace("\\rightarrow", "->")
    text = text.replace("\\leftarrow", "<-")
    # Remove escaped underscores (LaTeX \_ -> _)
    text = text.replace("\\_", "_")
    # Remove markdown headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers (** and * wrapping text)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    # Remove underscore bold/italic (only at word boundaries to preserve column names)
    text = re.sub(r"(?<!\w)_{1,3}([^_]+)_{1,3}(?!\w)", r"\1", text)
    # Remove bullet markers (- or * at line start, but not em-dash —)
    text = re.sub(r"^[\s]*[-*]\s+", "", text, flags=re.MULTILINE)
    # Remove numbered list markers (1. 2. etc. at line start)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\*{3,}$", "", text, flags=re.MULTILINE)
    # Clean up multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
