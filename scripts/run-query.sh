#!/usr/bin/env bash
# CLI entry point — runs a query through the full pipeline.
# Usage: ./scripts/run-query.sh "What are the deposit trends for Q3?"
# Usage: ./scripts/run-query.sh "query" "2024-10-01"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load .env
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="."

# Detect venv python location (Windows vs Unix)
if [ -f "$PROJECT_ROOT/.venv/Scripts/python.exe" ]; then
    VENV_PYTHON="$PROJECT_ROOT/.venv/Scripts/python.exe"
elif [ -f "$PROJECT_ROOT/.venv/bin/python" ]; then
    VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
    echo "Error: could not find venv Python (checked .venv/Scripts/python.exe and .venv/bin/python)" >&2
    exit 1
fi

exec "$VENV_PYTHON" -c "
import sys
from src.agents.manager_agent import build_graph
from src.utils.tracing import TraceContext

query = sys.argv[1] if len(sys.argv) > 1 else input('Enter query: ')
effective_date = sys.argv[2] if len(sys.argv) > 2 else '2024-10-01'

trace = TraceContext()
graph = build_graph()
result = graph.invoke({
    'query': query,
    'effective_date': effective_date,
    'trace': trace,
})
print(result.get('response', 'No response generated'))
" "$@"
