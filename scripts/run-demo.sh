#!/usr/bin/env bash
# Demo launcher — starts the Streamlit application.
# Usage: ./scripts/run-demo.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load .env for API credentials
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="."
export SIM_TODAY="2024-10-01"

# Detect venv python location (Windows vs Unix)
if [ -f "$PROJECT_ROOT/.venv/Scripts/python.exe" ]; then
    VENV_PYTHON="$PROJECT_ROOT/.venv/Scripts/python.exe"
elif [ -f "$PROJECT_ROOT/.venv/bin/python" ]; then
    VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
    echo "Error: could not find venv Python (checked .venv/Scripts/python.exe and .venv/bin/python)" >&2
    exit 1
fi

echo "============================================"
echo "  Deposit Insight Agents — Demo"
echo "============================================"
echo "  Effective date: ${SIM_TODAY}"
echo "  Portkey gateway: ${PORTKEY_GATEWAY_URL:-not set}"
echo "  Python: $("$VENV_PYTHON" --version 2>&1)"
echo "============================================"
echo ""

exec "$VENV_PYTHON" -m streamlit run src/app.py
