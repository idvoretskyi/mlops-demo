#!/usr/bin/env bash
set -euo pipefail

# Interactive Python environment setup for the MLOps demo
# - Creates .venv if missing
# - Installs requirements.txt
# - Verifies torch and fastapi imports

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$PROJECT_ROOT"

echo "==> Python environment setup"

# Choose Python executable
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "ERROR: Python is not installed. Please install Python 3.10+ first: https://www.python.org/downloads/" >&2
  exit 1
fi

# Create venv if not present
if [ ! -d .venv ]; then
  echo "Creating virtual environment at .venv ..."
  "$PY" -m venv .venv
else
  echo "Virtual environment .venv already exists."
fi

# Activate venv
# shellcheck source=/dev/null
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
if [ -f requirements.txt ]; then
  echo "Installing dependencies from requirements.txt ..."
  pip install -r requirements.txt
else
  echo "requirements.txt not found; nothing to install."
fi

# Verification
python - <<'PY'
try:
    import torch, fastapi, pydantic
    print("OK: torch version:", torch.__version__)
except Exception as e:
    raise SystemExit(f"Dependency check failed: {e}")
PY

echo "==> Done. To activate: source .venv/bin/activate"
