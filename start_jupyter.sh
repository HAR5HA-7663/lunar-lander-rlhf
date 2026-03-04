#!/bin/bash
# Start Jupyter Lab using the project .venv with a fixed token.
# VS Code: connect via http://localhost:8888/?token=har5ha
#   Command Palette → "Jupyter: Specify Jupyter Server for Connections" → enter URL above.
# Select kernel: "LunarLander RLHF (.venv)"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create / update .venv and install all deps from pyproject.toml
echo "==> Syncing .venv..."
uv sync --quiet

# Re-register kernel so it always points to this .venv's Python
.venv/bin/python -m ipykernel install \
    --user \
    --name lunarlander-rlhf \
    --display-name "LunarLander RLHF (.venv)" \
    --force \
    2>/dev/null

echo ""
echo "==> Jupyter Lab starting on http://localhost:8888"
echo "    Token  : har5ha"
echo "    URL    : http://localhost:8888/lab?token=har5ha"
echo "    VS Code: Jupyter → Specify Server → paste URL above"
echo "    Kernel : LunarLander RLHF (.venv)"
echo ""

.venv/bin/jupyter lab \
    --no-browser \
    --port=8888 \
    --IdentityProvider.token=har5ha \
    --ServerApp.root_dir="$SCRIPT_DIR"
