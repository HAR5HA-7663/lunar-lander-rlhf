#!/bin/bash
# Start Jupyter Lab using the project .venv with a fixed token.
#
# VS Code: open Command Palette → "Jupyter: Specify Jupyter Server for Connections"
#          paste → http://localhost:8888/lab?token=har5ha
# Kernel : select "LunarLander RLHF (.venv)" in any notebook

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Sync .venv / install any missing deps
echo "==> Syncing .venv..."
uv sync

# Register .venv Python as a named Jupyter kernel (safe to re-run)
echo "==> Registering kernel..."
.venv/bin/python -m ipykernel install \
    --user \
    --name lunarlander-rlhf \
    --display-name "LunarLander RLHF (.venv)"

echo ""
echo "==> Jupyter Lab → http://localhost:8888/lab?token=har5ha"
echo "    VS Code: Jupyter → Specify Server → paste URL above"
echo "    Kernel : LunarLander RLHF (.venv)"
echo ""

exec .venv/bin/jupyter lab \
    --no-browser \
    --port=8888 \
    --IdentityProvider.token=har5ha \
    --ServerApp.root_dir="$SCRIPT_DIR"
