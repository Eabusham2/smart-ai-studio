#!/usr/bin/env bash
# ==============================================================================
# Ternary AI — 1-Click macOS Desktop Launcher
# ==============================================================================
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "======================================================================"
echo "  Launching Ternary AI Autonomous Reasoning Studio"
echo "======================================================================"

if command -v python3 >/dev/null 2>&1; then
    exec python3 app_gui.py "$@"
elif command -v python >/dev/null 2>&1; then
    exec python app_gui.py "$@"
else
    echo "[!] Error: Python 3 not found in PATH."
    exit 1
fi
