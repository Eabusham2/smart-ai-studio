#!/usr/bin/env bash
# Double-click launcher for Smart AI Studio on macOS
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "============================================================"
echo "  🧠 Smart AI Studio — macOS Quick Launcher"
echo "============================================================"
echo ""

# 1. Clear quarantine attribute automatically if downloaded from browser
if [ -d "SmartAI.app" ]; then
    xattr -cr SmartAI.app 2>/dev/null || true
    chmod -R +x SmartAI.app/Contents/MacOS/* 2>/dev/null || true
fi
xattr -cr . 2>/dev/null || true

# 2. Source user profile
if [ -f "$HOME/.zprofile" ]; then source "$HOME/.zprofile" 2>/dev/null; fi
if [ -f "$HOME/.zshrc" ]; then source "$HOME/.zshrc" 2>/dev/null; fi

export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:$HOME/.pyenv/shims:$HOME/.pyenv/bin:$HOME/.local/bin:$HOME/miniconda3/bin:$HOME/anaconda3/bin:$PATH"

# 3. Find Python
PYTHON_CMD=""
CANDIDATES=(
    "$(which python3 2>/dev/null)"
    "/opt/homebrew/bin/python3"
    "/usr/local/bin/python3"
    "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
    "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
    "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3"
    "$(which python 2>/dev/null)"
)

for CAND in "${CANDIDATES[@]}"; do
    if [ -n "$CAND" ] && [ -x "$CAND" ]; then
        if "$CAND" -c "import tkinter" 2>/dev/null; then
            PYTHON_CMD="$CAND"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ Error: Python 3 with Tkinter not found."
    echo "Please install Python from https://www.python.org or via Homebrew: brew install python-tk"
    read -p "Press enter to exit..."
    exit 1
fi

echo "✓ Using Python: $PYTHON_CMD"
echo "🚀 Launching Smart AI Studio..."
echo ""

"$PYTHON_CMD" app_gui.py "$@"
