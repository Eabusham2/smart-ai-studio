#!/usr/bin/env bash
# Quick fix for macOS Gatekeeper / Quarantine on downloaded apps
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "============================================================"
echo "  Fixing macOS Permissions & Gatekeeper for Smart AI Studio"
echo "============================================================"
echo ""

xattr -cr SmartAI.app 2>/dev/null || true
xattr -cr . 2>/dev/null || true
chmod -R 755 SmartAI.app 2>/dev/null || true
chmod +x SmartAI.app/Contents/MacOS/* 2>/dev/null || true
chmod +x *.command 2>/dev/null || true

echo "✓ Gatekeeper quarantine cleared and permissions granted!"
echo "🚀 Opening SmartAI.app..."
open SmartAI.app || ./Launch_SmartAI_macOS.command
