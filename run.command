#!/bin/bash
# -------------------------------------------------------------------
# Cloud Task Placement Simulator - macOS One-Click Terminal Launcher
# Double-clicking this file in macOS Finder opens Terminal and runs
# the entire Python simulator setup and web server automatically.
# -------------------------------------------------------------------

# Navigate to project folder
cd "$(dirname "$0")" || exit 1

# Execute the pure Python orchestrator using the local environment
if [ -f "./.venv/bin/python" ]; then
    "./.venv/bin/python" run.py
else
    python3 run.py
fi
