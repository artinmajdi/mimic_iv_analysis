#!/bin/bash
set -e

echo "Uninstalling any existing installations of the package..."
uv pip uninstall -y mimic-iv-analysis mimic_iv_analysis 2>/dev/null || true

echo "Installing the package in development mode..."
uv pip install -e .

echo "Checking installation..."
uv pip list | grep mimic

echo "Done. Try running your application now."
