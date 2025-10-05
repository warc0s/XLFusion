#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Running XLFusion interactive mode..."
python XLFusion.py "$@"

