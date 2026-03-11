#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/run_batch.sh <config.yaml> [--template NAME]" >&2
  exit 1
fi

python XLFusion.py --batch "$@"

