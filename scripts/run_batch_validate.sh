#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/run_batch_validate.sh <config.yaml>" >&2
  exit 1
fi

python XLFusion.py --batch "$1" --validate-only

