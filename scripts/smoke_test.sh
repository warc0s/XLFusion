#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[1/4] Generating minimal test models..."
python scripts/gen_test_models.py

echo "[2/4] Running smoke batch (4 jobs: legacy2, legacy3, perres, hybrid)..."
python XLFusion.py --batch tests/test_batch_full.yaml

echo "[3/4] Cleaning test outputs..."
rm -f models/test_model_a.safetensors models/test_model_b.safetensors models/test_model_c.safetensors || true
rm -rf output/test_output || true
rm -f batch_log.txt || true

# Limpiar metadatos de batch de los tests
if [[ -d metadata ]]; then
  find metadata -maxdepth 1 -type d -name 'batch_meta_test_*' -exec rm -rf {} + 2>/dev/null || true
fi

echo "[4/4] Smoke test finished."
