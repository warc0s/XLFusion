# XLFusion V2.0

Professional SDXL checkpoint merger with CLI and GUI, focused on reproducibility, per-block control, and batch workflows. It includes three merge modes (Legacy, PerRes, and Hybrid), LoRA baking, advanced analysis, and complete metadata.

## What's New in V2

- Native GUI (Tk) with a step-by-step assistant and per-block preview.
- Hybrid mode: per-block assignment + weighted mixing.
- Batch processing with validation, logs, and reproducible YAML.
- Analysis tools (diff, compatibility, prediction, recommendations).
- Enriched metadata: BLAKE2 hashes of inputs, self-contained YAML, and automatic versioning.

## Merge Modes

- Legacy (classic weighted)
  - Global weights per model, optional coarse per-block control `down/mid/up` and group multipliers `down_0_1`, `down_2_3`, `mid`, `up_0_1`, `up_2_3`.
  - Cross-attention boost (improves prompt adherence).
  - LoRA baking into the result.
  - Memory-optimized streaming process.

- PerRes (by resolution)
  - 100% assignment per block group: `down_0_1`, `down_2_3`, `mid`, `up_0_1`, `up_2_3`.
  - Optional cross-attention locks (`down/mid/up`).

- Hybrid (PerRes + mixing)
  - Define per-block weights per model (sum ≈ 1.0), with lock support.
  - Ideal for transferring style in `up_*` while preserving composition in `down_*`.

Technical compatibility:
- SDXL-derived models (NoobAI, Illustrious, Pony, etc.).
- `.safetensors` inputs/outputs compatible with A1111/ComfyUI.

## Requirements and Installation

Requires Python 3.10+ and the packages listed in `requirements.txt`:

```
pip install -r requirements.txt
```

Main packages: `torch`, `safetensors`, `PyYAML`, `numpy`, `tqdm`, `psutil`.

Note: If you package the GUI on Windows, also install `pyinstaller` and use `scripts/build_gui_exe.py`.

## Folder Structure

```
XLFusion/
├── XLFusion.py                 # Main CLI entry point
├── gui_app.py                  # Graphical interface (V2)
├── config.yaml                 # Centralized configuration
├── Utils/                      # Internal modules (merge, lora, batch, analyzer, ...)
├── models/                     # Input checkpoints (.safetensors)
├── loras/                      # Optional LoRAs (.safetensors)
├── output/                     # Merged results
├── metadata/                   # Metadata and audit logs
├── scripts/                    # Utilities (batch, smoke, build exe)
└── tests/                      # Unit tests
```

## Quick Start (Interactive CLI)

1) Place `.safetensors` models in `models/` and (optionally) LoRAs in `loras/`.
2) Run:

```
python XLFusion.py
```

3) Select models, choose the merge mode, and adjust configuration.
4) The result is saved in `output/` and the audit in `metadata/`.

## GUI (Graphical Assistant)

Launch the GUI with:

```
python XLFusion.py --gui
```

GUI features:
- Model library with size, multi-selection, and sorting.
- Mode-guided configuration (Legacy, PerRes, Hybrid) and LoRAs.
- Per-block preview with weights/assignments.
- Real progress and safe cancellation.

## Batch Processing

Define multiple jobs in a `YAML` and process them sequentially with validation and logs.

- Run batch:

```
python XLFusion.py --batch batch_config_example.yaml
```

- Validate only:

```
python XLFusion.py --batch batch_config_example.yaml --validate-only
```

See `batch_config_example.yaml` and `tests/test_batch_full.yaml` for examples of:
- Legacy with `weights`, `block_multipliers`, `crossattn_boosts`, and `loras`.
- PerRes with `assignments` and `attn2_locks`.
- Hybrid with `hybrid_config` and `attn2_locks`.

There are also templates in `Utils/templates.py` and a `templates` section in the example YAML.

Shortcuts:
- `scripts/run_batch.sh <config.yaml>`
- `scripts/run_batch_validate.sh <config.yaml>`

## Analysis Mode (V1.3)

Tools to understand differences, compatibility, and predict merge characteristics.

Examples:

```
# Comparison between two models (by shown index)
python XLFusion.py --analyze --compare 0 1

# Recommendations for a specific goal
python XLFusion.py --analyze --recommend balanced

# Export report to JSON
python XLFusion.py --analyze --compare 0 1 --export-analysis report.json
```

Key metrics: cosine similarity per block, relative changes, architecture warnings, compatibility score, and recommendations.

## Configuration (`config.yaml`)

Centralizes the output name, versioning, paths, and defaults.

- `model_output`:
  - `base_name`: output filename prefix (e.g., `XLFusion_V1.safetensors`).
  - `version_prefix`: `V`, `v`, `Ver`, etc.
  - `file_extension`: always `.safetensors`.
  - `output_dir`, `metadata_dir`, `auto_version`.

- `directories`:
  - `models`, `loras`, `output`, `metadata`.

- `merge_defaults`:
  - `legacy`: default multipliers and `cross_attention_boost`.
  - `perres`: default `cross_attention_locks`.
  - `hybrid`: auto-normalization, minimum weights, default locks.

- `app`:
  - `tool_name`, `version` (included in metadata).

## Output and Metadata

- Models: `XLFusion_V{n}.safetensors` in `output/`.
- Metadata: folder `metadata/meta_{n}/` with:
  - `metadata.txt`: human-readable summary, BLAKE2 hashes of inputs (models and LoRAs), and exact kwargs.
  - `batch_config.yaml`: reproducible job configuration to recreate the result.

Metadata is also embedded in the `.safetensors` file itself.

## Best Practices and Performance

- Check the estimated memory notice before large merges.
- In Legacy, normalize weights; use `block_multipliers` and `crossattn_boosts` to fine-tune behavior.
- With PerRes/Hybrid, use `attn2_locks` for text consistency.
- The streaming mode avoids loading everything into GPU/CPU at once.

## Tests and Smoke Test

This repo includes unit tests in `tests/` and an automated smoke test that generates synthetic models and cleans up artifacts:

```
scripts/smoke_test.sh
```

The script creates test models, runs a batch with 4 jobs, and removes temporary artifacts at the end.

## Roadmap

See `ROADMAP.md` for V2.1+ goals (stronger validation, memory/compatibility preflight, performance and extensibility improvements).

## Credits and Contact

- Portfolio: https://warcos.dev/
- LinkedIn: https://www.linkedin.com/in/marcosgarest/
