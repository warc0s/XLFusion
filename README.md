# XLFusion

XLFusion is a Python tool for merging SDXL checkpoints with a reproducible workflow across CLI, GUI, and batch execution. As of V2.15, the product code lives in `xlfusion/` and the user runtime lives in `workspace/`. It focuses on reliable validation, per-block control, LoRA baking, metadata you can recreate later, and lightweight analysis before or after a merge.

## What It Does

- Merge SDXL-derived `.safetensors` checkpoints in three modes: `legacy`, `perres`, and `hybrid`
- Bake LoRAs into the merged result
- Validate configurations before execution in CLI, GUI, and batch
- Show a preflight plan with estimated memory, backbone, affected blocks, effective locks, and compatibility warnings
- Save reproducible metadata and a batch YAML that can recreate the run
- Analyze similarity, compatibility, and likely merge characteristics

## Merge Modes

- `legacy`
  Weighted merge across selected models with optional coarse `down/mid/up` multipliers and cross-attention boosts.

- `perres`
  Full assignment by block group: `down_0_1`, `down_2_3`, `mid`, `up_0_1`, `up_2_3`.

- `hybrid`
  Per-block weighted mixing with optional attention locks. Useful for moving style into `up_*` while preserving composition in `down_*`.

## Requirements

- Python 3.10+
- Packages from `requirements.txt`

```bash
pip install -r requirements.txt
```

Main dependencies: `torch`, `safetensors`, `PyYAML`, `numpy`, `tqdm`, `psutil`.

## Configuration

- `config.yaml` is optional.
- `config.yaml.example` is the distributable template.
- If `config.yaml` is missing, invalid, or partially defined, XLFusion falls back to safe built-in defaults.

## Project Layout

```text
XLFusion/
├── XLFusion.py
├── gui_app.py
├── config.yaml
├── config.yaml.example
├── xlfusion/
├── workspace/
│   ├── models/
│   ├── loras/
│   ├── output/
│   └── metadata/
├── scripts/
└── tests/
```

## Basic Usage

1. Place input checkpoints in `workspace/models/`.
2. Place optional LoRAs in `workspace/loras/`.
3. Run one of the entry points below.

Interactive CLI:

```bash
python XLFusion.py
```

GUI:

```bash
python XLFusion.py --gui
```

Batch:

```bash
python XLFusion.py --batch batch_config_example.yaml
python XLFusion.py --batch batch_config_example.yaml --validate-only
```

Analysis:

```bash
python XLFusion.py --analyze --compare 0 1
python XLFusion.py --analyze --recommend balanced
python XLFusion.py --analyze --compare 0 1 --export-analysis report.json
```

## CLI and GUI Workflow

The interactive flows now share the same execution guardrails:

- configuration is validated before merge execution
- invalid values do not reach `merge_*`
- a preflight plan is shown before running
- the preflight can be exported to `.txt` or `.json`

The GUI also provides:

- model list with size and timestamps
- per-block visual preview
- real-time progress and cancellation

## Batch Workflow

Batch mode uses the same validator as CLI and GUI. That means:

- file existence is checked before merge execution
- weights, assignments, locks, backbone and LoRAs are validated centrally
- memory and compatibility warnings are available during validation

See:

- `batch_config_example.yaml`
- `tests/test_batch_full.yaml`
- `xlfusion/templates.py`

Shortcuts:

- `scripts/run_batch.sh <config.yaml>`
- `scripts/run_batch_validate.sh <config.yaml>`

## Outputs and Metadata

Merged checkpoints are written to `workspace/output/` as versioned `.safetensors` files.

Each run also creates a metadata folder in `workspace/metadata/` containing:

- `metadata.txt`
- `batch_config.yaml`

The saved metadata includes source models, hashes, mode, backbone, and merge parameters. Metadata is also embedded in the resulting `.safetensors` file.

## Analysis

The analysis tools are meant to support merge decisions, not just report raw numbers. Current outputs include:

- cosine similarity by block
- compatibility score and architecture warnings
- difference summaries
- prediction and recommendation helpers

## Validation and Testing

Main validation command:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Smoke test:

```bash
scripts/smoke_test.sh
```

The smoke test generates synthetic models, runs a bounded batch scenario, and removes temporary artifacts afterwards.

## Future Work

`ROADMAP.md` now starts after the already implemented package/runtime reorganization and validation foundation. The remaining roadmap focuses on performance, reusable presets, metadata recovery, and deeper merge analysis.

## Credits

- Portfolio: https://warcos.dev/
- LinkedIn: https://www.linkedin.com/in/marcosgarest/
