# XLFusion

XLFusion is a Python tool for merging SDXL checkpoints with a reproducible workflow across CLI, GUI, and batch execution. As of V2.4, the product code lives in `xlfusion/` and the user runtime lives in `workspace/`. It focuses on reliable validation, per-block control, low-memory execution, reusable presets, recoverable metadata, actionable analysis, checkpoint algebra, and explicit control over non-UNet components.

## What It Does

- Merge SDXL-derived `.safetensors` checkpoints in three modes: `legacy`, `perres`, and `hybrid`
- Bake LoRAs into the merged result
- Validate configurations before execution in CLI, GUI, and batch
- Show a preflight plan with estimated memory, backbone, affected blocks, effective locks, component scope, compatibility warnings, and risk alerts
- Run merges in `standard` or `low-memory` execution mode with shared progress handling
- Save reusable presets as batch-compatible YAML and load them again in CLI, GUI, or batch
- Save reproducible metadata and a batch YAML that can recreate the run
- Analyze similarity, compatibility, region/submodule dominance, and merge starting points
- Apply checkpoint algebra with `A + alpha(B - C)` as an advanced workflow
- Bake compatible LoRA weights into UNet and text encoder targets with an audit trail

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
│   ├── metadata/
│   └── presets/
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

Recover a previous run from metadata:

```bash
python XLFusion.py --recover-metadata workspace/metadata/meta_1
python XLFusion.py --recover-metadata workspace/metadata/meta_1 --export-recovered recreated.yaml
python XLFusion.py --recover-metadata workspace/metadata/meta_1 --run-recovered
```

Analysis:

```bash
python XLFusion.py --analyze --compare 0 1
python XLFusion.py --analyze --recommend balanced
python XLFusion.py --analyze --recommend prompt_fidelity
python XLFusion.py --analyze --compare 0 1 --export-analysis report.json
```

Checkpoint algebra:

```bash
python XLFusion.py --algebra 0 1 2 --alpha 0.35 --algebra-output AlgebraMix
python XLFusion.py --algebra 0 1 2 --alpha 0.35 --include-non-unet
```

## CLI and GUI Workflow

The interactive flows now share the same execution guardrails:

- configuration is validated before merge execution
- invalid values do not reach `merge_*`
- a preflight plan is shown before running
- the preflight can be exported to `.txt` or `.json`
- reusable presets can be saved as single-job batch YAML files
- metadata folders can be used to recover the exact batch configuration later
- the merge scope for VAE, text encoder, and other non-UNet tensors is explicit

The GUI also provides:

- model list with size and timestamps
- per-block visual preview
- real-time progress and cancellation
- preset import/export and metadata recovery helpers
- visible non-UNet scope controls that stay aligned with preflight and metadata

## Batch Workflow

Batch mode uses the same validator as CLI and GUI. That means:

- file existence is checked before merge execution
- weights, assignments, locks, backbone and LoRAs are validated centrally
- memory, compatibility, and risk warnings are available during validation
- execution settings can be stored per job for `low-memory` or `standard` runs
- `only_unet` and `component_policy` are preserved in presets, batch YAML, and metadata recovery
- `block_mapping` can be set per job (defaults to `sdxl`) to prepare future partitions without rewriting the merge engine

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

The saved metadata includes source models, hashes, mode, backbone, execution settings, merge parameters, component scope, and LoRA/algebra audit details when relevant. Metadata is also embedded in the resulting `.safetensors` file.

## Analysis

The analysis tools are meant to support merge decisions, not just report raw numbers. Current outputs include:

- cosine similarity by region and submodule
- compatibility score, architecture warnings, and risk alerts
- histograms and summaries for structure, semantics, style, and detail-heavy regions
- recommendation profiles for `balanced`, `style_transfer`, `detail_recovery`, and `prompt_fidelity`
- predicted dominance by block from a suggested config

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

`ROADMAP.md` now starts after the already implemented package/runtime reorganization, shared validation and preflight, actionable analysis, checkpoint algebra, explicit component scope, low-memory execution, presets, metadata recovery, and the V2.4 platform refactor (shared merge runtime, shared internal types, and a block-mapping registry). The remaining roadmap focuses on broader regression coverage for the existing product.

## Credits

- Portfolio: https://warcos.dev/
- LinkedIn: https://www.linkedin.com/in/marcosgarest/
