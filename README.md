# XLFusion

[![CI](https://github.com/warc0s/XLFusion/actions/workflows/ci.yml/badge.svg)](https://github.com/warc0s/XLFusion/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

XLFusion is a Python toolkit for reproducible SDXL checkpoint merging. It is built for users and maintainers who need repeatable merge workflows across CLI, GUI, batch execution, analysis, metadata recovery, and checkpoint algebra instead of ad-hoc local scripts.

XLFusion does not include SDXL checkpoints, LoRAs, generated images, or model binaries. Place your own `.safetensors` files in `workspace/models/` and optional LoRAs in `workspace/loras/`.

## Quick Install

Use Python 3.10 or newer:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

This installs the package dependencies and exposes the `xlfusion` console script. `requirements.txt` mirrors the runtime dependencies for users who prefer inspecting them directly.

## Quick Usage

CLI help:

```bash
python XLFusion.py --help
python -m xlfusion --help
xlfusion --help
```

Interactive CLI:

```bash
python XLFusion.py
```

Batch validation:

```bash
python XLFusion.py --batch examples/minimal_batch.yaml --validate-only
```

Batch execution:

```bash
python XLFusion.py --batch examples/minimal_batch.yaml
```

GUI:

```bash
python XLFusion.py --gui
```

Run tests:

```bash
python -m compileall xlfusion XLFusion.py gui_app.py
python -m unittest discover -s tests -p "test_*.py"
```

## What It Does

- Merges SDXL-derived `.safetensors` checkpoints in `legacy`, `perres`, and `hybrid` modes.
- Bakes compatible LoRA weights into UNet and text encoder targets with an audit trail.
- Validates configurations before execution in CLI, GUI, and batch flows.
- Shows a preflight plan with estimated memory, backbone, affected blocks, effective locks, component scope, compatibility warnings, and risk alerts.
- Runs merges in `standard` or `low-memory` execution mode with shared progress handling.
- Saves reusable presets as batch-compatible YAML.
- Saves metadata and a batch YAML that can recreate the run.
- Recovers previous runs from metadata folders.
- Analyzes similarity, compatibility, region/submodule dominance, and merge starting points.
- Applies checkpoint algebra with `A + alpha(B - C)`.

## Who It Is For

XLFusion is for SDXL users, experimenters, and maintainers who care about:

- repeatable merge configurations
- validation before long-running local work
- metadata that explains how a checkpoint was produced
- batch configs that can be reviewed and rerun
- fast regression tests that do not require real checkpoints

It is not a model host and does not provide model recommendations.

## Why This Matters

XLFusion targets a niche but real workflow in the SDXL ecosystem: reproducible checkpoint merging. It focuses on validation, auditability, metadata recovery and repeatable batch configs rather than ad-hoc local scripts.

## Project Status

- Early-stage public OSS tool.
- Maintained by primary maintainer.
- Focus: reliability, regression tests, packaging and reproducible workflows.
- Current maturity release: `v2.4.1`.

## Merge Modes

`legacy` performs weighted checkpoint merging across selected models with optional coarse `down`, `mid`, and `up` multipliers plus cross-attention boosts.

`perres` assigns each SDXL block group to a source model: `down_0_1`, `down_2_3`, `mid`, `up_0_1`, and `up_2_3`.

`hybrid` performs per-block weighted mixing with optional attention locks. It is useful for preserving composition in down blocks while moving style or detail into up blocks.

## Batch Workflow

Batch mode uses the same validator as CLI and GUI:

- file existence is checked before merge execution
- weights, assignments, locks, backbone, and LoRAs are validated centrally
- memory, compatibility, and risk warnings are available during validation
- execution settings can be stored per job for `low-memory` or `standard` runs
- `only_unet`, `component_policy`, and `block_mapping` are preserved in presets, batch YAML, and metadata recovery

Examples:

- [examples/minimal_batch.yaml](examples/minimal_batch.yaml)
- [examples/hybrid_batch.yaml](examples/hybrid_batch.yaml)
- [batch_config_example.yaml](batch_config_example.yaml)

Shortcuts:

```bash
scripts/run_batch_validate.sh examples/minimal_batch.yaml
scripts/run_batch.sh examples/minimal_batch.yaml
```

## GUI Workflow

The GUI can be launched with:

```bash
python XLFusion.py --gui
```

It provides model listing, per-block preview, progress/cancellation, preset import/export, metadata recovery helpers, and component scope controls that stay aligned with CLI and batch metadata.

## Analysis and Algebra

Analysis examples:

```bash
python XLFusion.py --analyze --compare 0 1
python XLFusion.py --analyze --recommend balanced
python XLFusion.py --analyze --compare 0 1 --export-analysis report.json
```

Checkpoint algebra:

```bash
python XLFusion.py --algebra 0 1 2 --alpha 0.35 --algebra-output AlgebraMix
python XLFusion.py --algebra 0 1 2 --alpha 0.35 --include-non-unet
```

See [examples/algebra_usage.md](examples/algebra_usage.md).

## Outputs and Metadata

Merged checkpoints are written to `workspace/output/` as versioned `.safetensors` files. Each run also creates a metadata folder in `workspace/metadata/` containing:

- `metadata.txt`
- `batch_config.yaml`

The saved metadata includes source models, hashes, mode, backbone, execution settings, merge parameters, component scope, and LoRA/algebra audit details when relevant. Metadata is also embedded in the resulting `.safetensors` file.

See [examples/metadata_recovery.md](examples/metadata_recovery.md).

## Configuration

- `config.yaml` is optional and ignored by git.
- `config.yaml.example` is the distributable template.
- If `config.yaml` is missing, invalid, or partially defined, XLFusion falls back to safe built-in defaults.
- Runtime files belong under `workspace/`, which is ignored except for placeholders.

## Project Layout

```text
XLFusion/
├── XLFusion.py
├── gui_app.py
├── pyproject.toml
├── requirements.txt
├── xlfusion/
├── tests/
├── scripts/
├── examples/
├── docs/
└── workspace/
    ├── models/
    ├── loras/
    ├── output/
    ├── metadata/
    └── presets/
```

## Validation

Main validation command:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Smoke test with tiny synthetic models:

```bash
bash scripts/smoke_test.sh
```

The unittest suite is intended to stay fast and run on every change. The smoke test generates synthetic models, runs a bounded batch scenario, and removes temporary artifacts afterwards.

## Maintenance

- [Contributing](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
- [Changelog](CHANGELOG.md)
- [Release Notes](RELEASE_NOTES.md)
- [Roadmap](ROADMAP.md)
- [Maintainer Guide](docs/MAINTAINER_GUIDE.md)

## License

XLFusion is released under the [MIT License](LICENSE).

## Credits

- Portfolio: https://warcos.dev/
- LinkedIn: https://www.linkedin.com/in/marcosgarest/
