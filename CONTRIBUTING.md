# Contributing to XLFusion

Thanks for helping improve XLFusion. This project is an early-stage public OSS tool focused on reproducible SDXL checkpoint merging across CLI, GUI, batch, analysis, metadata recovery, and checkpoint algebra workflows.

## Development Setup

Use Python 3.10 or newer. A virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The editable install exposes the package and console script:

```bash
python XLFusion.py --help
python -m xlfusion --help
xlfusion --help
```

## Running Tests

Run the fast unittest suite before opening a pull request:

```bash
python -m compileall xlfusion XLFusion.py gui_app.py
python -m unittest discover -s tests -p "test_*.py"
python -m xlfusion --help
xlfusion --help
```

If your change affects batch execution, metadata recovery, LoRA baking, analysis, algebra, or merge behavior, add a focused regression test under `tests/`.

The smoke test uses tiny synthetic models and cleans up after itself:

```bash
bash scripts/smoke_test.sh
```

Do not use real SDXL checkpoints for repository tests.

## Adding Tests

- Prefer `unittest`, matching the existing suite.
- Use synthetic safetensors fixtures or small generated tensors.
- Keep tests deterministic and fast enough for CI.
- Cover public contracts: CLI behavior, batch validation, metadata shape, recovery, analysis output, and merge-mode invariants.
- Do not commit generated checkpoints, LoRAs, images, or large binary artifacts.

## Reporting Bugs

Open a bug report with:

- XLFusion version
- Python version
- operating system
- mode used: CLI, GUI, batch, analysis, or algebra
- exact command or sanitized YAML
- expected behavior
- actual behavior
- relevant logs or traceback

If real checkpoints are involved, provide only generic names, source family, tensor-shape notes, or partial hashes. Never upload proprietary models, LoRAs, generated images, or large binary files.

## Pull Request Style

- Keep PRs focused on one behavior, maintenance task, or documentation area.
- Explain why the change is needed and how it was validated.
- Preserve compatibility with `python XLFusion.py`, `python -m xlfusion`, and `xlfusion`.
- Avoid changing merge behavior unless the PR fixes a clear bug and includes tests.
- Update `README.md`, `CHANGELOG.md`, examples, or maintainer docs when behavior or user-facing commands change.
- Write new public documentation, code comments, identifiers, and tests in English.

## Artifact Policy

The repository must not contain real checkpoints, LoRAs, generated images, local configs, logs, private paths, API keys, or other large/proprietary artifacts. The `workspace/` directories are runtime locations only and are ignored except for placeholders.
