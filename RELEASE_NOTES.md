# XLFusion v2.4.1

XLFusion is a Python toolkit for reproducible SDXL checkpoint merging across CLI, GUI, batch execution, analysis, metadata recovery, and checkpoint algebra workflows.

This release is a repository maturity release. It does not add new merge modes or change the intended merge engine behavior.

## What Changed

- Added MIT licensing and package license metadata.
- Updated Python package metadata for editable installs.
- Added GitHub Actions CI on Ubuntu with Python 3.10 and 3.11.
- Added maintainer-facing project files: issue templates, PR template, contribution guide, security policy, code of conduct, maintainer guide, changelog, and release notes.
- Added OSS-safe examples for minimal batch, hybrid batch, checkpoint algebra, and metadata recovery.
- Clarified installation, testing, artifact policy, and release workflow documentation.

## Install

Use Python 3.10 or newer:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

## Run

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

GUI:

```bash
python XLFusion.py --gui
```

## Test

```bash
python -m compileall xlfusion XLFusion.py gui_app.py
python -m unittest discover -s tests -p "test_*.py"
python -m xlfusion --help
xlfusion --help
```

Optional smoke test with synthetic tiny models:

```bash
bash scripts/smoke_test.sh
```

## Models and Checkpoints

This release does not include SDXL checkpoints, LoRAs, generated images, or model binaries. Place your own `.safetensors` files in `workspace/models/` and optional LoRAs in `workspace/loras/`. Do not commit proprietary or large binary model files to the repository.

## Known Limitations

- XLFusion is an early-stage public OSS tool maintained by a primary maintainer.
- Tests use synthetic fixtures and do not guarantee quality of a merged real-world checkpoint.
- GPU access is not required for CI, but local merge performance depends on checkpoint size, RAM, storage speed, and PyTorch behavior.
- PyTorch installation can vary by Python version and platform. If dependency installation fails, check available PyTorch wheels for the target environment rather than hiding the failure.
- GUI behavior is covered by focused tests, but manual validation is still recommended before GUI-heavy releases.

## Manual Release Commands

Review the diff and CI status first. Do not run these commands until the maintainer has approved the release:

```bash
git tag -a v2.4.1 -m "XLFusion v2.4.1"
git push origin v2.4.1
gh release create v2.4.1 --title "XLFusion v2.4.1" --notes-file RELEASE_NOTES.md
```
