# XLFusion Maintainer Guide

This guide covers routine maintenance for the public OSS repository. It does not replace code review or release judgment.

## Release Checklist

1. Confirm the worktree is clean except for intentional release changes:

   ```bash
   git status --short --branch
   ```

2. Confirm version references are aligned:

   ```bash
   python - <<'PY'
   import tomllib
   from pathlib import Path
   from xlfusion.version import __version__

   data = tomllib.loads(Path("pyproject.toml").read_text())
   print("pyproject:", data["project"]["version"])
   print("package:", __version__)
   PY
   ```

3. Review `CHANGELOG.md` and `RELEASE_NOTES.md`.
4. Confirm no real checkpoints, LoRAs, generated images, local configs, logs, or large binary artifacts are staged.
5. Run the validation commands below.

## Test Checklist

Run these commands from a fresh virtual environment when possible:

```bash
python --version
python -m pip install --upgrade pip
python -m pip install -e .
python -m compileall xlfusion XLFusion.py gui_app.py
python -m unittest discover -s tests -p "test_*.py"
python -m xlfusion --help
xlfusion --help
```

If the change affects merge, LoRA baking, metadata, batch, analysis, or algebra behavior, also run:

```bash
bash scripts/smoke_test.sh
```

The smoke test uses tiny synthetic checkpoints generated locally. It must not require real SDXL models or GPU access.

## Validating a Release

Before tagging:

- inspect the diff and staged files
- check that `LICENSE`, `README.md`, `CHANGELOG.md`, `RELEASE_NOTES.md`, `CONTRIBUTING.md`, and `SECURITY.md` are current
- verify the GitHub Actions workflow passes on Python 3.10 and 3.11
- install the package with `python -m pip install -e .`
- verify all supported entrypoints:
  - `python XLFusion.py --help`
  - `python -m xlfusion --help`
  - `xlfusion --help`

If PyTorch installation fails in CI, do not hide the failure. Check the Python version, platform, and available PyTorch wheels. Prefer keeping the failure visible and documenting a narrow fix instead of skipping tests that need the installed dependency set.

## Creating a Local Tag

Only tag after review and successful validation:

```bash
git tag -a v2.4.1 -m "XLFusion v2.4.1"
```

Push only after human confirmation:

```bash
git push origin v2.4.1
```

## Publishing a GitHub Release

After the tag exists remotely and the release notes have been reviewed:

```bash
gh release create v2.4.1 --title "XLFusion v2.4.1" --notes-file RELEASE_NOTES.md
```

Do not publish a release automatically from local preparation work. A maintainer should review the final diff, CI status, and release notes first.

## Safe PR Review

- Check whether the PR changes merge behavior, metadata format, config parsing, path handling, or batch validation.
- Require tests for behavior changes and regression fixes.
- Reject committed checkpoints, LoRAs, generated images, logs, private configs, or large binary artifacts.
- Look for private paths, account names, API keys, and proprietary model names in examples or logs.
- Prefer small, reviewable PRs with direct validation output.
- Confirm CLI, GUI, batch, analysis, and algebra flows stay coherent when shared modules change.
- For security-sensitive changes, check path normalization, YAML parsing, recovered metadata inputs, and generated output locations.
