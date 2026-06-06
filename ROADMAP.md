# XLFusion Roadmap

Current status: `main` covers package/runtime reorganization, shared validation and preflight, actionable analysis, checkpoint algebra, explicit component scope, LoRA audit reporting, reproducible metadata, CLI/GUI/batch execution, presets, metadata recovery, V2.4 shared runtime/types/block-mapping work, V2.5 regression coverage, and V2.4.1 OSS readiness work.

This roadmap only tracks future improvements that still add real value to the current product.

## Completed: V2.4.1 OSS Readiness

V2.4.1 focused on making the repository installable, reviewable, and maintainable as an early-stage public OSS project without changing merge behavior.

Completed outcomes:
- MIT license and package license metadata
- editable install path with `python -m pip install -e .`
- console entrypoint through `xlfusion`
- GitHub Actions CI for Python 3.10 and 3.11
- contribution, security, code of conduct, maintainer, changelog, release notes, issue template, and PR template documentation
- OSS-safe examples for batch, hybrid, algebra, and metadata recovery workflows
- workspace ignore rules and placeholders for local runtime artifacts

## Completed: V2.5 Regression Coverage

V2.5 focused on protecting existing behavior with fast synthetic tests instead of adding new merge modes.

Completed outcomes:
- broader unittest coverage for blocks, execution, memory, batch runner, app entrypoints, analysis, algebra, metadata recovery, and LoRA flows
- regression tests for historical failure cases and stable contracts
- synthetic safetensors fixtures so validation does not require real checkpoints
- a smoke test path that generates bounded test models and cleans temporary outputs

## Next: V2.6 Internal Maintainability

Goal: continue reducing implementation risk without changing merge behavior.

Priorities:
- continue extracting large UI/orchestration modules into smaller internal modules without changing CLI, GUI, batch, metadata, or preset contracts
- keep package metadata and entrypoints aligned with `python XLFusion.py`, `python -m xlfusion`, `xlfusion`, and GUI workflows
- add a small release validation script once the manual release checklist has settled
- keep `config.yaml.example`, examples, and metadata recovery docs aligned with current behavior
- improve maintainer diagnostics for CI failures and dependency installation issues
- add focused regression tests whenever a cleanup changes module boundaries

Acceptance criteria:
- fresh clones have no local machine or agent artifacts
- the full unittest suite stays fast and passes without real SDXL checkpoints
- package entrypoints and direct script entrypoints remain equivalent
- README, roadmap, and build scripts describe the current product rather than legacy state

## Later Reliability Ideas

- split fast contract tests from heavier smoke/integration tests if the suite grows
- add optional lint/typecheck commands once the codebase is ready to enforce them consistently
- replace direct merge-engine `print()` calls with an injectable reporter/logger while preserving CLI output
- define a small compatibility checklist for future block mappings beyond SDXL
