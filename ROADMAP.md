# XLFusion Roadmap

Current status: `main` already covers package/runtime reorganization, shared validation and preflight, actionable analysis, checkpoint algebra, explicit component scope, LoRA audit reporting, reproducible metadata, CLI/GUI/batch execution, presets, metadata recovery, V2.4 shared runtime/types/block-mapping work, and V2.5 regression coverage.

This roadmap only tracks future improvements that still add real value to the current product.

## Completed: V2.5 Regression Coverage

V2.5 focused on protecting existing behavior with fast synthetic tests instead of adding new merge modes.

Completed outcomes:
- broader unittest coverage for blocks, execution, memory, batch runner, app entrypoints, analysis, algebra, metadata recovery, and LoRA flows
- regression tests for historical failure cases and stable contracts
- synthetic safetensors fixtures so validation does not require real checkpoints
- a smoke test path that generates bounded test models and cleans temporary outputs

## Next: V2.6 Repository Hygiene And Maintainability

Goal: make the repository easier to install, inspect, package, and maintain without changing merge behavior.

Priorities:
- keep local-only files out of git, especially `config.yaml`, agent metadata, caches, logs, and generated checkpoints
- maintain `config.yaml.example` as the distributable configuration template
- continue extracting large UI/orchestration modules into smaller internal modules without changing CLI, GUI, batch, metadata, or preset contracts
- keep package metadata and entrypoints aligned with the existing `python XLFusion.py`, `python -m xlfusion`, and GUI workflows
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
