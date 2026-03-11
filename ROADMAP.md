# XLFusion Roadmap

Current status: `main` already covers the package/runtime reorganization, shared validation and preflight, actionable analysis, checkpoint algebra, explicit component scope, LoRA audit reporting, reproducible metadata, CLI/GUI/batch execution, presets, metadata recovery, and the V2.4 platform work (shared merge runtime, shared types, and a block-mapping registry for future partitions). This roadmap only tracks future improvements that still add real value to the product.

## Principles For Upcoming Versions

- Prioritize reliability before adding more merge modes.
- Improve memory usage and speed without sacrificing reproducibility.
- Keep the experience aligned across CLI, GUI, and batch.
- Turn analysis into something that actively helps users make better merge decisions.

## V2.5 Full Coverage Of The Current Product

Goal: build a broad and useful test base that covers the code already in place so regressions are detected much earlier on every change.

Status: implemented on `main` (2026-03-11) by expanding the unittest suite with additional contract coverage (blocks/execution/memory/batch runner) and a metadata recovery roundtrip test.

### 1. Cover Every Relevant Module In The Current Product

- Review `xlfusion/` module by module and add tests wherever coverage is currently missing or clearly insufficient.
- Prioritize observable behavior and public contracts before fragile implementation details.
- Keep the focus on the current product, not on future features.

Acceptance criteria:
- The main areas of the product have automated tests covering normal behavior, expected failures, and useful edge cases.

### 2. Harden CLI, Batch, Workflow, Metadata, And Recovery

- Add dedicated tests for the flows most likely to break compatibility: configuration loading, validation, execution, persistence, presets, and metadata recovery.
- Verify that the same configuration produces coherent artifacts in CLI, batch, and GUI when they share the same common layer.
- Ensure output name handling, metadata, recreated batch YAML, and execution options remain stable through refactors.

Acceptance criteria:
- A change in configuration, workflow, or persistence breaks focused tests before it reaches the user.

### 3. Compare Execution Paths To Avoid Silent Divergence

- Add synthetic tests that compare results between `standard` and `low-memory`, across modes where appropriate, and between direct execution and metadata-based reconstruction.
- Include numerical equivalence checks within tolerance and output-structure checks.
- Define and verify minimum contracts for progress, cancellation, and warnings.

Acceptance criteria:
- If two execution paths that should behave the same start diverging, tests detect it immediately.

### 4. Real Coverage For Errors And Historical Regressions

- Turn previously found bugs into permanent tests before or alongside any fix.
- Cover shape errors, missing models, incompatible LoRAs, invalid YAML, incomplete presets, and partial metadata.
- Avoid having a suite made only of happy-path tests.

Acceptance criteria:
- Known or plausible failures have automated regression coverage and do not rely on team memory to avoid repeating them.

### 5. A Reliable Suite For Day-To-Day Development

- Keep the suite reasonably fast with small synthetic models so it can be run frequently.
- Split fast contract tests from heavier integration tests when needed, without losing useful coverage.
- Clearly document which commands validate the product and which ones are required before accepting a functional change.

Acceptance criteria:
- After any relevant change, there is a clear set of tests that provides real confidence about the state of the product.

## Recommended Priorities

1. Keep V2.5 coverage healthy (add regressions as bugs are found)
2. Define V2.6+ reliability goals
