# XLFusion Roadmap

Current status: `main` already covers the package/runtime reorganization, shared validation and preflight, actionable analysis, checkpoint algebra, explicit component scope, LoRA audit reporting, reproducible metadata, CLI/GUI/batch execution, presets, and metadata recovery. This roadmap only tracks future improvements that still add real value to the product.

## Principles For Upcoming Versions

- Prioritize reliability before adding more merge modes.
- Improve memory usage and speed without sacrificing reproducibility.
- Keep the experience aligned across CLI, GUI, and batch.
- Turn analysis into something that actively helps users make better merge decisions.

## V2.4 Platform And Internal Evolution

Goal: make it easier for the project to keep growing without repeating logic or introducing regressions.

### 1. Clearer Internal API

- Better separate the `config`, `merge`, `workflow`, `analysis`, and GUI layers.
- Reduce duplication between interactive CLI, batch, and GUI.
- Formalize shared configuration types.

### 2. Test Coverage Focused On Real Regressions

- Prioritize tests around validation, cancellation, reproducible metadata, low-memory modes, and checkpoint algebra.
- Add synthetic fixtures to compare outputs across different execution paths.

### 3. Future Architectures Without Touching The Core

- Prepare a registry of block mappings to support other partitions or derived architectures without changing the main engine.
- Keep SDXL as the primary path while avoiding unnecessary coupling.

## V2.5 Full Coverage Of The Current Product

Goal: build a broad and useful test base that covers the code already in place so regressions are detected much earlier on every change.

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

1. V2.4 Platform And Internal Evolution
2. V2.5 Full Coverage Of The Current Product
