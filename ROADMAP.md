# XLFusion Roadmap

Current status: `main` already covers V2.2 and V2.15 with CLI, GUI, batch mode, analysis, reproducible metadata, shared validation, preflight, batch-compatible presets, metadata recovery, and the `xlfusion/` + `workspace/` reorganization. This roadmap only tracks future improvements that still add real value to the product.

## Principles For Upcoming Versions

- Prioritize reliability before adding more merge modes.
- Improve memory usage and speed without sacrificing reproducibility.
- Keep the experience aligned across CLI, GUI, and batch.
- Turn analysis into something that actively helps users make better merge decisions.

## V2.3 Merge Quality And Actionable Analysis

Goal: move from "seeing data" to "making better merge decisions".

### 1. Submodule And Layer Analysis

- Extend the analyzer with metrics by submodule, histograms, and summaries by model region.
- Better separate structure, semantics, and style so recommendations are not just based on a single global score.

Acceptance criteria:
- The report helps explain which model dominates composition, detail, and style.

### 2. Weight And Block Recommender

- Generate initial suggestions for `hybrid_config`, `assignments`, and backbone selection from analysis results.
- Offer profiles such as `balanced`, `style transfer`, `detail recovery`, or `prompt fidelity`.

Acceptance criteria:
- The user can start from a reasonable proposal without configuring everything manually.

### 3. Compatibility Alerts Before Merging

- Detect potentially dangerous differences before execution: incompatible shapes, models that are too far apart, inconsistent locks, or combinations with low expected value.
- Integrate those alerts into preflight and GUI.

Acceptance criteria:
- High-risk combinations are detected before spending time and memory on a failed or poor merge.

### 4. Checkpoint Algebra

- Add operations like `A + alpha(B - C)` and compatible variants for `legacy` and `hybrid`.
- Reuse the streaming engine to avoid excessive memory usage.
- Expose it as an advanced mode, not as a replacement for the main workflow.

Acceptance criteria:
- Synthetic tests verify the tensor arithmetic and the output is fully audited in metadata.

### 5. Expanded LoRA Support

- Extend LoRA baking beyond UNet when compatible text-encoder keys exist.
- Add better shape validation and a report by submodule showing what was applied or skipped.

Acceptance criteria:
- The user knows exactly which parts of a LoRA were applied and which were not.

### 6. Explicit Non-UNet Mixing

- Turn `only_unet` into a visible, supported option across all modes.
- Allow explicit inclusion or exclusion of VAE and text encoder where it makes sense.

Acceptance criteria:
- The configuration clearly shows which components are being merged and that choice is preserved in metadata.

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

1. V2.2 Expert Workflow And High Performance
2. V2.3 Merge Quality And Actionable Analysis
3. V2.4 Platform And Internal Evolution
4. V2.5 Full Coverage Of The Current Product
