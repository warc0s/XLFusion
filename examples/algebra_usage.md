# Checkpoint Algebra Usage

XLFusion supports checkpoint algebra in the form:

```text
A + alpha(B - C)
```

This is an advanced workflow for applying a directional difference between two checkpoints to a base checkpoint.

## Setup

Place your checkpoints in `workspace/models/`. Example names:

```text
workspace/models/base_model_a.safetensors
workspace/models/base_model_b.safetensors
workspace/models/base_model_c.safetensors
```

Do not commit these files. The repository intentionally does not include models, LoRAs, generated images, or large binary artifacts.

## Inspect Model Indices

Run the interactive CLI or analysis commands to confirm model ordering:

```bash
python XLFusion.py
```

Model indices are based on the sorted `.safetensors` files found in `workspace/models/`.

## Run Algebra

Example:

```bash
python XLFusion.py --algebra 0 1 2 --alpha 0.35 --algebra-output AlgebraExample
```

This means:

- model index `0` is `A`, the base checkpoint
- model index `1` is `B`, the positive direction
- model index `2` is `C`, the negative direction
- `alpha` controls the strength of `B - C`

By default, checkpoint algebra focuses on UNet tensors. To include non-UNet tensors:

```bash
python XLFusion.py --algebra 0 1 2 --alpha 0.35 --algebra-output AlgebraExample --include-non-unet
```

## Validation Notes

- Use conservative `alpha` values first.
- Review generated metadata after each run.
- Keep source checkpoint names and hashes in your local notes for reproducibility.
- Do not publish proprietary model files or private output artifacts.
