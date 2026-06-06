# Metadata Recovery Usage

XLFusion writes reproducible metadata for merge runs. Metadata folders can be inspected and converted back into batch YAML so a run can be audited or repeated.

## Expected Runtime Files

After a merge, XLFusion writes output artifacts under:

```text
workspace/output/
workspace/metadata/
```

Each metadata folder should contain:

```text
metadata.txt
batch_config.yaml
```

The repository does not include generated metadata from real checkpoints. Do not commit private metadata folders if they reveal proprietary model names, local paths, or output details.

## Inspect a Metadata Folder

```bash
python XLFusion.py --recover-metadata workspace/metadata/meta_1
```

Replace `meta_1` with the local metadata folder you want to inspect.

## Export Recovered Batch YAML

```bash
python XLFusion.py --recover-metadata workspace/metadata/meta_1 --export-recovered recovered_batch.yaml
```

Review the recovered YAML before running it. Remove private paths or proprietary names before sharing.

## Run a Recovered Job

```bash
python XLFusion.py --recover-metadata workspace/metadata/meta_1 --run-recovered
```

Only run recovered jobs when the referenced checkpoints are available in `workspace/models/` and match the intended sources.

## Sharing Metadata Safely

When reporting issues:

- include only the relevant metadata excerpt
- remove private local paths
- replace proprietary checkpoint names with generic names
- provide partial hashes only when needed
- never upload model binaries, LoRAs, generated images, or merged checkpoints
