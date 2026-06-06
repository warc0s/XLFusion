# Pull Request

## Summary

- 

## Validation

Run the relevant commands and paste the results:

```bash
python -m compileall xlfusion XLFusion.py gui_app.py
python -m unittest discover -s tests -p "test_*.py"
python -m xlfusion --help
xlfusion --help
```

If the change affects batch, metadata recovery, LoRA baking, analysis, or merge behavior, include the focused test or smoke command you ran.

## Scope

- [ ] I did not change merge behavior unless the PR explicitly covers a bug or tested behavior change.
- [ ] I did not add checkpoints, LoRAs, generated images, or large binary artifacts.
- [ ] I removed private local paths, account names, tokens, and proprietary model names from examples or logs.
- [ ] Public documentation, comments, and new code are written in English.

## User-Facing Changes

Describe visible CLI, GUI, batch YAML, metadata, packaging, or documentation changes.

## Risks

List compatibility, security, migration, or release risks. Use `None` only when you have checked.
