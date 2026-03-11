#!/usr/bin/env python3
"""Public batch entry points and CLI wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .batch_runner import BatchProcessor
from .batch_schema import (
    BatchConfig,
    BatchJob,
    BatchValidator,
    apply_template,
    interpolate_params,
    load_batch_config,
)
from .config import resolve_app_context


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line interface for batch processing."""
    import argparse

    parser = argparse.ArgumentParser(description="XLFusion batch processor")
    parser.add_argument("config", type=Path, help="Batch configuration file")
    parser.add_argument("--validate-only", action="store_true", help="Only validate configuration, don't process")
    parser.add_argument("--template", help="Use specific template (overrides job templates)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}")
        return 1

    try:
        config = load_batch_config(args.config)

        if args.template:
            if args.template not in config.templates:
                print(f"Error: Template '{args.template}' not found in configuration")
                return 1

            template = config.templates[args.template]
            for job in config.batch_jobs:
                job_data = apply_template(job.__dict__.copy(), template)
                for key, value in job_data.items():
                    if hasattr(job, key):
                        setattr(job, key, value)
                job.template = args.template

        context = resolve_app_context(Path(__file__).resolve().parent.parent)
        validator = BatchValidator(context)
        is_valid = validator.validate_config(config)

        if validator.errors:
            print("Configuration validation errors:")
            for error in validator.errors:
                print(f"  ✗ {error}")

        if validator.warnings:
            print("Configuration warnings:")
            for warning in validator.warnings:
                print(f"  ⚠ {warning}")

        if not is_valid:
            print("Configuration is invalid. Aborting.")
            return 1

        if args.validate_only:
            print("Configuration validation successful!")
            return 0

        processor = BatchProcessor(config, context, validate_only=args.validate_only)
        results = processor.process_batch()

        print("\nBatch processing completed!")
        print(f"Total jobs: {results['total_jobs']}")
        print(f"Successful: {results['successful_jobs']}")
        print(f"Failed: {results['failed_jobs']}")
        return 0 if results["failed_jobs"] == 0 else 1
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


__all__ = [
    "BatchConfig",
    "BatchJob",
    "BatchProcessor",
    "BatchValidator",
    "apply_template",
    "interpolate_params",
    "load_batch_config",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
