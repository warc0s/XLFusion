"""Non-interactive application modes for the XLFusion CLI entrypoint."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import AppContext, list_safetensors
from .execution import execution_options_to_dict
from .merge import stream_checkpoint_algebra_from_paths
from .presets import inspect_recovery_source
from .validation import format_preflight_plan
from .workflow import save_merge_results

try:
    from .analyzer import (
        CompatibilityAnalyzer,
        FusionPredictor,
        ModelDiffAnalyzer,
        RecommendationEngine,
        export_analysis_json,
        generate_analysis_report,
    )

    ANALYZER_AVAILABLE = True
except ImportError:  # pragma: no cover - optional module handling
    ANALYZER_AVAILABLE = False


def analyze_mode(args: Any, models_dir: Path, output_dir: Path) -> int:
    """Execute analysis mode operations."""
    if not ANALYZER_AVAILABLE:
        print("Error: Analyzer module not available.")
        print("Make sure xlfusion/analyzer.py is available")
        return 1

    model_files = list_safetensors(models_dir)
    if not model_files:
        print(f"No models found in {models_dir}")
        return 1

    print("\nAvailable models:")
    for i, path in enumerate(model_files):
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  [{i}] {path.name} ({size_mb:.1f} MB)")

    results = {}

    if args.compare:
        if len(args.compare) != 2:
            print("Error: --compare requires exactly 2 model indices")
            return 1

        try:
            idx1, idx2 = int(args.compare[0]), int(args.compare[1])
            if not (0 <= idx1 < len(model_files) and 0 <= idx2 < len(model_files)):
                print(f"Error: Invalid model indices (0-{len(model_files)-1})")
                return 1
        except ValueError:
            print("Error: Model indices must be integers")
            return 1

        analyzer = ModelDiffAnalyzer()
        results["diff_analysis"] = analyzer.analyze_model_differences(
            model_files[idx1],
            model_files[idx2],
        )

        compat_analyzer = CompatibilityAnalyzer()
        results["compatibility"] = compat_analyzer.calculate_compatibility(
            [model_files[idx1], model_files[idx2]]
        )

    if args.recommend:
        goal = args.recommend
        if goal not in RecommendationEngine.GOALS:
            print(f"Error: Unknown goal '{goal}'")
            print(f"Available goals: {', '.join(RecommendationEngine.GOALS.keys())}")
            return 1

        raw_idx = input("Enter model indices to analyze (comma-separated): ").strip()
        if not raw_idx:
            print("No models selected")
            return 1

        selected_idx = []
        for tok in raw_idx.split(","):
            try:
                idx = int(tok.strip())
                if 0 <= idx < len(model_files):
                    selected_idx.append(idx)
            except ValueError:
                pass

        if len(selected_idx) < 2:
            print("Select at least 2 models")
            return 1

        selected_models = [model_files[i] for i in selected_idx]

        engine = RecommendationEngine()
        recommendations = engine.generate_recommendations(selected_models, goal)
        results["recommendations"] = recommendations

        predictor = FusionPredictor()
        suggested = (
            recommendations[0].suggested_config
            if recommendations
            else {"mode": "legacy", "weights": [0.5, 0.5]}
        )
        results["prediction"] = predictor.predict_fusion_characteristics(
            selected_models,
            suggested or {"mode": "legacy"},
        )

    if results:
        report = generate_analysis_report(results)
        print("\n" + report)

        if args.export_analysis:
            export_path = Path(args.export_analysis)
            export_analysis_json(results, export_path)

    return 0


def recovery_mode(args: Any, context: AppContext) -> int:
    """Inspect, export, or execute a run recovered from metadata."""
    try:
        inspection = inspect_recovery_source(args.recover_metadata, context)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print("\nRecovered metadata")
    print("=" * 60)
    print(f"Folder: {inspection.metadata_folder}")
    print(f"Batch config: {inspection.batch_config_path}")
    print(f"Job: {inspection.job.name}")
    print(f"Mode: {inspection.job.mode}")
    print(f"Models: {', '.join(inspection.job.models)}")
    print(f"Output name: {inspection.job.output_name or 'default'}")
    if inspection.missing_models:
        print(f"Missing models: {', '.join(inspection.missing_models)}")
    if inspection.missing_loras:
        print(f"Missing LoRAs: {', '.join(inspection.missing_loras)}")
    for warning in inspection.warnings:
        print(f"  WARNING: {warning}")

    if args.export_recovered:
        args.export_recovered.write_text(
            inspection.batch_config_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        print(f"Recovered YAML exported to: {args.export_recovered}")

    if args.run_recovered:
        from .batch_processor import BatchProcessor, BatchValidator, load_batch_config

        try:
            recovered_config = load_batch_config(inspection.batch_config_path)
        except Exception as exc:
            print(f"Error: could not load recovered batch config: {exc}")
            return 1

        validator = BatchValidator(context)
        if not validator.validate_config(recovered_config):
            print("Recovered configuration validation failed:")
            for error in validator.errors:
                print(f"  ERROR: {error}")
            for warning in validator.warnings:
                print(f"  WARNING: {warning}")
            return 1

        processor = BatchProcessor(recovered_config, context, validate_only=False)
        results = processor.process_batch()
        return 0 if results["failed_jobs"] == 0 else 1

    return 0


def checkpoint_algebra_mode(args: Any, context: AppContext) -> int:
    """Run checkpoint algebra from model indices."""
    model_files = list_safetensors(context.models_dir)
    if not model_files:
        print(f"No models found in {context.models_dir}")
        return 1
    try:
        a_idx, b_idx, c_idx = [int(item) for item in args.algebra]
    except ValueError:
        print("Error: checkpoint algebra indices must be integers")
        return 1
    for idx in (a_idx, b_idx, c_idx):
        if idx < 0 or idx >= len(model_files):
            print(f"Error: invalid checkpoint algebra index {idx} (0-{len(model_files) - 1})")
            return 1

    model_paths = [model_files[a_idx], model_files[b_idx], model_files[c_idx]]
    output_base_name = args.algebra_output or context.config["model_output"]["base_name"]
    only_unet = not args.include_non_unet
    component_policy = None if only_unet else {"vae": "merge", "text_encoder": "merge", "other": "merge"}
    merged, _stats, audit = stream_checkpoint_algebra_from_paths(
        model_paths,
        args.alpha,
        a_idx=0,
        b_idx=1,
        c_idx=2,
        only_unet=only_unet,
        component_policy=component_policy,
        execution=execution_options_to_dict(None),
    )
    yaml_kwargs = {
        "weights": [1.0, args.alpha, -args.alpha],
        "only_unet": only_unet,
        "component_policy": component_policy,
    }
    output_path, metadata_folder, version = save_merge_results(
        context.output_dir,
        context.metadata_dir,
        merged,
        [path.name for path in model_paths],
        "legacy",
        0,
        yaml_kwargs,
        model_paths=model_paths,
        output_base_name=output_base_name,
        extra_metadata={
            "operation": "checkpoint_algebra",
            "formula": "A + alpha(B - C)",
            "alpha": str(args.alpha),
        },
        execution=execution_options_to_dict(None),
        job_name="CLI_checkpoint_algebra",
        job_description="Advanced checkpoint algebra run",
        audit_sections={"checkpoint_algebra": audit},
    )
    print(f"Checkpoint algebra completed: {output_path.name} (V{version})")
    print(f"Metadata saved to: {metadata_folder}")
    return 0


def batch_mode(args: Any, context: AppContext) -> int:
    """Load, validate, and process a batch configuration."""
    from .batch_processor import BatchProcessor, BatchValidator, load_batch_config

    try:
        config = load_batch_config(args.batch)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    validator = BatchValidator(context)
    if not validator.validate_config(config):
        print("Batch configuration validation failed:")
        for error in validator.errors:
            print(f"  ERROR: {error}")
        for warning in validator.warnings:
            print(f"  WARNING: {warning}")
        return 1

    processor = BatchProcessor(config, context, args.validate_only)
    results = processor.process_batch()
    return 0 if results["failed_jobs"] == 0 else 1
