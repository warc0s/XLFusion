#!/usr/bin/env python3
"""
XLFusion - Advanced SDXL checkpoint merger

Interactive and graphical SDXL checkpoint merger with three fusion modes:
- Legacy: Classic weighted merge with down/mid/up blocks and optional LoRA baking
- PerRes: Resolution-based block control
- Hybrid: Combines PerRes assignment with weighted blending

Current product scope:
- Interactive CLI, guided GUI, and batch execution
- Shared validation and preflight before merge execution
- Actionable analysis tools for comparison, compatibility, risk detection, and recommendations
- Explicit component scope for UNet, VAE, text encoder, and other tensors
- Checkpoint algebra as an advanced CLI workflow
- Reproducible metadata and exportable batch recreation files

Expected folder structure:
  ./workspace/models     -> base checkpoints .safetensors
  ./workspace/loras      -> LoRA files .safetensors
  ./workspace/output     -> merged outputs .safetensors
  ./workspace/metadata   -> audit logs meta_*.txt

Requirements:
  pip install torch safetensors
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List

from .cli import (
    pick_backbone,
    prompt_block_merge,
    prompt_component_scope,
    prompt_crossattn_boost,
    prompt_execution_options,
    prompt_hybrid_config,
    prompt_loras,
    prompt_output_name,
    prompt_perres_assignments,
    prompt_select,
    prompt_weights,
)
from .config import list_safetensors, resolve_app_context
from .execution import execution_options_to_dict
from .merge import stream_checkpoint_algebra_from_paths
from .presets import (
    batch_job_to_runtime_state,
    inspect_recovery_source,
    load_single_job_preset,
    save_single_job_preset,
)
from .runtime import execute_merge_job
from .types import MergeJobConfig
from .validation import export_preflight_plan, format_preflight_plan, validate_merge_request
from .workflow import save_merge_results

# Optional analyzer
try:
    from .analyzer import (
        CompatibilityAnalyzer,
        FusionPredictor,
        ModelDiffAnalyzer,
        RecommendationEngine,
        generate_analysis_report, export_analysis_json
    )
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False

# Optional batch processor
try:
    from .batch_processor import BatchProcessor
    BATCH_PROCESSOR_AVAILABLE = True
except ImportError:
    BATCH_PROCESSOR_AVAILABLE = False


def analyze_mode(args, models_dir: Path, output_dir: Path) -> int:
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
    for i, p in enumerate(model_files):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  [{i}] {p.name} ({size_mb:.1f} MB)")
    
    results = {}
    
    # Model comparison
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
        results['diff_analysis'] = analyzer.analyze_model_differences(
            model_files[idx1], 
            model_files[idx2]
        )
        
        compat_analyzer = CompatibilityAnalyzer()
        results['compatibility'] = compat_analyzer.calculate_compatibility(
            [model_files[idx1], model_files[idx2]]
        )
    
    # Recommendations
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
        for tok in raw_idx.split(','):
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
        results['recommendations'] = recommendations

        predictor = FusionPredictor()
        suggested = recommendations[0].suggested_config if recommendations else {"mode": "legacy", "weights": [0.5, 0.5]}
        results['prediction'] = predictor.predict_fusion_characteristics(selected_models, suggested or {"mode": "legacy"})
    
    # Generate report
    if results:
        report = generate_analysis_report(results)
        print("\n" + report)
        
        if args.export_analysis:
            export_path = Path(args.export_analysis)
            export_analysis_json(results, export_path)
    
    return 0


def main() -> int:
    """Main orchestration function"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="XLFusion - Advanced SDXL checkpoint merger")
    
    # Batch mode arguments
    parser.add_argument("--batch", type=Path, help="Run in batch mode with configuration file")
    parser.add_argument("--validate-only", action="store_true", help="Batch mode: only validate, don't process")
    parser.add_argument("--template", help="Batch mode: override template for all jobs")
    
    # Analysis mode arguments
    parser.add_argument("--analyze", action="store_true", help="Run in analysis mode")
    parser.add_argument("--compare", nargs=2, metavar=('MODEL1', 'MODEL2'), help="Compare two models (indices)")
    parser.add_argument("--recommend", choices=['style_transfer', 'detail_recovery', 'prompt_fidelity', 'detail_enhance', 'balanced'], help="Generate recommendations for fusion goal")
    parser.add_argument("--export-analysis", type=Path, metavar='PATH', help="Export analysis to JSON file")
    parser.add_argument("--algebra", nargs=3, metavar=("A", "B", "C"), help="Run checkpoint algebra A + alpha(B - C) using model indices")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha factor for checkpoint algebra")
    parser.add_argument("--algebra-output", type=str, metavar="NAME", help="Output base name for checkpoint algebra")
    parser.add_argument("--include-non-unet", action="store_true", help="Include non-UNet tensors in checkpoint algebra")
    parser.add_argument("--gui", action="store_true", help="Launch graphical interface")
    parser.add_argument("--recover-metadata", type=Path, help="Inspect a metadata folder and reconstruct its batch job")
    parser.add_argument("--export-recovered", type=Path, metavar="PATH", help="Save the recovered batch YAML to a new path")
    parser.add_argument("--run-recovered", action="store_true", help="Execute the recovered batch job after inspection")

    args = parser.parse_args()

    # Setup directories
    context = resolve_app_context(Path(__file__).resolve().parent.parent)
    root = context.root_dir
    models_dir = context.models_dir
    loras_dir = context.loras_dir
    output_dir = context.output_dir
    metadata_dir = context.metadata_dir

    if args.gui:
        if any([args.batch, args.analyze, args.compare, args.recommend, args.recover_metadata, args.algebra]):
            print("The --gui option cannot be combined with batch or analysis modes.")
            return 1
        from .gui_app import launch_gui
        launch_gui(root)
        return 0

    if args.recover_metadata:
        if args.batch or args.analyze or args.compare or args.recommend or args.algebra:
            print("The recovery flow cannot be combined with batch or analysis modes.")
            return 1
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

    if args.algebra:
        model_files = list_safetensors(models_dir)
        if not model_files:
            print(f"No models found in {models_dir}")
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
            output_dir,
            metadata_dir,
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

    # Check for analysis mode
    if args.analyze or args.compare or args.recommend:
        return analyze_mode(args, models_dir, output_dir)

    # Check for batch mode
    if args.batch:
        if not BATCH_PROCESSOR_AVAILABLE:
            print("Error: Batch processor not available.")
            print("Make sure xlfusion/batch_processor.py is available")
            return 1
        
        # Load batch configuration
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

    # Interactive mode
    print("\n" + "="*60)
    print(" XLFusion - Advanced SDXL checkpoint merger")
    print("="*60)
    print(f"\nDetected structure:")
    print(f"  models:   {models_dir}")
    print(f"  loras:    {loras_dir}")
    print(f"  output:   {output_dir}")
    print(f"  metadata: {metadata_dir}")

    model_files = list_safetensors(models_dir)
    if not model_files:
        print(f"\nNo .safetensors files found in {models_dir}")
        return 1

    default_output_name = context.config["model_output"]["base_name"]
    execution_options = execution_options_to_dict(None)
    output_base_name = default_output_name

    weights: List[float] = []
    block_weights = None
    crossattn_boosts = None
    assignments = None
    attn2_locks = None
    hybrid_config = None
    lora_selections = []
    only_unet = None
    component_policy = None

    preset_source = input("\nLoad preset YAML or metadata folder [Enter to skip]: ").strip()
    loaded_runtime = None
    if preset_source:
        preset_path = Path(preset_source)
        try:
            if preset_path.is_dir():
                preset_path = inspect_recovery_source(preset_path, context).batch_config_path
            loaded_job = load_single_job_preset(preset_path)
            loaded_runtime = batch_job_to_runtime_state(loaded_job)
        except Exception as exc:
            print(f"Error loading preset: {exc}")
            return 1

    if loaded_runtime:
        name_to_idx = {path.name: idx for idx, path in enumerate(model_files)}
        missing_models = [name for name in loaded_runtime["models"] if name not in name_to_idx]
        if missing_models:
            print("Preset references models that are not available locally:")
            for model_name in missing_models:
                print(f"  - {model_name}")
            return 1

        model_idx = [name_to_idx[name] for name in loaded_runtime["models"]]
        model_paths = [model_files[i] for i in model_idx]
        model_names = [p.name for p in model_paths]
        mode_name = loaded_runtime["mode"]
        mode = ["legacy", "perres", "hybrid"].index(mode_name)
        execution_options = loaded_runtime.get("execution", execution_options)
        output_base_name = prompt_output_name(loaded_runtime.get("output_name") or default_output_name) or default_output_name

        config = loaded_runtime.get("config", {})
        if mode_name == "legacy":
            weights = list(config.get("weights", []))
            backbone_idx = int(config.get("backbone_idx", 0))
            block_weights = config.get("block_multipliers")
            crossattn_boosts = config.get("crossattn_boosts")
        elif mode_name == "perres":
            assignments = config.get("assignments")
            attn2_locks = config.get("attn2_locks")
            backbone_idx = list(assignments.values())[0] if assignments else 0
        else:
            hybrid_config = config.get("hybrid_config")
            attn2_locks = config.get("attn2_locks")
            backbone_idx = 0
        only_unet = config.get("only_unet")
        component_policy = config.get("component_policy")
        lora_selections = list(config.get("loras", []))

        adjust_exec = input("Adjust execution profile from the preset? [y/N]: ").strip().lower()
        if adjust_exec in {"y", "yes"}:
            execution_options = prompt_execution_options()
    else:
        model_idx = prompt_select(model_files, "Select models to merge (comma-separated indices):", [0, 1])
        if len(model_idx) < 2:
            print("\nSelect at least 2 models")
            return 1

        model_paths = [model_files[i] for i in model_idx]
        model_names = [p.name for p in model_paths]

        print("\nSelect fusion mode:")
        print("  [0] Legacy - Classic weighted merge")
        print("  [1] PerRes - Resolution-based block control")
        print("  [2] Hybrid - Combines PerRes assignment with weighted blending")

        mode_choice = input("Enter mode [0]: ").strip()
        if not mode_choice:
            mode_choice = "0"

        try:
            mode = int(mode_choice)
            if mode not in [0, 1, 2]:
                mode = 0
        except ValueError:
            mode = 0

        if mode == 0:  # Legacy
            weights = prompt_weights(model_names, [0.7, 0.3] + [0.0] * (len(model_names) - 2))
            backbone_idx = pick_backbone(model_names, weights)
            block_weights = prompt_block_merge(model_names)
            crossattn_boosts = prompt_crossattn_boost(model_names)
        elif mode == 1:  # PerRes
            assignments, attn2_locks = prompt_perres_assignments(model_names)
            backbone_idx = list(assignments.values())[0]
        else:  # Hybrid
            hybrid_config, attn2_locks = prompt_hybrid_config(model_names)
            backbone_idx = 0

        mode_name = ["legacy", "perres", "hybrid"][mode]
        only_unet, component_policy = prompt_component_scope(mode_name)

        lora_files = list_safetensors(loras_dir)
        if lora_files:
            lora_selections = prompt_loras(lora_files)

        output_base_name = prompt_output_name(default_output_name) or default_output_name
        execution_options = prompt_execution_options()

    if not loaded_runtime:
        mode_name = ["legacy", "perres", "hybrid"][mode]

    validation = validate_merge_request(
        mode=mode_name,
        model_paths=model_paths,
        backbone=backbone_idx,
        weights=weights if mode == 0 else None,
        assignments=assignments if mode == 1 else None,
        hybrid_config=hybrid_config if mode == 2 else None,
        attn2_locks=attn2_locks,
        block_multipliers=block_weights if mode == 0 else None,
        crossattn_boosts=crossattn_boosts if mode == 0 else None,
        loras=lora_selections,
        loras_dir=loras_dir,
        only_unet=only_unet,
        component_policy=component_policy,
        block_mapping=str(loaded_runtime.get("block_mapping", "sdxl")) if loaded_runtime else "sdxl",
    )

    if not validation.valid:
        print("\nInvalid configuration:")
        for issue in validation.errors:
            print(f"  ERROR: {issue.field}: {issue.message}")
        for issue in validation.warnings:
            print(f"  WARNING: {issue.field}: {issue.message}")
        return 1

    for issue in validation.warnings:
        print(f"  WARNING: {issue.field}: {issue.message}")

    if validation.preflight:
        print("\n" + format_preflight_plan(validation.preflight))
        export_path = input("\nExport preflight report (.txt/.json) [Enter to skip]: ").strip()
        if export_path:
            saved_preflight = export_preflight_plan(validation.preflight, Path(export_path))
            print(f"Preflight exported to: {saved_preflight}")
        preset_export = input(
            f"Save this configuration as reusable preset YAML in {context.presets_dir} [Enter to skip]: "
        ).strip()
        if preset_export:
            preset_path = Path(preset_export)
            if not preset_path.is_absolute():
                preset_path = context.presets_dir / preset_path
            if preset_path.suffix.lower() not in {".yaml", ".yml"}:
                preset_path = preset_path.with_suffix(".yaml")
            lora_yaml = [
                {"file": item["file"], "scale": item["scale"]}
                for item in validation.normalized["loras"]
            ] or None
            save_single_job_preset(
                preset_path,
                mode=mode_name,
                model_names=validation.normalized["model_names"],
                backbone_idx=validation.normalized["backbone_idx"],
                output_name=output_base_name,
                execution=execution_options,
                block_mapping=str(validation.normalized.get("block_mapping", "sdxl")),
                job_name=f"Preset_{mode_name}",
                description="Saved from interactive CLI",
                weights=validation.normalized.get("weights"),
                assignments=validation.normalized.get("assignments"),
                hybrid_config=validation.normalized.get("hybrid_config"),
                attn2_locks=validation.normalized.get("attn2_locks"),
                block_multipliers=validation.normalized.get("block_multipliers"),
                crossattn_boosts=validation.normalized.get("crossattn_boosts"),
                loras=lora_yaml,
                only_unet=validation.normalized.get("only_unet"),
                component_policy=validation.normalized.get("component_policy"),
            )
            print(f"Preset saved to: {preset_path}")
        proceed = input(
            f"Proceed with merge using output '{output_base_name}' and execution '{execution_options['mode']}'? [Y/n]: "
        ).strip().lower()
        if proceed in {"n", "no"}:
            print("Merge cancelled before execution.")
            return 0

    normalized = validation.normalized
    job = MergeJobConfig(
        mode=mode_name,
        model_paths=normalized["model_paths"],
        model_names=normalized["model_names"],
        backbone_idx=normalized["backbone_idx"],
        block_mapping=str(normalized.get("block_mapping", "sdxl")),
        output_base_name=output_base_name,
        weights=normalized.get("weights"),
        assignments=normalized.get("assignments"),
        hybrid_config=normalized.get("hybrid_config"),
        attn2_locks=normalized.get("attn2_locks"),
        block_multipliers=normalized.get("block_multipliers"),
        crossattn_boosts=normalized.get("crossattn_boosts"),
        loras=normalized.get("loras"),
        only_unet=bool(normalized.get("only_unet")),
        component_policy=normalized.get("component_policy"),
        execution=execution_options,
        job_name=f"Interactive_{mode_name}",
        job_description="Interactive CLI run",
    )
    result = execute_merge_job(output_dir, metadata_dir, job)

    for report in result.lora_reports:
        applied_by_component = report.get("applied_by_component", {})
        print(
            f"LoRA {report.get('lora_file')}: applied {report.get('applied_pairs')}, "
            f"skipped {report.get('skipped_pairs')}, by component={applied_by_component}"
        )

    print(f"\nSaved: {result.output_path.name}")
    print(f"Version: {result.version}")
    print(f"Metadata folder: {result.metadata_folder.name}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
