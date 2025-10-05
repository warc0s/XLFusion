#!/usr/bin/env python3
"""
XLFusion V2.0 - Advanced SDXL checkpoint merger

Interactive and graphical SDXL checkpoint merger with three fusion modes:
- Legacy: Classic weighted merge with down/mid/up blocks and optional LoRA baking
- PerRes: Resolution-based block control
- Hybrid: Combines PerRes assignment with weighted blending

Batch mode (V1.2):
- Process multiple fusions from YAML configuration
- Templates for common fusion patterns
- Validation and progress tracking
- Automated workflow processing

Analysis capabilities (V1.3):
- Model difference analysis with block-level statistics
- Compatibility checking between multiple models
- Fusion result prediction
- Intelligent configuration recommendations

Graphical interface (V2.0):
- Guided wizard with model library and metadata
- Visual preview of block assignments
- Integrated execution workflow with progress logging

Expected folder structure:
  ./models     -> base checkpoints .safetensors
  ./loras      -> LoRA files .safetensors (legacy mode only)
  ./output     -> merged outputs .safetensors
  ./metadata   -> audit logs meta_*.txt

Requirements:
  pip install torch safetensors
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List

# Import all utilities from Utils
from Utils.config import ensure_dirs, list_safetensors
from Utils.merge import merge_hybrid, merge_perres, stream_weighted_merge_from_paths
from Utils.lora import apply_single_lora
from Utils.cli import (
    prompt_select, prompt_weights, pick_backbone, 
    prompt_hybrid_config, prompt_perres_assignments,
    prompt_block_merge, prompt_crossattn_boost, prompt_loras
)
from Utils.workflow import save_merge_results

# Optional analyzer
try:
    from Utils.analyzer import (
        ModelDiffAnalyzer, CompatibilityAnalyzer,
        FusionPredictor, RecommendationEngine,
        generate_analysis_report, export_analysis_json
    )
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False

# Optional batch processor
try:
    from Utils.batch_processor import BatchProcessor
    BATCH_PROCESSOR_AVAILABLE = True
except ImportError:
    BATCH_PROCESSOR_AVAILABLE = False


def analyze_mode(args, models_dir: Path, output_dir: Path) -> int:
    """Execute analysis mode operations"""
    if not ANALYZER_AVAILABLE:
        print("Error: Analyzer module not available.")
        print("Make sure Utils/analyzer.py is available")
        return 1
    
    model_files = list_safetensors(models_dir)
    if not model_files:
        print("No models found in ./models directory")
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
        results['recommendations'] = engine.generate_recommendations(
            selected_models,
            goal
        )
        
        predictor = FusionPredictor()
        results['prediction'] = predictor.predict_fusion_characteristics(
            selected_models,
            {'mode': 'balanced'}
        )
    
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
    
    # Analysis mode arguments (V1.3)
    parser.add_argument("--analyze", action="store_true", help="Run in analysis mode")
    parser.add_argument("--compare", nargs=2, metavar=('MODEL1', 'MODEL2'), help="Compare two models (indices)")
    parser.add_argument("--recommend", choices=['style_transfer', 'detail_enhance', 'balanced'], help="Generate recommendations for fusion goal")
    parser.add_argument("--export-analysis", type=Path, metavar='PATH', help="Export analysis to JSON file")
    parser.add_argument("--gui", action="store_true", help="Launch graphical interface (V2.0)")

    args = parser.parse_args()

    # Setup directories
    root = Path(__file__).resolve().parent
    models_dir, loras_dir, output_dir, metadata_dir = ensure_dirs(root)

    if args.gui:
        if any([args.batch, args.analyze, args.compare, args.recommend]):
            print("The --gui option cannot be combined with batch or analysis modes.")
            return 1
        from gui_app import launch_gui
        launch_gui(root)
        return 0

    # Check for analysis mode
    if args.analyze or args.compare or args.recommend:
        return analyze_mode(args, models_dir, output_dir)

    # Check for batch mode
    if args.batch:
        if not BATCH_PROCESSOR_AVAILABLE:
            print("Error: Batch processor not available.")
            print("Make sure Utils/batch_processor.py is available")
            return 1
        
        # Load batch configuration
        from Utils.batch_processor import load_batch_config, BatchValidator, BatchProcessor
        config = load_batch_config(args.batch)
        validator = BatchValidator(root)
        if not validator.validate_config(config):
            print("Batch configuration validation failed:")
            for error in validator.errors:
                print(f"  ERROR: {error}")
            for warning in validator.warnings:
                print(f"  WARNING: {warning}")
            return 1
        
        processor = BatchProcessor(config, root, args.validate_only)
        return processor.process_batch()

    # Interactive mode
    print("\n" + "="*60)
    print(" XLFusion V2.0 - Advanced SDXL checkpoint merger")
    print("="*60)
    print(f"\nDetected structure:")
    print(f"  models:   {models_dir}")
    print(f"  loras:    {loras_dir}")
    print(f"  output:   {output_dir}")
    print(f"  metadata: {metadata_dir}")

    # Select models
    model_files = list_safetensors(models_dir)
    if not model_files:
        print("\nNo .safetensors files found in ./models directory")
        return 1

    model_idx = prompt_select(model_files, "Select models to merge (comma-separated indices):", [0, 1])
    if len(model_idx) < 2:
        print("\nSelect at least 2 models")
        return 1

    model_paths = [model_files[i] for i in model_idx]
    model_names = [p.name for p in model_paths]

    # Select fusion mode
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

    weights: List[float] = []
    block_weights = None
    crossattn_boosts = None
    assignments = None
    attn2_locks = None
    hybrid_config = None
    lora_selections = []

    # Process based on mode
    if mode == 0:  # Legacy
        weights = prompt_weights(model_names, [0.7, 0.3] + [0.0] * (len(model_names) - 2))
        backbone_idx = pick_backbone(model_names, weights)
        
        # Optional block-specific weights
        block_weights = prompt_block_merge(model_names)
        
        # Optional cross-attention boost
        crossattn_boosts = prompt_crossattn_boost(model_names)
        
        # Perform merge (with optional enhancements)
        merged, _legacy_stats = stream_weighted_merge_from_paths(
            model_paths,
            weights,
            backbone_idx,
            only_unet=True,
            block_multipliers=block_weights,
            crossattn_boosts=crossattn_boosts,
        )
        
    elif mode == 1:  # PerRes
        assignments, attn2_locks = prompt_perres_assignments(model_names)
        backbone_idx = list(assignments.values())[0]  # Use first assignment as backbone
        
        # Perform merge
        merged = merge_perres(model_paths, assignments, backbone_idx, attn2_locks)
        
    else:  # Hybrid
        hybrid_config, attn2_locks = prompt_hybrid_config(model_names)
        backbone_idx = 0  # Default to first model
        
        # Perform merge
        merged = merge_hybrid(model_paths, hybrid_config, backbone_idx, attn2_locks)

    # Apply LoRAs if available (todos los modos)
    lora_files = list_safetensors(loras_dir)
    if lora_files:
        lora_selections = prompt_loras(lora_files)
        for lora_path, scale in lora_selections:
            applied, skipped = apply_single_lora(merged, lora_path, scale)
            print(f"LoRA {lora_path.name}: applied {applied}, skipped {skipped}")

    mode_name = ["legacy", "perres", "hybrid"][mode]
    yaml_kwargs = {}

    if mode == 0:
        yaml_kwargs["weights"] = weights
        if block_weights:
            yaml_kwargs["block_multipliers"] = block_weights
        if crossattn_boosts:
            yaml_kwargs["crossattn_boosts"] = crossattn_boosts
    elif mode == 1:
        yaml_kwargs["assignments"] = assignments
        if attn2_locks:
            yaml_kwargs["attn2_locks"] = attn2_locks
    else:
        yaml_kwargs["hybrid_config"] = hybrid_config
        if attn2_locks:
            yaml_kwargs["attn2_locks"] = attn2_locks

    if lora_selections:
        yaml_kwargs["loras"] = [
            {"file": path.name, "scale": scale}
            for path, scale in lora_selections
        ]

    output_path, metadata_folder, version = save_merge_results(
        output_dir,
        metadata_dir,
        merged,
        model_names,
        mode_name,
        backbone_idx,
        yaml_kwargs,
        model_paths=model_paths,
        lora_paths=[p for p, _s in lora_selections] if lora_selections else None,
    )

    print(f"\nSaved: {output_path.name}")
    print(f"Version: {version}")
    print(f"Metadata folder: {metadata_folder.name}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
