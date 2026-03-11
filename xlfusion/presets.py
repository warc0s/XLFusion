"""Preset export/import and metadata recovery helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .batch_schema import BatchJob, BatchValidator, load_batch_config
from .config import AppContext, YAML_AVAILABLE, generate_batch_config_yaml, resolve_app_context
from .execution import execution_options_to_dict


@dataclass
class RecoveryInspection:
    metadata_folder: Path
    batch_config_path: Path
    job: BatchJob
    missing_models: list[str]
    missing_loras: list[str]
    warnings: list[str]


def save_single_job_preset(
    destination: Path,
    *,
    mode: str,
    model_names: list[str],
    backbone_idx: int,
    output_name: Optional[str] = None,
    execution: Optional[Dict[str, Any]] = None,
    job_name: str = "SavedPreset",
    description: str = "Saved preset from XLFusion",
    weights: Optional[list[float]] = None,
    assignments: Optional[Dict[str, Any]] = None,
    hybrid_config: Optional[Dict[str, Any]] = None,
    attn2_locks: Optional[Dict[str, Any]] = None,
    block_multipliers: Optional[list[Dict[str, Any]]] = None,
    crossattn_boosts: Optional[list[Dict[str, Any]]] = None,
    loras: Optional[list[Dict[str, Any]]] = None,
    only_unet: Optional[bool] = None,
    component_policy: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a reusable single-job batch preset."""
    if not YAML_AVAILABLE:
        raise RuntimeError("PyYAML is required to save preset files.")

    destination = Path(destination)
    preset_yaml = generate_batch_config_yaml(
        mode=mode,
        model_names=model_names,
        backbone_idx=backbone_idx,
        version=1,
        job_name=job_name,
        job_description=description,
        output_name=output_name,
        execution=execution_options_to_dict(execution),
        weights=weights,
        assignments=assignments,
        hybrid_config=hybrid_config,
        attn2_locks=attn2_locks,
        block_multipliers=block_multipliers,
        crossattn_boosts=crossattn_boosts,
        loras=loras,
        only_unet=only_unet,
        component_policy=component_policy,
    )
    destination.write_text(preset_yaml, encoding="utf-8")
    return destination


def load_single_job_preset(source: Path) -> BatchJob:
    """Load a single-job preset or metadata batch config."""
    config = load_batch_config(Path(source))
    if len(config.batch_jobs) != 1:
        raise ValueError("Preset loading requires exactly one batch job.")
    return config.batch_jobs[0]


def batch_job_to_runtime_state(job: BatchJob) -> Dict[str, Any]:
    """Convert a batch job into GUI/CLI runtime state."""
    execution = execution_options_to_dict(job.execution)
    runtime: Dict[str, Any] = {
        "mode": job.mode,
        "models": list(job.models),
        "output_name": job.output_name,
        "execution": execution,
    }
    if job.mode == "legacy":
        runtime["config"] = {
            "weights": job.weights or [],
            "backbone_idx": int(job.backbone),
            "block_multipliers": job.block_multipliers,
            "crossattn_boosts": job.crossattn_boosts,
            "only_unet": job.only_unet,
            "component_policy": job.component_policy,
            "loras": [
                {"file": item["file"], "scale": item.get("scale", 1.0)}
                for item in (job.loras or [])
            ],
        }
    elif job.mode == "perres":
        runtime["config"] = {
            "assignments": job.assignments or {},
            "attn2_locks": job.attn2_locks,
            "only_unet": job.only_unet,
            "component_policy": job.component_policy,
            "loras": [
                {"file": item["file"], "scale": item.get("scale", 1.0)}
                for item in (job.loras or [])
            ],
        }
    else:
        runtime["config"] = {
            "hybrid_config": job.hybrid_config or {},
            "attn2_locks": job.attn2_locks,
            "only_unet": job.only_unet,
            "component_policy": job.component_policy,
            "loras": [
                {"file": item["file"], "scale": item.get("scale", 1.0)}
                for item in (job.loras or [])
            ],
        }
    return runtime


def resolve_metadata_folder(source: Path, metadata_root: Optional[Path] = None) -> Path:
    """Resolve a metadata folder reference from a folder, file or meta_* name."""
    source = Path(source)

    if source.is_dir() and (source / "batch_config.yaml").exists():
        return source
    if source.is_file() and source.name == "batch_config.yaml":
        return source.parent

    if metadata_root is not None:
        candidate = Path(metadata_root) / source
        if candidate.is_dir() and (candidate / "batch_config.yaml").exists():
            return candidate

    raise FileNotFoundError(f"Could not resolve metadata folder from: {source}")


def inspect_recovery_source(source: Path, context: AppContext | Path) -> RecoveryInspection:
    """Load and inspect a metadata folder to detect missing inputs."""
    app_context = context if isinstance(context, AppContext) else resolve_app_context(context)
    metadata_folder = resolve_metadata_folder(Path(source), app_context.metadata_dir)
    batch_config_path = metadata_folder / "batch_config.yaml"
    job = load_single_job_preset(batch_config_path)

    missing_models = [
        model_name
        for model_name in job.models
        if not (app_context.models_dir / model_name).exists()
    ]
    missing_loras = [
        item["file"]
        for item in (job.loras or [])
        if not (app_context.loras_dir / item["file"]).exists()
    ]

    validator = BatchValidator(app_context)
    config = load_batch_config(batch_config_path)
    validator.validate_config(config)

    warnings = list(validator.warnings)
    if missing_models:
        warnings.append(f"Missing models: {', '.join(missing_models)}")
    if missing_loras:
        warnings.append(f"Missing LoRAs: {', '.join(missing_loras)}")

    return RecoveryInspection(
        metadata_folder=metadata_folder,
        batch_config_path=batch_config_path,
        job=job,
        missing_models=missing_models,
        missing_loras=missing_loras,
        warnings=warnings,
    )
