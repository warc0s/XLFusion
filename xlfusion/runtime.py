"""Shared merge runtime for CLI, GUI and batch.

This module centralizes the execution path:
- run merge engine (legacy/perres/hybrid)
- optionally bake LoRAs
- persist artifacts + reproducible metadata

Front-ends should rely on this module to avoid drift.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .execution import execution_options_to_dict
from .lora import apply_single_lora_with_report
from .merge import merge_hybrid, merge_perres, stream_weighted_merge_from_paths
from .types import MergeJobConfig, MergeJobResult, ProgressCallback
from .workflow import save_merge_results


def _build_yaml_kwargs(job: MergeJobConfig) -> Dict[str, Any]:
    yaml_kwargs: Dict[str, Any] = {
        "only_unet": bool(job.only_unet),
        "component_policy": job.component_policy,
        "block_mapping": job.block_mapping,
    }

    if job.mode == "legacy":
        if job.weights is not None:
            yaml_kwargs["weights"] = job.weights
        if job.block_multipliers:
            yaml_kwargs["block_multipliers"] = job.block_multipliers
        if job.crossattn_boosts:
            yaml_kwargs["crossattn_boosts"] = job.crossattn_boosts
    elif job.mode == "perres":
        if job.assignments is not None:
            yaml_kwargs["assignments"] = job.assignments
        if job.attn2_locks:
            yaml_kwargs["attn2_locks"] = job.attn2_locks
    elif job.mode == "hybrid":
        if job.hybrid_config is not None:
            yaml_kwargs["hybrid_config"] = job.hybrid_config
        if job.attn2_locks:
            yaml_kwargs["attn2_locks"] = job.attn2_locks
    else:
        raise ValueError(f"Unsupported mode: {job.mode}")

    if job.loras:
        yaml_kwargs["loras"] = [{"file": item["file"], "scale": item["scale"]} for item in job.loras]
    return yaml_kwargs


def _resolve_lora_paths(job: MergeJobConfig) -> Optional[List[Path]]:
    if not job.loras:
        return None
    paths: List[Path] = []
    for item in job.loras:
        path = item.get("path")
        if isinstance(path, Path):
            paths.append(path)
            continue
        file_name = item.get("file")
        if isinstance(file_name, str):
            paths.append(Path(file_name))
    return paths or None


def _execute_merge_engine(
    job: MergeJobConfig,
    *,
    progress_cb: Optional[ProgressCallback],
    cancel_event: Optional[threading.Event],
) -> Dict[str, Any]:
    if job.mode == "legacy":
        if not job.weights:
            raise ValueError("Legacy mode requires weights")
        merged, _stats = stream_weighted_merge_from_paths(
            job.model_paths,
            job.weights,
            job.backbone_idx,
            only_unet=bool(job.only_unet),
            component_policy=job.component_policy,
            block_multipliers=job.block_multipliers,
            crossattn_boosts=job.crossattn_boosts,
            execution=job.execution,
            progress_cb=progress_cb,
            cancel_event=cancel_event,
            block_mapping=job.block_mapping,
        )
        return merged

    if job.mode == "perres":
        if job.assignments is None:
            raise ValueError("PerRes mode requires assignments")
        return merge_perres(
            job.model_paths,
            job.assignments,
            job.backbone_idx,
            job.attn2_locks,
            only_unet=bool(job.only_unet),
            component_policy=job.component_policy,
            execution=job.execution,
            progress_cb=progress_cb,
            cancel_event=cancel_event,
            block_mapping=job.block_mapping,
        )

    if job.mode == "hybrid":
        if job.hybrid_config is None:
            raise ValueError("Hybrid mode requires hybrid_config")
        return merge_hybrid(
            job.model_paths,
            job.hybrid_config,
            job.backbone_idx,
            job.attn2_locks,
            only_unet=bool(job.only_unet),
            component_policy=job.component_policy,
            execution=job.execution,
            progress_cb=progress_cb,
            cancel_event=cancel_event,
            block_mapping=job.block_mapping,
        )

    raise ValueError(f"Unsupported mode: {job.mode}")


def _apply_loras(merged_state: Dict[str, Any], job: MergeJobConfig) -> List[Dict[str, Any]]:
    if not job.loras:
        return []

    reports: List[Dict[str, Any]] = []
    for lora_spec in job.loras:
        path = lora_spec.get("path")
        if not isinstance(path, Path):
            file_name = lora_spec.get("file")
            if not isinstance(file_name, str) or not file_name.strip():
                raise ValueError("LoRA spec is missing a valid path or file name")
            path = Path(file_name)
        scale = float(lora_spec.get("scale", 1.0))
        report = apply_single_lora_with_report(merged_state, path, scale)
        reports.append(report.to_dict())
    return reports


def execute_merge_job(
    output_dir: Path,
    metadata_dir: Path,
    job: MergeJobConfig,
    *,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> MergeJobResult:
    """Execute a normalized merge job and persist artifacts."""
    if len(job.model_paths) < 2:
        raise ValueError("At least two model paths are required")

    merged_state = _execute_merge_engine(job, progress_cb=progress_cb, cancel_event=cancel_event)
    lora_reports = _apply_loras(merged_state, job)
    yaml_kwargs = _build_yaml_kwargs(job)
    lora_paths = _resolve_lora_paths(job)

    output_path, metadata_folder, version = save_merge_results(
        output_dir,
        metadata_dir,
        merged_state,
        job.model_names,
        job.mode,
        job.backbone_idx,
        yaml_kwargs,
        model_paths=job.model_paths,
        lora_paths=lora_paths,
        output_base_name=job.output_base_name,
        execution=execution_options_to_dict(job.execution),
        job_name=job.job_name,
        job_description=job.job_description,
        audit_sections={"lora_application": lora_reports} if lora_reports else None,
    )

    return MergeJobResult(
        output_path=output_path,
        metadata_folder=metadata_folder,
        version=int(version),
        keys_processed=len(merged_state),
        lora_reports=lora_reports,
    )
