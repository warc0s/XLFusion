#!/usr/bin/env python3
"""
XLFusion V1.3 - Batch Processor
===============================

Processes multiple model fusions from YAML configuration files.
Supports all three fusion modes: Legacy, PerRes, and Hybrid.

Usage:
    python XLFusion.py --batch config.yaml
    python XLFusion.py --batch config.yaml --validate-only
    python XLFusion.py --batch config.yaml --template style_transfer
"""

from __future__ import annotations
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime

import yaml

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Import existing XLFusion functions
from .config import load_config, ensure_dirs, list_safetensors, next_version_path
from .merge import merge_hybrid, merge_perres, stream_weighted_merge_from_paths, validate_hybrid_config
from .lora import apply_single_lora
from .memory import save_state, estimate_memory_requirement, check_memory_availability
from .blocks import UNET_PREFIX


@dataclass
class BatchJob:
    """Represents a single batch job configuration"""
    name: str
    mode: str
    description: str = ""
    models: List[str] = field(default_factory=list)
    backbone: Union[int, str] = 0
    output_name: Optional[str] = None

    # Mode-specific configurations
    weights: Optional[List[float]] = None
    assignments: Optional[Dict[str, int]] = None
    hybrid_config: Optional[Dict[str, Dict[str, float]]] = None
    attn2_locks: Optional[Dict[str, int]] = None
    block_multipliers: Optional[List[Dict[str, float]]] = None
    crossattn_boosts: Optional[List[Dict[str, float]]] = None
    loras: Optional[List[Dict[str, Any]]] = None

    # Template support
    template: Optional[str] = None
    template_params: Optional[Dict[str, Any]] = None

    # Internal state
    model_paths: List[Path] = field(default_factory=list)
    backbone_idx: int = 0
    output_path: Optional[Path] = None
    version: Optional[int] = None

    # Results
    success: bool = False
    error_message: str = ""
    processing_time: float = 0.0
    keys_processed: int = 0


@dataclass
class BatchConfig:
    """Complete batch configuration"""
    version: str
    global_settings: Dict[str, Any]
    batch_jobs: List[BatchJob]
    templates: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class BatchValidator:
    """Validates batch configuration before execution"""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.models_dir, self.loras_dir, self.output_dir, self.metadata_dir = ensure_dirs(root_dir)
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_config(self, config: BatchConfig) -> bool:
        """Validate entire batch configuration"""
        self.errors = []
        self.warnings = []

        # Validate version
        if config.version not in ["1.2", "1.3"]:
            self.errors.append(f"Unsupported config version: {config.version} (expected 1.2 or 1.3)")

        # Validate global settings
        self._validate_global_settings(config.global_settings)

        # Validate each job
        for i, job in enumerate(config.batch_jobs):
            self._validate_job(job, i)

        # Resolve model paths for memory validation
        for job in config.batch_jobs:
            job.model_paths = [self.models_dir / name for name in job.models]

        # Validate memory requirements
        self._validate_memory_requirements(config.batch_jobs)

        return len(self.errors) == 0

    def _validate_global_settings(self, settings: Dict[str, Any]) -> None:
        """Validate global settings"""
        required_keys = ["output_base", "continue_on_error", "max_parallel"]
        for key in required_keys:
            if key not in settings:
                self.errors.append(f"Missing global setting: {key}")

        if "max_parallel" in settings and settings["max_parallel"] != 1:
            self.warnings.append("Parallel processing not yet implemented, max_parallel will be ignored")

    def _validate_job(self, job: BatchJob, index: int) -> None:
        """Validate individual job"""
        prefix = f"Job {index} ({job.name}):"

        # Validate mode
        valid_modes = ["legacy", "perres", "hybrid"]
        if job.mode not in valid_modes:
            self.errors.append(f"{prefix} Invalid mode '{job.mode}'. Must be one of: {valid_modes}")

        # Validate models exist
        for model_name in job.models:
            model_path = self.models_dir / model_name
            if not model_path.exists():
                self.errors.append(f"{prefix} Model file not found: {model_path}")

        # Mode-specific validation
        if job.mode == "legacy":
            self._validate_legacy_job(job, prefix)
        elif job.mode == "perres":
            self._validate_perres_job(job, prefix)
        elif job.mode == "hybrid":
            self._validate_hybrid_job(job, prefix)

    def _validate_legacy_job(self, job: BatchJob, prefix: str) -> None:
        """Validate legacy mode job"""
        if not job.weights:
            self.errors.append(f"{prefix} Legacy mode requires 'weights' array")
        elif len(job.weights) != len(job.models):
            self.errors.append(f"{prefix} Weights array length ({len(job.weights)}) must match models array ({len(job.models)})")

        if job.weights:
            total_weight = sum(job.weights)
            if abs(total_weight - 1.0) > 0.01:
                self.warnings.append(f"{prefix} Weights sum to {total_weight:.3f} (expected ~1.0)")

        # Validate LoRAs if specified
        if job.loras:
            for lora in job.loras:
                if "file" not in lora:
                    self.errors.append(f"{prefix} LoRA entry missing 'file' field")
                else:
                    lora_path = self.loras_dir / lora["file"]
                    if not lora_path.exists():
                        self.errors.append(f"{prefix} LoRA file not found: {lora_path}")

    def _validate_perres_job(self, job: BatchJob, prefix: str) -> None:
        """Validate PerRes mode job"""
        if not job.assignments:
            self.errors.append(f"{prefix} PerRes mode requires 'assignments' dict")

        required_blocks = ["down_0_1", "down_2_3", "mid", "up_0_1", "up_2_3"]
        if job.assignments:
            for block in required_blocks:
                if block not in job.assignments:
                    self.errors.append(f"{prefix} Missing assignment for block: {block}")
                elif job.assignments[block] >= len(job.models):
                    self.errors.append(f"{prefix} Assignment for {block} ({job.assignments[block]}) exceeds model count ({len(job.models)})")
        # Validate LoRAs if specified
        if job.loras:
            for lora in job.loras:
                if "file" not in lora:
                    self.errors.append(f"{prefix} LoRA entry missing 'file' field")
                else:
                    lora_path = self.loras_dir / lora["file"]
                    if not lora_path.exists():
                        self.errors.append(f"{prefix} LoRA file not found: {lora_path}")

    def _validate_hybrid_job(self, job: BatchJob, prefix: str) -> None:
        """Validate hybrid mode job"""
        if not job.hybrid_config:
            self.errors.append(f"{prefix} Hybrid mode requires 'hybrid_config' dict")

        if job.hybrid_config:
            warnings = validate_hybrid_config(job.hybrid_config, len(job.models))
            for warning in warnings:
                self.warnings.append(f"{prefix} {warning}")
        # Validate LoRAs if specified
        if job.loras:
            for lora in job.loras:
                if "file" not in lora:
                    self.errors.append(f"{prefix} LoRA entry missing 'file' field")
                else:
                    lora_path = self.loras_dir / lora["file"]
                    if not lora_path.exists():
                        self.errors.append(f"{prefix} LoRA file not found: {lora_path}")

    def _validate_memory_requirements(self, jobs: List[BatchJob]) -> None:
        """Validate memory requirements for all jobs"""
        for job in jobs:
            if not job.model_paths:
                continue

            needed_indices = set()
            if job.mode == "legacy":
                needed_indices = set(range(len(job.model_paths)))
            elif job.mode == "perres" and job.assignments:
                needed_indices = set(job.assignments.values())
            elif job.mode == "hybrid" and job.hybrid_config:
                for block_config in job.hybrid_config.values():
                    needed_indices.update(block_config.keys())

            if job.attn2_locks:
                needed_indices.update(job.attn2_locks.values())

            needed_indices.add(job.backbone_idx)

            required_memory = estimate_memory_requirement(job.model_paths, needed_indices)
            if not check_memory_availability(required_memory):
                self.warnings.append(f"Job {job.name}: Estimated memory requirement {required_memory:.1f}GB may exceed available memory")


class BatchProcessor:
    """Processes batch jobs sequentially"""

    def __init__(self, config: BatchConfig, root_dir: Path, validate_only: bool = False):
        self.config = config
        self.root_dir = root_dir
        self.validate_only = validate_only
        self.models_dir, self.loras_dir, self.output_dir, self.metadata_dir = ensure_dirs(root_dir)

        # Setup logging
        self._setup_logging()

        # Progress tracking
        self.total_jobs = len(config.batch_jobs)
        self.completed_jobs = 0
        self.failed_jobs = 0

    def _setup_logging(self) -> None:
        """Setup structured logging"""
        log_level = getattr(logging, self.config.global_settings.get("log_level", "INFO").upper())
        handlers = [logging.StreamHandler(sys.stdout)]
        if self.config.global_settings.get("log_to_file", False):
            handlers.insert(0, logging.FileHandler(self.root_dir / "batch_log.txt"))

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers,
        )
        self.logger = logging.getLogger("XLFusion.Batch")

    def process_batch(self) -> Dict[str, Any]:
        """Process all batch jobs"""
        start_time = time.time()
        self.logger.info(f"Starting batch processing of {self.total_jobs} jobs")

        results = {
            "total_jobs": self.total_jobs,
            "successful_jobs": 0,
            "failed_jobs": 0,
            "total_time": 0.0,
            "jobs": []
        }

        # Create progress bar if available
        progress_bar = None
        if TQDM_AVAILABLE and not self.validate_only:
            progress_bar = tqdm(total=self.total_jobs, desc="Batch Progress", unit="job")

        for job in self.config.batch_jobs:
            job_start = time.time()

            try:
                self.logger.info(f"Processing job: {job.name}")
                if not self.validate_only:
                    self._process_job(job)
                else:
                    job.success = True  # Validation-only mode

                job.processing_time = time.time() - job_start
                self.completed_jobs += 1
                results["successful_jobs"] += 1

                self.logger.info(f"Job {job.name} completed successfully in {job.processing_time:.2f}s")

            except Exception as e:
                job.success = False
                job.error_message = str(e)
                job.processing_time = time.time() - job_start
                self.failed_jobs += 1
                results["failed_jobs"] += 1

                self.logger.error(f"Job {job.name} failed: {e}")

                if not self.config.global_settings.get("continue_on_error", True):
                    self.logger.error("Stopping batch due to error (continue_on_error=false)")
                    break

            # Update progress
            if progress_bar:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "completed": self.completed_jobs,
                    "failed": self.failed_jobs
                })

            # Record job result
            results["jobs"].append({
                "name": job.name,
                "success": job.success,
                "processing_time": job.processing_time,
                "keys_processed": job.keys_processed,
                "error": job.error_message if not job.success else None
            })

        if progress_bar:
            progress_bar.close()

        results["total_time"] = time.time() - start_time
        # Summary report generation disabled
        # self._generate_summary_report(results)

        return results

    def _process_job(self, job: BatchJob) -> None:
        """Process a single job"""
        # Resolve model paths
        job.model_paths = [self.models_dir / name for name in job.models]

        # Resolve backbone index
        if isinstance(job.backbone, str):
            try:
                job.backbone_idx = job.models.index(job.backbone)
            except ValueError:
                raise ValueError(f"Backbone model '{job.backbone}' not found in models list")
        else:
            job.backbone_idx = job.backbone

        # Process based on mode
        merged_state = None

        if job.mode == "legacy":
            if job.weights is None:
                raise ValueError("Legacy mode requires weights")
            merged_state, _ = stream_weighted_merge_from_paths(
                job.model_paths, job.weights, job.backbone_idx,
                only_unet=True,
                block_multipliers=job.block_multipliers,
                crossattn_boosts=job.crossattn_boosts
            )

            # Apply LoRAs if specified
            if job.loras:
                for lora_spec in job.loras:
                    lora_path = self.loras_dir / lora_spec["file"]
                    scale = lora_spec.get("scale", 0.3)
                    applied, skipped = apply_single_lora(merged_state, lora_path, scale)
                    self.logger.info(f"Applied LoRA {lora_spec['file']}: {applied} applied, {skipped} skipped")

        elif job.mode == "perres":
            if job.assignments is None:
                raise ValueError("PerRes mode requires assignments")
            merged_state = merge_perres(
                job.model_paths, job.assignments, job.backbone_idx, job.attn2_locks
            )
            # Apply LoRAs if specified
            if job.loras:
                for lora_spec in job.loras:
                    lora_path = self.loras_dir / lora_spec["file"]
                    scale = lora_spec.get("scale", 0.3)
                    applied, skipped = apply_single_lora(merged_state, lora_path, scale)
                    self.logger.info(f"Applied LoRA {lora_spec['file']}: {applied} applied, {skipped} skipped")

        elif job.mode == "hybrid":
            if job.hybrid_config is None:
                raise ValueError("Hybrid mode requires hybrid_config")
            merged_state = merge_hybrid(
                job.model_paths, job.hybrid_config, job.backbone_idx, job.attn2_locks
            )
            # Apply LoRAs if specified
            if job.loras:
                for lora_spec in job.loras:
                    lora_path = self.loras_dir / lora_spec["file"]
                    scale = lora_spec.get("scale", 0.3)
                    applied, skipped = apply_single_lora(merged_state, lora_path, scale)
                    self.logger.info(f"Applied LoRA {lora_spec['file']}: {applied} applied, {skipped} skipped")

        if merged_state is None:
            raise ValueError(f"Unsupported mode: {job.mode}")

        # Determine output path
        output_dir = self.output_dir / self.config.global_settings.get("output_base", "batch_output")
        output_dir.mkdir(parents=True, exist_ok=True)

        if job.output_name:
            # Custom name provided
            job.output_path, job.version = next_version_path(output_dir)
            # Replace the auto-generated name with custom name
            if job.version is not None:
                custom_name = f"{job.output_name}_V{job.version}.safetensors"
                job.output_path = output_dir / custom_name
        else:
            # Auto-generated name
            job.output_path, job.version = next_version_path(output_dir)

        # Prepare metadata
        config = load_config()
        app_cfg = config["app"]

        # Verify output_path before using it
        if job.output_path is None:
            raise ValueError("Output path not determined")

        meta_embed = {
            "title": job.output_path.stem,
            "format": "sdxl-a1111-like",
            "merge_mode": job.mode,
            "batch_job": job.name,
            "backbone": job.models[job.backbone_idx],
            "models": json.dumps(job.models, ensure_ascii=False),
            "created": datetime.now().isoformat(timespec='seconds'),
            "tool": app_cfg.get("tool_name", "XLFusion"),
            "version": app_cfg.get("version", "1.3")
        }

        # Add mode-specific metadata
        if job.mode == "legacy":
            meta_embed.update({
                "weights": json.dumps(job.weights, ensure_ascii=False),
                "block_multipliers": json.dumps(job.block_multipliers, ensure_ascii=False) if job.block_multipliers else "",
                "crossattn_boosts": json.dumps(job.crossattn_boosts, ensure_ascii=False) if job.crossattn_boosts else "",
                "loras": json.dumps(job.loras, ensure_ascii=False) if job.loras else "",
            })
        elif job.mode == "perres":
            meta_embed.update({
                "assignments": json.dumps(job.assignments, ensure_ascii=False),
                "attn2_locks": json.dumps(job.attn2_locks, ensure_ascii=False) if job.attn2_locks else "",
                "loras": json.dumps(job.loras, ensure_ascii=False) if job.loras else "",
            })
        elif job.mode == "hybrid":
            meta_embed.update({
                "hybrid_config": json.dumps(job.hybrid_config, ensure_ascii=False),
                "attn2_locks": json.dumps(job.attn2_locks, ensure_ascii=False) if job.attn2_locks else "",
                "loras": json.dumps(job.loras, ensure_ascii=False) if job.loras else "",
            })

        # Save model
        if job.output_path is None:
            raise ValueError("Output path not determined")
        self.logger.info(f"Saving model to {job.output_path}")
        save_state(job.output_path, merged_state, meta_embed)

        # Create audit log
        self._create_audit_log(job, meta_embed)

        job.keys_processed = len(merged_state)
        job.success = True

    def _create_audit_log(self, job: BatchJob, meta_embed: Dict[str, str]) -> None:
        """Create detailed audit log for the job"""
        config = load_config()
        app_cfg = config["app"]

        output_name = job.output_path.name if job.output_path else "unknown"
        log_lines = [
            f"{app_cfg['tool_name']} V{app_cfg['version']} - Batch Mode - Job: {job.name}",
            f"Date: {datetime.now().isoformat(timespec='seconds')}",
            f"Output: {output_name}",
            "",
            f"Description: {job.description}",
            "",
            f"Mode: {job.mode}",
            "",
            "Base models:",
        ]

        for i, name in enumerate(job.models):
            marker = " <-- BACKBONE" if i == job.backbone_idx else ""
            log_lines.append(f"  [{i}] {name}{marker}")
        log_lines.append("")

        # Mode-specific details
        if job.mode == "legacy":
            log_lines.append("Weights:")
            if job.weights:
                for name, weight in zip(job.models, job.weights):
                    log_lines.append(f"  {name}: {weight:.6f}")
            if job.block_multipliers:
                log_lines.append("")
                log_lines.append("Block multipliers:")
                for name, mults in zip(job.models, job.block_multipliers):
                    log_lines.append(f"  {name}: down={mults.get('down',1.0):.3f} mid={mults.get('mid',1.0):.3f} up={mults.get('up',1.0):.3f}")
            if job.loras:
                log_lines.append("")
                log_lines.append("Baked LoRAs:")
                for lora in job.loras:
                    log_lines.append(f"  {lora['file']}  scale={lora.get('scale', 0.3)}")

        elif job.mode == "perres":
            log_lines.append("Resolution assignments:")
            if job.assignments:
                log_lines.append(f"  Down 0,1: {job.models[job.assignments['down_0_1']]}")
                log_lines.append(f"  Down 2,3: {job.models[job.assignments['down_2_3']]}")
                log_lines.append(f"  Mid:      {job.models[job.assignments['mid']]}")
                log_lines.append(f"  Up 0,1:   {job.models[job.assignments['up_0_1']]}")
                log_lines.append(f"  Up 2,3:   {job.models[job.assignments['up_2_3']]}")
            if job.attn2_locks:
                log_lines.append("")
                log_lines.append("Cross-attention locks:")
                log_lines.append(f"  Down: {job.models[job.attn2_locks['down']]}")
                log_lines.append(f"  Mid:  {job.models[job.attn2_locks['mid']]}")
                log_lines.append(f"  Up:   {job.models[job.attn2_locks['up']]}")
            if job.loras:
                log_lines.append("")
                log_lines.append("Baked LoRAs:")
                for lora in job.loras:
                    log_lines.append(f"  {lora['file']}  scale={lora.get('scale', 0.3)}")

        elif job.mode == "hybrid":
            log_lines.append("Hybrid block configuration:")
            if job.hybrid_config:
                for block_name, weights in job.hybrid_config.items():
                    log_lines.append(f"  {block_name}:")
                    for idx, weight in weights.items():
                        log_lines.append(f"    {job.models[idx]}: {weight:.3f}")
            if job.attn2_locks:
                log_lines.append("")
                log_lines.append("Cross-attention locks:")
                log_lines.append(f"  Down: {job.models[job.attn2_locks['down']]}")
                log_lines.append(f"  Mid:  {job.models[job.attn2_locks['mid']]}")
                log_lines.append(f"  Up:   {job.models[job.attn2_locks['up']]}")
            if job.loras:
                log_lines.append("")
                log_lines.append("Baked LoRAs:")
                for lora in job.loras:
                    log_lines.append(f"  {lora['file']}  scale={lora.get('scale', 0.3)}")

        log_lines.append("")
        log_lines.append(f"Total keys: {job.keys_processed}")
        # Count UNet keys from the actual merged state, not metadata
        if job.keys_processed > 0:
            # Estimate UNet keys (typically around 70-80% of total keys in SDXL models)
            unet_keys = int(job.keys_processed * 0.75)
        else:
            unet_keys = 0
        log_lines.append(f"UNet keys: {unet_keys}")

        # Create structured metadata folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metadata_folder = self.metadata_dir / f"batch_meta_{job.name}_{timestamp}"
        metadata_folder.mkdir(parents=True, exist_ok=True)

        # 1. Write text audit log
        audit_txt_path = metadata_folder / "metadata.txt"
        audit_txt_path.write_text("\n".join(log_lines), encoding="utf-8")

        # 2. Generate and save batch config YAML for recreation
        from .config import generate_batch_config_yaml

        yaml_params = {
            'mode': job.mode,
            'model_names': job.models,
            'backbone_idx': job.backbone_idx,
            'version': job.version if job.version is not None else 1
        }

        # Add mode-specific configurations
        if job.mode == "legacy":
            if job.weights:
                yaml_params['weights'] = job.weights
            if job.block_multipliers:
                yaml_params['block_multipliers'] = job.block_multipliers
            if job.crossattn_boosts:
                yaml_params['crossattn_boosts'] = job.crossattn_boosts
            if job.loras:
                yaml_params['loras'] = job.loras
        elif job.mode == "perres":
            if job.assignments:
                yaml_params['assignments'] = job.assignments
            if job.attn2_locks:
                yaml_params['attn2_locks'] = job.attn2_locks
            if job.loras:
                yaml_params['loras'] = job.loras
        elif job.mode == "hybrid":
            if job.hybrid_config:
                yaml_params['hybrid_config'] = job.hybrid_config
            if job.attn2_locks:
                yaml_params['attn2_locks'] = job.attn2_locks
            if job.loras:
                yaml_params['loras'] = job.loras

        try:
            batch_yaml = generate_batch_config_yaml(**yaml_params)
            batch_yaml_path = metadata_folder / "batch_config.yaml"
            batch_yaml_path.write_text(batch_yaml, encoding="utf-8")
            self.logger.info(f"Metadata saved to: {metadata_folder.name}/")
        except Exception as e:
            self.logger.warning(f"Could not generate batch config YAML: {e}")
            self.logger.info(f"Text metadata saved to: {metadata_folder.name}/metadata.txt")

    def _generate_summary_report(self, results: Dict[str, Any]) -> None:
        """Generate batch summary report - DISABLED"""
        return  # Summary report generation disabled
        summary_path = self.root_dir / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        lines = [
            "="*60,
            "XLFusion V1.3 - Batch Processing Summary",
            "="*60,
            f"Date: {datetime.now().isoformat(timespec='seconds')}",
            f"Total jobs: {results['total_jobs']}",
            f"Successful: {results['successful_jobs']}",
            f"Failed: {results['failed_jobs']}",
            f"Total time: {results['total_time']:.2f} seconds",
            f"Average time per job: {results['total_time']/max(results['total_jobs'], 1):.2f} seconds",
            "",
            "Job Details:",
        ]

        for job_result in results["jobs"]:
            status = "✓ SUCCESS" if job_result["success"] else "✗ FAILED"
            lines.append(f"  {job_result['name']}: {status} ({job_result['processing_time']:.2f}s)")
            if not job_result["success"] and job_result["error"]:
                lines.append(f"    Error: {job_result['error']}")

        lines.append("")
        lines.append("="*60)

        summary_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"Batch summary saved to: {summary_path}")


def load_batch_config(config_path: Path) -> BatchConfig:
    """Load and parse batch configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Parse jobs
    jobs = []
    for job_data in data.get("batch_jobs", []):
        # Check if using template
        if "template" in job_data:
            template_name = job_data["template"]
            if template_name not in data.get("templates", {}):
                raise ValueError(f"Template '{template_name}' not found")

            # Apply template
            job_data = apply_template(job_data, data["templates"][template_name])

        # Ensure required fields are present
        if 'mode' not in job_data:
            raise ValueError(f"Job missing required field 'mode': {job_data.get('name', 'unnamed')}")
        if 'name' not in job_data:
            raise ValueError("Job missing required field 'name'")
        
        job = BatchJob(**job_data)
        jobs.append(job)

    return BatchConfig(
        version=data.get("version", "1.3"),
        global_settings=data.get("global_settings", {}),
        batch_jobs=jobs,
        templates=data.get("templates", {})
    )


def apply_template(job_data: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
    """Apply template configuration to job data"""
    # Start with template config
    result = template.get("config_template", {}).copy()

    # Override with job-specific data
    result.update(job_data)

    # Apply template parameters
    template_params = job_data.get("template_params", {})
    default_params = template.get("default_params", {})

    # Merge parameters (job overrides defaults)
    params = {**default_params, **template_params}

    # Interpolate parameters in config
    result = interpolate_params(result, params)

    return result


def interpolate_params(data: Any, params: Dict[str, Any]) -> Any:
    """Recursively interpolate parameters in nested data structures with safe expression evaluation"""
    if isinstance(data, str):
        # First, replace simple parameter placeholders
        result = data
        for key, value in params.items():
            placeholder = "{{" + key + "}}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        
        # Then evaluate mathematical expressions in {{...}} format using ast.literal_eval
        import re
        import ast
        pattern = r'\{\{([^}]+)\}\}'
        
        def evaluate_expression(match):
            expr = match.group(1).strip()
            
            # Only allow safe mathematical operations
            allowed_chars = set('0123456789.+-*/() ')
            if not all(c in allowed_chars or c.isalpha() or c == '_' for c in expr):
                # If contains unsafe characters, don't evaluate
                return match.group(0)
            
            try:
                # Replace parameter names with values
                safe_expr = expr
                for key, value in params.items():
                    if key in safe_expr:
                        try:
                            safe_expr = safe_expr.replace(key, str(float(value)))
                        except (ValueError, TypeError):
                            safe_expr = safe_expr.replace(key, str(value))
                
                # Use ast.literal_eval for safe evaluation
                result = ast.literal_eval(safe_expr)
                return str(result)
            except Exception:
                # If evaluation fails, return the original expression
                return match.group(0)
        
        result = re.sub(pattern, evaluate_expression, result)
        return result
    elif isinstance(data, dict):
        return {k: interpolate_params(v, params) for k, v in data.items()}
    elif isinstance(data, list):
        return [interpolate_params(item, params) for item in data]
    else:
        return data


def main():
    """Command line interface for batch processing"""
    import argparse

    parser = argparse.ArgumentParser(description="XLFusion V1.3 - Batch Processor")
    parser.add_argument("config", type=Path, help="Batch configuration file")
    parser.add_argument("--validate-only", action="store_true", help="Only validate configuration, don't process")
    parser.add_argument("--template", help="Use specific template (overrides job templates)")

    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}")
        return 1

    try:
        # Load configuration
        config = load_batch_config(args.config)

        # Override template if specified
        if args.template:
            if args.template not in config.templates:
                print(f"Error: Template '{args.template}' not found in configuration")
                return 1
            
            template = config.templates[args.template]
            for job in config.batch_jobs:
                # Apply the template to each job
                job_data = job.__dict__.copy()
                job_data = apply_template(job_data, template)
                
                # Update job with new configuration
                for key, value in job_data.items():
                    if hasattr(job, key):
                        setattr(job, key, value)
                
                job.template = args.template

        # Validate configuration
        root_dir = Path(__file__).resolve().parent
        validator = BatchValidator(root_dir)
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

        # Process batch
        processor = BatchProcessor(config, root_dir, validate_only=args.validate_only)
        results = processor.process_batch()

        # Print summary
        print(f"\nBatch processing completed!")
        print(f"Total jobs: {results['total_jobs']}")
        print(f"Successful: {results['successful_jobs']}")
        print(f"Failed: {results['failed_jobs']}")
        print(f"Total time: {results['total_time']:.2f} seconds")

        return 0 if results['failed_jobs'] == 0 else 1

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
