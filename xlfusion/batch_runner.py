"""Batch execution runtime."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Union

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .batch_schema import BatchConfig, BatchJob
from .config import AppContext, resolve_app_context
from .execution import execution_options_to_dict
from .lora import apply_single_lora
from .merge import merge_hybrid, merge_perres, stream_weighted_merge_from_paths
from .validation import format_preflight_plan
from .workflow import save_merge_results


class BatchProcessor:
    """Processes batch jobs sequentially."""

    def __init__(self, config: BatchConfig, context: Union[AppContext, Path], validate_only: bool = False):
        self.config = config
        self.context = context if isinstance(context, AppContext) else resolve_app_context(context)
        self.validate_only = validate_only

        self.total_jobs = len(config.batch_jobs)
        self.completed_jobs = 0
        self.failed_jobs = 0
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        log_level = getattr(logging, self.config.global_settings.get("log_level", "INFO").upper())
        handlers = [logging.StreamHandler(sys.stdout)]
        if self.config.global_settings.get("log_to_file", False):
            handlers.insert(0, logging.FileHandler(self.context.root_dir / "batch_log.txt"))

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=handlers,
        )
        return logging.getLogger("XLFusion.Batch")

    def process_batch(self) -> Dict[str, Any]:
        start_time = time.time()
        self.logger.info(f"Starting batch processing of {self.total_jobs} jobs")

        results = {
            "total_jobs": self.total_jobs,
            "successful_jobs": 0,
            "failed_jobs": 0,
            "total_time": 0.0,
            "jobs": [],
        }

        progress_bar = None
        if TQDM_AVAILABLE and not self.validate_only and sys.stderr.isatty():
            progress_bar = tqdm(total=self.total_jobs, desc="Batch Progress", unit="job")

        for job in self.config.batch_jobs:
            job_start = time.time()
            try:
                self.logger.info(f"Processing job: {job.name}")
                if job.preflight:
                    self.logger.info("\n" + format_preflight_plan(job.preflight))
                if self.validate_only:
                    job.success = True
                else:
                    self._process_job(job)

                job.processing_time = time.time() - job_start
                self.completed_jobs += 1
                results["successful_jobs"] += 1
                self.logger.info(f"Job {job.name} completed successfully in {job.processing_time:.2f}s")
            except Exception as exc:
                job.success = False
                job.error_message = str(exc)
                job.processing_time = time.time() - job_start
                self.failed_jobs += 1
                results["failed_jobs"] += 1
                self.logger.error(f"Job {job.name} failed: {exc}")
                if not self.config.global_settings.get("continue_on_error", True):
                    self.logger.error("Stopping batch due to error (continue_on_error=false)")
                    break

            if progress_bar:
                progress_bar.update(1)
                progress_bar.set_postfix({"completed": self.completed_jobs, "failed": self.failed_jobs})
            elif not self.validate_only:
                self.logger.info(
                    f"Batch progress: {self.completed_jobs + self.failed_jobs}/{self.total_jobs} "
                    f"(ok={self.completed_jobs}, failed={self.failed_jobs})"
                )

            results["jobs"].append(
                {
                    "name": job.name,
                    "success": job.success,
                    "processing_time": job.processing_time,
                    "keys_processed": job.keys_processed,
                    "error": job.error_message if not job.success else None,
                }
            )

        if progress_bar:
            progress_bar.close()

        results["total_time"] = time.time() - start_time
        return results

    def _process_job(self, job: BatchJob) -> None:
        if not job.model_paths:
            job.model_paths = [self.context.models_dir / name for name in job.models]
        if not isinstance(job.backbone_idx, int):
            raise ValueError("Validated backbone index is missing")

        merged_state = self._merge_job(job)
        if merged_state is None:
            raise ValueError(f"Unsupported mode: {job.mode}")

        self._apply_loras(job, merged_state)

        output_dir = self.context.output_dir / self.config.global_settings.get("output_base", "batch_output")
        output_dir.mkdir(parents=True, exist_ok=True)

        job.keys_processed = len(merged_state)
        output_path, metadata_folder, version = save_merge_results(
            output_dir,
            self.context.metadata_dir,
            merged_state,
            job.models,
            job.mode,
            job.backbone_idx,
            self._build_yaml_kwargs(job),
            model_paths=job.model_paths,
            lora_paths=[self.context.loras_dir / spec["file"] for spec in (job.loras or [])] or None,
            output_base_name=job.output_name,
            extra_metadata={"batch_job": job.name, "description": job.description},
            execution=execution_options_to_dict(job.execution),
            job_name=job.name,
            job_description=job.description,
        )
        job.output_path = output_path
        job.version = version
        job.success = True
        self.logger.info(f"Saved model to {job.output_path}")
        self.logger.info(f"Metadata saved to: {metadata_folder.name}/")

    def _merge_job(self, job: BatchJob):
        if job.mode == "legacy":
            if job.weights is None:
                raise ValueError("Legacy mode requires weights")
            merged_state, _ = stream_weighted_merge_from_paths(
                job.model_paths,
                job.weights,
                job.backbone_idx,
                only_unet=True,
                block_multipliers=job.block_multipliers,
                crossattn_boosts=job.crossattn_boosts,
                execution=job.execution,
            )
            return merged_state

        if job.mode == "perres":
            if job.assignments is None:
                raise ValueError("PerRes mode requires assignments")
            return merge_perres(
                job.model_paths,
                job.assignments,
                job.backbone_idx,
                job.attn2_locks,
                execution=job.execution,
            )

        if job.mode == "hybrid":
            if job.hybrid_config is None:
                raise ValueError("Hybrid mode requires hybrid_config")
            return merge_hybrid(
                job.model_paths,
                job.hybrid_config,
                job.backbone_idx,
                job.attn2_locks,
                execution=job.execution,
            )

        return None

    def _apply_loras(self, job: BatchJob, merged_state: Dict[str, Any]) -> None:
        if not job.loras:
            return

        for lora_spec in job.loras:
            lora_path = self.context.loras_dir / lora_spec["file"]
            scale = lora_spec.get("scale", 0.3)
            applied, skipped = apply_single_lora(merged_state, lora_path, scale)
            self.logger.info(f"Applied LoRA {lora_spec['file']}: {applied} applied, {skipped} skipped")

    def _build_yaml_kwargs(self, job: BatchJob) -> Dict[str, Any]:
        yaml_kwargs: Dict[str, Any] = {}

        if job.mode == "legacy":
            if job.weights:
                yaml_kwargs["weights"] = job.weights
            if job.block_multipliers:
                yaml_kwargs["block_multipliers"] = job.block_multipliers
            if job.crossattn_boosts:
                yaml_kwargs["crossattn_boosts"] = job.crossattn_boosts
        elif job.mode == "perres":
            if job.assignments:
                yaml_kwargs["assignments"] = job.assignments
            if job.attn2_locks:
                yaml_kwargs["attn2_locks"] = job.attn2_locks
        elif job.mode == "hybrid":
            if job.hybrid_config:
                yaml_kwargs["hybrid_config"] = job.hybrid_config
            if job.attn2_locks:
                yaml_kwargs["attn2_locks"] = job.attn2_locks

        if job.loras:
            yaml_kwargs["loras"] = job.loras
        if job.execution:
            yaml_kwargs["execution"] = execution_options_to_dict(job.execution)
        return yaml_kwargs
