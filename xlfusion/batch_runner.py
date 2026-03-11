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
from .validation import format_preflight_plan
from .runtime import execute_merge_job
from .types import MergeJobConfig


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

        output_dir = self.context.output_dir / self.config.global_settings.get("output_base", "batch_output")
        output_dir.mkdir(parents=True, exist_ok=True)

        loras = None
        if job.loras:
            loras = [
                {"file": item["file"], "scale": item.get("scale", 1.0), "path": self.context.loras_dir / item["file"]}
                for item in job.loras
            ]

        merge_job = MergeJobConfig(
            mode=job.mode,
            model_paths=list(job.model_paths),
            model_names=list(job.models),
            backbone_idx=int(job.backbone_idx),
            block_mapping=str(getattr(job, "block_mapping", "sdxl")),
            output_base_name=job.output_name,
            weights=job.weights,
            assignments=job.assignments,
            hybrid_config=job.hybrid_config,
            attn2_locks=job.attn2_locks,
            block_multipliers=job.block_multipliers,
            crossattn_boosts=job.crossattn_boosts,
            loras=loras,
            only_unet=bool(job.only_unet) if job.only_unet is not None else True,
            component_policy=job.component_policy,
            execution=job.execution,
            job_name=job.name,
            job_description=job.description,
        )

        result = execute_merge_job(output_dir, self.context.metadata_dir, merge_job)
        job.keys_processed = int(result.keys_processed)
        job.output_path = result.output_path
        job.version = int(result.version)
        job.success = True
        self.logger.info(f"Saved model to {job.output_path}")
        self.logger.info(f"Metadata saved to: {result.metadata_folder.name}/")
