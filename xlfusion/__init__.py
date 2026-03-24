"""Minimal public package surface for XLFusion."""

from .batch_processor import BatchProcessor, BatchValidator, load_batch_config
from .config import AppContext, ensure_dirs, list_safetensors, load_config, resolve_app_context
from .execution import execution_options_to_dict, normalize_execution_options
from .merge import merge_hybrid, merge_perres, stream_checkpoint_algebra_from_paths, stream_weighted_merge_from_paths
from .presets import inspect_recovery_source, load_single_job_preset, save_single_job_preset
from .runtime import execute_merge_job
from .types import MergeJobConfig, MergeJobResult
from .validation import export_preflight_plan, format_preflight_plan, validate_merge_request
from .version import __version__
from .workflow import save_merge_results

__all__ = [
    "AppContext",
    "BatchProcessor",
    "BatchValidator",
    "ensure_dirs",
    "execution_options_to_dict",
    "export_preflight_plan",
    "format_preflight_plan",
    "inspect_recovery_source",
    "list_safetensors",
    "load_single_job_preset",
    "load_batch_config",
    "load_config",
    "merge_hybrid",
    "merge_perres",
    "MergeJobConfig",
    "MergeJobResult",
    "normalize_execution_options",
    "resolve_app_context",
    "save_single_job_preset",
    "execute_merge_job",
    "save_merge_results",
    "stream_checkpoint_algebra_from_paths",
    "stream_weighted_merge_from_paths",
    "validate_merge_request",
    "__version__",
]
