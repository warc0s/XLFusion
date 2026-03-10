"""Minimal public package surface for XLFusion."""

from .batch_processor import BatchProcessor, BatchValidator, load_batch_config
from .config import AppContext, ensure_dirs, list_safetensors, load_config, resolve_app_context
from .merge import merge_hybrid, merge_perres, stream_weighted_merge_from_paths
from .validation import export_preflight_plan, format_preflight_plan, validate_merge_request
from .workflow import save_merge_results

__all__ = [
    "AppContext",
    "BatchProcessor",
    "BatchValidator",
    "ensure_dirs",
    "export_preflight_plan",
    "format_preflight_plan",
    "list_safetensors",
    "load_batch_config",
    "load_config",
    "merge_hybrid",
    "merge_perres",
    "resolve_app_context",
    "save_merge_results",
    "stream_weighted_merge_from_paths",
    "validate_merge_request",
]
