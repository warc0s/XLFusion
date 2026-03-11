"""Shared internal types for XLFusion.

These dataclasses provide a stable contract between validation, execution,
and the CLI/GUI/batch front-ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


ProgressCallback = Callable[[str, int], None]


@dataclass(frozen=True)
class MergeJobConfig:
    """Normalized merge configuration ready for execution.

    This structure is intentionally close to what ``validate_merge_request`` returns
    under ``result.normalized`` so that all front-ends can share the same execution path.
    """

    mode: str
    model_paths: List[Path]
    model_names: List[str]
    backbone_idx: int
    block_mapping: str = "sdxl"

    output_base_name: Optional[str] = None

    weights: Optional[List[float]] = None
    assignments: Optional[Dict[str, int]] = None
    hybrid_config: Optional[Dict[str, Dict[str, float]]] = None
    attn2_locks: Optional[Dict[str, int]] = None
    block_multipliers: Optional[List[Dict[str, float]]] = None
    crossattn_boosts: Optional[List[Dict[str, float]]] = None

    loras: Optional[List[Dict[str, Any]]] = None

    only_unet: bool = True
    component_policy: Optional[Dict[str, str]] = None
    execution: Optional[Dict[str, Any]] = None

    job_name: Optional[str] = None
    job_description: Optional[str] = None


@dataclass(frozen=True)
class MergeJobResult:
    """Artifacts produced by executing a merge job."""

    output_path: Path
    metadata_folder: Path
    version: int
    keys_processed: int
    lora_reports: List[Dict[str, Any]]
