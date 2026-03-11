"""Batch configuration schema, parsing and validation."""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import YAML_AVAILABLE, AppContext, resolve_app_context
from .validation import validate_merge_request

try:
    import yaml
except ImportError:  # pragma: no cover - handled through YAML_AVAILABLE
    yaml = None


@dataclass
class BatchJob:
    """Represents a single batch job configuration."""

    name: str
    mode: str
    block_mapping: str = "sdxl"
    description: str = ""
    models: List[str] = field(default_factory=list)
    backbone: Union[int, str] = 0
    output_name: Optional[str] = None

    weights: Optional[List[float]] = None
    assignments: Optional[Dict[str, int]] = None
    hybrid_config: Optional[Dict[str, Dict[str, float]]] = None
    attn2_locks: Optional[Dict[str, int]] = None
    block_multipliers: Optional[List[Dict[str, float]]] = None
    crossattn_boosts: Optional[List[Dict[str, float]]] = None
    loras: Optional[List[Dict[str, Any]]] = None
    execution: Optional[Dict[str, Any]] = None
    only_unet: Optional[bool] = None
    component_policy: Optional[Dict[str, str]] = None

    template: Optional[str] = None
    template_params: Optional[Dict[str, Any]] = None

    model_paths: List[Path] = field(default_factory=list)
    backbone_idx: int = 0
    output_path: Optional[Path] = None
    version: Optional[int] = None
    preflight: Optional[Any] = None

    success: bool = False
    error_message: str = ""
    processing_time: float = 0.0
    keys_processed: int = 0


@dataclass
class BatchConfig:
    """Complete batch configuration."""

    global_settings: Dict[str, Any]
    batch_jobs: List[BatchJob]
    templates: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class BatchValidator:
    """Validates batch configuration before execution."""

    def __init__(self, context: Union[AppContext, Path]):
        self.context = context if isinstance(context, AppContext) else resolve_app_context(context)
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_config(self, config: BatchConfig) -> bool:
        self.errors = []
        self.warnings = []

        self._validate_global_settings(config.global_settings)
        for index, job in enumerate(config.batch_jobs):
            self._validate_job(job, index)

        return len(self.errors) == 0

    def _validate_global_settings(self, settings: Dict[str, Any]) -> None:
        required_keys = ["output_base", "continue_on_error", "max_parallel"]
        for key in required_keys:
            if key not in settings:
                self.errors.append(f"Missing global setting: {key}")

        if "max_parallel" in settings and settings["max_parallel"] != 1:
            self.warnings.append("Parallel processing not yet implemented, max_parallel will be ignored")

    def _validate_job(self, job: BatchJob, index: int) -> None:
        prefix = f"Job {index} ({job.name}):"
        valid_modes = ["legacy", "perres", "hybrid"]
        if job.mode not in valid_modes:
            self.errors.append(f"{prefix} Invalid mode '{job.mode}'. Must be one of: {valid_modes}")
            return

        job.model_paths = [self.context.models_dir / name for name in job.models]
        validation = validate_merge_request(
            mode=job.mode,
            model_paths=job.model_paths,
            backbone=job.backbone,
            weights=job.weights,
            assignments=job.assignments,
            hybrid_config=job.hybrid_config,
            attn2_locks=job.attn2_locks,
            block_multipliers=job.block_multipliers,
            crossattn_boosts=job.crossattn_boosts,
            loras=job.loras,
            loras_dir=self.context.loras_dir,
            only_unet=job.only_unet,
            component_policy=job.component_policy,
            block_mapping=job.block_mapping,
        )

        for issue in validation.errors:
            self.errors.append(f"{prefix} {issue.field}: {issue.message}")
        for issue in validation.warnings:
            self.warnings.append(f"{prefix} {issue.field}: {issue.message}")

        if not validation.valid:
            return

        job.model_paths = validation.normalized["model_paths"]
        job.backbone_idx = validation.normalized["backbone_idx"]
        job.backbone = job.backbone_idx
        job.weights = validation.normalized.get("weights")
        job.assignments = validation.normalized.get("assignments")
        job.hybrid_config = validation.normalized.get("hybrid_config")
        job.attn2_locks = validation.normalized.get("attn2_locks")
        job.block_multipliers = validation.normalized.get("block_multipliers")
        job.crossattn_boosts = validation.normalized.get("crossattn_boosts")
        if validation.normalized.get("loras"):
            job.loras = [
                {"file": item["file"], "scale": item["scale"]}
                for item in validation.normalized["loras"]
            ]
        if job.execution is not None and not isinstance(job.execution, dict):
            self.errors.append(f"{prefix} execution: must be a mapping when provided")
            return
        job.only_unet = validation.normalized.get("only_unet")
        job.component_policy = validation.normalized.get("component_policy")
        job.preflight = validation.preflight
        job.block_mapping = str(validation.normalized.get("block_mapping", job.block_mapping))


def load_batch_config(config_path: Path) -> BatchConfig:
    """Load and parse batch configuration file."""
    if not YAML_AVAILABLE:
        raise RuntimeError("PyYAML is required to read batch configuration files.")

    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Batch configuration file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Batch configuration is not valid YAML: {exc}") from exc

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Batch configuration must contain a top-level mapping")

    jobs: List[BatchJob] = []
    for job_data in data.get("batch_jobs", []):
        if "template" in job_data:
            template_name = job_data["template"]
            if template_name not in data.get("templates", {}):
                raise ValueError(f"Template '{template_name}' not found")
            job_data = apply_template(job_data, data["templates"][template_name])

        if "mode" not in job_data:
            raise ValueError(f"Job missing required field 'mode': {job_data.get('name', 'unnamed')}")
        if "name" not in job_data:
            raise ValueError("Job missing required field 'name'")

        jobs.append(BatchJob(**job_data))

    return BatchConfig(
        global_settings=data.get("global_settings", {}),
        batch_jobs=jobs,
        templates=data.get("templates", {}),
    )


def apply_template(job_data: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
    """Apply template configuration to job data."""
    result = copy.deepcopy(template.get("config_template", {}))
    result.update(job_data)

    template_params = job_data.get("template_params", {})
    default_params = template.get("default_params", {})
    params = {**default_params, **template_params}
    return interpolate_params(result, params)


def interpolate_params(data: Any, params: Dict[str, Any]) -> Any:
    """Recursively interpolate parameters in nested data structures."""

    def evaluate_numeric_expression(expr: str) -> float:
        tree = ast.parse(expr, mode="eval")

        def _eval(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            if isinstance(node, ast.Name):
                if node.id not in params:
                    raise ValueError(f"Unknown parameter '{node.id}'")
                value = params[node.id]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"Parameter '{node.id}' is not numeric")
                return float(value)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                operand = _eval(node.operand)
                return operand if isinstance(node.op, ast.UAdd) else -operand
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                left = _eval(node.left)
                right = _eval(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                return left / right
            raise ValueError("Unsupported template expression")

        return _eval(tree)

    if isinstance(data, str):
        import re

        pattern = r"\{\{([^}]+)\}\}"
        exact_match = re.fullmatch(pattern, data.strip())
        if exact_match:
            expr = exact_match.group(1).strip()
            if expr in params:
                return params[expr]
            try:
                return evaluate_numeric_expression(expr)
            except Exception:
                return data

        def replace_expression(match: re.Match[str]) -> str:
            expr = match.group(1).strip()
            if expr in params:
                return str(params[expr])
            try:
                return str(evaluate_numeric_expression(expr))
            except Exception:
                return match.group(0)

        return re.sub(pattern, replace_expression, data)
    if isinstance(data, dict):
        return {key: interpolate_params(value, params) for key, value in data.items()}
    if isinstance(data, list):
        return [interpolate_params(item, params) for item in data]
    return data
