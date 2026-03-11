"""
Shared validation and preflight utilities for CLI, GUI and batch flows.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from safetensors.torch import safe_open

from .blocks import (
    SDXL_ATTN_BLOCKS,
    SDXL_BLOCK_GROUPS,
    SDXL_LEGACY_GROUPS,
    classify_component_key,
    get_block_assignment,
    get_block_mapping,
)
from .memory import check_memory_availability, estimate_memory_requirement, get_available_memory_gb
from .merge import COMPONENT_POLICY_DEFAULTS, NON_UNET_COMPONENTS

BLOCK_GROUPS = list(SDXL_BLOCK_GROUPS)
LEGACY_GROUPS = list(SDXL_LEGACY_GROUPS)
ATTN_BLOCKS = list(SDXL_ATTN_BLOCKS)
LEGACY_NON_UNET_ACTIONS = {"exclude", "merge", "backbone"}
BACKBONE_ONLY_ACTIONS = {"exclude", "backbone"}


@dataclass
class ValidationIssue:
    field: str
    message: str


@dataclass
class PreflightPlan:
    mode: str
    model_names: List[str]
    backbone_idx: int
    backbone_name: str
    selected_model_indices: List[int]
    selected_models: List[str]
    loaded_model_indices: List[int]
    loaded_models: List[str]
    affected_blocks: List[str]
    effective_locks: Dict[str, str]
    loras: List[Dict[str, Any]]
    only_unet: bool
    component_policy: Dict[str, str]
    estimated_memory_gb: float
    available_memory_gb: Optional[float]
    memory_check_ok: bool
    compatibility_warnings: List[str] = field(default_factory=list)
    risk_alerts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    valid: bool
    normalized: Dict[str, Any]
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    preflight: Optional[PreflightPlan] = None


def _error(result: ValidationResult, field: str, message: str) -> None:
    result.errors.append(ValidationIssue(field=field, message=message))
    result.valid = False


def _warning(result: ValidationResult, field: str, message: str) -> None:
    result.warnings.append(ValidationIssue(field=field, message=message))


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid integer index")
    return int(value)


def _safe_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid numeric value")
    return float(value)


def _resolve_backbone(backbone: Any, model_names: Sequence[str]) -> int:
    if isinstance(backbone, str):
        if backbone not in model_names:
            raise ValueError(f"unknown backbone '{backbone}'")
        return model_names.index(backbone)
    return _safe_int(backbone)


def _validate_index(result: ValidationResult, value: Any, field: str, model_count: int) -> Optional[int]:
    try:
        idx = _safe_int(value)
    except (TypeError, ValueError) as exc:
        _error(result, field, f"must be an integer model index ({exc})")
        return None
    if idx < 0 or idx >= model_count:
        _error(result, field, f"index {idx} out of range for {model_count} selected models")
        return None
    return idx


def _validate_path_list(result: ValidationResult, model_paths: Sequence[Path]) -> List[Path]:
    normalized_paths: List[Path] = []
    if len(model_paths) < 2:
        _error(result, "models", "select at least two model files")
        return normalized_paths

    for idx, path in enumerate(model_paths):
        if not isinstance(path, Path):
            path = Path(path)
        if not path.exists():
            _error(result, f"models[{idx}]", f"file not found: {path}")
            continue
        if path.suffix.lower() != ".safetensors":
            _error(result, f"models[{idx}]", f"unsupported file type: {path.name}")
            continue
        normalized_paths.append(path)
    return normalized_paths


def _validate_loras(result: ValidationResult, loras: Optional[Sequence[Any]], loras_dir: Optional[Path]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if not loras:
        return normalized

    for idx, entry in enumerate(loras):
        field = f"loras[{idx}]"
        path: Optional[Path] = None
        scale_raw: Any = 1.0

        if isinstance(entry, tuple) and len(entry) == 2:
            path = Path(entry[0])
            scale_raw = entry[1]
        elif isinstance(entry, dict):
            file_name = entry.get("file")
            scale_raw = entry.get("scale", 1.0)
            if not isinstance(file_name, str) or not file_name.strip():
                _error(result, f"{field}.file", "must be a non-empty filename")
                continue
            path = (loras_dir / file_name) if loras_dir else Path(file_name)
        else:
            _error(result, field, "must be a tuple(path, scale) or a mapping with file/scale")
            continue

        try:
            scale = _safe_float(scale_raw)
        except (TypeError, ValueError):
            _error(result, f"{field}.scale", "must be numeric")
            continue

        if path is None:
            _error(result, field, "could not resolve LoRA path")
            continue
        if not path.exists():
            _error(result, f"{field}.file", f"file not found: {path}")
            continue
        if path.suffix.lower() != ".safetensors":
            _error(result, f"{field}.file", f"unsupported file type: {path.name}")
            continue

        normalized.append({"file": path.name, "scale": scale, "path": path})
    return normalized


def _validate_legacy_maps(
    result: ValidationResult,
    items: Optional[Sequence[Dict[str, Any]]],
    field: str,
    model_count: int,
    *,
    allowed_groups: Sequence[str] = LEGACY_GROUPS,
) -> Optional[List[Dict[str, float]]]:
    if items is None:
        return None
    if not isinstance(items, list):
        _error(result, field, "must be a list of per-model mappings")
        return None
    if len(items) != model_count:
        _error(result, field, f"must contain {model_count} entries, one per selected model")
        return None

    normalized: List[Dict[str, float]] = []
    for model_idx, mapping in enumerate(items):
        if not isinstance(mapping, dict):
            _error(result, f"{field}[{model_idx}]", "must be a mapping")
            normalized.append({})
            continue
        per_model: Dict[str, float] = {}
        for key, value in mapping.items():
            if key not in allowed_groups:
                _error(result, f"{field}[{model_idx}].{key}", f"unsupported block group '{key}'")
                continue
            try:
                per_model[key] = _safe_float(value)
            except (TypeError, ValueError):
                _error(result, f"{field}[{model_idx}].{key}", "must be numeric")
        normalized.append(per_model)
    return normalized


def _normalize_component_policy(
    result: ValidationResult,
    *,
    mode: str,
    only_unet: Optional[Any],
    component_policy: Optional[Dict[str, Any]],
) -> tuple[bool, Dict[str, str]]:
    default_only_unet = mode == "legacy"
    if only_unet is None:
        normalized_only_unet = default_only_unet
    elif isinstance(only_unet, bool):
        normalized_only_unet = only_unet
    else:
        _error(result, "only_unet", "must be a boolean when provided")
        normalized_only_unet = default_only_unet

    if normalized_only_unet:
        return True, {component: "exclude" for component in NON_UNET_COMPONENTS}

    allowed_actions = LEGACY_NON_UNET_ACTIONS if mode == "legacy" else BACKBONE_ONLY_ACTIONS
    normalized_policy = dict(COMPONENT_POLICY_DEFAULTS.get(mode, COMPONENT_POLICY_DEFAULTS["legacy"]))

    if component_policy is not None:
        if not isinstance(component_policy, dict):
            _error(result, "component_policy", "must be a mapping of component to action")
        else:
            for component, action in component_policy.items():
                if component not in NON_UNET_COMPONENTS:
                    _error(result, f"component_policy.{component}", "unsupported component")
                    continue
                if not isinstance(action, str) or action not in allowed_actions:
                    _error(
                        result,
                        f"component_policy.{component}",
                        f"must be one of {sorted(allowed_actions)} for mode '{mode}'",
                    )
                    continue
                normalized_policy[component] = action

    normalized_only_unet = all(normalized_policy[component] == "exclude" for component in NON_UNET_COMPONENTS)
    return normalized_only_unet, normalized_policy


def _collect_structure_compatibility(model_paths: Sequence[Path]) -> List[str]:
    if len(model_paths) < 2:
        return []

    warnings: List[str] = []
    reference_path = model_paths[0]
    with safe_open(str(reference_path), framework="pt", device="cpu") as ref_handle:
        reference_keys = list(ref_handle.keys())
        reference_key_set = set(reference_keys)
        reference_shapes = {key: tuple(ref_handle.get_slice(key).get_shape()) for key in reference_keys}

    for model_path in model_paths[1:]:
        with safe_open(str(model_path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            key_set = set(keys)
            missing = len(reference_key_set - key_set)
            extra = len(key_set - reference_key_set)
            if missing or extra:
                warnings.append(
                    f"{model_path.name}: structure differs from {reference_path.name} (missing={missing}, extra={extra})"
                )

            mismatches = 0
            for key in reference_key_set & key_set:
                if tuple(handle.get_slice(key).get_shape()) != reference_shapes[key]:
                    mismatches += 1
                    if mismatches >= 3:
                        break
            if mismatches:
                warnings.append(f"{model_path.name}: found shape mismatches against {reference_path.name}")
    return warnings


def _sample_pairwise_similarity(model_a: Path, model_b: Path, limit: int = 12) -> float:
    with safe_open(str(model_a), framework="pt", device="cpu") as handle_a, safe_open(str(model_b), framework="pt", device="cpu") as handle_b:
        common_keys = sorted(set(handle_a.keys()) & set(handle_b.keys()))
        similarities: List[float] = []
        for key in common_keys:
            if len(similarities) >= limit:
                break
            shape_a = tuple(handle_a.get_slice(key).get_shape())
            shape_b = tuple(handle_b.get_slice(key).get_shape())
            if shape_a != shape_b:
                continue
            tensor_a = handle_a.get_tensor(key).flatten().to(torch.float32)
            tensor_b = handle_b.get_tensor(key).flatten().to(torch.float32)
            if tensor_a.numel() == 0 or tensor_b.numel() == 0:
                continue
            similarity = torch.nn.functional.cosine_similarity(tensor_a.unsqueeze(0), tensor_b.unsqueeze(0)).item()
            similarities.append(float(similarity))
        if not similarities:
            return 0.0
        return sum(similarities) / len(similarities)


def _collect_risk_alerts(
    model_paths: Sequence[Path],
    *,
    mode: str,
    normalized: Dict[str, Any],
    compatibility_warnings: Sequence[str],
    block_groups: Sequence[str] = BLOCK_GROUPS,
) -> List[str]:
    alerts: List[str] = []

    if compatibility_warnings:
        alerts.append("Model structure differs across inputs; the merge may fall back to the backbone or skip tensors.")

    if len(model_paths) >= 2:
        try:
            similarity = _sample_pairwise_similarity(model_paths[0], model_paths[1])
        except Exception:
            similarity = 0.0
        if similarity < 0.35:
            alerts.append(
                f"Sampled similarity is very low ({similarity:.3f}); this combination is high-risk for unstable or low-value results."
            )
        elif similarity > 0.995:
            alerts.append(
                f"Sampled similarity is extremely high ({similarity:.3f}); the merge may add very little value over the backbone."
            )

    if mode == "legacy":
        weights = normalized.get("weights") or []
        if weights and max(weights) >= 0.9:
            alerts.append("One model dominates the legacy weights; the output may be too close to a single source model.")
    elif mode == "perres":
        assignments = normalized.get("assignments") or {}
        if len(set(assignments.values())) == 1 and len(assignments) == len(block_groups):
            alerts.append("All PerRes blocks are assigned to the same model; block mode adds no diversity in this configuration.")
    elif mode == "hybrid":
        hybrid_config = normalized.get("hybrid_config") or {}
        concentrated_blocks = 0
        for weights in hybrid_config.values():
            if weights and max(weights.values()) >= 0.95:
                concentrated_blocks += 1
        if concentrated_blocks >= 4:
            alerts.append("Most hybrid blocks are effectively single-source; consider whether hybrid mode is adding enough value.")

    locks = normalized.get("attn2_locks") or {}
    if locks and mode == "legacy":
        alerts.append("Cross-attention locks are ignored in legacy mode.")

    component_policy = normalized.get("component_policy") or {}
    if normalized.get("only_unet"):
        alerts.append("Only UNet tensors will be written; VAE, text encoder and other components are excluded explicitly.")
    elif mode != "legacy" and any(action == "exclude" for action in component_policy.values()):
        alerts.append("Some non-UNet components will be excluded instead of copied from the backbone.")

    return alerts


def validate_merge_request(
    *,
    mode: str,
    model_paths: Sequence[Path],
    backbone: Any = 0,
    weights: Optional[Sequence[Any]] = None,
    assignments: Optional[Dict[str, Any]] = None,
    hybrid_config: Optional[Dict[str, Dict[str, Any]]] = None,
    attn2_locks: Optional[Dict[str, Any]] = None,
    block_multipliers: Optional[Sequence[Dict[str, Any]]] = None,
    crossattn_boosts: Optional[Sequence[Dict[str, Any]]] = None,
    loras: Optional[Sequence[Any]] = None,
    loras_dir: Optional[Path] = None,
    only_unet: Optional[Any] = None,
    component_policy: Optional[Dict[str, Any]] = None,
    block_mapping: str = "sdxl",
) -> ValidationResult:
    mapping = get_block_mapping(block_mapping)
    block_groups = list(mapping.block_groups)
    attn_blocks = set(mapping.attn_blocks)
    legacy_groups = list(mapping.legacy_groups)

    result = ValidationResult(valid=True, normalized={"mode": mode, "block_mapping": mapping.name})
    normalized_paths = _validate_path_list(result, model_paths)
    model_names = [path.name for path in normalized_paths]
    model_count = len(normalized_paths)

    result.normalized["model_paths"] = normalized_paths
    result.normalized["model_names"] = model_names

    valid_modes = {"legacy", "perres", "hybrid"}
    if mode not in valid_modes:
        _error(result, "mode", f"must be one of {sorted(valid_modes)}")

    backbone_idx: Optional[int] = None
    if model_count:
        try:
            backbone_idx = _resolve_backbone(backbone, model_names)
        except ValueError as exc:
            _error(result, "backbone", str(exc))
        else:
            if backbone_idx < 0 or backbone_idx >= model_count:
                _error(result, "backbone", f"index {backbone_idx} out of range for {model_count} selected models")
                backbone_idx = None
    result.normalized["backbone_idx"] = backbone_idx

    normalized_only_unet, normalized_component_policy = _normalize_component_policy(
        result,
        mode=mode,
        only_unet=only_unet,
        component_policy=component_policy,
    )
    result.normalized["only_unet"] = normalized_only_unet
    result.normalized["component_policy"] = normalized_component_policy

    normalized_loras = _validate_loras(result, loras, loras_dir)
    result.normalized["loras"] = normalized_loras

    if mode == "legacy":
        normalized_weights: List[float] = []
        if weights is None:
            _error(result, "weights", "legacy mode requires a weights array")
        elif len(weights) != model_count:
            _error(result, "weights", f"expected {model_count} weights, got {len(weights)}")
        else:
            for idx, value in enumerate(weights):
                try:
                    weight = _safe_float(value)
                except (TypeError, ValueError):
                    _error(result, f"weights[{idx}]", "must be numeric")
                    continue
                if weight < 0:
                    _error(result, f"weights[{idx}]", "cannot be negative")
                    continue
                normalized_weights.append(weight)

        if normalized_weights and sum(normalized_weights) <= 0:
            _error(result, "weights", "sum of weights must be greater than zero")
        elif normalized_weights:
            total_weight = sum(normalized_weights)
            if abs(total_weight - 1.0) > 0.01:
                _warning(result, "weights", f"weights sum to {total_weight:.3f}; the merge will normalize them")

        result.normalized["weights"] = normalized_weights
        result.normalized["block_multipliers"] = _validate_legacy_maps(
            result,
            block_multipliers,
            "block_multipliers",
            model_count,
            allowed_groups=legacy_groups,
        )
        result.normalized["crossattn_boosts"] = _validate_legacy_maps(
            result,
            crossattn_boosts,
            "crossattn_boosts",
            model_count,
            allowed_groups=legacy_groups,
        )

    elif mode == "perres":
        normalized_assignments: Dict[str, int] = {}
        if not isinstance(assignments, dict):
            _error(result, "assignments", "perres mode requires an assignments mapping")
        else:
            for block in block_groups:
                if block not in assignments:
                    _error(result, f"assignments.{block}", "missing required assignment")
                    continue
                idx = _validate_index(result, assignments[block], f"assignments.{block}", model_count)
                if idx is not None:
                    normalized_assignments[block] = idx
        for block in assignments or {}:
            if block not in block_groups:
                _warning(result, f"assignments.{block}", "unknown block will be ignored")
        result.normalized["assignments"] = normalized_assignments

    elif mode == "hybrid":
        normalized_hybrid: Dict[str, Dict[int, float]] = {}
        if not isinstance(hybrid_config, dict):
            _error(result, "hybrid_config", "hybrid mode requires a hybrid_config mapping")
        else:
            for block in block_groups:
                weights_by_model = hybrid_config.get(block) if hybrid_config else None
                if not isinstance(weights_by_model, dict):
                    _error(result, f"hybrid_config.{block}", "missing required block mapping")
                    continue
                normalized_block: Dict[int, float] = {}
                for model_idx, value in weights_by_model.items():
                    idx = _validate_index(result, model_idx, f"hybrid_config.{block}.{model_idx}", model_count)
                    if idx is None:
                        continue
                    try:
                        weight = _safe_float(value)
                    except (TypeError, ValueError):
                        _error(result, f"hybrid_config.{block}.{model_idx}", "weight must be numeric")
                        continue
                    if weight < 0:
                        _error(result, f"hybrid_config.{block}.{idx}", "weight cannot be negative")
                        continue
                    normalized_block[idx] = weight

                if not normalized_block:
                    _error(result, f"hybrid_config.{block}", "must contain at least one valid model weight")
                else:
                    total_weight = sum(normalized_block.values())
                    if total_weight == 0:
                        _warning(result, f"hybrid_config.{block}", "all weights are zero; backbone fallback would be used")
                    elif abs(total_weight - 1.0) > 0.01:
                        _warning(
                            result,
                            f"hybrid_config.{block}",
                            f"weights sum to {total_weight:.3f}; the merge will normalize them per block",
                        )
                normalized_hybrid[block] = normalized_block
        result.normalized["hybrid_config"] = normalized_hybrid

    normalized_locks: Dict[str, int] = {}
    if attn2_locks is not None:
        if not isinstance(attn2_locks, dict):
            _error(result, "attn2_locks", "must be a mapping of down/mid/up to model indices")
        else:
            for block_type, value in attn2_locks.items():
                if block_type not in attn_blocks:
                    _error(result, f"attn2_locks.{block_type}", "unsupported lock group")
                    continue
                idx = _validate_index(result, value, f"attn2_locks.{block_type}", model_count)
                if idx is not None:
                    normalized_locks[block_type] = idx
    result.normalized["attn2_locks"] = normalized_locks or None

    if not result.valid or backbone_idx is None:
        return result

    if mode == "legacy":
        selected_indices = sorted({idx for idx, weight in enumerate(result.normalized["weights"]) if weight > 0} | {backbone_idx})
        loaded_indices = list(range(model_count))
        affected_blocks = ["down", "mid", "up"]
    elif mode == "perres":
        assignments_map = result.normalized.get("assignments", {})
        selected_indices = sorted(set(assignments_map.values()) | set(normalized_locks.values()) | {backbone_idx})
        loaded_indices = selected_indices[:]
        affected_blocks = [block for block in block_groups if block in assignments_map]
    else:
        hybrid_map = result.normalized.get("hybrid_config", {})
        selected = {backbone_idx}
        for block_weights in hybrid_map.values():
            selected.update(idx for idx, weight in block_weights.items() if weight > 0)
        selected.update(normalized_locks.values())
        selected_indices = sorted(selected)
        loaded_indices = selected_indices[:]
        affected_blocks = [block for block in block_groups if block in hybrid_map]

    try:
        compatibility_warnings = _collect_structure_compatibility(normalized_paths)
    except Exception as exc:
        _error(result, "models", f"could not inspect model structure for preflight: {exc}")
        return result
    for warning_text in compatibility_warnings:
        _warning(result, "compatibility", warning_text)

    risk_alerts = _collect_risk_alerts(
        normalized_paths,
        mode=mode,
        normalized=result.normalized,
        compatibility_warnings=compatibility_warnings,
        block_groups=block_groups,
    )
    for alert in risk_alerts:
        _warning(result, "risk", alert)

    estimated_indices = set(loaded_indices if mode != "legacy" else range(model_count))
    estimated_memory_gb = estimate_memory_requirement(normalized_paths, estimated_indices)
    available_memory_gb = get_available_memory_gb()
    memory_ok = check_memory_availability(estimated_memory_gb)
    if not memory_ok:
        _warning(
            result,
            "memory",
            f"estimated memory requirement is {estimated_memory_gb:.2f} GB and may exceed available memory",
        )

    effective_locks = {block_type: model_names[idx] for block_type, idx in normalized_locks.items()}

    result.preflight = PreflightPlan(
        mode=mode,
        model_names=model_names,
        backbone_idx=backbone_idx,
        backbone_name=model_names[backbone_idx],
        selected_model_indices=selected_indices,
        selected_models=[model_names[idx] for idx in selected_indices],
        loaded_model_indices=loaded_indices if mode != "legacy" else list(range(model_count)),
        loaded_models=[model_names[idx] for idx in (loaded_indices if mode != "legacy" else list(range(model_count)))],
        affected_blocks=affected_blocks,
        effective_locks=effective_locks,
        loras=[{"file": item["file"], "scale": item["scale"]} for item in normalized_loras],
        only_unet=normalized_only_unet,
        component_policy=normalized_component_policy,
        estimated_memory_gb=estimated_memory_gb,
        available_memory_gb=available_memory_gb,
        memory_check_ok=memory_ok,
        compatibility_warnings=compatibility_warnings,
        risk_alerts=risk_alerts,
        warnings=[issue.message for issue in result.warnings],
    )
    return result


def preflight_to_dict(plan: PreflightPlan) -> Dict[str, Any]:
    return asdict(plan)


def format_preflight_plan(plan: PreflightPlan) -> str:
    available = f"{plan.available_memory_gb:.2f} GB" if plan.available_memory_gb is not None else "unknown"
    component_policy = json.dumps(plan.component_policy, ensure_ascii=False) if plan.component_policy else "none"
    lines = [
        "Fusion preflight",
        "=" * 60,
        f"Mode: {plan.mode}",
        f"Backbone: [{plan.backbone_idx}] {plan.backbone_name}",
        f"Selected models: {', '.join(plan.selected_models) or 'none'}",
        f"Loaded models: {', '.join(plan.loaded_models) or 'none'}",
        f"Affected blocks: {', '.join(plan.affected_blocks) or 'none'}",
        f"Only UNet: {'yes' if plan.only_unet else 'no'}",
        f"Component policy: {component_policy}",
        f"Effective locks: {json.dumps(plan.effective_locks, ensure_ascii=False) if plan.effective_locks else 'none'}",
        f"LoRAs: {json.dumps(plan.loras, ensure_ascii=False) if plan.loras else 'none'}",
        f"Estimated memory: {plan.estimated_memory_gb:.2f} GB",
        f"Available memory: {available}",
        f"Memory check: {'OK' if plan.memory_check_ok else 'warning'}",
    ]

    if plan.compatibility_warnings:
        lines.append("")
        lines.append("Compatibility warnings:")
        for warning_text in plan.compatibility_warnings:
            lines.append(f"  - {warning_text}")

    if plan.risk_alerts:
        lines.append("")
        lines.append("Risk alerts:")
        for alert in plan.risk_alerts:
            lines.append(f"  - {alert}")

    extra_warnings = [
        warning_text
        for warning_text in plan.warnings
        if warning_text not in plan.compatibility_warnings and warning_text not in plan.risk_alerts
    ]
    if extra_warnings:
        lines.append("")
        lines.append("Other warnings:")
        for warning_text in extra_warnings:
            lines.append(f"  - {warning_text}")

    return "\n".join(lines)


def export_preflight_plan(plan: PreflightPlan, destination: Path) -> Path:
    destination = Path(destination)
    if destination.suffix.lower() == ".json":
        destination.write_text(json.dumps(preflight_to_dict(plan), indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        destination.write_text(format_preflight_plan(plan), encoding="utf-8")
    return destination
