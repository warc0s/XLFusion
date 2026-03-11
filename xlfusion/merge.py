"""
Merge utilities for XLFusion.
"""
from __future__ import annotations

import copy
import gc
import threading
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
from safetensors.torch import safe_open

from .blocks import (
    UNET_PREFIX,
    classify_component_key,
    get_attn2_block_type,
    get_block_assignment,
    group_for_key,
    is_cross_attn_key,
)
from .execution import ProgressReporter, build_processing_order, normalize_execution_options
from .memory import check_memory_availability, estimate_memory_requirement

NON_UNET_COMPONENTS = ("vae", "text_encoder", "other")
COMPONENT_POLICY_DEFAULTS = {
    "legacy": {"vae": "exclude", "text_encoder": "exclude", "other": "exclude"},
    "perres": {"vae": "backbone", "text_encoder": "backbone", "other": "backbone"},
    "hybrid": {"vae": "backbone", "text_encoder": "backbone", "other": "backbone"},
    "algebra": {"vae": "exclude", "text_encoder": "exclude", "other": "exclude"},
}


class MergeCancelled(Exception):
    """Signal to cancel an ongoing merge process."""


def _accumulation_dtype_for(tensor: torch.Tensor) -> torch.dtype:
    if tensor.dtype == torch.float64:
        return torch.float64
    return torch.float32


def _prepare_tensor_for_merge(tensor: torch.Tensor, device: str) -> Tuple[torch.Tensor, torch.dtype]:
    reference_dtype = tensor.dtype
    if device != "cpu":
        tensor = tensor.to(device)
    if tensor.dtype.is_floating_point:
        tensor = tensor.to(_accumulation_dtype_for(tensor))
    else:
        tensor = tensor.to(torch.float32)
    return tensor, reference_dtype


def _finalize_tensor_dtype(tensor: torch.Tensor, reference_dtype: torch.dtype) -> torch.Tensor:
    if tensor.dtype != reference_dtype:
        tensor = tensor.to(reference_dtype)
    return tensor


def _resolve_component_policy(
    mode: str,
    only_unet: bool = False,
    component_policy: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    if only_unet:
        return {component: "exclude" for component in NON_UNET_COMPONENTS}

    policy = dict(COMPONENT_POLICY_DEFAULTS.get(mode, COMPONENT_POLICY_DEFAULTS["legacy"]))
    for component, action in (component_policy or {}).items():
        if component in NON_UNET_COMPONENTS and isinstance(action, str):
            policy[component] = action
    return policy


def _should_copy_from_backbone(component: str, policy: Dict[str, str]) -> bool:
    return component != "unet" and policy.get(component) == "backbone"


def _should_merge_component(component: str, policy: Dict[str, str]) -> bool:
    return component == "unet" or policy.get(component) == "merge"


def _to_cpu_if_needed(tensor: torch.Tensor, device: str) -> torch.Tensor:
    return tensor.cpu() if device != "cpu" and tensor.device.type != "cpu" else tensor


def _update_stat(stats: Dict[str, int], key: str) -> None:
    stats[key] = stats.get(key, 0) + 1


def _copy_backbone_key(
    merged: Dict[str, torch.Tensor],
    key: str,
    base_handle: Any,
    base_keys: Iterable[str],
    *,
    device: str,
) -> bool:
    if key not in base_keys:
        return False
    merged[key] = _to_cpu_if_needed(base_handle.get_tensor(key).to(device), device)
    return True


def validate_hybrid_config(hybrid_config: Dict[str, Any], model_count: int) -> List[str]:
    """Validate hybrid configuration and return warnings."""
    config = copy.deepcopy(hybrid_config)
    warnings: List[str] = []
    errors: List[str] = []

    required_blocks = ["down_0_1", "down_2_3", "mid", "up_0_1", "up_2_3"]
    for block in required_blocks:
        if block not in config:
            errors.append(f"Missing required block: {block}")

    for block_name, weights in config.items():
        if not isinstance(weights, dict):
            errors.append(f"Block {block_name}: weights must be a dictionary, got {type(weights).__name__}")
            continue

        valid_weights: Dict[int, float] = {}
        for model_idx, weight in weights.items():
            try:
                idx = int(model_idx)
            except (TypeError, ValueError):
                errors.append(f"Block {block_name}: invalid model index '{model_idx}' (must be integer)")
                continue
            if idx < 0 or idx >= model_count:
                errors.append(f"Block {block_name}: model index {idx} out of range (0-{model_count - 1})")
                continue
            try:
                value = float(weight)
            except (TypeError, ValueError):
                errors.append(f"Block {block_name}: invalid weight '{weight}' for model {idx} (must be number)")
                continue
            if value < 0:
                errors.append(f"Block {block_name}: weight for model {idx} cannot be negative ({value})")
                continue
            valid_weights[idx] = value

        config[block_name] = valid_weights
        if not valid_weights:
            errors.append(f"Block {block_name}: no valid weights found")
            continue

        total_weight = sum(valid_weights.values())
        if total_weight < 0.95:
            warnings.append(f"Block {block_name}: weights sum to {total_weight:.3f} (expected ~1.0)")
        if total_weight == 0:
            warnings.append(f"Block {block_name}: all weights are zero - backbone fallback will be used")

    if errors:
        raise ValueError("Hybrid configuration validation failed:\n" + "\n".join(errors))
    return warnings


def _open_handles(stack: ExitStack, model_paths: List[Path], indices: Iterable[int], device: str) -> Tuple[Dict[int, Any], Dict[int, set[str]]]:
    handles: Dict[int, Any] = {}
    key_sets: Dict[int, set[str]] = {}
    for idx in sorted(set(indices)):
        if idx < 0 or idx >= len(model_paths):
            continue
        print(f"Opening model {idx} ({model_paths[idx].name})...")
        handle = stack.enter_context(safe_open(str(model_paths[idx]), framework="pt", device=device))
        handles[idx] = handle
        key_sets[idx] = set(handle.keys())
    return handles, key_sets


def _tick_progress(
    reporter: ProgressReporter,
    *,
    processed: List[int],
    cancel_event: Optional[threading.Event],
    cancel_every: int,
) -> None:
    processed[0] += 1
    reporter.step(1)
    if cancel_event and (processed[0] % max(1, cancel_every) == 0) and cancel_event.is_set():
        raise MergeCancelled("Merge cancelled by the user")


def merge_hybrid(
    model_paths: List[Path],
    hybrid_config: Dict[str, Dict[str, float]],
    backbone_idx: int,
    attn2_locks: Optional[Dict[str, int]] = None,
    *,
    only_unet: bool = False,
    component_policy: Optional[Dict[str, str]] = None,
    execution: Optional[Dict[str, object]] = None,
    progress_cb: Optional[Callable[[str, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    cancel_every: int = 1,
) -> Dict[str, torch.Tensor]:
    """Hybrid mode fusion with optional non-UNet backbone preservation."""
    execution_options = normalize_execution_options(execution)
    print(f"\nStarting Hybrid fusion ({execution_options.mode})...")

    warnings = validate_hybrid_config(hybrid_config, len(model_paths))
    if warnings:
        print("Configuration warnings:")
        for warning in warnings:
            print(f"  ⚠ {warning}")

    normalized_config: Dict[str, Dict[int, float]] = {}
    for block_name, weights in hybrid_config.items():
        normalized_config[block_name] = {int(model_idx): float(weight) for model_idx, weight in weights.items()}

    effective_component_policy = _resolve_component_policy("hybrid", only_unet, component_policy)

    needed_indices = {backbone_idx}
    for block_weights in normalized_config.values():
        needed_indices.update(block_weights.keys())
    if attn2_locks:
        needed_indices.update(attn2_locks.values())

    required_memory = estimate_memory_requirement(model_paths, needed_indices)
    if not check_memory_availability(required_memory):
        print(f"  ⚠ Warning: Estimated memory requirement {required_memory:.1f}GB may exceed available memory")

    stats: Dict[str, int] = {
        "down_0_1": 0,
        "down_2_3": 0,
        "mid": 0,
        "up_0_1": 0,
        "up_2_3": 0,
        "attn2_locks": 0,
        "non_unet_backbone": 0,
        "excluded_non_unet": 0,
    }
    merged: Dict[str, torch.Tensor] = {}

    with ExitStack() as stack:
        handles, key_sets = _open_handles(stack, model_paths, needed_indices, "cpu")
        if backbone_idx not in handles:
            raise IndexError(f"Backbone index {backbone_idx} unavailable among opened models")

        base_handle = handles[backbone_idx]
        base_keys = key_sets[backbone_idx]
        ordered_keys = build_processing_order(list(base_keys), key_sets.values(), sort_keys=execution_options.sort_keys)
        processed = [0]

        with ProgressReporter(len(ordered_keys), "Hybrid merge", execution_options, progress_cb=progress_cb) as reporter:
            for key in ordered_keys:
                component = classify_component_key(key)
                if component != "unet":
                    if _should_copy_from_backbone(component, effective_component_policy):
                        if _copy_backbone_key(merged, key, base_handle, base_keys, device="cpu"):
                            _update_stat(stats, "non_unet_backbone")
                    else:
                        _update_stat(stats, "excluded_non_unet")
                    _tick_progress(reporter, processed=processed, cancel_event=cancel_event, cancel_every=cancel_every)
                    continue

                if attn2_locks and is_cross_attn_key(key):
                    block_type = get_attn2_block_type(key)
                    if block_type and block_type in attn2_locks:
                        lock_idx = attn2_locks[block_type]
                        handle = handles.get(lock_idx)
                        if handle and key in key_sets.get(lock_idx, set()):
                            merged[key] = handle.get_tensor(key)
                            _update_stat(stats, "attn2_locks")
                            _tick_progress(reporter, processed=processed, cancel_event=cancel_event, cancel_every=cancel_every)
                            continue

                block_group = get_block_assignment(key)
                if block_group and block_group in normalized_config:
                    weighted_sum: Optional[torch.Tensor] = None
                    total_weight = 0.0
                    reference_dtype: Optional[torch.dtype] = None
                    for model_idx, weight in normalized_config[block_group].items():
                        handle = handles.get(model_idx)
                        if handle is None or key not in key_sets.get(model_idx, set()):
                            continue
                        tensor, tensor_dtype = _prepare_tensor_for_merge(handle.get_tensor(key), "cpu")
                        if reference_dtype is None:
                            reference_dtype = tensor_dtype
                        contribution = tensor * weight
                        weighted_sum = contribution if weighted_sum is None else weighted_sum + contribution
                        total_weight += weight

                    if weighted_sum is not None and total_weight > 0.0:
                        if abs(total_weight - 1.0) > 1e-6:
                            weighted_sum = weighted_sum / total_weight
                        if reference_dtype is not None:
                            weighted_sum = _finalize_tensor_dtype(weighted_sum, reference_dtype)
                        merged[key] = weighted_sum
                        _update_stat(stats, block_group)
                        _tick_progress(reporter, processed=processed, cancel_event=cancel_event, cancel_every=cancel_every)
                        continue

                if _copy_backbone_key(merged, key, base_handle, base_keys, device="cpu"):
                    _update_stat(stats, "other")
                _tick_progress(reporter, processed=processed, cancel_event=cancel_event, cancel_every=cancel_every)

    gc.collect()
    print("\nHybrid merge statistics:")
    for block, count in stats.items():
        if count > 0:
            print(f"  {block}: {count} tensors")
    return merged


def merge_perres(
    model_paths: List[Path],
    assignments: Dict[str, int],
    backbone_idx: int,
    attn2_locks: Optional[Dict[str, int]] = None,
    *,
    only_unet: bool = False,
    component_policy: Optional[Dict[str, str]] = None,
    execution: Optional[Dict[str, object]] = None,
    progress_cb: Optional[Callable[[str, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    cancel_every: int = 1,
) -> Dict[str, torch.Tensor]:
    """PerRes mode fusion with explicit non-UNet scope control."""
    execution_options = normalize_execution_options(execution)
    print(f"\nStarting PerRes fusion ({execution_options.mode})...")

    effective_component_policy = _resolve_component_policy("perres", only_unet, component_policy)

    needed_indices = set(assignments.values()) | {backbone_idx}
    if attn2_locks:
        needed_indices.update(attn2_locks.values())

    required_memory = estimate_memory_requirement(model_paths, needed_indices)
    if not check_memory_availability(required_memory):
        print(f"  ⚠ Warning: Estimated memory requirement {required_memory:.1f}GB may exceed available memory")

    stats: Dict[str, int] = {
        "down_0_1": 0,
        "down_2_3": 0,
        "mid": 0,
        "up_0_1": 0,
        "up_2_3": 0,
        "attn2_locks": 0,
        "non_unet_backbone": 0,
        "excluded_non_unet": 0,
    }
    merged: Dict[str, torch.Tensor] = {}

    with ExitStack() as stack:
        handles, key_sets = _open_handles(stack, model_paths, needed_indices, "cpu")
        if backbone_idx not in handles:
            raise IndexError(f"Backbone index {backbone_idx} unavailable among opened models")

        base_handle = handles[backbone_idx]
        base_keys = key_sets[backbone_idx]
        ordered_keys = build_processing_order(list(base_keys), key_sets.values(), sort_keys=execution_options.sort_keys)
        processed = [0]

        with ProgressReporter(len(ordered_keys), "PerRes merge", execution_options, progress_cb=progress_cb) as reporter:
            for key in ordered_keys:
                component = classify_component_key(key)
                if component != "unet":
                    if _should_copy_from_backbone(component, effective_component_policy):
                        if _copy_backbone_key(merged, key, base_handle, base_keys, device="cpu"):
                            _update_stat(stats, "non_unet_backbone")
                    else:
                        _update_stat(stats, "excluded_non_unet")
                    _tick_progress(reporter, processed=processed, cancel_event=cancel_event, cancel_every=cancel_every)
                    continue

                if attn2_locks and is_cross_attn_key(key):
                    block_type = get_attn2_block_type(key)
                    if block_type and block_type in attn2_locks:
                        lock_idx = attn2_locks[block_type]
                        handle = handles.get(lock_idx)
                        if handle and key in key_sets.get(lock_idx, set()):
                            merged[key] = handle.get_tensor(key)
                            _update_stat(stats, "attn2_locks")
                            _tick_progress(reporter, processed=processed, cancel_event=cancel_event, cancel_every=cancel_every)
                            continue

                block_group = get_block_assignment(key)
                if block_group and block_group in assignments:
                    model_idx = assignments[block_group]
                    handle = handles.get(model_idx)
                    if handle and key in key_sets.get(model_idx, set()):
                        merged[key] = handle.get_tensor(key)
                        _update_stat(stats, block_group)
                        _tick_progress(reporter, processed=processed, cancel_event=cancel_event, cancel_every=cancel_every)
                        continue

                if _copy_backbone_key(merged, key, base_handle, base_keys, device="cpu"):
                    _update_stat(stats, "other")
                _tick_progress(reporter, processed=processed, cancel_event=cancel_event, cancel_every=cancel_every)

    gc.collect()
    print("\nPerRes merge statistics:")
    for block, count in stats.items():
        if count > 0:
            print(f"  {block}: {count} tensors")
    return merged


def stream_weighted_merge_from_paths(
    model_paths: List[Path],
    weights: List[float],
    base_idx: int = 0,
    device: str = "cpu",
    *,
    only_unet: bool = True,
    component_policy: Optional[Dict[str, str]] = None,
    block_multipliers: Optional[List[Dict[str, float]]] = None,
    crossattn_boosts: Optional[List[Dict[str, float]]] = None,
    execution: Optional[Dict[str, object]] = None,
    progress_cb: Optional[Callable[[str, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    cancel_every: int = 1,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Legacy weighted merge with explicit component policy support."""
    execution_options = normalize_execution_options(execution)
    print(f"\nStarting Legacy weighted merge ({execution_options.mode})...")

    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Sum of weights cannot be zero")
    weights = [weight / total_weight for weight in weights]
    effective_component_policy = _resolve_component_policy("legacy", only_unet, component_policy)

    needed_indices = set(range(len(model_paths)))
    required_memory = estimate_memory_requirement(model_paths, needed_indices)
    if not check_memory_availability(required_memory):
        print(f"  ⚠ Warning: Estimated memory requirement {required_memory:.1f}GB may exceed available memory")

    key_sets: List[set[str]] = []
    stats: Dict[str, int] = {
        "down": 0,
        "mid": 0,
        "up": 0,
        "other": 0,
        "non_unet_backbone": 0,
        "excluded_non_unet": 0,
    }

    with ExitStack() as stack:
        handles = []
        for idx, path in enumerate(model_paths):
            print(f"Opening model {idx} ({path.name})...")
            handle = stack.enter_context(safe_open(str(path), framework="pt", device=device))
            handles.append(handle)
            key_sets.append(set(handle.keys()))

        if base_idx >= len(handles):
            raise IndexError(f"Base index {base_idx} is out of range for {len(handles)} models")

        base_handle = handles[base_idx]
        base_keys = key_sets[base_idx]
        ordered_keys = build_processing_order(list(base_keys), key_sets, sort_keys=execution_options.sort_keys)
        merged: Dict[str, torch.Tensor] = {}
        processed = [0]

        def process_key(key: str) -> None:
            component = classify_component_key(key)
            if component != "unet" and not _should_merge_component(component, effective_component_policy):
                if _should_copy_from_backbone(component, effective_component_policy):
                    if _copy_backbone_key(merged, key, base_handle, base_keys, device=device):
                        _update_stat(stats, "non_unet_backbone")
                else:
                    _update_stat(stats, "excluded_non_unet")
                return

            group = group_for_key(key) or "other"
            weighted_sum: Optional[torch.Tensor] = None
            active_weight = 0.0
            reference_dtype: Optional[torch.dtype] = None

            for idx, (handle, weight, keys) in enumerate(zip(handles, weights, key_sets)):
                if weight == 0.0 or key not in keys:
                    continue

                effective_weight = float(weight)
                if component == "unet" and block_multipliers and idx < len(block_multipliers):
                    multiplier = block_multipliers[idx].get(group, 1.0)
                    try:
                        effective_weight *= float(multiplier)
                    except Exception:
                        pass
                if component == "unet" and crossattn_boosts and is_cross_attn_key(key) and idx < len(crossattn_boosts):
                    boost = crossattn_boosts[idx].get(group, 1.0)
                    try:
                        effective_weight *= float(boost)
                    except Exception:
                        pass

                tensor, tensor_dtype = _prepare_tensor_for_merge(handle.get_tensor(key), device)
                if reference_dtype is None:
                    reference_dtype = tensor_dtype
                contribution = tensor * effective_weight
                weighted_sum = contribution if weighted_sum is None else weighted_sum + contribution
                active_weight += effective_weight

            if weighted_sum is not None and active_weight > 0.0:
                if abs(active_weight - 1.0) > 1e-6:
                    weighted_sum = weighted_sum / active_weight
                if reference_dtype is not None:
                    weighted_sum = _finalize_tensor_dtype(weighted_sum, reference_dtype)
                merged[key] = _to_cpu_if_needed(weighted_sum, device)
                _update_stat(stats, group)
                return

            if _copy_backbone_key(merged, key, base_handle, base_keys, device=device):
                _update_stat(stats, "other" if component == "unet" else "non_unet_backbone")

        with ProgressReporter(len(ordered_keys), "Legacy merge", execution_options, progress_cb=progress_cb) as reporter:
            for key in ordered_keys:
                process_key(key)
                _tick_progress(reporter, processed=processed, cancel_event=cancel_event, cancel_every=cancel_every)

    gc.collect()
    return merged, stats


def stream_checkpoint_algebra_from_paths(
    model_paths: List[Path],
    alpha: float,
    *,
    a_idx: int = 0,
    b_idx: int = 1,
    c_idx: int = 2,
    device: str = "cpu",
    only_unet: bool = True,
    component_policy: Optional[Dict[str, str]] = None,
    execution: Optional[Dict[str, object]] = None,
    progress_cb: Optional[Callable[[str, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    cancel_every: int = 1,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int], Dict[str, Any]]:
    """Apply checkpoint algebra A + alpha(B - C) in a streaming-safe way."""
    execution_options = normalize_execution_options(execution)
    print(f"\nStarting checkpoint algebra ({execution_options.mode})...")

    indices = {a_idx, b_idx, c_idx}
    required_memory = estimate_memory_requirement(model_paths, indices)
    if not check_memory_availability(required_memory):
        print(f"  ⚠ Warning: Estimated memory requirement {required_memory:.1f}GB may exceed available memory")

    effective_component_policy = _resolve_component_policy("algebra", only_unet, component_policy)
    stats: Dict[str, int] = {
        "formula_applied": 0,
        "backbone_fallback": 0,
        "shape_mismatch": 0,
        "missing_operand": 0,
        "non_unet_backbone": 0,
        "excluded_non_unet": 0,
    }
    audit: Dict[str, Any] = {
        "formula": "A + alpha(B - C)",
        "alpha": float(alpha),
        "indices": {"A": a_idx, "B": b_idx, "C": c_idx},
        "shape_mismatch_keys": [],
        "missing_operand_keys": [],
    }

    with ExitStack() as stack:
        handles, key_sets = _open_handles(stack, model_paths, indices, device)
        if a_idx not in handles or b_idx not in handles or c_idx not in handles:
            raise IndexError("Checkpoint algebra requires valid A, B and C indices")

        handle_a = handles[a_idx]
        handle_b = handles[b_idx]
        handle_c = handles[c_idx]
        keys_a = key_sets[a_idx]
        ordered_keys = build_processing_order(list(keys_a), (), sort_keys=execution_options.sort_keys)
        merged: Dict[str, torch.Tensor] = {}
        processed = [0]

        with ProgressReporter(len(ordered_keys), "Checkpoint algebra", execution_options, progress_cb=progress_cb) as reporter:
            for key in ordered_keys:
                component = classify_component_key(key)
                if component != "unet" and not _should_merge_component(component, effective_component_policy):
                    if _should_copy_from_backbone(component, effective_component_policy):
                        if _copy_backbone_key(merged, key, handle_a, keys_a, device=device):
                            _update_stat(stats, "non_unet_backbone")
                    else:
                        _update_stat(stats, "excluded_non_unet")
                    _tick_progress(reporter, processed=processed, cancel_event=cancel_event, cancel_every=cancel_every)
                    continue

                if key not in key_sets[b_idx] or key not in key_sets[c_idx]:
                    if _copy_backbone_key(merged, key, handle_a, keys_a, device=device):
                        _update_stat(stats, "backbone_fallback")
                        _update_stat(stats, "missing_operand")
                        if len(audit["missing_operand_keys"]) < 10:
                            audit["missing_operand_keys"].append(key)
                    _tick_progress(reporter, processed=processed, cancel_event=cancel_event, cancel_every=cancel_every)
                    continue

                tensor_a, dtype_a = _prepare_tensor_for_merge(handle_a.get_tensor(key), device)
                tensor_b, _ = _prepare_tensor_for_merge(handle_b.get_tensor(key), device)
                tensor_c, _ = _prepare_tensor_for_merge(handle_c.get_tensor(key), device)
                if tensor_a.shape != tensor_b.shape or tensor_a.shape != tensor_c.shape:
                    if _copy_backbone_key(merged, key, handle_a, keys_a, device=device):
                        _update_stat(stats, "backbone_fallback")
                        _update_stat(stats, "shape_mismatch")
                        if len(audit["shape_mismatch_keys"]) < 10:
                            audit["shape_mismatch_keys"].append(key)
                    _tick_progress(reporter, processed=processed, cancel_event=cancel_event, cancel_every=cancel_every)
                    continue

                combined = tensor_a + float(alpha) * (tensor_b - tensor_c)
                merged[key] = _to_cpu_if_needed(_finalize_tensor_dtype(combined, dtype_a), device)
                _update_stat(stats, "formula_applied")
                _tick_progress(reporter, processed=processed, cancel_event=cancel_event, cancel_every=cancel_every)

    gc.collect()
    audit["stats"] = dict(stats)
    return merged, stats, audit
