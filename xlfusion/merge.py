"""
Merge utilities for XLFusion
Contains all fusion mode implementations
"""
import copy
import gc
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from contextlib import ExitStack
from safetensors.torch import safe_open
from tqdm import tqdm
import threading

from .blocks import (
    UNET_PREFIX,
    get_block_assignment,
    is_cross_attn_key,
    get_attn2_block_type,
    group_for_key,
)
from .memory import estimate_memory_requirement, check_memory_availability


class MergeCancelled(Exception):
    """Signal to cancel an ongoing merge process"""
    pass


def validate_hybrid_config(hybrid_config: Dict, model_count: int) -> List[str]:
    """Validate hybrid configuration and return warnings"""
    config = copy.deepcopy(hybrid_config)  # No mutar input
    warnings = []
    errors = []
    
    # Validate required blocks
    required_blocks = ['down_0_1', 'down_2_3', 'mid', 'up_0_1', 'up_2_3']
    for block in required_blocks:
        if block not in config:
            errors.append(f"Missing required block: {block}")
    
    for block_name, weights in config.items():
        # Validate that weights is a dictionary
        if not isinstance(weights, dict):
            errors.append(f"Block {block_name}: weights must be a dictionary, got {type(weights).__name__}")
            continue
        
        # Validate model indices and weight values
        valid_weights = {}
        for model_idx, weight in weights.items():
            # Try to convert model_idx to integer
            try:
                idx = int(model_idx)
                if idx < 0 or idx >= model_count:
                    errors.append(f"Block {block_name}: model index {idx} out of range (0-{model_count-1})")
                    continue
            except (ValueError, TypeError):
                errors.append(f"Block {block_name}: invalid model index '{model_idx}' (must be integer)")
                continue
            
            # Try to convert weight to float
            try:
                w = float(weight)
                if w < 0:
                    errors.append(f"Block {block_name}: weight for model {idx} cannot be negative ({w})")
                    continue
                valid_weights[idx] = w
            except (ValueError, TypeError):
                errors.append(f"Block {block_name}: invalid weight '{weight}' for model {idx} (must be number)")
                continue
        
        # Replace with validated weights
        config[block_name] = valid_weights
        
        # Check weight sum
        if valid_weights:
            total_weight = sum(valid_weights.values())
            if total_weight < 0.95:  # Allow for floating point precision
                warnings.append(f"Block {block_name}: weights sum to {total_weight:.3f} (expected ~1.0)")
            if total_weight == 0:
                warnings.append(f"Block {block_name}: all weights are zero - will use backbone fallback")
        else:
            errors.append(f"Block {block_name}: no valid weights found")
    
    # Combine errors and warnings, raise exception for errors
    if errors:
        raise ValueError("Hybrid configuration validation failed:\n" + "\n".join(errors))
    
    return warnings


def merge_hybrid(
    model_paths: List[Path],
    hybrid_config: Dict[str, Dict[str, float]],  # {"down_0_1": {0: 0.7, 1: 0.3}, ...}
    backbone_idx: int,
    attn2_locks: Optional[Dict[str, int]] = None,  # {"down": 0, "mid": 1, "up": 2}
    *,
    progress_cb: Optional[Callable[[str, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    cancel_every: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    Hybrid mode fusion: Combines PerRes assignment with weighted blending.

    For each block group:
    - Primary model gets the assigned weight
    - Secondary models share remaining weight
    - Cross-attention locks supported
    """
    print("\nStarting Hybrid fusion...")

    warnings = validate_hybrid_config(hybrid_config, len(model_paths))
    if warnings:
        print("Configuration warnings:")
        for warning in warnings:
            print(f"  ⚠ {warning}")

    # Normalise configuration for consistent processing
    normalized_config: Dict[str, Dict[int, float]] = {}
    for block_name, weights in hybrid_config.items():
        normalized_weights: Dict[int, float] = {}
        for model_idx, weight in weights.items():
            try:
                idx = int(model_idx)
                normalized_weights[idx] = float(weight)
            except (TypeError, ValueError):
                continue
        normalized_config[block_name] = normalized_weights

    # Validate cross-attention lock consistency
    if attn2_locks is not None:
        for block_type, lock_idx in attn2_locks.items():
            lock_used = False
            for block_weights in normalized_config.values():
                if lock_idx in block_weights:
                    lock_used = True
                    break
            if not lock_used:
                print(f"  ⚠ Cross-attention lock model {lock_idx} ({model_paths[lock_idx].name}) is not used in any block weights")

    # Memory check
    needed_indices = set()
    for block_weights in normalized_config.values():
        needed_indices.update(block_weights.keys())
    if attn2_locks:
        needed_indices.update(attn2_locks.values())
    needed_indices.add(backbone_idx)

    required_memory = estimate_memory_requirement(model_paths, needed_indices)
    if not check_memory_availability(required_memory):
        print(f"  ⚠ Warning: Estimated memory requirement {required_memory:.1f}GB may exceed available memory")

    def to_cpu(t: torch.Tensor) -> torch.Tensor:
        return t.cpu() if t.device.type != "cpu" else t

    stats = {
        "down_0_1": 0, "down_2_3": 0, "mid": 0,
        "up_0_1": 0, "up_2_3": 0, "other": 0,
        "attn2_locks": 0
    }

    merged: Dict[str, torch.Tensor] = {}

    with ExitStack() as stack:
        handles: Dict[int, any] = {}
        key_sets: Dict[int, set] = {}

        for idx in sorted(needed_indices):
            if idx >= len(model_paths):
                continue
            print(f"Opening model {idx} ({model_paths[idx].name})...")
            handle = stack.enter_context(safe_open(str(model_paths[idx]), framework="pt", device="cpu"))
            handles[idx] = handle
            key_sets[idx] = set(handle.keys())

        if backbone_idx not in handles:
            raise IndexError(f"Backbone index {backbone_idx} unavailable among opened models")

        base_handle = handles[backbone_idx]
        base_keys = list(base_handle.keys())
        if progress_cb:
            try:
                progress_cb("total", len(base_keys))
            except Exception:
                pass

        processed = 0

        def _tick() -> None:
            nonlocal processed
            processed += 1
            if progress_cb:
                try:
                    progress_cb("tick", 1)
                except Exception:
                    pass
            if cancel_event and (processed % max(1, cancel_every) == 0) and cancel_event.is_set():
                raise MergeCancelled("Merge cancelled by the user")

        for key in tqdm(base_keys, desc="Hybrid merge", unit="tensor"):
            if not key.startswith(UNET_PREFIX):
                merged[key] = to_cpu(base_handle.get_tensor(key))
                _tick()
                continue

            if attn2_locks and is_cross_attn_key(key):
                block_type = get_attn2_block_type(key)
                if block_type and block_type in attn2_locks:
                    lock_idx = attn2_locks[block_type]
                    handle = handles.get(lock_idx)
                    if handle and key in key_sets.get(lock_idx, set()):
                        merged[key] = to_cpu(handle.get_tensor(key))
                        stats["attn2_locks"] += 1
                        _tick()
                        continue

            block_group = get_block_assignment(key)

            if block_group and block_group in normalized_config:
                weighted_sum: Optional[torch.Tensor] = None
                total_weight = 0.0
                for model_idx, weight in normalized_config[block_group].items():
                    handle = handles.get(model_idx)
                    if handle is None or key not in key_sets.get(model_idx, set()):
                        continue
                    tensor = handle.get_tensor(key)
                    contribution = tensor * weight
                    if weighted_sum is None:
                        weighted_sum = contribution
                    else:
                        weighted_sum += contribution
                    total_weight += weight

                if weighted_sum is not None and total_weight > 0.0:
                    if abs(total_weight - 1.0) > 1e-6:
                        weighted_sum = weighted_sum / total_weight
                    merged[key] = to_cpu(weighted_sum)
                    stats[block_group] += 1
                    _tick()
                    continue

            merged[key] = to_cpu(base_handle.get_tensor(key))
            stats["other"] += 1
            _tick()

    gc.collect()

    print("\nHybrid merge statistics:")
    for block, count in stats.items():
        if count > 0:
            print(f"  {block}: {count} tensors")

    return merged


def merge_perres(
    model_paths: List[Path],
    assignments: Dict[str, int],  # {"down_0_1": 0, "down_2_3": 1, "mid": 0, "up_0_1": 1, "up_2_3": 0}
    backbone_idx: int,
    attn2_locks: Optional[Dict[str, int]] = None,  # {"down": 0, "mid": 1, "up": 2}
    *,
    progress_cb: Optional[Callable[[str, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    cancel_every: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    PerRes mode fusion: 100% assignment by block pairs based on resolution.
    """
    print("\nStarting PerRes fusion...")

    # Memory check
    needed_indices = set(assignments.values())
    if attn2_locks:
        needed_indices.update(attn2_locks.values())
    needed_indices.add(backbone_idx)

    required_memory = estimate_memory_requirement(model_paths, needed_indices)
    if not check_memory_availability(required_memory):
        print(f"  ⚠ Warning: Estimated memory requirement {required_memory:.1f}GB may exceed available memory")

    def to_cpu(t: torch.Tensor) -> torch.Tensor:
        return t.cpu() if t.device.type != "cpu" else t

    stats = {
        "down_0_1": 0, "down_2_3": 0, "mid": 0,
        "up_0_1": 0, "up_2_3": 0, "other": 0,
        "attn2_locks": 0
    }

    merged: Dict[str, torch.Tensor] = {}

    with ExitStack() as stack:
        handles: Dict[int, any] = {}
        key_sets: Dict[int, set] = {}

        for idx in sorted(needed_indices):
            if idx >= len(model_paths):
                continue
            print(f"Opening model {idx} ({model_paths[idx].name})...")
            handle = stack.enter_context(safe_open(str(model_paths[idx]), framework="pt", device="cpu"))
            handles[idx] = handle
            key_sets[idx] = set(handle.keys())

        if backbone_idx not in handles:
            raise IndexError(f"Backbone index {backbone_idx} unavailable among opened models")

        base_handle = handles[backbone_idx]
        base_keys = list(base_handle.keys())
        if progress_cb:
            try:
                progress_cb("total", len(base_keys))
            except Exception:
                pass

        processed = 0

        def _tick() -> None:
            nonlocal processed
            processed += 1
            if progress_cb:
                try:
                    progress_cb("tick", 1)
                except Exception:
                    pass
            if cancel_event and (processed % max(1, cancel_every) == 0) and cancel_event.is_set():
                raise MergeCancelled("Merge cancelled by the user")

        for key in tqdm(base_keys, desc="PerRes merge", unit="tensor"):
            if not key.startswith(UNET_PREFIX):
                merged[key] = to_cpu(base_handle.get_tensor(key))
                _tick()
                continue

            if attn2_locks and is_cross_attn_key(key):
                block_type = get_attn2_block_type(key)
                if block_type and block_type in attn2_locks:
                    lock_idx = attn2_locks[block_type]
                    handle = handles.get(lock_idx)
                    if handle and key in key_sets.get(lock_idx, set()):
                        merged[key] = to_cpu(handle.get_tensor(key))
                        stats["attn2_locks"] += 1
                        _tick()
                        continue

            block_group = get_block_assignment(key)

            if block_group and block_group in assignments:
                model_idx = assignments[block_group]
                handle = handles.get(model_idx)
                if handle and key in key_sets.get(model_idx, set()):
                    merged[key] = to_cpu(handle.get_tensor(key))
                    stats[block_group] += 1
                    _tick()
                    continue

            merged[key] = to_cpu(base_handle.get_tensor(key))
            stats["other"] += 1
            _tick()

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
    block_multipliers: Optional[List[Dict[str, float]]] = None,  # per-model {down, mid, up}
    crossattn_boosts: Optional[List[Dict[str, float]]] = None,   # per-model {down, mid, up}
    progress_cb: Optional[Callable[[str, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    cancel_every: int = 1,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """
    Legacy weighted merge with streaming to reduce memory usage.
    Supports optional per-block multipliers and cross-attention boosts.

    Returns (merged_state, stats)
    """
    print("\nStarting Legacy weighted merge...")
    
    # Normalize weights
    total = sum(weights)
    if total == 0:
        raise ValueError("Sum of weights cannot be zero")
    weights = [w / total for w in weights]

    # Memory check
    needed_indices = set(range(len(model_paths)))
    required_memory = estimate_memory_requirement(model_paths, needed_indices)
    if not check_memory_availability(required_memory):
        print(f"  ⚠ Warning: Estimated memory requirement {required_memory:.1f}GB may exceed available memory")

    def to_cpu_if_needed(t: torch.Tensor) -> torch.Tensor:
        return t.cpu() if device != "cpu" and t.device.type != "cpu" else t

    print(f"Opening models for streaming merge ({len(model_paths)} total)...")

    key_sets: List[set] = []
    stats = {"down": 0, "mid": 0, "up": 0, "other": 0}

    with ExitStack() as stack:
        # Open all safetensor files lazily to avoid full memory load
        handles = []
        for idx, path in enumerate(model_paths):
            print(f"Opening model {idx} ({path.name})...")
            handle = stack.enter_context(safe_open(str(path), framework="pt", device=device))
            handles.append(handle)
            key_sets.append(set(handle.keys()))

        if base_idx >= len(handles):
            raise IndexError(f"Base index {base_idx} is out of range for {len(handles)} models")

        base_keys = list(handles[base_idx].keys())
        if progress_cb:
            try:
                progress_cb("total", len(base_keys))
            except Exception:
                pass
        merged: Dict[str, torch.Tensor] = {}
        seen = set()

        processed = 0

        def process_key(key: str) -> None:
            # Skip non-UNet keys if requested
            if only_unet and not key.startswith(UNET_PREFIX):
                return

            # Determine block group for multipliers/stats
            group = group_for_key(key) or "other"

            weighted_sum: Optional[torch.Tensor] = None
            active_weight = 0.0

            for i, (handle, weight, keys) in enumerate(zip(handles, weights, key_sets)):
                if weight == 0.0 or key not in keys:
                    continue

                eff_weight = float(weight)

                # Apply per-block multipliers per model if provided
                if block_multipliers and i < len(block_multipliers):
                    mult = block_multipliers[i].get(group, 1.0)
                    try:
                        eff_weight *= float(mult)
                    except Exception:
                        pass

                # Apply cross-attention boosts only for attn2 keys
                if crossattn_boosts and is_cross_attn_key(key) and i < len(crossattn_boosts):
                    boost = crossattn_boosts[i].get(group, 1.0)
                    try:
                        eff_weight *= float(boost)
                    except Exception:
                        pass

                tensor = handle.get_tensor(key)
                tensor = tensor.to(device)
                contribution = tensor * eff_weight
                if weighted_sum is None:
                    weighted_sum = contribution
                else:
                    weighted_sum += contribution
                active_weight += eff_weight

            if weighted_sum is not None and active_weight > 0.0:
                if abs(active_weight - 1.0) > 1e-6:
                    weighted_sum = weighted_sum / active_weight
                merged[key] = to_cpu_if_needed(weighted_sum)
                stats[group] = stats.get(group, 0) + 1
            else:
                if key in key_sets[base_idx]:
                    fallback = handles[base_idx].get_tensor(key)
                    merged[key] = to_cpu_if_needed(fallback.to(device))
                    stats[group] = stats.get(group, 0) + 1
            if progress_cb:
                try:
                    progress_cb("tick", 1)
                except Exception:
                    pass

        for key in tqdm(base_keys, desc="Legacy merge", unit="tensor"):
            process_key(key)
            processed += 1
            if cancel_event and (processed % max(1, cancel_every) == 0) and cancel_event.is_set():
                raise MergeCancelled("Merge cancelled by the user")
            seen.add(key)

        # Include keys that exist only in non-base models
        for keys in key_sets:
            for key in keys:
                if key not in seen:
                    process_key(key)
                    seen.add(key)

    gc.collect()
    return merged, stats
