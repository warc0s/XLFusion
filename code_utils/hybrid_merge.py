"""
Hybrid Merge Module for XLFusion V1.1

Combines weighted merging (Legacy) with resolution-based control (PerRes).
Applies different weights to different resolution blocks for maximum flexibility.
"""
import gc
from pathlib import Path
from typing import Dict, List, Optional
import torch
from .common import (
    UNET_PREFIX,
    load_state,
    get_block_assignment,
    is_cross_attn_key,
    get_attn2_block_type,
    is_cross_attn_key_legacy
)


# Common functions imported from .common module


def merge_hybrid(
    model_paths: List[Path],
    weights: List[float],
    block_weights: Dict[str, List[float]],  # {"down_0_1": [0.7, 0.3], "down_2_3": [0.5, 0.5], ...}
    backbone_idx: int,
    cross_attention_boost: float = 1.0,
    attn2_locks: Optional[Dict[str, int]] = None
) -> Dict[str, torch.Tensor]:
    """
    Hybrid mode fusion: Combines weighted merging (Legacy) with resolution-based control (PerRes).
    Applies different weights to different resolution blocks for maximum flexibility.
    """
    print("\nStarting Hybrid fusion...")

    # Validate backbone index
    if backbone_idx < 0 or backbone_idx >= len(model_paths):
        print(f"Error: Backbone index {backbone_idx} is out of range (0-{len(model_paths)-1})")
        return {}

    # Load all models
    states = {}
    model_names = [p.name for p in model_paths]

    for i, path in enumerate(model_paths):
        print(f"Loading model {i} ({path.name})...")
        try:
            states[i] = load_state(path)
        except Exception as e:
            print(f"Error loading model {i}: {e}")
            print("Consider using fewer models or freeing memory.")
            # Clean up partially loaded states before failing
            for idx in list(states.keys()):
                del states[idx]
            gc.collect()
            raise

    # Start with backbone for complete structure
    merged = {}
    backbone_state = states[backbone_idx]

    # Statistics
    stats = {
        "down_0_1": 0, "down_2_3": 0, "mid": 0,
        "up_0_1": 0, "up_2_3": 0, "other": 0,
        "attn2_locks": 0, "cross_attn_boost": 0
    }

    for key in backbone_state.keys():
        # For non-UNet keys, use backbone
        if not key.startswith(UNET_PREFIX):
            merged[key] = backbone_state[key]
            continue

        # Check if it's a cross-attention key with lock
        if attn2_locks and is_cross_attn_key(key):
            block_type = get_attn2_block_type(key)
            if block_type and block_type in attn2_locks:
                lock_idx = attn2_locks[block_type]
                if key in states[lock_idx]:
                    merged[key] = states[lock_idx][key]
                    stats["attn2_locks"] += 1
                    continue

        # Get block assignment for this key
        block_group = get_block_assignment(key)

        if block_group and block_group in block_weights:
            # Apply hybrid weighted merge for this block group
            block_weight_list = block_weights[block_group]

            # Validate weight list length matches model count
            if len(block_weight_list) != len(model_paths):
                print(f"Warning: Block weights for {block_group} has {len(block_weight_list)} entries but {len(model_paths)} models provided")
                # Fallback to backbone
                merged[key] = backbone_state[key]
                continue

            # Ensure we have tensors from all models for this key
            tensors = []
            used_weights = []
            for i, weight in enumerate(block_weight_list):
                if weight > 0 and i < len(model_paths) and key in states[i]:
                    tensors.append(states[i][key])
                    used_weights.append(weight)

            if tensors:
                # Normalize weights
                total_weight = sum(used_weights)
                if total_weight > 0:
                    normalized_weights = [w / total_weight for w in used_weights]

                    # Weighted sum
                    result = tensors[0] * normalized_weights[0]
                    for j in range(1, len(tensors)):
                        result += tensors[j] * normalized_weights[j]

                    # Apply cross-attention boost if applicable
                    if cross_attention_boost != 1.0 and is_cross_attn_key_legacy(key):
                        result *= cross_attention_boost
                        stats["cross_attn_boost"] += 1

                    merged[key] = result
                    stats[block_group] = stats.get(block_group, 0) + 1
                else:
                    # Fallback to backbone
                    merged[key] = backbone_state[key]
            else:
                # Fallback to backbone
                merged[key] = backbone_state[key]
        else:
            # Other UNet components, use backbone
            merged[key] = backbone_state[key]
            if block_group:
                stats["other"] += 1

    # Report statistics
    print("\nHybrid fusion completed:")
    print(f"  Down 0,1: {stats['down_0_1']} keys")
    print(f"  Down 2,3: {stats['down_2_3']} keys")
    print(f"  Mid:      {stats['mid']} keys")
    print(f"  Up 0,1:   {stats['up_0_1']} keys")
    print(f"  Up 2,3:   {stats['up_2_3']} keys")
    print(f"  Other:    {stats['other']} keys")
    if attn2_locks:
        print(f"  Attn2 locks: {stats['attn2_locks']} keys")
    if cross_attention_boost != 1.0:
        print(f"  Cross-attn boost: {stats['cross_attn_boost']} keys")

    # Free memory
    del states
    gc.collect()

    return merged


def prompt_hybrid_weights(model_names: List[str]) -> Dict[str, List[float]]:
    """
    Prompt user for hybrid mode weights per resolution block.
    """
    print("\n" + "="*60)
    print("HYBRID MODE CONFIGURATION")
    print("="*60)
    print("Configure weights for each resolution block group.")
    print("Each block can have different weight distributions.")
    print("Sum of weights per block will be normalized to 1.0.")

    block_groups = ["down_0_1", "down_2_3", "mid", "up_0_1", "up_2_3"]
    block_descriptions = {
        "down_0_1": "Down 0,1 (64x, 32x) - Composition & Structure",
        "down_2_3": "Down 2,3 (16x, 8x) - Semantic Details",
        "mid": "Mid (8x) - Abstract Processing",
        "up_0_1": "Up 0,1 (8x, 16x) - Reconstruction",
        "up_2_3": "Up 2,3 (32x, 64x) - Final Style & Textures"
    }

    block_weights = {}

    # Default weights: equal distribution
    default_weight = 1.0 / len(model_names)

    for block in block_groups:
        print(f"\n{block_descriptions[block]}:")
        print("Models:")
        for i, name in enumerate(model_names):
            print(f"  [{i}] {name}")

        # Prompt for weights
        weights_str = input(f"Weights for {block} [{', '.join([f'{default_weight:.2f}'] * len(model_names))}]: ").strip()

        if not weights_str:
            # Use default equal weights
            weights = [default_weight] * len(model_names)
        else:
            try:
                weights = [float(x.strip()) for x in weights_str.split(",")]
                if len(weights) != len(model_names):
                    print(f"Warning: Expected {len(model_names)} weights, got {len(weights)}. Using defaults.")
                    weights = [default_weight] * len(model_names)
            except ValueError:
                print("Invalid input. Using default weights.")
                weights = [default_weight] * len(model_names)

        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [default_weight] * len(model_names)

        block_weights[block] = weights

        print(f"Normalized weights: {[f'{w:.3f}' for w in weights]}")

    return block_weights