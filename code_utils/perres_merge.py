"""
PerRes Merge Module for XLFusion V1.1

Resolution-based block assignment for precise control over different resolution tiers.
"""
import gc
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import torch
from .common import (
    UNET_PREFIX,
    load_state,
    get_block_assignment,
    is_cross_attn_key,
    get_attn2_block_type
)


# Functions imported from common module


def merge_perres(
    model_paths: List[Path],
    assignments: Dict[str, int],  # {"down_0_1": 0, "down_2_3": 1, ...}
    backbone_idx: int,
    attn2_locks: Optional[Dict[str, int]] = None  # {"down": 0, "mid": 1, "up": 2}
) -> Dict[str, torch.Tensor]:
    """
    PerRes mode fusion: 100% assignment by block groups.
    With optional cross-attention locks.
    """
    print("\nStarting PerRes fusion...")

    # Validate backbone index
    if backbone_idx < 0 or backbone_idx >= len(model_paths):
        print(f"Error: Backbone index {backbone_idx} is out of range (0-{len(model_paths)-1})")
        return {}

    # Load backbone for complete structure
    print(f"Loading backbone ({model_paths[backbone_idx].name})...")
    try:
        backbone_state = load_state(model_paths[backbone_idx])
    except Exception as e:
        print(f"Error loading backbone model: {e}")
        return {}

    # Load all needed models
    states = {}
    needed_indices = set(assignments.values())
    if attn2_locks:
        needed_indices.update(attn2_locks.values())

    # Validate indices are within range
    for idx in needed_indices:
        if idx < 0 or idx >= len(model_paths):
            print(f"Error: Model index {idx} is out of range (0-{len(model_paths)-1})")
            return {}

    for idx in needed_indices:
        if idx not in states:
            print(f"Loading model {idx} ({model_paths[idx].name})...")
            try:
                states[idx] = load_state(model_paths[idx])
            except Exception as e:
                print(f"Error loading model {idx}: {e}")
                # Clean up loaded states
                for loaded_idx in list(states.keys()):
                    del states[loaded_idx]
                gc.collect()
                raise

    # Build final state
    merged = {}

    # Statistics
    stats = {
        "down_0_1": 0, "down_2_3": 0, "mid": 0,
        "up_0_1": 0, "up_2_3": 0, "other": 0,
        "attn2_locks": 0
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

        # Normal block assignment
        block_group = get_block_assignment(key)

        if block_group and block_group in assignments:
            model_idx = assignments[block_group]
            if key in states[model_idx]:
                merged[key] = states[model_idx][key]
                stats[block_group] = stats.get(block_group, 0) + 1
            else:
                # Fallback to backbone if model doesn't have this key
                merged[key] = backbone_state[key]
        else:
            # Other UNet components, use backbone
            merged[key] = backbone_state[key]
            if block_group:
                stats[block_group] = stats.get(block_group, 0) + 1

    # Report statistics
    print("\nFusion completed:")
    print(f"  Down 0,1: {stats['down_0_1']} keys")
    print(f"  Down 2,3: {stats['down_2_3']} keys")
    print(f"  Mid:      {stats['mid']} keys")
    print(f"  Up 0,1:   {stats['up_0_1']} keys")
    print(f"  Up 2,3:   {stats['up_2_3']} keys")
    print(f"  Other:    {stats['other']} keys")
    if attn2_locks:
        print(f"  Attn2 locks: {stats['attn2_locks']} keys")

    # Free memory
    del states
    del backbone_state
    gc.collect()

    return merged


def prompt_perres_assignments(model_names: List[str]) -> Tuple[Dict[str, int], Optional[Dict[str, int]]]:
    """
    Prompts for PerRes assignments and attn2 locks
    Returns: (assignments, attn2_locks)
    """
    print("\n" + "="*60)
    print("PERRES CONFIGURATION - Resolution-based control")
    print("="*60)

    print("\nAvailable models:")
    for i, name in enumerate(model_names):
        print(f"  [{i}] {name}")

    assignments = {}

    # DOWN BLOCKS
    print("\n" + "-"*50)
    print("DOWN BLOCKS - Image encoding")
    print("-"*50)
    print("\nBlocks 0,1 (resolutions 64x, 32x)")
    print("  -> Control overall composition and basic shapes")
    print("  Recommended: model with best structural understanding")
    raw = input(f"Which model for down blocks 0,1? [0-{len(model_names)-1}]: ").strip()
    try:
        idx = int(raw)
        if 0 <= idx < len(model_names):
            assignments["down_0_1"] = idx
        else:
            print("Invalid index, using model 0")
            assignments["down_0_1"] = 0
    except:
        print("Invalid input, using model 0")
        assignments["down_0_1"] = 0

    print(f"\nBlocks 2,3 (resolutions 16x, 8x)")
    print("  -> Fine semantic details and prompt adherence")
    print("  Recommended: model with best prompt following")
    raw = input(f"Which model for down blocks 2,3? [0-{len(model_names)-1}]: ").strip()
    try:
        idx = int(raw)
        if 0 <= idx < len(model_names):
            assignments["down_2_3"] = idx
        else:
            print("Invalid index, using model 0")
            assignments["down_2_3"] = 0
    except:
        print("Invalid input, using model 0")
        assignments["down_2_3"] = 0

    # MID BLOCK
    print("\n" + "-"*50)
    print("MID BLOCK - Latent representation")
    print("-"*50)
    print("  -> Processes the most abstract information (8x latent)")
    print("  Recommended: most versatile or generalist model")
    raw = input(f"Which model for mid block? [0-{len(model_names)-1}]: ").strip()
    try:
        idx = int(raw)
        if 0 <= idx < len(model_names):
            assignments["mid"] = idx
        else:
            print("Invalid index, using model 0")
            assignments["mid"] = 0
    except:
        print("Invalid input, using model 0")
        assignments["mid"] = 0

    # UP BLOCKS
    print("\n" + "-"*50)
    print("UP BLOCKS - Reconstruction and style")
    print("-"*50)
    print("\nBlocks 0,1 (resolutions 8x, 16x)")
    print("  -> Begin reconstruction from abstract")
    print("  Recommended: model with good medium detail")
    raw = input(f"Which model for up blocks 0,1? [0-{len(model_names)-1}]: ").strip()
    try:
        idx = int(raw)
        if 0 <= idx < len(model_names):
            assignments["up_0_1"] = idx
        else:
            print("Invalid index, using model 0")
            assignments["up_0_1"] = 0
    except:
        print("Invalid input, using model 0")
        assignments["up_0_1"] = 0

    print(f"\nBlocks 2,3 (resolutions 32x, 64x)")
    print("  -> Define final style, textures and visual finish")
    print("  Recommended: model with best artistic style")
    raw = input(f"Which model for up blocks 2,3? [0-{len(model_names)-1}]: ").strip()
    try:
        idx = int(raw)
        if 0 <= idx < len(model_names):
            assignments["up_2_3"] = idx
        else:
            print("Invalid index, using model 0")
            assignments["up_2_3"] = 0
    except:
        print("Invalid input, using model 0")
        assignments["up_2_3"] = 0

    # ATTN2 LOCKS
    print("\n" + "="*60)
    print("CROSS-ATTENTION LOCKS (attn2)")
    print("="*60)
    print("Locks fix text attention layers to a specific model")
    print("This improves consistency and prompt adherence")

    raw = input("\nEnable cross-attention locks? [n]: ").strip().lower()
    attn2_locks = None

    if raw in ("y", "yes"):
        attn2_locks = {}

        print("\nAvailable models:")
        for i, name in enumerate(model_names):
            print(f"  [{i}] {name}")

        print("\nLock for DOWN blocks (attn2)")
        print("  Fixes prompt interpretation in encoding")
        raw = input(f"Model for attn2 in down? [0-{len(model_names)-1}]: ").strip()
        try:
            idx = int(raw)
            if 0 <= idx < len(model_names):
                attn2_locks["down"] = idx
            else:
                print("Invalid index, using model 0")
                attn2_locks["down"] = 0
        except:
            print("Invalid input, using model 0")
            attn2_locks["down"] = 0

        print("\nLock for MID block (attn2)")
        print("  Fixes attention in latent space")
        raw = input(f"Model for attn2 in mid? [0-{len(model_names)-1}]: ").strip()
        try:
            idx = int(raw)
            if 0 <= idx < len(model_names):
                attn2_locks["mid"] = idx
            else:
                print("Invalid index, using model 0")
                attn2_locks["mid"] = 0
        except:
            print("Invalid input, using model 0")
            attn2_locks["mid"] = 0

        print("\nLock for UP blocks (attn2)")
        print("  Fixes how text influences reconstruction")
        raw = input(f"Model for attn2 in up? [0-{len(model_names)-1}]: ").strip()
        try:
            idx = int(raw)
            if 0 <= idx < len(model_names):
                attn2_locks["up"] = idx
            else:
                print("Invalid index, using model 0")
                attn2_locks["up"] = 0
        except:
            print("Invalid input, using model 0")
            attn2_locks["up"] = 0

    # Summary
    print("\n" + "="*60)
    print("FINAL CONFIGURATION")
    print("="*60)
    print(f"Down 0,1: {model_names[assignments['down_0_1']]}")
    print(f"Down 2,3: {model_names[assignments['down_2_3']]}")
    print(f"Mid:      {model_names[assignments['mid']]}")
    print(f"Up 0,1:   {model_names[assignments['up_0_1']]}")
    print(f"Up 2,3:   {model_names[assignments['up_2_3']]}")

    if attn2_locks:
        print("\nAttn2 locks:")
        print(f"  Down: {model_names[attn2_locks['down']]}")
        print(f"  Mid:  {model_names[attn2_locks['mid']]}")
        print(f"  Up:   {model_names[attn2_locks['up']]}")

    return assignments, attn2_locks