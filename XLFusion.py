#!/usr/bin/env python3
"""
XLFusion V1.1

Interactive SDXL checkpoint merger with three fusion modes:
- Legacy: Classic weighted merge with down/mid/up blocks
- PerRes: Resolution-based block control
- Hybrid: Combined weighted + resolution control (V1.1)

PerRes (per-resolution) mode:
- 100% assignment by block pairs based on resolution
- Down 0,1 (64x, 32x) and 2,3 (16x, 8x)
- Mid (8x latent)
- Up 0,1 (8x, 16x) and 2,3 (32x, 64x)
- Optional cross-attention (attn2) locks

Expected folder structure:
  ./models     -> base checkpoints .safetensors
  ./loras      -> LoRA files .safetensors (legacy mode only)
  ./output     -> merged outputs .safetensors
  ./metadata   -> audit logs meta_*.txt

Requirements:
  pip install torch safetensors
"""
from __future__ import annotations
import sys
import re
import gc
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime

import torch
import yaml
from safetensors.torch import load_file as st_load, save_file as st_save

# SDXL A1111 style naming constants
UNET_PREFIX = "model.diffusion_model."

# Configuration loader
def load_config() -> dict:
    """Load configuration from config.yaml"""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Fallback configuration if config.yaml doesn't exist
        return {
            "model_output": {
                "base_name": "XLFusion",
                "version_prefix": "V",
                "file_extension": ".safetensors",
                "output_dir": "output",
                "metadata_dir": "metadata",
                "auto_version": True
            },
            "directories": {
                "models": "models",
                "loras": "loras",
                "output": "output",
                "metadata": "metadata"
            },
            "merge_defaults": {
                "legacy": {
                    "cross_attention_boost": 1.0,
                    "down_blocks_multiplier": 1.0,
                    "mid_blocks_multiplier": 1.0,
                    "up_blocks_multiplier": 1.0
                },
                "perres": {
                    "cross_attention_locks": False
                },
                "hybrid": {
                    "cross_attention_boost": 1.0,
                    "cross_attention_locks": False
                }
            },
            "app": {
                "tool_name": "XLFusion",
                "version": "1.1"
            }
        }

# ---------------- I/O Utilities ----------------

def ensure_dirs(root: Path) -> Tuple[Path, Path, Path, Path]:
    config = load_config()
    dirs = config["directories"]

    models = root / dirs["models"]
    loras = root / dirs["loras"]
    output = root / dirs["output"]
    metadata = root / dirs["metadata"]
    for p in [models, loras, output, metadata]:
        p.mkdir(parents=True, exist_ok=True)
    return models, loras, output, metadata


def list_safetensors(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("*.safetensors") if p.is_file()])


def next_version_path(output_dir: Path) -> Tuple[Path, int]:
    config = load_config()
    output_cfg = config["model_output"]

    base_name = output_cfg["base_name"]
    version_prefix = output_cfg["version_prefix"]
    file_extension = output_cfg["file_extension"]

    pattern = re.compile(fr"{re.escape(base_name)}_{re.escape(version_prefix)}(\d+){re.escape(file_extension)}$")
    max_v = 0

    for p in output_dir.glob(f"{base_name}_{version_prefix}*{file_extension}"):
        m = pattern.search(p.name)
        if m:
            try:
                v = int(m.group(1))
                max_v = max(max_v, v)
            except ValueError:
                pass

    next_v = max_v + 1
    return output_dir / f"{base_name}_{version_prefix}{next_v}{file_extension}", next_v


# ---------------- Loading and saving ----------------

def load_state(path: Path) -> Dict[str, torch.Tensor]:
    state = st_load(str(path), device="cpu")
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if v.dtype in (torch.float16, torch.bfloat16):
            out[k] = v.to(torch.float32)
        else:
            out[k] = v
    return out


def save_state(path: Path, state: Dict[str, torch.Tensor], meta: Dict[str, str]) -> None:
    compact: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if v.dtype == torch.float32 and v.dim() >= 2:
            compact[k] = v.to(torch.float16)
        else:
            compact[k] = v
    st_save(compact, str(path), metadata=meta)


# ---------------- Block detection and resolutions ----------------

def get_block_assignment(key: str) -> Optional[str]:
    """
    Determines which block group a key belongs to.
    Returns: "down_0_1", "down_2_3", "mid", "up_0_1", "up_2_3", None
    """
    if not key.startswith(UNET_PREFIX):
        return None

    s = key[len(UNET_PREFIX):]

    # Down blocks
    if "down_blocks.0." in s or "down_blocks.1." in s:
        return "down_0_1"
    if "down_blocks.2." in s or "down_blocks.3." in s:
        return "down_2_3"

    # Mid block
    if "mid_block." in s or "middle_block." in s:
        return "mid"

    # Up blocks
    if "up_blocks.0." in s or "up_blocks.1." in s:
        return "up_0_1"
    if "up_blocks.2." in s or "up_blocks.3." in s:
        return "up_2_3"

    # Other UNet components (time embed, conv in/out, etc)
    return "other"


def is_cross_attn_key(key: str) -> bool:
    """Detects if a key is cross-attention (attn2)"""
    if not key.startswith(UNET_PREFIX):
        return False

    # Search for cross-attention patterns
    attn2_patterns = [
        ".attn2.to_q.",
        ".attn2.to_k.",
        ".attn2.to_v.",
        ".attn2.to_out.0.",
    ]

    return any(pattern in key for pattern in attn2_patterns)


def get_attn2_block_type(key: str) -> Optional[str]:
    """Determines if an attn2 key is in down, mid or up"""
    if not is_cross_attn_key(key):
        return None

    s = key[len(UNET_PREFIX):]

    if "down_blocks." in s:
        return "down"
    elif "mid_block." in s or "middle_block." in s:
        return "mid"
    elif "up_blocks." in s:
        return "up"

    return None


# ---------------- PerRes Mode (Per-Resolution) ----------------

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

    # Load backbone for complete structure
    print(f"Loading backbone ({model_paths[backbone_idx].name})...")
    backbone_state = load_state(model_paths[backbone_idx])

    # Load all needed models
    states = {}
    needed_indices = set(assignments.values())
    if attn2_locks:
        needed_indices.update(attn2_locks.values())

    for idx in needed_indices:
        if idx not in states:
            print(f"Loading model {idx} ({model_paths[idx].name})...")
            states[idx] = load_state(model_paths[idx])

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


# ---------------- Hybrid Mode (V1.1) ----------------

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


# ---------------- Legacy Merge (original mode) ----------------

def should_merge_key(k: str, only_unet: bool = True) -> bool:
    return k.startswith(UNET_PREFIX) if only_unet else True


def group_for_key(k: str) -> Optional[str]:
    if not k.startswith(UNET_PREFIX):
        return None
    if f"{UNET_PREFIX}down_blocks." in k or f"{UNET_PREFIX}input_blocks." in k:
        return "down"
    if f"{UNET_PREFIX}mid_block." in k or f"{UNET_PREFIX}middle_block." in k:
        return "mid"
    if f"{UNET_PREFIX}up_blocks." in k or f"{UNET_PREFIX}output_blocks." in k:
        return "up"
    return "other"

CROSS_TOKENS = (".attn2.",)
CROSS_PROJ = (".to_q.", ".to_k.", ".to_v.", ".to_out.0.")

def is_cross_attn_key_legacy(k: str) -> bool:
    if not k.startswith(UNET_PREFIX):
        return False
    s = k[len(UNET_PREFIX):]
    if not any(tok in s for tok in CROSS_TOKENS):
        return False
    return any(proj in s for proj in CROSS_PROJ)


def stream_weighted_merge_from_paths(
    model_paths: List[Path],
    weights: List[float],
    backbone_idx: int,
    only_unet: bool = True,
    block_multipliers: Optional[List[Dict[str, float]]] = None,
    crossattn_boosts: Optional[List[Dict[str, float]]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Legacy fusion with memory containment"""
    assert len(model_paths) == len(weights) and len(model_paths) > 0
    if block_multipliers is None:
        block_multipliers = [dict(down=1.0, mid=1.0, up=1.0, other=1.0) for _ in model_paths]
    if crossattn_boosts is None:
        crossattn_boosts = [dict(down=1.0, mid=1.0, up=1.0) for _ in model_paths]

    print("Loading backbone for CLIP and VAE...")
    backbone_state = load_state(model_paths[backbone_idx])

    acc_sum: Dict[str, torch.Tensor] = {}
    acc_w: Dict[str, float] = {}
    acc_shape: Dict[str, torch.Size] = {}

    for i, mp in enumerate(model_paths):
        print(f"  Merging model {i}: {mp.name}")
        s = load_state(mp)
        w_global = float(weights[i])
        mults = block_multipliers[i]
        boosts = crossattn_boosts[i]
        for k, t in s.items():
            if only_unet and not should_merge_key(k, True):
                continue
            grp = group_for_key(k) or "other"
            eff = w_global * float(mults.get(grp, 1.0))
            if is_cross_attn_key_legacy(k):
                eff *= float(boosts.get(grp, 1.0))
            if eff <= 0:
                continue
            if k not in acc_sum:
                acc_sum[k] = eff * t.to(torch.float32)
                acc_w[k] = eff
                acc_shape[k] = t.shape
            else:
                if t.shape != acc_shape[k]:
                    continue
                acc_sum[k].add_(eff * t.to(torch.float32))
                acc_w[k] += eff
        del s
        gc.collect()

    merged: Dict[str, torch.Tensor] = {}
    for k, ssum in acc_sum.items():
        wsum = acc_w[k]
        if wsum > 0:
            merged[k] = (ssum / wsum).to(torch.float32)

    for k, v in backbone_state.items():
        if only_unet and should_merge_key(k, True):
            if k not in merged:
                merged[k] = v
        else:
            merged[k] = v

    return merged, backbone_state


# ---------------- LoRA baking (legacy only) ----------------
DOWN_PAT = re.compile(r"\.lora_down\.weight$")
UP_PAT = re.compile(r"\.lora_up\.weight$")
ALPHA_KEYS = ["alpha", "lora_alpha", "ss_network_alpha", "scale"]


def lora_pairs_from_state(lora_state: Dict[str, torch.Tensor]) -> List[Tuple[str, str]]:
    downs = [k for k in lora_state.keys() if DOWN_PAT.search(k)]
    pairs: List[Tuple[str, str]] = []
    for d in downs:
        u = d.replace(".lora_down.weight", ".lora_up.weight")
        if u in lora_state:
            pairs.append((d, u))
    return pairs


def parse_lora_alpha_rank(meta: Dict[str, str], down: torch.Tensor) -> Tuple[float, int]:
    rank = int(down.shape[0]) if down.dim() >= 2 else int(down.numel())
    alpha = None
    for k in ALPHA_KEYS:
        if k in meta:
            try:
                alpha = float(meta[k])
                break
            except Exception:
                pass
    if alpha is None or alpha <= 0:
        alpha = float(rank)
    return alpha, rank


def map_lora_key_to_base(d_key: str) -> Optional[str]:
    if not d_key.startswith("lora_unet_"):
        return None
    base = d_key[len("lora_unet_"):]
    base = base.replace("_", ".")
    base = base.replace(".to.", ".to_")
    base = base.replace(".lora_down.weight", ".weight").replace(".lora_up.weight", ".weight")
    base = base.replace(".to_q.", ".to_q.").replace(".to_k.", ".to_k.").replace(".to_v.", ".to_v.")
    base = base.replace(".to_out.0.", ".to_out.0.")
    return UNET_PREFIX + base


def apply_single_lora(
    base_state: Dict[str, torch.Tensor],
    lora_path: Path,
    scale: float,
) -> Tuple[int, int]:
    lora_state = st_load(str(lora_path), device="cpu")
    meta = getattr(lora_state, "metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    pairs = lora_pairs_from_state(lora_state)
    applied = 0
    skipped = 0

    def as_2d_if_1x1(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 4 and t.shape[2] == 1 and t.shape[3] == 1:
            return t.view(t.shape[0], t.shape[1])
        return t

    for d_key, u_key in pairs:
        if not d_key.startswith("lora_unet_"):
            skipped += 1
            continue
        down = lora_state[d_key].to(torch.float32)
        up = lora_state[u_key].to(torch.float32)
        alpha, rank = parse_lora_alpha_rank(meta, down)
        scale_eff = scale * (alpha / max(rank, 1))
        base_w_key = map_lora_key_to_base(d_key)
        if base_w_key is None or base_w_key not in base_state:
            skipped += 1
            continue
        W = base_state[base_w_key].to(torch.float32)

        delta = None
        if W.dim() == 2:
            down2 = as_2d_if_1x1(down)
            up2 = as_2d_if_1x1(up)
            if down2.dim() == 2 and up2.dim() == 2 and down2.shape[0] == up2.shape[1]:
                delta = torch.matmul(up2, down2) * scale_eff
        elif W.dim() == 4:
            kh, kw = W.shape[2], W.shape[3]
            d_is_1x1 = down.dim() == 4 and down.shape[2] == 1 and down.shape[3] == 1
            u_is_1x1 = up.dim() == 4 and up.shape[2] == 1 and up.shape[3] == 1
            if down.dim() == 2 and up.dim() == 2:
                delta2d = torch.matmul(up, down) * scale_eff
                delta = delta2d.view(W.shape[0], W.shape[1], 1, 1)
            elif down.dim() == 4 and not d_is_1x1 and (up.dim() == 2 or u_is_1x1):
                up2 = up.view(up.shape[0], up.shape[1]) if up.dim() == 4 else up
                delta = torch.einsum('or,rijk->oijk', up2, down) * scale_eff
            elif up.dim() == 4 and not u_is_1x1 and (down.dim() == 2 or d_is_1x1):
                down2 = down.view(down.shape[0], down.shape[1]) if down.dim() == 4 else down
                delta = torch.einsum('orhw,ri->oihw', up, down2) * scale_eff
            else:
                delta = None
        if delta is None or delta.shape != W.shape:
            skipped += 1
            continue
        base_state[base_w_key] = (W + delta).to(torch.float32)
        applied += 1
    return applied, skipped


# ---------------- Interactive CLI ----------------

def prompt_select(items: List[Path], title: str, default_idx: List[int]) -> List[int]:
    print(f"\n{title}")
    for i, p in enumerate(items):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  [{i}] {p.name}  ({size_mb:.1f} MB)")
    if not items:
        return []
    def_str = ",".join(str(i) for i in default_idx if 0 <= i < len(items))
    raw = input(f"Select indices separated by comma [{def_str}]: ").strip()
    if not raw:
        raw = def_str
    idx = []
    for tok in raw.split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            k = int(tok)
            if 0 <= k < len(items):
                idx.append(k)
        except ValueError:
            pass
    idx = sorted(list(dict.fromkeys(idx)))
    return idx


def prompt_weights(names: List[str], suggestion: List[float]) -> List[float]:
    print("\nUNet weights per checkpoint.")
    ws: List[float] = []
    for name, sug in zip(names, suggestion):
        raw = input(f"Weight for {name} [{sug}]: ").strip()
        if raw == "":
            ws.append(float(sug))
        else:
            try:
                ws.append(float(raw))
            except ValueError:
                print("Invalid input. Using suggestion.")
                ws.append(float(sug))
    s = sum(ws)
    if s <= 0:
        print("All weights were zero or invalid. Assigning uniform distribution.")
        ws = [1.0 for _ in ws]
        s = sum(ws)
    ws_norm = [w / s for w in ws]
    print("Normalized weights:")
    for name, w in zip(names, ws_norm):
        print(f"  {name}: {w:.4f}")
    return ws_norm


def pick_backbone(names: List[str], weights: Optional[List[float]] = None) -> int:
    if weights:
        by_w = sorted(list(enumerate(weights)), key=lambda x: x[1], reverse=True)
        default_idx = by_w[0][0]
    else:
        default_idx = 0

    print(f"\nBackbone for CLIP and VAE")
    print("The backbone provides text and image encoders. Choose the most versatile model.")
    raw = input(f"Default: {names[default_idx]} [Enter to accept, or index 0..{len(names)-1}]: ").strip()
    if raw == "":
        return default_idx
    try:
        idx = int(raw)
        if 0 <= idx < len(names):
            return idx
    except ValueError:
        pass
    print("Invalid input. Using default.")
    return default_idx


def prompt_block_merge(names: List[str]) -> Optional[List[Dict[str, float]]]:
    print("\nMerge by down/mid/up blocks")
    raw = input("Enable block merge? [n] ").strip().lower()
    if raw not in ("y", "yes"):
        return None
    mults: List[Dict[str, float]] = []
    for name in names:
        s_down = input(f"Down multiplier for {name} [1.0]: ").strip()
        s_mid  = input(f"Mid multiplier for {name}  [1.0]: ").strip()
        s_up   = input(f"Up multiplier for {name}   [1.0]: ").strip()
        try:
            m_down = float(s_down) if s_down else 1.0
            m_mid  = float(s_mid) if s_mid else 1.0
            m_up   = float(s_up) if s_up else 1.0
        except ValueError:
            print("Invalid input. Using 1.0, 1.0, 1.0.")
            m_down, m_mid, m_up = 1.0, 1.0, 1.0
        mults.append(dict(down=m_down, mid=m_mid, up=m_up, other=1.0))
    return mults


def prompt_crossattn_boost(names: List[str]) -> Optional[List[Dict[str, float]]]:
    print("\nCross-attention boost")
    raw = input("Configure cross-attention boost? [n] ").strip().lower()
    if raw not in ("y", "yes"):
        return None
    print("Models:")
    for i, n in enumerate(names):
        print(f"  [{i}] {n}")
    raw_idx = input("Indices to boost separated by comma: ").strip()
    idxs: List[int] = []
    for tok in raw_idx.split(',') if raw_idx else []:
        tok = tok.strip()
        if not tok:
            continue
        try:
            k = int(tok)
            if 0 <= k < len(names):
                idxs.append(k)
        except ValueError:
            pass
    idxs = sorted(list(dict.fromkeys(idxs)))
    boosts = [dict(down=1.0, mid=1.0, up=1.0) for _ in names]
    if not idxs:
        print("No model selected for boost.")
        return None
    for k in idxs:
        raw_f = input(f"Factor for {names[k]} [1.20]: ").strip()
        try:
            factor = float(raw_f) if raw_f else 1.20
        except ValueError:
            print("Invalid input. Using 1.20.")
            factor = 1.20
        raw_where = input("Blocks to boost [down,mid]: ").strip().lower()
        if not raw_where:
            where = {"down", "mid"}
        else:
            where = {w.strip() for w in raw_where.split(',') if w.strip() in {"down", "mid", "up"}}
            if not where:
                where = {"down", "mid"}
        for g in where:
            boosts[k][g] = factor
    return boosts


def prompt_loras(lora_files: List[Path]) -> List[Tuple[Path, float]]:
    if not lora_files:
        return []
    print("\nLoRA selection to bake")
    for i, p in enumerate(lora_files):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  [{i}] {p.name}  ({size_mb:.1f} MB)")
    raw_idx = input("Select indices separated by comma [Enter for none]: ").strip()
    if not raw_idx:
        return []
    idxs = []
    for tok in raw_idx.split(','):
        tok = tok.strip()
        if tok == "":
            continue
        try:
            k = int(tok)
            if 0 <= k < len(lora_files):
                idxs.append(k)
        except ValueError:
            pass
    idxs = sorted(list(dict.fromkeys(idxs)))
    chosen = []
    for k in idxs:
        p = lora_files[k]
        raw_s = input(f"Scale for {p.name} [0.30]: ").strip()
        try:
            s = 0.30 if raw_s == "" else float(raw_s)
        except ValueError:
            print("Invalid input. Using 0.30.")
            s = 0.30
        chosen.append((p, s))
    return chosen


# ---------------- PerRes CLI ----------------

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


# ---------------- Orchestration ----------------

def main() -> int:
    root = Path(__file__).resolve().parent
    models_dir, loras_dir, output_dir, metadata_dir = ensure_dirs(root)

    print("\n" + "="*60)
    print(" XLFusion V1.1 - Advanced SDXL checkpoint merger")
    print("="*60)
    print(f"\nDetected structure:")
    print(f"  models:   {models_dir}")
    print(f"  loras:    {loras_dir}")
    print(f"  output:   {output_dir}")
    print(f"  metadata: {metadata_dir}")

    model_files = list_safetensors(models_dir)
    if not model_files:
        print("\nNo checkpoints found in ./models")
        print("   Place at least one and run again.")
        return 1

    # Mode selection
    print("\n" + "="*60)
    print("FUSION MODE")
    print("="*60)
    print("[1] Legacy - Classic weighted merge (3 blocks + LoRAs)")
    print("[2] PerRes - Resolution-based control (no LoRAs)")
    print("[3] Hybrid - Combined weighting + resolution control (NEW)")
    print("\n    Legacy: Traditional weighted blending")
    print("    PerRes: Complete block assignment by resolution")
    print("    Hybrid: Best of both - weights applied per resolution")

    mode_raw = input("\nSelect mode [1]: ").strip()
    if mode_raw == "2":
        mode = "perres"
    elif mode_raw == "3":
        mode = "hybrid"
    else:
        mode = "legacy"

    if mode == "perres":
        # PERRES MODE
        print("\n" + "="*60)
        print("PERRES MODE ACTIVATED")
        print("="*60)

        # Model selection
        print("\nAvailable checkpoints:")
        for i, p in enumerate(model_files):
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  [{i}] {p.name}  ({size_mb:.1f} MB)")

        print("\nRecommendation: use 2-3 complementary models")
        print("   Ex: one generalist, one with good style, one realistic")

        default_model_idx = list(range(min(3, len(model_files))))
        selected_idx = prompt_select(model_files, "Select models to merge:", default_model_idx)

        if len(selected_idx) < 2:
            print("\nPerRes requires at least 2 models. Switching to Legacy mode...")
            mode = "legacy"
        else:
            selected_models = [model_files[i] for i in selected_idx]
            model_names = [p.name for p in selected_models]

            # Configure PerRes assignments
            assignments, attn2_locks = prompt_perres_assignments(model_names)

            # Backbone
            backbone_idx = pick_backbone(model_names)

            # PerRes fusion
            merged = merge_perres(
                selected_models,
                assignments,
                backbone_idx,
                attn2_locks
            )

            # Prepare metadata
            out_path, version = next_version_path(output_dir)

            config = load_config()
            app_cfg = config["app"]
            output_cfg = config["model_output"]

            meta_embed = {
                "title": f"{output_cfg['base_name']}_{output_cfg['version_prefix']}{version}",
                "format": "sdxl-a1111-like",
                "merge_mode": "perres",
                "backbone": model_names[backbone_idx],
                "models": json.dumps(model_names, ensure_ascii=False),
                "assignments": json.dumps(assignments, ensure_ascii=False),
                "attn2_locks": json.dumps(attn2_locks, ensure_ascii=False) if attn2_locks else "",
                "created": datetime.now().isoformat(timespec='seconds'),
                "tool": "XLFusion",
            }

            # Save
            print(f"\nSaving model to {out_path.name}...")
            save_state(out_path, merged, meta_embed)

            # Audit log
            meta_txt = metadata_dir / f"meta_{output_cfg['base_name']}_{output_cfg['version_prefix']}{version}.txt"
            lines = [
                f"{app_cfg['tool_name']} V{app_cfg['version']} - PerRes Mode",
                f"Date: {datetime.now().isoformat(timespec='seconds')}",
                f"Output: {out_path.name}",
                "",
                "Base models:",
            ]
            for i, name in enumerate(model_names):
                lines.append(f"  [{i}] {name}")
            lines.append("")
            lines.append(f"Backbone (CLIP/VAE): {model_names[backbone_idx]}")
            lines.append("")
            lines.append("Resolution assignments:")
            lines.append(f"  Down 0,1: {model_names[assignments['down_0_1']]}")
            lines.append(f"  Down 2,3: {model_names[assignments['down_2_3']]}")
            lines.append(f"  Mid:      {model_names[assignments['mid']]}")
            lines.append(f"  Up 0,1:   {model_names[assignments['up_0_1']]}")
            lines.append(f"  Up 2,3:   {model_names[assignments['up_2_3']]}")

            if attn2_locks:
                lines.append("")
                lines.append("Cross-attention locks (attn2):")
                lines.append(f"  Down: {model_names[attn2_locks['down']]}")
                lines.append(f"  Mid:  {model_names[attn2_locks['mid']]}")
                lines.append(f"  Up:   {model_names[attn2_locks['up']]}")

            lines.append("")
            lines.append(f"Total keys: {len(merged)}")
            lines.append(f"UNet keys: {len([k for k in merged.keys() if k.startswith(UNET_PREFIX)])}")

            meta_txt.write_text("\n".join(lines), encoding="utf-8")
            print(f"Metadata log: {meta_txt.name}")

            print("\nPerRes fusion completed successfully")
            print("Validate with different seeds and prompts to verify consistency")
            return 0

    # HYBRID MODE
    elif mode == "hybrid":
        print("\n" + "="*60)
        print("HYBRID MODE ACTIVATED")
        print("="*60)
        print("Combines weighted blending with resolution-based control")
        print("Configure different weights for each resolution block")

        # Model selection
        print("\nAvailable checkpoints:")
        for i, p in enumerate(model_files):
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  [{i}] {p.name}  ({size_mb:.1f} MB)")

        print("\nRecommendation: use 2-4 models with complementary strengths")
        default_model_idx = list(range(min(3, len(model_files))))
        selected_idx = prompt_select(model_files, "Select models to merge:", default_model_idx)

        if len(selected_idx) < 2:
            print("\nHybrid mode requires at least 2 models. Switching to Legacy mode...")
            mode = "legacy"
        else:
            selected_models = [model_files[i] for i in selected_idx]
            model_names = [p.name for p in selected_models]

            # Configure hybrid weights per resolution block
            block_weights = prompt_hybrid_weights(model_names)

            # Backbone selection
            backbone_idx = pick_backbone(model_names)

            # Cross-attention boost
            try:
                config = load_config()
                default_boost = config["merge_defaults"]["hybrid"]["cross_attention_boost"]
            except (KeyError, TypeError):
                default_boost = 1.0  # Fallback if hybrid config missing
            boost_str = input(f"\nCross-attention boost [{default_boost}]: ").strip()
            if boost_str:
                try:
                    cross_attention_boost = float(boost_str)
                    if cross_attention_boost < 0:
                        print("Warning: Negative boost value, using default.")
                        cross_attention_boost = default_boost
                except ValueError:
                    print("Invalid boost value, using default.")
                    cross_attention_boost = default_boost
            else:
                cross_attention_boost = default_boost

            # Optional cross-attention locks
            use_locks = input("\nUse cross-attention locks? [n]: ").strip().lower() in ['y', 'yes']
            attn2_locks = None
            if use_locks:
                print("\nCross-attention locks (override block weights for attn2 layers):")
                locks = {}
                for block_type in ["down", "mid", "up"]:
                    idx_str = input(f"  {block_type.capitalize()} attn2 model [0]: ").strip()
                    if idx_str.isdigit():
                        idx = int(idx_str)
                        if 0 <= idx < len(selected_models):
                            locks[block_type] = idx
                        else:
                            print(f"Invalid index {idx}, using 0")
                            locks[block_type] = 0
                    else:
                        locks[block_type] = 0
                attn2_locks = locks

            # Execute hybrid merge
            print(f"\nExecuting hybrid merge...")
            merged = merge_hybrid(
                selected_models,
                [1.0] * len(selected_models),  # Not used in hybrid, but kept for compatibility
                block_weights,
                backbone_idx,
                cross_attention_boost,
                attn2_locks
            )

            # Prepare metadata
            out_path, version = next_version_path(output_dir)

            app_cfg = config["app"]
            output_cfg = config["model_output"]

            meta_embed = {
                "title": f"{output_cfg['base_name']}_{output_cfg['version_prefix']}{version}",
                "format": "sdxl-a1111-like",
                "merge_mode": "hybrid",
                "backbone": model_names[backbone_idx],
                "models": json.dumps(model_names, ensure_ascii=False),
                "block_weights": json.dumps(block_weights, ensure_ascii=False),
                "cross_attention_boost": str(cross_attention_boost),
                "attn2_locks": json.dumps(attn2_locks, ensure_ascii=False) if attn2_locks else "",
                "created": datetime.now().isoformat(timespec='seconds'),
                "tool": "XLFusion",
            }

            # Save
            print(f"\nSaving model to {out_path.name}...")
            save_state(out_path, merged, meta_embed)

            # Audit log
            meta_txt = metadata_dir / f"meta_{output_cfg['base_name']}_{output_cfg['version_prefix']}{version}.txt"
            lines = [
                f"{app_cfg['tool_name']} V{app_cfg['version']} - Hybrid Mode",
                f"Date: {datetime.now().isoformat(timespec='seconds')}",
                f"Output: {out_path.name}",
                "",
                "Base models:",
            ]
            for i, name in enumerate(model_names):
                lines.append(f"  [{i}] {name}")
            lines.append("")
            lines.append(f"Backbone (CLIP/VAE): {model_names[backbone_idx]}")
            lines.append("")
            lines.append("Block weights:")
            for block, weights in block_weights.items():
                lines.append(f"  {block}: {[f'{w:.3f}' for w in weights]}")

            if cross_attention_boost != 1.0:
                lines.append(f"\nCross-attention boost: {cross_attention_boost}")

            if attn2_locks:
                lines.append("")
                lines.append("Cross-attention locks (attn2):")
                lines.append(f"  Down: {model_names[attn2_locks['down']]}")
                lines.append(f"  Mid:  {model_names[attn2_locks['mid']]}")
                lines.append(f"  Up:   {model_names[attn2_locks['up']]}")

            lines.append("")
            lines.append(f"Total keys: {len(merged)}")
            lines.append(f"UNet keys: {len([k for k in merged.keys() if k.startswith(UNET_PREFIX)])}")

            meta_txt.write_text("\n".join(lines), encoding="utf-8")
            print(f"Metadata log: {meta_txt.name}")

            print("\nHybrid fusion completed successfully")
            print("Validate with different seeds and prompts to verify consistency")
            return 0

    # LEGACY MODE
    if mode == "legacy":
        print("\nLegacy mode activated")

        # Checkpoint selection
        default_model_idx = [0, 1] if len(model_files) >= 2 else [0]
        selected_idx = prompt_select(model_files, "Available checkpoints:", default_model_idx)
        if not selected_idx:
            print("No checkpoint selected. Aborting.")
            return 1

        selected_models = [model_files[i] for i in selected_idx]
        model_names = [p.name for p in selected_models]

        # Weights
        if len(selected_models) == 1:
            suggestions = [1.0]
        elif len(selected_models) == 2:
            suggestions = [0.65, 0.35]
        elif len(selected_models) == 3:
            suggestions = [0.62, 0.28, 0.10]
        else:
            suggestions = [1.0 / len(selected_models) for _ in selected_models]
        weights = prompt_weights(model_names, suggestions)

        # Backbone
        backbone_idx = pick_backbone(model_names, weights)

        # Block merge
        block_multipliers = prompt_block_merge(model_names)

        # Cross-attention boost
        crossattn_boosts = prompt_crossattn_boost(model_names)

        print("\nFusion with memory containment...")
        merged, backbone_state = stream_weighted_merge_from_paths(
            selected_models, weights, backbone_idx,
            only_unet=True,
            block_multipliers=block_multipliers,
            crossattn_boosts=crossattn_boosts,
        )

        # LoRAs
        lora_files = list_safetensors(loras_dir)
        baked_info = []
        if lora_files:
            chosen_loras = prompt_loras(lora_files)
            if chosen_loras:
                for p, s in chosen_loras:
                    print(f"Baking {p.name} with scale {s}...")
                    applied, skipped = apply_single_lora(merged, p, s)
                    print(f"  Applied {applied}, skipped {skipped}")
                    baked_info.append((p.name, float(s), int(applied), int(skipped)))

        # Save
        out_path, version = next_version_path(output_dir)

        config = load_config()
        app_cfg = config["app"]
        output_cfg = config["model_output"]

        meta_embed = {
            "title": f"{output_cfg['base_name']}_{output_cfg['version_prefix']}{version}",
            "format": "sdxl-a1111-like",
            "unet_merge": "weighted_blocked_crossboost_stream",
            "backbone": model_names[backbone_idx],
            "bases": json.dumps([{ "file": n, "weight": float(w) } for n, w in zip(model_names, weights)], ensure_ascii=False),
            "block_multipliers": json.dumps(block_multipliers, ensure_ascii=False) if block_multipliers else "",
            "crossattn_boosts": json.dumps(crossattn_boosts, ensure_ascii=False) if crossattn_boosts else "",
            "loras": json.dumps([{ "file": n, "scale": s, "applied": a, "skipped": k } for n, s, a, k in baked_info], ensure_ascii=False),
            "created": datetime.now().isoformat(timespec='seconds'),
            "tool": "XLFusion",
        }

        print(f"\nSaving model to {out_path.name}...")
        save_state(out_path, merged, meta_embed)

        # Audit log
        meta_txt = metadata_dir / f"meta_{output_cfg['base_name']}_{output_cfg['version_prefix']}{version}.txt"
        lines = [
            f"{app_cfg['tool_name']} V{app_cfg['version']} - Legacy Mode",
            f"Date: {datetime.now().isoformat(timespec='seconds')}",
            f"Output: {out_path.name}",
            "",
            f"Backbone (CLIP/VAE): {model_names[backbone_idx]}",
            "",
            "Base weights UNet:",
        ]
        for n, w in zip(model_names, weights):
            lines.append(f"  {n}: {w:.6f}")

        if block_multipliers:
            lines.append("")
            lines.append("Block multipliers:")
            for n, m in zip(model_names, block_multipliers):
                lines.append(f"  {n}: down={m.get('down',1.0):.3f} mid={m.get('mid',1.0):.3f} up={m.get('up',1.0):.3f}")

        if crossattn_boosts:
            lines.append("")
            lines.append("Cross-attention boost:")
            for n, b in zip(model_names, crossattn_boosts):
                lines.append(f"  {n}: down={b.get('down',1.0):.3f} mid={b.get('mid',1.0):.3f} up={b.get('up',1.0):.3f}")

        if baked_info:
            lines.append("")
            lines.append("Baked LoRAs:")
            for n, s, a, k in baked_info:
                lines.append(f"  {n}  scale={s}  applied={a}  skipped={k}")

        lines.append("")
        lines.append(f"Total keys: {len(merged)}")
        lines.append(f"UNet keys: {len([k for k in merged.keys() if k.startswith(UNET_PREFIX)])}")

        meta_txt.write_text("\n".join(lines), encoding="utf-8")
        print(f"Metadata log: {meta_txt.name}")

        print("\nLegacy fusion completed")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)