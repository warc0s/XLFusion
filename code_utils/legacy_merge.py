"""
Legacy Merge Module for XLFusion V1.1

Traditional weighted merging with block-level control and LoRA baking support.
"""
import re
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from safetensors.torch import load_file as st_load

from .common import UNET_PREFIX, load_state, is_cross_attn_key_legacy

# LoRA patterns
DOWN_PAT = re.compile(r"\.lora_down\.weight$")
UP_PAT = re.compile(r"\.lora_up\.weight$")
ALPHA_KEYS = ["alpha", "lora_alpha", "ss_network_alpha", "scale"]

# Cross-attention patterns
CROSS_TOKENS = (".attn2.",)
CROSS_PROJ = (".to_q.", ".to_k.", ".to_v.", ".to_out.0.")


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


# is_cross_attn_key_legacy is now imported from common module




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
                    print(f"Warning: Shape mismatch for key {k}: expected {acc_shape[k]}, got {t.shape}")
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


# LoRA baking functions
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
    try:
        lora_state = st_load(str(lora_path), device="cpu")
    except Exception as e:
        print(f"Error loading LoRA file {lora_path.name}: {e}")
        return 0, 0

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