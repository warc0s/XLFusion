"""
LoRA utilities for XLFusion
Handles LoRA application and key mapping
"""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file as st_load

from .blocks import UNET_PREFIX

# LoRA patterns and constants
DOWN_PAT = re.compile(r"\.lora_down\.weight$")
UP_PAT = re.compile(r"\.lora_up\.weight$")
ALPHA_KEYS = ["alpha", "lora_alpha", "ss_network_alpha", "scale"]
CROSS_TOKENS = (".attn2.",)
CROSS_PROJ = (".to_q.", ".to_k.", ".to_v.", ".to_out.0.")


def lora_pairs_from_state(lora_state: Dict[str, torch.Tensor]) -> List[Tuple[str, str]]:
    downs = [k for k in lora_state.keys() if DOWN_PAT.search(k)]
    pairs: List[Tuple[str, str]] = []
    for d in downs:
        u = d.replace(".lora_down.weight", ".lora_up.weight")
        if u in lora_state:
            pairs.append((d, u))
    return pairs


def parse_lora_alpha_rank(meta: Dict[str, str], down: torch.Tensor) -> Tuple[float, int]:
    if not isinstance(meta, dict):
        raise ValueError("meta must be a dict")
    
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
    # Primero convertir guiones bajos a puntos (estilo comun en LoRAs)
    base = base.replace("_", ".")
    # Corregir tokens especiales donde el nombre real usa guion bajo
    # Bloques y submódulos comunes de SDXL
    base = base.replace(".down.blocks.", ".down_blocks.")
    base = base.replace("down.blocks.", "down_blocks.")
    base = base.replace(".up.blocks.", ".up_blocks.")
    base = base.replace("up.blocks.", "up_blocks.")
    base = base.replace(".mid.block.", ".mid_block.")
    base = base.replace("mid.block.", "mid_block.")
    base = base.replace(".middle.block.", ".middle_block.")
    base = base.replace("middle.block.", "middle_block.")
    base = base.replace(".input.blocks.", ".input_blocks.")
    base = base.replace("input.blocks.", "input_blocks.")
    base = base.replace(".output.blocks.", ".output_blocks.")
    base = base.replace("output.blocks.", "output_blocks.")
    base = base.replace(".transformer.blocks.", ".transformer_blocks.")
    base = base.replace("transformer.blocks.", "transformer_blocks.")
    # Restaurar proyecciones 'to_x' que deben llevar guion bajo
    base = base.replace(".to.", ".to_")
    # Sustituir sufijos de LoRA por el peso real (soportar variantes con punto o guion bajo)
    base = base.replace(".lora_down.weight", ".weight").replace(".lora_up.weight", ".weight")
    base = base.replace(".lora.down.weight", ".weight").replace(".lora.up.weight", ".weight")
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
