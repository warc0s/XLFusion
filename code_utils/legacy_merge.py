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
from .progress_simple import track_merge_progress, track_tensor_progress, show_phase_start, show_phase_complete, show_merge_complete

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
    """Legacy fusion with memory containment and lightweight progress tracking"""
    assert len(model_paths) == len(weights) and len(model_paths) > 0
    if block_multipliers is None:
        block_multipliers = [dict(down=1.0, mid=1.0, up=1.0, other=1.0) for _ in model_paths]
    if crossattn_boosts is None:
        crossattn_boosts = [dict(down=1.0, mid=1.0, up=1.0) for _ in model_paths]

    merge_start = show_phase_start(f"Legacy merge of {len(model_paths)} models")

    # Phase 1: Load backbone
    backbone_start = show_phase_start("Loading backbone for CLIP and VAE")
    backbone_state = load_state(model_paths[backbone_idx])
    show_phase_complete("Backbone loading", backbone_start)

    acc_sum: Dict[str, torch.Tensor] = {}
    acc_w: Dict[str, float] = {}
    acc_shape: Dict[str, torch.Size] = {}

    # Phase 2: Merge models
    models_start = show_phase_start("Processing models")
    model_progress = track_merge_progress(model_paths, "Legacy")

    for i, mp in enumerate(model_paths):
        print(f"  Model {i+1}/{len(model_paths)}: {mp.name}")
        s = load_state(mp)
        w_global = float(weights[i])
        mults = block_multipliers[i]
        boosts = crossattn_boosts[i]

        # Process tensors from this model
        tensor_keys = list(s.keys())
        tensor_progress = track_tensor_progress(len(tensor_keys), f"    Tensors")

        for k, t in s.items():
            if only_unet and not should_merge_key(k, True):
                if tensor_progress:
                    tensor_progress.update()
                continue
            grp = group_for_key(k) or "other"
            eff = w_global * float(mults.get(grp, 1.0))
            if is_cross_attn_key_legacy(k):
                eff *= float(boosts.get(grp, 1.0))
            if eff <= 0:
                if tensor_progress:
                    tensor_progress.update()
                continue
            if k not in acc_sum:
                acc_sum[k] = eff * t.to(torch.float32)
                acc_w[k] = eff
                acc_shape[k] = t.shape
            else:
                if t.shape != acc_shape[k]:
                    print(f"    Warning: Shape mismatch for key {k}")
                    if tensor_progress:
                        tensor_progress.update()
                    continue
                acc_sum[k].add_(eff * t.to(torch.float32))
                acc_w[k] += eff
            if tensor_progress:
                tensor_progress.update()

        if tensor_progress:
            tensor_progress.finish()
        if model_progress:
            model_progress.update()
        del s
        gc.collect()

    if model_progress:
        model_progress.finish()
    show_phase_complete("Model processing", models_start)

    # Phase 3: Normalize and finalize
    finalize_start = show_phase_start("Finalizing merge")
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

    show_phase_complete("Merge finalization", finalize_start)
    show_merge_complete(merge_start, "Legacy")

    return merged, backbone_state


def stream_weighted_merge_memory_efficient(
    model_paths: List[Path],
    weights: List[float],
    backbone_idx: int,
    only_unet: bool = True,
    block_multipliers: Optional[List[Dict[str, float]]] = None,
    crossattn_boosts: Optional[List[Dict[str, float]]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Memory-efficient legacy fusion that loads tensors one at a time.
    Reduces RAM spikes by avoiding full model loading.
    """
    from .memory_efficient import MemoryEfficientLoader

    assert len(model_paths) == len(weights) and len(model_paths) > 0
    if block_multipliers is None:
        block_multipliers = [dict(down=1.0, mid=1.0, up=1.0, other=1.0) for _ in model_paths]
    if crossattn_boosts is None:
        crossattn_boosts = [dict(down=1.0, mid=1.0, up=1.0) for _ in model_paths]

    # Initialize progress tracker
    tracker = MergeProgressTracker(model_paths, "Legacy Memory-Efficient")

    # Phase 1: Load backbone
    progress_bar = tracker.start_phase("Loading backbone for CLIP and VAE")
    backbone_state = load_state(model_paths[backbone_idx])
    tracker.finish_phase(progress_bar)

    # Phase 2: Scan UNet keys
    all_unet_keys = set()
    progress_bar = tracker.start_phase("Scanning UNet keys", len(model_paths))

    for i, path in enumerate(model_paths):
        with MemoryEfficientLoader(path) as loader:
            for key in loader.keys():
                if should_merge_key(key, only_unet):
                    all_unet_keys.add(key)
        progress_bar.update()

    tracker.finish_phase(progress_bar)

    # Phase 3: Merge tensors
    merged_unet = {}
    progress_bar = tracker.start_phase("Processing tensors", len(all_unet_keys))

    for key in all_unet_keys:
        acc_sum = None
        acc_weight = 0.0
        key_shape = None

        # Process this key across all models
        for i, path in enumerate(model_paths):
            w_global = float(weights[i])
            mults = block_multipliers[i]
            boosts = crossattn_boosts[i]

            # Calculate effective weight
            grp = group_for_key(key) or "other"
            eff = w_global * float(mults.get(grp, 1.0))
            if is_cross_attn_key_legacy(key):
                eff *= float(boosts.get(grp, 1.0))

            if eff <= 0:
                continue

            # Load tensor from this model
            with MemoryEfficientLoader(path) as loader:
                try:
                    tensor = loader.get_tensor(key, preserve_dtype=True)

                    # Shape validation
                    if key_shape is None:
                        key_shape = tensor.shape
                    elif tensor.shape != key_shape:
                        print(f"Warning: Shape mismatch for key {key}: expected {key_shape}, got {tensor.shape}")
                        continue

                    # Convert to float32 for accumulation
                    tensor_f32 = tensor.to(torch.float32)

                    # Accumulate
                    if acc_sum is None:
                        acc_sum = eff * tensor_f32
                    else:
                        acc_sum.add_(eff * tensor_f32)

                    acc_weight += eff

                except Exception:
                    # Key doesn't exist in this model
                    continue

        # Store merged result
        if acc_sum is not None and acc_weight > 0:
            merged_unet[key] = (acc_sum / acc_weight).to(torch.float32)

        progress_bar.update()

        # Periodic garbage collection to manage memory
        if progress_bar.current % 50 == 0:
            gc.collect()

    tracker.finish_phase(progress_bar)

    # Phase 4: Combine with backbone
    progress_bar = tracker.start_phase("Finalizing merge", len(backbone_state))

    merged = {}
    for k, v in backbone_state.items():
        if only_unet and should_merge_key(k, True):
            if k in merged_unet:
                merged[k] = merged_unet[k]
            else:
                merged[k] = v
        else:
            merged[k] = v
        progress_bar.update()

    tracker.finish_phase(progress_bar)
    tracker.finish()

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

    # Simple progress for LoRA application
    progress_bar = track_tensor_progress(len(pairs), f"LoRA: {lora_path.name}")

    def as_2d_if_1x1(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 4 and t.shape[2] == 1 and t.shape[3] == 1:
            return t.view(t.shape[0], t.shape[1])
        return t

    for d_key, u_key in pairs:
        if not d_key.startswith("lora_unet_"):
            skipped += 1
            if progress_bar:
                progress_bar.update()
            continue
        down = lora_state[d_key].to(torch.float32)
        up = lora_state[u_key].to(torch.float32)
        alpha, rank = parse_lora_alpha_rank(meta, down)
        scale_eff = scale * (alpha / max(rank, 1))
        base_w_key = map_lora_key_to_base(d_key)
        if base_w_key is None or base_w_key not in base_state:
            skipped += 1
            if progress_bar:
                progress_bar.update()
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
            if progress_bar:
                progress_bar.update()
            continue
        base_state[base_w_key] = (W + delta).to(torch.float32)
        applied += 1
        if progress_bar:
            progress_bar.update()

    if progress_bar:
        progress_bar.finish()
    return applied, skipped