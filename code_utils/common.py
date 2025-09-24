"""
Common Utilities for XLFusion V1.1

Shared functions used across all merge modes to avoid code duplication.
"""
from pathlib import Path
from typing import Dict, Optional
import torch
from safetensors.torch import load_file as st_load, save_file as st_save

# SDXL A1111 style naming constants
UNET_PREFIX = "model.diffusion_model."


def load_state(path: Path) -> Dict[str, torch.Tensor]:
    """Load model state from safetensors file and convert to float32."""
    state = st_load(str(path), device="cpu")
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if v.dtype in (torch.float16, torch.bfloat16):
            out[k] = v.to(torch.float32)
        else:
            out[k] = v
    return out


def load_state_memory_efficient(path: Path, preserve_dtype: bool = False) -> Dict[str, torch.Tensor]:
    """
    Memory-efficient state loading with optional dtype preservation.
    For large models, consider using MemoryEfficientLoader directly.
    """
    from .memory_efficient import MemoryEfficientLoader

    state = {}
    with MemoryEfficientLoader(path) as loader:
        for key in loader.keys():
            tensor = loader.get_tensor(key, preserve_dtype=preserve_dtype)
            if not preserve_dtype and tensor.dtype in (torch.float16, torch.bfloat16):
                tensor = tensor.to(torch.float32)
            state[key] = tensor
    return state


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


def is_cross_attn_key_legacy(k: str) -> bool:
    """Legacy cross-attention detection for boost compatibility"""
    if not k.startswith(UNET_PREFIX):
        return False
    s = k[len(UNET_PREFIX):]
    cross_tokens = (".attn2.",)
    cross_proj = (".to_q.", ".to_k.", ".to_v.", ".to_out.0.")
    if not any(tok in s for tok in cross_tokens):
        return False
    return any(proj in s for proj in cross_proj)


def save_state(path: Path, state: Dict[str, torch.Tensor], meta: Dict[str, str]) -> None:
    """Save model state to safetensors file with metadata."""
    # Import here to avoid circular import
    from .progress_simple import track_tensor_progress, show_phase_start, show_phase_complete

    # Convert tensors to FP16 for storage efficiency
    convert_start = show_phase_start(f"Preparing {len(state)} tensors for save")
    progress_bar = track_tensor_progress(len(state), "Converting to FP16")

    compact: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if v.dtype == torch.float32 and v.dim() >= 2:
            compact[k] = v.to(torch.float16)
        else:
            compact[k] = v
        if progress_bar:
            progress_bar.update()

    if progress_bar:
        progress_bar.finish()
    show_phase_complete("Tensor conversion", convert_start)

    # Write to disk
    print(f"Writing to {path.name}...")
    st_save(compact, str(path), metadata=meta)

    # Show final size
    file_size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Saved {len(compact)} tensors ({file_size_mb:.1f} MB)")