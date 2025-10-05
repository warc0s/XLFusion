"""
Memory management utilities for XLFusion
Handles memory estimation, state loading/saving, and system checks
"""
import gc
import time
from pathlib import Path
from typing import Dict, List, Set

import torch
from safetensors.torch import load_file as st_load, save_file as st_save

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def estimate_memory_requirement(model_paths: List[Path], needed_indices: Set[int]) -> float:
    """Estimate total memory needed for loading models in GB"""
    total_size = sum(model_paths[i].stat().st_size for i in needed_indices if i < len(model_paths))
    return total_size / (1024**3)  # Convert to GB


def check_memory_availability(required_gb: float) -> bool:
    """Check if system has enough available memory"""
    if not PSUTIL_AVAILABLE:
        return True  # Can't check, assume OK
    available = psutil.virtual_memory().available / (1024**3)
    return available > (required_gb * 1.5)  # 50% safety margin


def load_state_with_fallback(path: Path, max_retries: int = 3) -> Dict[str, torch.Tensor]:
    """Load model state with retry mechanism"""
    for attempt in range(max_retries):
        try:
            return load_state(path)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Load attempt {attempt + 1} failed, retrying... ({e})")
            gc.collect()
            time.sleep(1)


def load_state(path: Path) -> Dict[str, torch.Tensor]:
    """Load tensors while preserving their on-disk dtype."""
    state = st_load(str(path), device="cpu")
    # Keep original dtype to avoid doubling memory usage; callers can upcast selectively
    return dict(state.items())


def save_state(path: Path, state: Dict[str, torch.Tensor], meta: Dict[str, str]) -> None:
    compact: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if v.dtype == torch.float32 and v.dim() >= 2:
            compact[k] = v.to(torch.float16)
        else:
            compact[k] = v
    st_save(compact, str(path), metadata=meta)
