"""
Memory-Efficient Weight Loading for XLFusion V1.1

Reduces RAM spikes during large SDXL merges by:
- Loading tensors one at a time from safetensors
- Preserving original dtypes and casting only during accumulation
- Avoiding default FP32 elevation for entire models
"""
from pathlib import Path
from typing import Dict, Iterator, Tuple, Optional, Any
import torch
from safetensors import safe_open
import gc
from .progress import MergeProgressTracker, create_tensor_progress


class MemoryEfficientLoader:
    """
    Memory-efficient loader that iterates through tensors without loading entire state.
    RAM consumption is proportional to the current tensor, not the complete model.
    """

    def __init__(self, path: Path, device: str = "cpu"):
        self.path = path
        self.device = device
        self._file_handle = None
        self._tensor_keys = None

    def __enter__(self):
        self._file_handle = safe_open(str(self.path), framework="pt", device=self.device)
        self._tensor_keys = list(self._file_handle.keys())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file_handle:
            # safetensors handles will be closed automatically
            self._file_handle = None
            self._tensor_keys = None

    def keys(self) -> list[str]:
        """Get all tensor keys in the file"""
        if self._tensor_keys is None:
            raise RuntimeError("Loader not properly initialized. Use with context manager.")
        return self._tensor_keys.copy()

    def get_tensor(self, key: str, preserve_dtype: bool = True) -> torch.Tensor:
        """
        Load a single tensor with optional dtype preservation.

        Args:
            key: Tensor key to load
            preserve_dtype: If True, keep original dtype. If False, convert to FP32.
        """
        if self._file_handle is None:
            raise RuntimeError("Loader not properly initialized. Use with context manager.")

        tensor = self._file_handle.get_tensor(key)

        if not preserve_dtype and tensor.dtype in (torch.float16, torch.bfloat16):
            tensor = tensor.to(torch.float32)

        return tensor

    def iter_tensors(self, preserve_dtype: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        Iterate through all tensors one by one.
        Memory usage stays constant per tensor instead of growing with model size.
        """
        if self._tensor_keys is None:
            raise RuntimeError("Loader not properly initialized. Use with context manager.")

        for key in self._tensor_keys:
            yield key, self.get_tensor(key, preserve_dtype=preserve_dtype)


def memory_efficient_weighted_merge(
    model_paths: list[Path],
    weights: list[float],
    backbone_idx: int,
    only_unet: bool = True,
    accumulation_dtype: torch.dtype = torch.float32
) -> Dict[str, torch.Tensor]:
    """
    Memory-efficient weighted merge that processes tensors one at a time.

    Args:
        model_paths: List of model file paths
        weights: Weight for each model
        backbone_idx: Index of backbone model (for non-UNet components)
        only_unet: If True, only merge UNet components
        accumulation_dtype: Dtype to use for accumulation (usually float32)

    Returns:
        Merged model state dict
    """
    assert len(model_paths) == len(weights) and len(model_paths) > 0

    from .common import UNET_PREFIX

    # Initialize progress tracker
    tracker = MergeProgressTracker(model_paths, "Memory-Efficient Weighted")

    # Phase 1: Scan tensor keys
    all_keys = set()
    progress_bar = tracker.start_phase("Scanning tensor keys", len(model_paths))

    for i, path in enumerate(model_paths):
        with MemoryEfficientLoader(path) as loader:
            all_keys.update(loader.keys())
        progress_bar.update()

    tracker.finish_phase(progress_bar)

    # Filter keys if only_unet is True
    if only_unet:
        unet_keys = {k for k in all_keys if k.startswith(UNET_PREFIX)}
        non_unet_keys = {k for k in all_keys if not k.startswith(UNET_PREFIX)}
    else:
        unet_keys = all_keys
        non_unet_keys = set()

    print(f"Found {len(unet_keys)} UNet keys, {len(non_unet_keys)} non-UNet keys")

    # Phase 2: Load backbone for non-UNet components
    merged_state = {}
    if non_unet_keys:
        progress_bar = tracker.start_phase("Loading backbone components", len(non_unet_keys))
        with MemoryEfficientLoader(model_paths[backbone_idx]) as backbone_loader:
            for key in non_unet_keys:
                try:
                    merged_state[key] = backbone_loader.get_tensor(key, preserve_dtype=False)
                except Exception:
                    # Key might not exist in backbone
                    pass
                progress_bar.update()
        tracker.finish_phase(progress_bar)

    # Phase 3: Merge UNet tensors one by one
    progress_bar = tracker.start_phase("Merging UNet tensors", len(unet_keys))

    for key in unet_keys:
        # Collect tensors for this key from all models
        tensors_and_weights = []

        for i, (path, weight) in enumerate(zip(model_paths, weights)):
            if weight <= 0:
                continue

            with MemoryEfficientLoader(path) as loader:
                try:
                    tensor = loader.get_tensor(key, preserve_dtype=True)
                    tensors_and_weights.append((tensor, weight))
                except Exception:
                    # Key might not exist in this model
                    continue

        if not tensors_and_weights:
            progress_bar.update()
            continue

        # Merge tensors for this key
        result = None
        total_weight = 0.0

        for tensor, weight in tensors_and_weights:
            # Cast to accumulation dtype only during computation
            tensor_acc = tensor.to(accumulation_dtype) if tensor.dtype != accumulation_dtype else tensor

            if result is None:
                result = tensor_acc * weight
            else:
                # Ensure shapes match
                if tensor_acc.shape != result.shape:
                    print(f"Warning: Shape mismatch for {key}, skipping")
                    progress_bar.update()
                    continue
                result.add_(tensor_acc * weight)

            total_weight += weight

        if result is not None and total_weight > 0:
            # Normalize by total weight
            result.div_(total_weight)
            merged_state[key] = result

        progress_bar.update()

        # Force garbage collection periodically
        if progress_bar.current % 50 == 0:
            gc.collect()

    tracker.finish_phase(progress_bar)
    tracker.finish()

    return merged_state


def memory_efficient_hybrid_merge(
    model_paths: list[Path],
    block_weights: Dict[str, list[float]],
    backbone_idx: int,
    cross_attention_boost: float = 1.0,
    accumulation_dtype: torch.dtype = torch.float32
) -> Dict[str, torch.Tensor]:
    """
    Memory-efficient hybrid merge with block-specific weights.
    """
    from .common import UNET_PREFIX, get_block_assignment, is_cross_attn_key_legacy

    # Initialize progress tracker
    tracker = MergeProgressTracker(model_paths, "Memory-Efficient Hybrid")

    # Phase 1: Scan tensor keys
    all_keys = set()
    progress_bar = tracker.start_phase("Scanning tensor keys", len(model_paths))

    for i, path in enumerate(model_paths):
        with MemoryEfficientLoader(path) as loader:
            all_keys.update(loader.keys())
        progress_bar.update()

    tracker.finish_phase(progress_bar)

    # Separate UNet and non-UNet keys
    unet_keys = {k for k in all_keys if k.startswith(UNET_PREFIX)}
    non_unet_keys = {k for k in all_keys if not k.startswith(UNET_PREFIX)}

    print(f"Found {len(unet_keys)} UNet keys, {len(non_unet_keys)} non-UNet keys")

    # Phase 2: Load backbone for non-UNet components
    merged_state = {}
    if non_unet_keys:
        progress_bar = tracker.start_phase("Loading backbone components", len(non_unet_keys))
        with MemoryEfficientLoader(model_paths[backbone_idx]) as backbone_loader:
            for key in non_unet_keys:
                try:
                    merged_state[key] = backbone_loader.get_tensor(key, preserve_dtype=False)
                except Exception:
                    pass
                progress_bar.update()
        tracker.finish_phase(progress_bar)

    # Phase 3: Merge UNet tensors with hybrid weights
    progress_bar = tracker.start_phase("Processing UNet tensors with hybrid weights", len(unet_keys))
    stats = {block: 0 for block in block_weights}
    stats["cross_attn_boost"] = 0
    stats["other"] = 0

    for key in unet_keys:
        # Get block assignment for this key
        block_group = get_block_assignment(key)

        if block_group and block_group in block_weights:
            # Use block-specific weights
            weights_for_block = block_weights[block_group]

            if len(weights_for_block) != len(model_paths):
                # Fallback to backbone
                with MemoryEfficientLoader(model_paths[backbone_idx]) as loader:
                    try:
                        merged_state[key] = loader.get_tensor(key, preserve_dtype=False)
                    except Exception:
                        pass
                progress_bar.update()
                continue

            # Collect tensors with block-specific weights
            tensors_and_weights = []
            for i, (path, weight) in enumerate(zip(model_paths, weights_for_block)):
                if weight <= 0:
                    continue

                with MemoryEfficientLoader(path) as loader:
                    try:
                        tensor = loader.get_tensor(key, preserve_dtype=True)
                        tensors_and_weights.append((tensor, weight))
                    except Exception:
                        continue

            if not tensors_and_weights:
                progress_bar.update()
                continue

            # Merge tensors
            result = None
            total_weight = 0.0

            for tensor, weight in tensors_and_weights:
                tensor_acc = tensor.to(accumulation_dtype) if tensor.dtype != accumulation_dtype else tensor

                if result is None:
                    result = tensor_acc * weight
                else:
                    if tensor_acc.shape != result.shape:
                        progress_bar.update()
                        continue
                    result.add_(tensor_acc * weight)

                total_weight += weight

            if result is not None and total_weight > 0:
                result.div_(total_weight)

                # Apply cross-attention boost if applicable
                if cross_attention_boost != 1.0 and is_cross_attn_key_legacy(key):
                    result.mul_(cross_attention_boost)
                    stats["cross_attn_boost"] += 1

                merged_state[key] = result
                stats[block_group] = stats.get(block_group, 0) + 1
        else:
            # Use backbone for other keys
            with MemoryEfficientLoader(model_paths[backbone_idx]) as loader:
                try:
                    merged_state[key] = loader.get_tensor(key, preserve_dtype=False)
                    stats["other"] += 1
                except Exception:
                    pass

        progress_bar.update()

        # Periodic garbage collection
        if progress_bar.current % 50 == 0:
            gc.collect()

    tracker.finish_phase(progress_bar)

    # Print statistics
    print("\nHybrid merge statistics:")
    for block, count in stats.items():
        if count > 0:
            print(f"  {block}: {count} keys")

    tracker.finish()
    return merged_state