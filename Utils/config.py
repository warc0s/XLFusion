"""
Configuration utilities for XLFusion
"""
import re
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import yaml


def load_config() -> dict:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    try:
        with open(config_path, "r") as f:
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
            "app": {
                "tool_name": "XLFusion",
                "version": "2.0"
            }
        }


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


def generate_batch_config_yaml(
    mode: str,
    model_names: List[str],
    backbone_idx: int,
    version: int,
    weights: Optional[List[float]] = None,
    assignments: Optional[Dict[str, Any]] = None,
    hybrid_config: Optional[Dict[str, Any]] = None,
    attn2_locks: Optional[Dict[str, Any]] = None,
    block_multipliers: Optional[List[Dict[str, Any]]] = None,
    crossattn_boosts: Optional[List[Dict[str, Any]]] = None,
    loras: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Generate a batch configuration YAML compatible with XLFusion format.
    
    Args:
        mode: Fusion mode ('legacy', 'perres', 'hybrid')
        model_names: List of model filenames
        backbone_idx: Index of backbone model
        version: Version number for output naming
        weights: Weights for legacy mode (optional)
        assignments: PerRes assignments for perres mode (optional)
        hybrid_config: Hybrid configuration for hybrid mode (optional)
        attn2_locks: Cross-attention locks (optional)
        block_multipliers: Block multipliers for legacy mode (optional)
        crossattn_boosts: Cross-attention boosts for legacy mode (optional)
        loras: LoRAs to bake (optional)
    
    Returns:
        Formatted YAML string
    """
    # Base configuration structure
    config = {
        "version": "2.0",
        "global_settings": {
            "output_base": "recreated_output",
            "continue_on_error": True,
            "max_parallel": 1,
            "log_level": "INFO"
        },
        "batch_jobs": []
    }
    
    # Job configuration
    job = {
        "name": f"Recreated_Model_V{version}",
        "mode": mode,
        "description": "Recreated from metadata backup",
        "models": model_names,
        "backbone": backbone_idx,
        "output_name": f"Recreated_V{version}"
    }
    
    # Add mode-specific fields
    if mode == "legacy":
        if weights is not None:
            job["weights"] = weights
        if block_multipliers is not None:
            job["block_multipliers"] = block_multipliers
        if crossattn_boosts is not None:
            job["crossattn_boosts"] = crossattn_boosts
        if loras is not None:
            job["loras"] = loras
    
    elif mode == "perres":
        if assignments is not None:
            job["assignments"] = assignments
        if attn2_locks is not None:
            job["attn2_locks"] = attn2_locks
        if loras is not None:
            job["loras"] = loras
    
    elif mode == "hybrid":
        if hybrid_config is not None:
            job["hybrid_config"] = hybrid_config
        if attn2_locks is not None:
            job["attn2_locks"] = attn2_locks
        if loras is not None:
            job["loras"] = loras
    
    config["batch_jobs"].append(job)
    
    return yaml.dump(config, default_flow_style=False, sort_keys=False)
