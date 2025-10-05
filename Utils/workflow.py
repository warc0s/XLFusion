"""Shared orchestration utilities between CLI and GUI."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, List
import hashlib
import torch

from .config import generate_batch_config_yaml, load_config, next_version_path
from .memory import save_state


def _build_metadata_header(version: str, mode: str, model_names: Iterable[str]) -> str:
    """Generate a human-readable header for metadata.txt."""
    lines = [
        f"XLFusion V{version} - Merge Metadata",
        "=" * 50,
        "",
        f"Mode: {mode}",
        f"Models: {', '.join(model_names)}",
    ]
    return "\n".join(lines) + "\n"


def _blake2b_file(path: Path, digest_size: int = 16) -> str:
    h = hashlib.blake2b(digest_size=digest_size)
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_merge_results(
    output_dir: Path,
    metadata_dir: Path,
    merged_state: Dict[str, Any],
    model_names: Iterable[str],
    mode: str,
    backbone_idx: int,
    yaml_kwargs: Optional[Dict[str, Any]] = None,
    *,
    model_paths: Optional[List[Path]] = None,
    lora_paths: Optional[List[Path]] = None,
) -> Tuple[Path, Path, int]:
    """Unified persistence of the merged model and auxiliary files.

    Args:
        output_dir: Destination directory for the merged checkpoint.
        metadata_dir: Root directory for structured metadata.
        merged_state: State dict of the resulting model.
        model_names: List of source model names.
        mode: Fusion mode used (legacy/perres/hybrid).
        backbone_idx: Index of the backbone model within the selection.
        yaml_kwargs: Extra parameters to reconstruct the configuration via
            ``generate_batch_config_yaml`` (e.g., ``weights`` or ``assignments``).

    Returns:
        A tuple ``(output_path, metadata_folder, version)``.
    """

    model_names = list(model_names)
    yaml_kwargs = yaml_kwargs or {}

    output_path, version = next_version_path(output_dir)
    config = load_config()

    meta = {
        "models": ",".join(model_names),
        "mode": mode,
        "tool": config["app"]["tool_name"],
        "version": config["app"]["version"],
        "timestamp": str(time.time()),
    }

    save_state(output_path, merged_state, meta)

    metadata_folder = metadata_dir / f"meta_{version}"
    metadata_folder.mkdir(parents=True, exist_ok=True)

    metadata_txt_path = metadata_folder / "metadata.txt"
    with open(metadata_txt_path, "w", encoding="utf-8") as fh:
        fh.write(_build_metadata_header(config["app"]["version"], mode, model_names))
        fh.write(f"Output: {output_path.name}\n")
        fh.write(f"Backbone: {model_names[backbone_idx]} (idx {backbone_idx})\n")
        fh.write(f"Timestamp: {meta['timestamp']}\n")
        try:
            fh.write(f"Torch: {torch.__version__}\n")
        except Exception:
            pass
        fh.write("\nInputs:\n")
        # Hashes de modelos
        if model_paths:
            for p in model_paths:
                try:
                    digest = _blake2b_file(p)
                    fh.write(f"  MODEL {p.name}  blake2b={digest}\n")
                except Exception:
                    fh.write(f"  MODEL {p.name}  blake2b=ERROR\n")
        else:
            for name in model_names:
                fh.write(f"  MODEL {name}\n")
        # Hashes de LoRAs
        if lora_paths:
            for lp in lora_paths:
                try:
                    digest = _blake2b_file(lp)
                    fh.write(f"  LORA  {lp.name}  blake2b={digest}\n")
                except Exception:
                    fh.write(f"  LORA  {lp.name}  blake2b=ERROR\n")

        # Exact configuration used (as raw kwargs)
        if yaml_kwargs:
            fh.write("\nConfiguration (kwargs):\n")
            for k, v in yaml_kwargs.items():
                fh.write(f"  {k}: {v}\n")

    yaml_params = {
        "mode": mode,
        "model_names": model_names,
        "backbone_idx": backbone_idx,
        "version": version,
    }
    yaml_params.update(yaml_kwargs)

    try:
        batch_yaml = generate_batch_config_yaml(**yaml_params)
    except Exception as exc:  # pragma: no cover - fail gracefully
        warning_path = metadata_folder / "batch_config.error.txt"
        with open(warning_path, "w", encoding="utf-8") as fh:
            fh.write(f"Could not generate batch_config.yaml: {exc}\n")
        return output_path, metadata_folder, version

    batch_yaml_path = metadata_folder / "batch_config.yaml"
    with open(batch_yaml_path, "w", encoding="utf-8") as fh:
        fh.write(batch_yaml)

    return output_path, metadata_folder, version
