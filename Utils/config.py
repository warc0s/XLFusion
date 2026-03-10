"""
Configuration utilities for XLFusion.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised indirectly
    yaml = None
    YAML_AVAILABLE = False


ConfigReporter = Optional[Callable[[str], None]]

DEFAULT_CONFIG: Dict[str, Any] = {
    "model_output": {
        "base_name": "XLFusion",
        "version_prefix": "V",
        "file_extension": ".safetensors",
        "output_dir": "output",
        "metadata_dir": "metadata",
        "auto_version": True,
    },
    "directories": {
        "models": "models",
        "loras": "loras",
        "output": "output",
        "metadata": "metadata",
    },
    "merge_defaults": {
        "legacy": {
            "cross_attention_boost": 1.0,
            "down_blocks_multiplier": 1.0,
            "mid_blocks_multiplier": 1.0,
            "up_blocks_multiplier": 1.0,
        },
        "perres": {
            "cross_attention_locks": False,
        },
        "hybrid": {
            "primary_model_weight": 0.7,
            "cross_attention_locks": False,
            "auto_normalize_weights": True,
            "minimum_weight_threshold": 0.01,
        },
    },
    "app": {
        "tool_name": "XLFusion",
        "version": "2.1",
    },
}


def _clone_defaults() -> Dict[str, Any]:
    return copy.deepcopy(DEFAULT_CONFIG)


def _resolve_config_root(root_dir: Path) -> Path:
    for candidate in [root_dir, *root_dir.parents]:
        if (candidate / "config.yaml").exists():
            return candidate
        if (candidate / "config.yaml.example").exists() and (candidate / "XLFusion.py").exists():
            return candidate
    return root_dir


def _report(reporter: ConfigReporter, message: str) -> None:
    if reporter:
        reporter(message)


def _matches_expected_type(value: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(value, bool)
    if isinstance(expected, int) and not isinstance(expected, bool):
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(expected, float):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(expected, str):
        return isinstance(value, str)
    if isinstance(expected, list):
        return isinstance(value, list)
    if isinstance(expected, dict):
        return isinstance(value, dict)
    return isinstance(value, type(expected))


def _merge_known_config(
    defaults: Dict[str, Any],
    override: Dict[str, Any],
    *,
    reporter: ConfigReporter = print,
    prefix: str = "",
) -> Dict[str, Any]:
    merged = copy.deepcopy(defaults)

    for key, value in override.items():
        path = f"{prefix}{key}"
        if key not in defaults:
            merged[key] = value
            continue

        default_value = defaults[key]
        if isinstance(default_value, dict):
            if not isinstance(value, dict):
                _report(
                    reporter,
                    f"Warning: config field '{path}' must be a mapping. Using defaults for that section.",
                )
                continue
            merged[key] = _merge_known_config(
                default_value,
                value,
                reporter=reporter,
                prefix=f"{path}.",
            )
            continue

        if not _matches_expected_type(value, default_value):
            expected_name = type(default_value).__name__
            _report(
                reporter,
                f"Warning: config field '{path}' has invalid type. Expected {expected_name}; using default value.",
            )
            continue

        if isinstance(default_value, float):
            merged[key] = float(value)
        else:
            merged[key] = value

    return merged


def load_config(
    *,
    root: Optional[Path] = None,
    reporter: ConfigReporter = print,
) -> Dict[str, Any]:
    """Load ``config.yaml`` with safe fallbacks and partial overrides."""
    root_dir = _resolve_config_root(root or Path(__file__).resolve().parent.parent)
    config_path = root_dir / "config.yaml"
    config = _clone_defaults()

    if not config_path.exists():
        _report(
            reporter,
            "Info: config.yaml not found. Using built-in defaults. Copy config.yaml.example if you want local overrides.",
        )
        return config

    if not YAML_AVAILABLE:
        _report(
            reporter,
            "Warning: PyYAML is not installed. config.yaml cannot be read, so XLFusion will use built-in defaults.",
        )
        return config

    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            raw_data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        _report(
            reporter,
            f"Warning: config.yaml is not valid YAML ({exc}). Using built-in defaults.",
        )
        return config
    except OSError as exc:
        _report(
            reporter,
            f"Warning: config.yaml could not be read ({exc}). Using built-in defaults.",
        )
        return config

    if raw_data is None:
        return config

    if not isinstance(raw_data, dict):
        _report(
            reporter,
            "Warning: config.yaml must contain a top-level mapping. Using built-in defaults.",
        )
        return config

    return _merge_known_config(config, raw_data, reporter=reporter)


def ensure_dirs(root: Path) -> Tuple[Path, Path, Path, Path]:
    config = load_config(root=root)
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
    config = load_config(root=output_dir.parent, reporter=None)
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
    loras: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate a batch configuration YAML compatible with XLFusion format."""
    if not YAML_AVAILABLE:
        raise RuntimeError("PyYAML is required to generate batch YAML files.")

    config = {
        "version": "2.1",
        "global_settings": {
            "output_base": "recreated_output",
            "continue_on_error": True,
            "max_parallel": 1,
            "log_level": "INFO",
        },
        "batch_jobs": [],
    }

    job = {
        "name": f"Recreated_Model_V{version}",
        "mode": mode,
        "description": "Recreated from metadata backup",
        "models": model_names,
        "backbone": backbone_idx,
        "output_name": f"Recreated_V{version}",
    }

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

    return yaml.safe_dump(config, default_flow_style=False, sort_keys=False)
