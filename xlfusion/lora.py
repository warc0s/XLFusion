"""
LoRA utilities for XLFusion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from safetensors.torch import load_file as st_load

from .blocks import classify_component_key, classify_submodule_key

DOWN_PAT = re.compile(r"\.lora_down\.weight$")
UP_PAT = re.compile(r"\.lora_up\.weight$")
ALPHA_KEYS = ["alpha", "lora_alpha", "ss_network_alpha", "scale"]

_UNET_REPLACEMENTS = (
    (".down.blocks.", ".down_blocks."),
    ("down.blocks.", "down_blocks."),
    (".up.blocks.", ".up_blocks."),
    ("up.blocks.", "up_blocks."),
    (".mid.block.", ".mid_block."),
    ("mid.block.", "mid_block."),
    (".middle.block.", ".middle_block."),
    ("middle.block.", "middle_block."),
    (".input.blocks.", ".input_blocks."),
    ("input.blocks.", "input_blocks."),
    (".output.blocks.", ".output_blocks."),
    ("output.blocks.", "output_blocks."),
    (".transformer.blocks.", ".transformer_blocks."),
    ("transformer.blocks.", "transformer_blocks."),
)


@dataclass
class LoraApplyReport:
    """Structured LoRA application audit."""

    lora_file: str
    scale: float
    total_pairs: int = 0
    applied_pairs: int = 0
    skipped_pairs: int = 0
    applied_by_component: Dict[str, int] = field(default_factory=dict)
    applied_by_submodule: Dict[str, int] = field(default_factory=dict)
    skipped_by_reason: Dict[str, int] = field(default_factory=dict)
    skipped_examples: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "lora_file": self.lora_file,
            "scale": self.scale,
            "total_pairs": self.total_pairs,
            "applied_pairs": self.applied_pairs,
            "skipped_pairs": self.skipped_pairs,
            "applied_by_component": dict(sorted(self.applied_by_component.items())),
            "applied_by_submodule": dict(sorted(self.applied_by_submodule.items())),
            "skipped_by_reason": dict(sorted(self.skipped_by_reason.items())),
            "skipped_examples": self.skipped_examples[:10],
        }


def lora_pairs_from_state(lora_state: Dict[str, torch.Tensor]) -> List[Tuple[str, str]]:
    downs = [k for k in lora_state.keys() if DOWN_PAT.search(k)]
    pairs: List[Tuple[str, str]] = []
    for d_key in downs:
        u_key = d_key.replace(".lora_down.weight", ".lora_up.weight")
        if u_key in lora_state:
            pairs.append((d_key, u_key))
    return pairs


def parse_lora_alpha_rank(meta: Dict[str, str], down: torch.Tensor) -> Tuple[float, int]:
    if not isinstance(meta, dict):
        raise ValueError("meta must be a dict")

    rank = int(down.shape[0]) if down.dim() >= 2 else int(down.numel())
    alpha = None
    for key in ALPHA_KEYS:
        if key in meta:
            try:
                alpha = float(meta[key])
                break
            except Exception:
                continue
    if alpha is None or alpha <= 0:
        alpha = float(rank)
    return alpha, rank


def _record_skip(report: LoraApplyReport, reason: str, example: str) -> None:
    report.skipped_pairs += 1
    report.skipped_by_reason[reason] = report.skipped_by_reason.get(reason, 0) + 1
    if len(report.skipped_examples) < 10:
        report.skipped_examples.append(example)


def _convert_lora_body_to_dotted(body: str) -> str:
    dotted = body.replace("_", ".")
    for src, dst in _UNET_REPLACEMENTS:
        dotted = dotted.replace(src, dst)
    dotted = dotted.replace(".text.model.", ".text_model.")
    dotted = dotted.replace("text.model.", "text_model.")
    dotted = dotted.replace(".to.", ".to_")
    dotted = dotted.replace(".lora_down.weight", ".weight")
    dotted = dotted.replace(".lora_up.weight", ".weight")
    dotted = dotted.replace(".lora.down.weight", ".weight")
    dotted = dotted.replace(".lora.up.weight", ".weight")
    return dotted


def _text_encoder_prefix_candidates(prefix: str) -> Sequence[str]:
    if prefix in {"lora_te1_", "lora_te_"}:
        return (
            "conditioner.embedders.0.transformer.",
            "text_encoder.",
            "cond_stage_model.",
        )
    if prefix == "lora_te2_":
        return (
            "conditioner.embedders.1.model.",
            "text_encoder_2.",
            "text_encoder2.",
        )
    if prefix == "lora_text_encoder_":
        return ("text_encoder.", "conditioner.embedders.0.transformer.")
    if prefix == "lora_text_encoder_2_":
        return ("text_encoder_2.", "conditioner.embedders.1.model.")
    return ()


def candidate_base_keys_from_lora_key(d_key: str) -> List[str]:
    candidates: List[str] = []
    if d_key.startswith("lora_unet_"):
        body = d_key[len("lora_unet_"):]
        candidates.append("model.diffusion_model." + _convert_lora_body_to_dotted(body))
        return candidates

    for prefix in ("lora_te1_", "lora_te2_", "lora_te_", "lora_text_encoder_2_", "lora_text_encoder_"):
        if d_key.startswith(prefix):
            body = d_key[len(prefix):]
            dotted = _convert_lora_body_to_dotted(body)
            for base_prefix in _text_encoder_prefix_candidates(prefix):
                candidates.append(base_prefix + dotted)
            return candidates

    return candidates


def map_lora_key_to_base(d_key: str) -> Optional[str]:
    candidates = candidate_base_keys_from_lora_key(d_key)
    return candidates[0] if candidates else None


def resolve_lora_base_key(d_key: str, base_state: Dict[str, torch.Tensor]) -> Optional[str]:
    for candidate in candidate_base_keys_from_lora_key(d_key):
        if candidate in base_state:
            return candidate
    return None


def _as_2d_if_1x1(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 4 and tensor.shape[2] == 1 and tensor.shape[3] == 1:
        return tensor.view(tensor.shape[0], tensor.shape[1])
    return tensor


def _compute_lora_delta(weight: torch.Tensor, down: torch.Tensor, up: torch.Tensor, scale_eff: float) -> Optional[torch.Tensor]:
    if weight.dim() == 2:
        down_2d = _as_2d_if_1x1(down)
        up_2d = _as_2d_if_1x1(up)
        if down_2d.dim() == 2 and up_2d.dim() == 2 and down_2d.shape[0] == up_2d.shape[1]:
            return torch.matmul(up_2d, down_2d) * scale_eff
        return None

    if weight.dim() != 4:
        return None

    d_is_1x1 = down.dim() == 4 and down.shape[2] == 1 and down.shape[3] == 1
    u_is_1x1 = up.dim() == 4 and up.shape[2] == 1 and up.shape[3] == 1

    if down.dim() == 2 and up.dim() == 2:
        delta_2d = torch.matmul(up, down) * scale_eff
        return delta_2d.view(weight.shape[0], weight.shape[1], 1, 1)

    if down.dim() == 4 and not d_is_1x1 and (up.dim() == 2 or u_is_1x1):
        up_2d = up.view(up.shape[0], up.shape[1]) if up.dim() == 4 else up
        return torch.einsum("or,rijk->oijk", up_2d, down) * scale_eff

    if up.dim() == 4 and not u_is_1x1 and (down.dim() == 2 or d_is_1x1):
        down_2d = down.view(down.shape[0], down.shape[1]) if down.dim() == 4 else down
        return torch.einsum("orhw,ri->oihw", up, down_2d) * scale_eff

    return None


def apply_single_lora_with_report(
    base_state: Dict[str, torch.Tensor],
    lora_path: Path,
    scale: float,
) -> LoraApplyReport:
    lora_state = st_load(str(lora_path), device="cpu")
    meta = getattr(lora_state, "metadata", {})
    if not isinstance(meta, dict):
        meta = {}

    report = LoraApplyReport(lora_file=lora_path.name, scale=float(scale))
    pairs = lora_pairs_from_state(lora_state)
    report.total_pairs = len(pairs)

    for d_key, u_key in pairs:
        down = lora_state[d_key].to(torch.float32)
        up = lora_state[u_key].to(torch.float32)
        alpha, rank = parse_lora_alpha_rank(meta, down)
        scale_eff = float(scale) * (alpha / max(rank, 1))

        base_key = resolve_lora_base_key(d_key, base_state)
        if base_key is None:
            _record_skip(report, "missing_target", d_key)
            continue

        weight = base_state[base_key].to(torch.float32)
        delta = _compute_lora_delta(weight, down, up, scale_eff)
        if delta is None:
            _record_skip(report, "unsupported_shape", base_key)
            continue
        if delta.shape != weight.shape:
            _record_skip(report, "shape_mismatch", base_key)
            continue

        base_state[base_key] = (weight + delta).to(torch.float32)
        report.applied_pairs += 1

        component = classify_component_key(base_key)
        submodule = classify_submodule_key(base_key)
        report.applied_by_component[component] = report.applied_by_component.get(component, 0) + 1
        report.applied_by_submodule[submodule] = report.applied_by_submodule.get(submodule, 0) + 1

    return report


def apply_single_lora(
    base_state: Dict[str, torch.Tensor],
    lora_path: Path,
    scale: float,
) -> Tuple[int, int]:
    report = apply_single_lora_with_report(base_state, lora_path, scale)
    return report.applied_pairs, report.skipped_pairs
