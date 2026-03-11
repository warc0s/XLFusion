"""
Block and component utilities for XLFusion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

# SDXL A1111 style naming constants
UNET_PREFIX = "model.diffusion_model."
VAE_PREFIXES = ("first_stage_model.", "vae.")
TEXT_ENCODER_PREFIXES = (
    "conditioner.embedders.0.transformer.",
    "conditioner.embedders.1.model.",
    "cond_stage_model.",
    "text_encoder.",
    "text_encoder_2.",
)

SDXL_BLOCK_GROUPS = ("down_0_1", "down_2_3", "mid", "up_0_1", "up_2_3")
SDXL_LEGACY_GROUPS = ("down", "mid", "up", "other")
SDXL_ATTN_BLOCKS = ("down", "mid", "up")


@dataclass(frozen=True)
class BlockMapping:
    """Block mapping registry entry for a checkpoint architecture/partition.

    For now, XLFusion ships with a single SDXL mapping. The registry exists so
    derived architectures can be added without rewriting the core merge logic.
    """

    name: str
    block_groups: tuple[str, ...]
    legacy_groups: tuple[str, ...]
    attn_blocks: tuple[str, ...]
    block_assignment_fn: Callable[[str], Optional[str]]
    coarse_group_fn: Callable[[str], Optional[str]]
    is_cross_attn_fn: Callable[[str], bool]
    attn_block_type_fn: Callable[[str], Optional[str]]

    def get_block_assignment(self, key: str) -> Optional[str]:
        return self.block_assignment_fn(key)

    def group_for_key(self, key: str) -> Optional[str]:
        return self.coarse_group_fn(key)

    def is_cross_attn_key(self, key: str) -> bool:
        return self.is_cross_attn_fn(key)

    def get_attn2_block_type(self, key: str) -> Optional[str]:
        return self.attn_block_type_fn(key)


_BLOCK_MAPPINGS = {
    "sdxl": BlockMapping(
        name="sdxl",
        block_groups=SDXL_BLOCK_GROUPS,
        legacy_groups=SDXL_LEGACY_GROUPS,
        attn_blocks=SDXL_ATTN_BLOCKS,
        block_assignment_fn=lambda key: get_block_assignment(key),
        coarse_group_fn=lambda key: group_for_key(key),
        is_cross_attn_fn=lambda key: is_cross_attn_key(key),
        attn_block_type_fn=lambda key: get_attn2_block_type(key),
    )
}


def get_block_mapping(name: str = "sdxl") -> BlockMapping:
    """Return a registered block mapping by name."""
    try:
        return _BLOCK_MAPPINGS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown block mapping '{name}'. Available: {sorted(_BLOCK_MAPPINGS)}") from exc


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


CROSS_TOKENS = (".attn2.",)
CROSS_PROJ = (".to_q.", ".to_k.", ".to_v.", ".to_out.0.")


def is_cross_attn_key_legacy(k: str) -> bool:
    if not k.startswith(UNET_PREFIX):
        return False
    s = k[len(UNET_PREFIX):]
    if not any(tok in s for tok in CROSS_TOKENS):
        return False
    return any(proj in s for proj in CROSS_PROJ)


def classify_component_key(key: str) -> str:
    """Classify a checkpoint key into a stable high-level component."""
    if key.startswith(UNET_PREFIX):
        return "unet"
    if any(key.startswith(prefix) for prefix in VAE_PREFIXES):
        return "vae"
    if any(key.startswith(prefix) for prefix in TEXT_ENCODER_PREFIXES):
        return "text_encoder"
    return "other"


def classify_submodule_key(key: str) -> str:
    """Classify a tensor key into a coarse functional submodule."""
    lowered = key.lower()
    if ".attn2." in lowered:
        return "cross_attention"
    if ".attn1." in lowered:
        return "self_attention"
    if ".to_q." in lowered or ".to_k." in lowered or ".to_v." in lowered or ".to_out.0." in lowered:
        return "attention_projection"
    if ".mlp." in lowered or ".ff." in lowered:
        return "mlp"
    if ".resnets." in lowered or ".conv" in lowered:
        return "convolution"
    if ".norm" in lowered:
        return "normalization"
    if "embeddings" in lowered or ".embed" in lowered:
        return "embedding"
    return "other"
