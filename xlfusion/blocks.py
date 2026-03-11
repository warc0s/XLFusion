"""
Block and component utilities for XLFusion.
"""
from typing import Optional

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
