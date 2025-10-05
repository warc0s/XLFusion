#!/usr/bin/env python3
import os
from pathlib import Path
import torch
from safetensors.torch import save_file as st_save

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

def make_state(seed: int):
    torch.manual_seed(seed)
    state = {}
    # Minimal set of UNet keys across blocks and attn2/non-attn2
    keys = [
        # Down block attn2
        "model.diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.weight",
        "model.diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight",
        "model.diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v.weight",
        "model.diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_out.0.weight",
        # Mid block attn2
        "model.diffusion_model.mid_block.attentions.0.transformer_blocks.0.attn2.to_q.weight",
        # Up block attn2
        "model.diffusion_model.up_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.weight",
        # Some non-attn UNet weights
        "model.diffusion_model.input_blocks.0.0.weight",
        "model.diffusion_model.output_blocks.0.0.weight",
    ]
    for k in keys:
        state[k] = torch.randn(4, 4)
    # Include a couple non-UNet keys (will be ignored in only_unet=True)
    state["first_stage_model.decoder.conv_in.weight"] = torch.randn(4, 4)
    return state

def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    specs = [
        ("test_model_a.safetensors", 42),
        ("test_model_b.safetensors", 1234),
        ("test_model_c.safetensors", 9876),
    ]
    for name, seed in specs:
        path = MODELS / name
        state = make_state(seed)
        meta = {"title": name, "format": "test"}
        st_save(state, str(path), metadata=meta)
        print(f"Created {path}")

if __name__ == "__main__":
    main()

