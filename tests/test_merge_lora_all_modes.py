import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file as st_save

from Utils.merge import (
    stream_weighted_merge_from_paths,
    merge_perres,
    merge_hybrid,
)
from Utils.lora import apply_single_lora, map_lora_key_to_base


def _make_tensors(value: float) -> dict:
    """Crea un estado minimo con algunas claves UNet relevantes."""
    W = torch.full((8, 8), value, dtype=torch.float32)
    C = torch.full((4, 4, 1, 1), value, dtype=torch.float32)
    state = {
        # Cross-attn (down block)
        "model.diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.weight": W.clone(),
        "model.diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight": W.clone(),
        # Mid block cross-attn
        "model.diffusion_model.mid_block.attentions.0.transformer_blocks.0.attn2.to_v.weight": W.clone(),
        # Up block conv (no lora target, solo para variar)
        "model.diffusion_model.up_blocks.0.resnets.0.conv1.weight": C.clone(),
    }
    return state


def _save_model(path: Path, value: float) -> None:
    meta = {"title": path.stem}
    st_save(_make_tensors(value), str(path), metadata=meta)


def _save_lora(path: Path, rank: int = 2, scale: float = 1.0) -> None:
    """Crea una LoRA minima que afecte al peso to_q del down block 0."""
    # Base layer shape (8, 8); down (r, in), up (out, r)
    down = torch.zeros((rank, 8), dtype=torch.float32)
    up = torch.zeros((8, rank), dtype=torch.float32)
    # Introduce una delta simple
    down[0, 0] = 1.0
    up[0, 0] = 1.0

    state = {
        # Usar sufijo con punto antes de lora_down/lora_up para que el regex coincida
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_q.lora_down.weight": down,
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_q.lora_up.weight": up,
    }
    meta = {"alpha": str(scale * rank)}
    st_save(state, str(path), metadata=meta)


class MergeLoRATests(unittest.TestCase):
    def test_lora_applies_to_plain_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            lora = tmp / "l1.safetensors"
            _save_lora(lora, rank=2, scale=1.0)
            state = _make_tensors(1.0)
            key = "model.diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.weight"
            before = state[key].clone()
            d_key = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_q_lora_down.weight"
            mapped = map_lora_key_to_base(d_key)
            self.assertIn(mapped, state)
            applied, skipped = apply_single_lora(state, lora, 0.5)
            self.assertGreater(applied, 0)
            self.assertTrue(torch.any(state[key] != before))
    def test_legacy_merge_with_lora(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            m1 = tmp / "m1.safetensors"
            m2 = tmp / "m2.safetensors"
            _save_model(m1, 1.0)
            _save_model(m2, 0.0)
            lora = tmp / "l1.safetensors"
            _save_lora(lora, rank=2, scale=1.0)

            merged, _ = stream_weighted_merge_from_paths([m1, m2], [0.25, 0.75], base_idx=0)

            key = "model.diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.weight"
            # Verifica mapeo de clave LoRA -> base
            d_key = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_q_lora_down.weight"
            self.assertEqual(map_lora_key_to_base(d_key), key)
            before = merged[key].clone()
            # Verifica que el mapeo devuelva una clave existente en el merge
            d_key = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_q_lora_down.weight"
            mapped = map_lora_key_to_base(d_key)
            self.assertIn(mapped, merged)
            applied, skipped = apply_single_lora(merged, lora, 0.5)
            self.assertGreater(applied, 0)
            self.assertTrue(torch.any(merged[key] != before))

    def test_perres_merge_with_lora(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            m1 = tmp / "m1.safetensors"
            m2 = tmp / "m2.safetensors"
            _save_model(m1, 1.0)  # modelo A
            _save_model(m2, 2.0)  # modelo B
            lora = tmp / "l1.safetensors"
            _save_lora(lora, rank=2, scale=1.0)

            assigns = {
                "down_0_1": 1,  # usar m2 (valor 2.0) para down
                "down_2_3": 0,
                "mid": 0,
                "up_0_1": 1,   # usar m2 (valor 2.0) para up
                "up_2_3": 0,
            }
            merged = merge_perres([m1, m2], assigns, backbone_idx=0)

            # Clave de down debe provenir de m2 -> valor base 2.0 antes de LoRA
            k_down_q = "model.diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.weight"
            self.assertTrue(torch.allclose(merged[k_down_q], torch.full_like(merged[k_down_q], 2.0)))

            before = merged[k_down_q].clone()
            applied, _ = apply_single_lora(merged, lora, 0.5)
            self.assertGreater(applied, 0)
            self.assertTrue(torch.any(merged[k_down_q] != before))

    def test_hybrid_merge_with_lora(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            m1 = tmp / "m1.safetensors"
            m2 = tmp / "m2.safetensors"
            _save_model(m1, 1.0)
            _save_model(m2, 3.0)
            lora = tmp / "l1.safetensors"
            _save_lora(lora, rank=2, scale=1.0)

            cfg = {
                "down_0_1": {0: 0.25, 1: 0.75},
                "down_2_3": {0: 1.0},
                "mid": {1: 1.0},
                "up_0_1": {0: 0.5, 1: 0.5},
                "up_2_3": {0: 1.0},
            }
            merged = merge_hybrid([m1, m2], cfg, backbone_idx=0)

            # Verifica una media ponderada en una clave down (valor esperado 0.25*1 + 0.75*3 = 2.5)
            k_down_k = "model.diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight"
            self.assertTrue(torch.allclose(merged[k_down_k], torch.full_like(merged[k_down_k], 2.5), atol=1e-6))

            # Aplica LoRA y comprueba cambio
            before = merged[k_down_k].clone()
            applied, _ = apply_single_lora(merged, lora, 0.4)
            self.assertGreater(applied, 0)
            # k_down_k no es objetivo del lora definido (to_q), podria no cambiar
            # Verificamos cambio en to_q
            k_down_q = "model.diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.weight"
            self.assertTrue(torch.any(merged[k_down_q] != before))


if __name__ == "__main__":
    unittest.main()
