import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file as st_save

from xlfusion.lora import apply_single_lora_with_report, resolve_lora_base_key


def _save_text_encoder_lora(path: Path) -> None:
    st_save(
        {
            "lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_up.weight": torch.tensor([[1.0], [0.0]], dtype=torch.float32),
        },
        str(path),
        metadata={"alpha": "1"},
    )


class LoraTextEncoderTests(unittest.TestCase):
    def test_text_encoder_lora_is_resolved_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            lora = root / "text.safetensors"
            _save_text_encoder_lora(lora)
            state = {
                "conditioner.embedders.0.transformer.text_model.encoder.layers.0.mlp.fc1.weight": torch.ones((2, 2), dtype=torch.float32),
            }

            base_key = resolve_lora_base_key(
                "lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight",
                state,
            )
            self.assertEqual(
                base_key,
                "conditioner.embedders.0.transformer.text_model.encoder.layers.0.mlp.fc1.weight",
            )

            before = state[base_key].clone()
            report = apply_single_lora_with_report(state, lora, 0.5)

            self.assertEqual(report.applied_pairs, 1)
            self.assertEqual(report.applied_by_component["text_encoder"], 1)
            self.assertEqual(report.applied_by_submodule["mlp"], 1)
            self.assertTrue(torch.any(state[base_key] != before))


if __name__ == "__main__":
    unittest.main()
