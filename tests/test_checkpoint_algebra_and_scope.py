import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file as st_save

from xlfusion.merge import merge_perres, stream_checkpoint_algebra_from_paths, stream_weighted_merge_from_paths
from xlfusion.validation import validate_merge_request


def _save_scope_model(path: Path, unet_value: float, vae_value: float, text_value: float) -> None:
    st_save(
        {
            "model.diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.weight": torch.full((2, 2), unet_value),
            "first_stage_model.decoder.conv_in.weight": torch.full((2, 2, 1, 1), vae_value),
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.mlp.fc1.weight": torch.full((2, 2), text_value),
        },
        str(path),
    )


class CheckpointAlgebraTests(unittest.TestCase):
    def test_checkpoint_algebra_matches_formula(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = root / "a.safetensors"
            b = root / "b.safetensors"
            c = root / "c.safetensors"
            _save_scope_model(a, 1.0, 10.0, 20.0)
            _save_scope_model(b, 3.0, 12.0, 22.0)
            _save_scope_model(c, 2.0, 8.0, 18.0)

            merged, stats, audit = stream_checkpoint_algebra_from_paths([a, b, c], 0.5, only_unet=False, component_policy={"vae": "merge", "text_encoder": "merge", "other": "merge"})

            key = "model.diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.weight"
            self.assertTrue(torch.allclose(merged[key], torch.full_like(merged[key], 1.5), atol=1e-6))
            self.assertGreater(stats["formula_applied"], 0)
            self.assertEqual(audit["formula"], "A + alpha(B - C)")

    def test_component_scope_is_respected_in_merges(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = root / "a.safetensors"
            b = root / "b.safetensors"
            _save_scope_model(a, 1.0, 10.0, 20.0)
            _save_scope_model(b, 3.0, 30.0, 40.0)

            merged, _ = stream_weighted_merge_from_paths(
                [a, b],
                [0.5, 0.5],
                base_idx=0,
                only_unet=False,
                component_policy={"vae": "merge", "text_encoder": "exclude", "other": "exclude"},
            )

            self.assertIn("first_stage_model.decoder.conv_in.weight", merged)
            self.assertNotIn("conditioner.embedders.0.transformer.text_model.encoder.layers.0.mlp.fc1.weight", merged)

    def test_perres_scope_copies_backbone_components(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = root / "a.safetensors"
            b = root / "b.safetensors"
            _save_scope_model(a, 1.0, 10.0, 20.0)
            _save_scope_model(b, 3.0, 30.0, 40.0)

            merged = merge_perres(
                [a, b],
                {"down_0_1": 1, "down_2_3": 1, "mid": 1, "up_0_1": 1, "up_2_3": 1},
                backbone_idx=0,
                only_unet=False,
                component_policy={"vae": "backbone", "text_encoder": "exclude", "other": "exclude"},
            )

            self.assertTrue(torch.allclose(merged["first_stage_model.decoder.conv_in.weight"], torch.full((2, 2, 1, 1), 10.0)))
            self.assertNotIn("conditioner.embedders.0.transformer.text_model.encoder.layers.0.mlp.fc1.weight", merged)

    def test_validation_normalizes_component_scope(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = root / "a.safetensors"
            b = root / "b.safetensors"
            _save_scope_model(a, 1.0, 10.0, 20.0)
            _save_scope_model(b, 3.0, 30.0, 40.0)

            result = validate_merge_request(
                mode="legacy",
                model_paths=[a, b],
                backbone=0,
                weights=[0.5, 0.5],
                only_unet=False,
                component_policy={"vae": "merge", "text_encoder": "exclude", "other": "exclude"},
            )

            self.assertTrue(result.valid, result.errors)
            self.assertFalse(result.normalized["only_unet"])
            self.assertEqual(result.normalized["component_policy"]["vae"], "merge")


if __name__ == "__main__":
    unittest.main()
