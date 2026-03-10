import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file as st_save

from xlfusion.batch_processor import load_batch_config
from xlfusion.config import load_config
from xlfusion.validation import format_preflight_plan, validate_merge_request


def _save_model(path: Path, value: float, shape=(2, 2)) -> None:
    st_save(
        {
            "model.diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.weight": torch.full(shape, value),
            "model.diffusion_model.mid_block.attentions.0.transformer_blocks.0.attn2.to_v.weight": torch.full(shape, value),
            "model.diffusion_model.up_blocks.0.resnets.0.conv1.weight": torch.full(shape, value),
        },
        str(path),
    )


def _save_lora(path: Path) -> None:
    st_save(
        {
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_q.lora_down.weight": torch.ones((1, 2)),
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_q.lora_up.weight": torch.ones((2, 1)),
        },
        str(path),
        metadata={"alpha": "1"},
    )


class ConfigLoadingTests(unittest.TestCase):
    def test_missing_config_uses_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            messages = []
            config = load_config(root=Path(tmp), reporter=messages.append)
            self.assertEqual(config["app"]["version"], "2.15")
            self.assertEqual(config["directories"]["models"], "workspace/models")
            self.assertTrue(any("config.yaml not found" in message for message in messages))

    def test_partial_config_merges_with_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.yaml").write_text(
                "model_output:\n  base_name: CustomFusion\napp:\n  tool_name: LabFusion\n",
                encoding="utf-8",
            )
            config = load_config(root=root, reporter=None)
            self.assertEqual(config["model_output"]["base_name"], "CustomFusion")
            self.assertEqual(config["app"]["tool_name"], "LabFusion")
            self.assertEqual(config["directories"]["metadata"], "workspace/metadata")

    def test_invalid_yaml_falls_back_to_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.yaml").write_text("model_output: [broken\n", encoding="utf-8")
            messages = []
            config = load_config(root=root, reporter=messages.append)
            self.assertEqual(config["model_output"]["base_name"], "XLFusion")
            self.assertTrue(any("not valid YAML" in message for message in messages))


class ValidationTests(unittest.TestCase):
    def test_legacy_validation_builds_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m1 = root / "a.safetensors"
            m2 = root / "b.safetensors"
            lora = root / "style.safetensors"
            _save_model(m1, 1.0)
            _save_model(m2, 0.0)
            _save_lora(lora)

            result = validate_merge_request(
                mode="legacy",
                model_paths=[m1, m2],
                backbone=0,
                weights=[1.0, 0.0],
                loras=[(lora, 0.5)],
            )

            self.assertTrue(result.valid)
            self.assertIsNotNone(result.preflight)
            self.assertEqual(result.preflight.selected_models, [m1.name])
            self.assertEqual(result.preflight.loaded_models, [m1.name, m2.name])
            self.assertIn("Fusion preflight", format_preflight_plan(result.preflight))
            self.assertEqual(result.normalized["loras"][0]["file"], lora.name)

    def test_invalid_perres_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m1 = root / "a.safetensors"
            m2 = root / "b.safetensors"
            _save_model(m1, 1.0)
            _save_model(m2, 0.0)

            result = validate_merge_request(
                mode="perres",
                model_paths=[m1, m2],
                backbone=0,
                assignments={"down_0_1": 0, "mid": 1},
                attn2_locks={"down": 5},
            )

            self.assertFalse(result.valid)
            fields = {item.field for item in result.errors}
            self.assertIn("assignments.down_2_3", fields)
            self.assertIn("attn2_locks.down", fields)

    def test_hybrid_validation_reports_compatibility_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m1 = root / "a.safetensors"
            m2 = root / "b.safetensors"
            _save_model(m1, 1.0, shape=(2, 2))
            _save_model(m2, 2.0, shape=(3, 3))

            result = validate_merge_request(
                mode="hybrid",
                model_paths=[m1, m2],
                backbone=0,
                hybrid_config={
                    "down_0_1": {0: 0.7, 1: 0.3},
                    "down_2_3": {0: 1.0},
                    "mid": {0: 0.5, 1: 0.5},
                    "up_0_1": {1: 1.0},
                    "up_2_3": {0: 1.0},
                },
            )

            self.assertTrue(result.valid)
            self.assertTrue(any(item.field == "compatibility" for item in result.warnings))


class BatchConfigLoadTests(unittest.TestCase):
    def test_invalid_batch_yaml_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.yaml"
            path.write_text("version: [bad\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not valid YAML"):
                load_batch_config(path)


if __name__ == "__main__":
    unittest.main()
