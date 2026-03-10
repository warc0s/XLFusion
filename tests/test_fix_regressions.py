import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file as st_save

from xlfusion.batch_processor import BatchConfig, BatchJob, BatchProcessor, load_batch_config
from xlfusion.cli import prompt_block_merge
from xlfusion.validation import validate_merge_request


def _save_tiny_model(path: Path, value: float) -> None:
    st_save(
        {
            "model.diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.weight": torch.full((2, 2), value),
            "model.diffusion_model.mid_block.attentions.0.transformer_blocks.0.attn2.to_v.weight": torch.full((2, 2), value),
            "model.diffusion_model.up_blocks.0.resnets.0.conv1.weight": torch.full((2, 2, 1, 1), value),
        },
        str(path),
    )


class LegacyContractTests(unittest.TestCase):
    def test_prompt_block_merge_and_validation_use_coarse_groups(self) -> None:
        with patch("builtins.input", side_effect=["0:1.0,1:0.5", "0:0.8,1:0.2", "1:1.1"]):
            block_multipliers = prompt_block_merge(["model_a", "model_b"])

        self.assertEqual(
            block_multipliers,
            [
                {"down": 1.0, "mid": 0.8},
                {"down": 0.5, "mid": 0.2, "up": 1.1},
            ],
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            m1 = root / "a.safetensors"
            m2 = root / "b.safetensors"
            _save_tiny_model(m1, 1.0)
            _save_tiny_model(m2, 0.0)

            result = validate_merge_request(
                mode="legacy",
                model_paths=[m1, m2],
                backbone=0,
                weights=[0.5, 0.5],
                block_multipliers=block_multipliers,
            )

            self.assertTrue(result.valid, result.errors)
            self.assertEqual(result.normalized["block_multipliers"], block_multipliers)


class BatchFixRegressionTests(unittest.TestCase):
    def test_template_arithmetic_resolves_via_load_batch_config(self) -> None:
        config_text = """
version: "2.1"
global_settings:
  output_base: "out"
  continue_on_error: true
  max_parallel: 1
batch_jobs:
  - name: "templated"
    template: "style_transfer"
    models: ["a.safetensors", "b.safetensors"]
templates:
  style_transfer:
    default_params:
      primary_weight: 0.7
    config_template:
      mode: "hybrid"
      backbone: 0
      hybrid_config:
        down_0_1: {0: "{{primary_weight}}", 1: "{{1 - primary_weight}}"}
        down_2_3: {0: "{{primary_weight}}", 1: "{{1 - primary_weight}}"}
        mid: {0: 0.5, 1: 0.5}
        up_0_1: {1: "{{primary_weight}}", 0: "{{1 - primary_weight}}"}
        up_2_3: {1: "{{primary_weight}}", 0: "{{1 - primary_weight}}"}
"""
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "batch.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            config = load_batch_config(config_path)
            job = config.batch_jobs[0]

            self.assertAlmostEqual(job.hybrid_config["down_0_1"][0], 0.7, places=6)
            self.assertAlmostEqual(job.hybrid_config["down_0_1"][1], 0.3, places=6)
            self.assertAlmostEqual(job.hybrid_config["up_0_1"][0], 0.3, places=6)
            self.assertAlmostEqual(job.hybrid_config["up_0_1"][1], 0.7, places=6)

    def test_batch_job_with_partial_locks_saves_metadata_without_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.yaml").write_text('app:\n  tool_name: XLFusion\n  version: "2.15"\n', encoding="utf-8")
            models_dir = root / "workspace" / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            m1 = models_dir / "a.safetensors"
            m2 = models_dir / "b.safetensors"
            _save_tiny_model(m1, 1.0)
            _save_tiny_model(m2, 0.0)

            job = BatchJob(
                name="partial-locks",
                mode="perres",
                models=["a.safetensors", "b.safetensors"],
                backbone=0,
                assignments={"down_0_1": 0, "down_2_3": 0, "mid": 0, "up_0_1": 0, "up_2_3": 0},
                attn2_locks={"up": 1},
                output_name="PartialLocks",
            )
            config = BatchConfig(
                version="2.1",
                global_settings={"output_base": "batch_output", "continue_on_error": True, "max_parallel": 1},
                batch_jobs=[job],
            )
            validator = BatchProcessor(config, root, validate_only=True)
            del validator

            from xlfusion.batch_processor import BatchValidator

            batch_validator = BatchValidator(root)
            self.assertTrue(batch_validator.validate_config(config), batch_validator.errors)

            processor = BatchProcessor(config, root, validate_only=False)
            processor._process_job(job)

            self.assertTrue(job.success)
            self.assertIsNotNone(job.output_path)
            self.assertTrue(job.output_path.exists())
            self.assertTrue(job.output_path.name.startswith("PartialLocks_V"))

            meta_dirs = sorted((root / "workspace" / "metadata").glob("meta_*"))
            self.assertTrue(meta_dirs)
            meta_txt = (meta_dirs[-1] / "metadata.txt").read_text(encoding="utf-8")
            self.assertIn("blake2b=", meta_txt)
            self.assertIn("batch_config.yaml", "\n".join(p.name for p in meta_dirs[-1].iterdir()))

    def test_batch_metadata_includes_hashes_and_custom_output_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.yaml").write_text('app:\n  tool_name: XLFusion\n  version: "2.15"\n', encoding="utf-8")
            models_dir = root / "workspace" / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            m1 = models_dir / "a.safetensors"
            m2 = models_dir / "b.safetensors"
            _save_tiny_model(m1, 1.0)
            _save_tiny_model(m2, 0.0)

            job = BatchJob(
                name="hash-check",
                mode="legacy",
                models=["a.safetensors", "b.safetensors"],
                backbone=0,
                weights=[0.5, 0.5],
                output_name="BatchCustom",
            )
            config = BatchConfig(
                version="2.1",
                global_settings={"output_base": "batch_output", "continue_on_error": True, "max_parallel": 1},
                batch_jobs=[job],
            )

            from xlfusion.batch_processor import BatchValidator

            batch_validator = BatchValidator(root)
            self.assertTrue(batch_validator.validate_config(config), batch_validator.errors)

            processor = BatchProcessor(config, root, validate_only=False)
            processor._process_job(job)

            self.assertTrue(job.output_path.exists())
            self.assertTrue(job.output_path.name.startswith("BatchCustom_V"))
            self.assertGreater(job.keys_processed, 0)

            meta_dirs = sorted((root / "workspace" / "metadata").glob("meta_*"))
            self.assertTrue(meta_dirs)
            metadata_folder = meta_dirs[-1]
            self.assertTrue((metadata_folder / "batch_config.yaml").exists())
            meta_txt = (metadata_folder / "metadata.txt").read_text(encoding="utf-8")
            self.assertIn("blake2b=", meta_txt)
            self.assertIn("a.safetensors", meta_txt)
            self.assertIn("b.safetensors", meta_txt)


if __name__ == "__main__":
    unittest.main()
