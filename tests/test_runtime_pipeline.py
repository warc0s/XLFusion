import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file as st_save

from xlfusion.merge import MergeCancelled
from xlfusion.runtime import execute_merge_job
from xlfusion.types import MergeJobConfig
from xlfusion.validation import validate_merge_request


def _save_model(path: Path, value: float) -> None:
    st_save(
        {
            "model.diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.weight": torch.full((2, 2), value),
            "model.diffusion_model.mid_block.attentions.0.transformer_blocks.0.attn2.to_v.weight": torch.full((2, 2), value),
            "model.diffusion_model.up_blocks.0.resnets.0.conv1.weight": torch.full((2, 2, 1, 1), value),
        },
        str(path),
    )


def _save_lora(path: Path) -> None:
    # Base layer shape (2, 2); down (r, in), up (out, r)
    down = torch.zeros((1, 2), dtype=torch.float32)
    up = torch.zeros((2, 1), dtype=torch.float32)
    down[0, 0] = 1.0
    up[0, 0] = 1.0
    st_save(
        {
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_q.lora_down.weight": down,
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_q.lora_up.weight": up,
        },
        str(path),
        metadata={"alpha": "1"},
    )


class RuntimePipelineTests(unittest.TestCase):
    def test_runtime_executes_legacy_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "output"
            meta_dir = root / "metadata"
            out_dir.mkdir()
            meta_dir.mkdir()

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
                weights=[0.75, 0.25],
                loras=[(lora, 0.5)],
                only_unet=True,
            )
            self.assertTrue(result.valid, result.errors)

            job = MergeJobConfig(
                mode="legacy",
                model_paths=result.normalized["model_paths"],
                model_names=result.normalized["model_names"],
                backbone_idx=result.normalized["backbone_idx"],
                output_base_name="RuntimeLegacy",
                weights=result.normalized["weights"],
                block_multipliers=result.normalized.get("block_multipliers"),
                crossattn_boosts=result.normalized.get("crossattn_boosts"),
                loras=result.normalized.get("loras"),
                only_unet=bool(result.normalized.get("only_unet")),
                component_policy=result.normalized.get("component_policy"),
                execution={"mode": "low-memory", "progress": "quiet"},
                job_name="TestLegacy",
                job_description="Runtime legacy test",
            )
            artifacts = execute_merge_job(out_dir, meta_dir, job)
            self.assertTrue(artifacts.output_path.exists())
            self.assertTrue((artifacts.metadata_folder / "metadata.txt").exists())
            self.assertTrue((artifacts.metadata_folder / "batch_config.yaml").exists())
            self.assertGreater(artifacts.keys_processed, 0)
            self.assertEqual(len(artifacts.lora_reports), 1)
            self.assertEqual(artifacts.lora_reports[0]["lora_file"], lora.name)

    def test_runtime_executes_perres(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "output"
            meta_dir = root / "metadata"
            out_dir.mkdir()
            meta_dir.mkdir()

            m1 = root / "a.safetensors"
            m2 = root / "b.safetensors"
            _save_model(m1, 1.0)
            _save_model(m2, 2.0)

            assigns = {"down_0_1": 1, "down_2_3": 1, "mid": 1, "up_0_1": 1, "up_2_3": 1}
            result = validate_merge_request(
                mode="perres",
                model_paths=[m1, m2],
                backbone=0,
                assignments=assigns,
                only_unet=True,
            )
            self.assertTrue(result.valid, result.errors)

            job = MergeJobConfig(
                mode="perres",
                model_paths=result.normalized["model_paths"],
                model_names=result.normalized["model_names"],
                backbone_idx=result.normalized["backbone_idx"],
                output_base_name="RuntimePerRes",
                assignments=result.normalized.get("assignments"),
                attn2_locks=result.normalized.get("attn2_locks"),
                loras=result.normalized.get("loras"),
                only_unet=bool(result.normalized.get("only_unet")),
                component_policy=result.normalized.get("component_policy"),
                execution={"mode": "standard", "progress": "quiet"},
                job_name="TestPerRes",
                job_description="Runtime perres test",
            )
            artifacts = execute_merge_job(out_dir, meta_dir, job)
            self.assertTrue(artifacts.output_path.exists())
            self.assertGreater(artifacts.keys_processed, 0)

    def test_runtime_propagates_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "output"
            meta_dir = root / "metadata"
            out_dir.mkdir()
            meta_dir.mkdir()

            m1 = root / "a.safetensors"
            m2 = root / "b.safetensors"
            _save_model(m1, 1.0)
            _save_model(m2, 2.0)

            hybrid_cfg = {
                "down_0_1": {0: 1.0},
                "down_2_3": {0: 1.0},
                "mid": {0: 0.5, 1: 0.5},
                "up_0_1": {0: 1.0},
                "up_2_3": {0: 1.0},
            }
            result = validate_merge_request(
                mode="hybrid",
                model_paths=[m1, m2],
                backbone=0,
                hybrid_config=hybrid_cfg,
                only_unet=True,
            )
            self.assertTrue(result.valid, result.errors)

            import threading

            cancel_event = threading.Event()
            cancel_event.set()
            job = MergeJobConfig(
                mode="hybrid",
                model_paths=result.normalized["model_paths"],
                model_names=result.normalized["model_names"],
                backbone_idx=result.normalized["backbone_idx"],
                output_base_name="RuntimeHybrid",
                hybrid_config=result.normalized.get("hybrid_config"),
                attn2_locks=result.normalized.get("attn2_locks"),
                only_unet=bool(result.normalized.get("only_unet")),
                component_policy=result.normalized.get("component_policy"),
                execution={"mode": "low-memory", "progress": "quiet"},
                job_name="TestHybrid",
                job_description="Runtime hybrid cancel test",
            )

            with self.assertRaises(MergeCancelled):
                execute_merge_job(out_dir, meta_dir, job, cancel_event=cancel_event)


if __name__ == "__main__":
    unittest.main()

