import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file as st_load
from safetensors.torch import save_file as st_save

from xlfusion.config import ensure_dirs
from xlfusion.presets import batch_job_to_runtime_state, inspect_recovery_source, load_single_job_preset
from xlfusion.runtime import execute_merge_job
from xlfusion.types import MergeJobConfig
from xlfusion.validation import validate_merge_request


def _save_model(path: Path, value: float) -> None:
    st_save(
        {
            "model.diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.weight": torch.full((2, 2), value),
            "model.diffusion_model.mid_block.attentions.0.transformer_blocks.0.attn2.to_v.weight": torch.full((2, 2), value),
            "model.diffusion_model.up_blocks.0.resnets.0.conv1.weight": torch.full((2, 2, 1, 1), value),
            "first_stage_model.decoder.conv_in.weight": torch.full((2, 2), value),
        },
        str(path),
        metadata={"title": path.stem},
    )


class MetadataRecoveryRoundtripTests(unittest.TestCase):
    def test_execute_then_recover_and_reexecute_matches(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            root.joinpath("config.yaml").write_text("app:\n  tool_name: XLFusion\n", encoding="utf-8")
            models_dir, _loras_dir, output_dir, metadata_dir = ensure_dirs(root)

            m1 = models_dir / "a.safetensors"
            m2 = models_dir / "b.safetensors"
            _save_model(m1, 1.0)
            _save_model(m2, 3.0)

            with patch("xlfusion.memory.PSUTIL_AVAILABLE", False):
                validation = validate_merge_request(
                    mode="legacy",
                    model_paths=[m1, m2],
                    backbone=0,
                    weights=[0.25, 0.75],
                    only_unet=False,
                    component_policy={"vae": "merge", "text_encoder": "exclude", "other": "exclude"},
                )
                self.assertTrue(validation.valid, validation.errors)

                job = MergeJobConfig(
                    mode="legacy",
                    model_paths=validation.normalized["model_paths"],
                    model_names=validation.normalized["model_names"],
                    backbone_idx=validation.normalized["backbone_idx"],
                    output_base_name="RoundTrip",
                    weights=validation.normalized["weights"],
                    block_multipliers=validation.normalized.get("block_multipliers"),
                    crossattn_boosts=validation.normalized.get("crossattn_boosts"),
                    loras=validation.normalized.get("loras"),
                    only_unet=bool(validation.normalized.get("only_unet")),
                    component_policy=validation.normalized.get("component_policy"),
                    execution={"mode": "low-memory", "progress": "quiet"},
                    job_name="RoundTrip",
                    job_description="Metadata roundtrip test",
                )

                first = execute_merge_job(output_dir, metadata_dir, job)
                self.assertTrue(first.output_path.exists())
                self.assertTrue((first.metadata_folder / "batch_config.yaml").exists())

                inspection = inspect_recovery_source(first.metadata_folder, root)
                self.assertEqual(inspection.missing_models, [])
                self.assertEqual(inspection.missing_loras, [])

                recovered_job = load_single_job_preset(inspection.batch_config_path)
                runtime_state = batch_job_to_runtime_state(recovered_job)

                recovered_validation = validate_merge_request(
                    mode=runtime_state["mode"],
                    model_paths=[models_dir / name for name in runtime_state["models"]],
                    backbone=runtime_state["config"]["backbone_idx"],
                    weights=runtime_state["config"]["weights"],
                    only_unet=bool(runtime_state["config"]["only_unet"]),
                    component_policy=runtime_state["config"]["component_policy"],
                )
                self.assertTrue(recovered_validation.valid, recovered_validation.errors)

                job2 = MergeJobConfig(
                    mode=recovered_validation.normalized["mode"],
                    model_paths=recovered_validation.normalized["model_paths"],
                    model_names=recovered_validation.normalized["model_names"],
                    backbone_idx=recovered_validation.normalized["backbone_idx"],
                    output_base_name="RoundTrip",
                    weights=recovered_validation.normalized["weights"],
                    only_unet=bool(recovered_validation.normalized.get("only_unet")),
                    component_policy=recovered_validation.normalized.get("component_policy"),
                    execution=runtime_state["execution"],
                    job_name="RoundTripRecovered",
                    job_description="Recovered metadata execution",
                )

                second = execute_merge_job(output_dir, metadata_dir, job2)
                state_first = st_load(str(first.output_path), device="cpu")
                state_second = st_load(str(second.output_path), device="cpu")

                self.assertEqual(set(state_first.keys()), set(state_second.keys()))
                for key in state_first:
                    self.assertTrue(torch.allclose(state_first[key], state_second[key], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
