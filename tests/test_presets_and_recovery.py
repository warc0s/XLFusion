import tempfile
import unittest
from pathlib import Path

from xlfusion.presets import (
    batch_job_to_runtime_state,
    inspect_recovery_source,
    load_single_job_preset,
    save_single_job_preset,
)


class PresetRecoveryTests(unittest.TestCase):
    def test_single_job_preset_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            preset_path = Path(td) / "preset.yaml"
            save_single_job_preset(
                preset_path,
                mode="hybrid",
                model_names=["a.safetensors", "b.safetensors"],
                backbone_idx=0,
                output_name="PresetOutput",
                execution={"mode": "low-memory", "progress": "simple", "log_every": 50},
                only_unet=False,
                component_policy={"vae": "backbone", "text_encoder": "exclude", "other": "exclude"},
                hybrid_config={
                    "down_0_1": {0: 0.7, 1: 0.3},
                    "down_2_3": {0: 0.6, 1: 0.4},
                    "mid": {0: 0.5, 1: 0.5},
                    "up_0_1": {0: 0.3, 1: 0.7},
                    "up_2_3": {1: 1.0},
                },
                attn2_locks={"up": 1},
            )

            job = load_single_job_preset(preset_path)
            runtime = batch_job_to_runtime_state(job)

            self.assertEqual(job.output_name, "PresetOutput")
            self.assertEqual(job.execution["mode"], "low-memory")
            self.assertEqual(runtime["mode"], "hybrid")
            self.assertEqual(runtime["config"]["attn2_locks"], {"up": 1})
            self.assertFalse(runtime["config"]["only_unet"])
            self.assertEqual(runtime["config"]["component_policy"]["vae"], "backbone")

    def test_metadata_recovery_detects_missing_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.yaml").write_text("app:\n  tool_name: XLFusion\n", encoding="utf-8")
            models_dir = root / "workspace" / "models"
            loras_dir = root / "workspace" / "loras"
            metadata_dir = root / "workspace" / "metadata" / "meta_7"
            models_dir.mkdir(parents=True, exist_ok=True)
            loras_dir.mkdir(parents=True, exist_ok=True)
            metadata_dir.mkdir(parents=True, exist_ok=True)

            (models_dir / "available.safetensors").write_text("placeholder", encoding="utf-8")
            save_single_job_preset(
                metadata_dir / "batch_config.yaml",
                mode="legacy",
                model_names=["available.safetensors", "missing.safetensors"],
                backbone_idx=0,
                output_name="Recovered",
                weights=[0.5, 0.5],
                loras=[{"file": "missing_lora.safetensors", "scale": 0.4}],
            )

            inspection = inspect_recovery_source(metadata_dir, root)

            self.assertEqual(inspection.metadata_folder, metadata_dir)
            self.assertIn("missing.safetensors", inspection.missing_models)
            self.assertIn("missing_lora.safetensors", inspection.missing_loras)
            self.assertTrue(any("Missing models" in item for item in inspection.warnings))


if __name__ == "__main__":
    unittest.main()
