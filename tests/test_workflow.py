import tempfile
import unittest
from pathlib import Path

import torch

from Utils.workflow import save_merge_results


class WorkflowTests(unittest.TestCase):
    def test_save_merge_results_creates_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            output_dir = base / "output"
            metadata_dir = base / "metadata"
            output_dir.mkdir()
            metadata_dir.mkdir()

            state = {"linear.weight": torch.zeros(1)}
            models = ["model_a.safetensors", "model_b.safetensors"]

            output_path, metadata_folder, version = save_merge_results(
                output_dir,
                metadata_dir,
                state,
                models,
                "legacy",
                0,
                {"weights": [0.7, 0.3]},
            )

            self.assertTrue(output_path.exists())
            self.assertIn("meta_", metadata_folder.name)
            self.assertTrue((metadata_folder / "metadata.txt").exists())
            self.assertTrue((metadata_folder / "batch_config.yaml").exists())
            self.assertGreater(version, 0)

    def test_metadata_hashes_and_kwargs(self) -> None:
        import tempfile
        from safetensors.torch import save_file as st_save
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            output_dir = base / "output"
            metadata_dir = base / "metadata"
            output_dir.mkdir()
            metadata_dir.mkdir()

            # crear dos archivos de modelo y un lora
            m1 = base / "a.safetensors"
            m2 = base / "b.safetensors"
            st_save({"x": torch.zeros(1)}, str(m1))
            st_save({"y": torch.ones(1)}, str(m2))
            lora = base / "l.safetensors"
            st_save({"z": torch.ones(1)}, str(lora))

            state = {"linear.weight": torch.zeros(1)}
            output_path, metadata_folder, version = save_merge_results(
                output_dir,
                metadata_dir,
                state,
                [m1.name, m2.name],
                "legacy",
                0,
                {"weights": [0.6, 0.4]},
                model_paths=[m1, m2],
                lora_paths=[lora],
            )
            meta_txt = (metadata_folder / "metadata.txt").read_text(encoding="utf-8")
            self.assertIn("blake2b=", meta_txt)
            self.assertIn(m1.name, meta_txt)
            self.assertIn(lora.name, meta_txt)
            self.assertIn("weights", meta_txt)


if __name__ == "__main__":
    unittest.main()
