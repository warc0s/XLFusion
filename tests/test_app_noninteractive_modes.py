import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from safetensors.torch import save_file as st_save

from xlfusion.app import main as app_main
from xlfusion.config import resolve_app_context
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
        metadata={"title": path.stem},
    )


class AppNonInteractiveModesTests(unittest.TestCase):
    def test_batch_mode_success_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            root.joinpath("config.yaml").write_text("app:\n  tool_name: XLFusion\n", encoding="utf-8")
            context = resolve_app_context(root, reporter=None)
            batch_path = root / "batch.yaml"
            batch_path.write_text("global_settings: {}\nbatch_jobs: []\n", encoding="utf-8")

            fake_config = MagicMock()
            fake_validator = MagicMock()
            fake_validator.validate_config.return_value = True
            fake_validator.errors = []
            fake_validator.warnings = []

            processor_instance = MagicMock()
            processor_instance.process_batch.return_value = {"failed_jobs": 0}

            with patch("xlfusion.app.resolve_app_context", return_value=context):
                with patch("xlfusion.batch_processor.load_batch_config", return_value=fake_config):
                    with patch("xlfusion.batch_processor.BatchValidator", return_value=fake_validator):
                        with patch("xlfusion.batch_processor.BatchProcessor", return_value=processor_instance) as proc_cls:
                            code = app_main(["--batch", str(batch_path), "--validate-only"])

            self.assertEqual(code, 0)
            proc_cls.assert_called_once()

    def test_batch_mode_invalid_config_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            root.joinpath("config.yaml").write_text("app:\n  tool_name: XLFusion\n", encoding="utf-8")
            context = resolve_app_context(root, reporter=None)
            batch_path = root / "batch.yaml"
            batch_path.write_text("global_settings: {}\nbatch_jobs: []\n", encoding="utf-8")

            fake_config = MagicMock()
            fake_validator = MagicMock()
            fake_validator.validate_config.return_value = False
            fake_validator.errors = ["bad"]
            fake_validator.warnings = ["warn"]

            with patch("xlfusion.app.resolve_app_context", return_value=context):
                with patch("xlfusion.batch_processor.load_batch_config", return_value=fake_config):
                    with patch("xlfusion.batch_processor.BatchValidator", return_value=fake_validator):
                        with patch("xlfusion.batch_processor.BatchProcessor") as proc_cls:
                            code = app_main(["--batch", str(batch_path)])

            self.assertEqual(code, 1)
            proc_cls.assert_not_called()

    def test_recovery_export_and_run_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            root.joinpath("config.yaml").write_text("app:\n  tool_name: XLFusion\n", encoding="utf-8")
            context = resolve_app_context(root, reporter=None)

            m1 = context.models_dir / "a.safetensors"
            m2 = context.models_dir / "b.safetensors"
            _save_model(m1, 1.0)
            _save_model(m2, 0.0)

            validation = validate_merge_request(
                mode="legacy",
                model_paths=[m1, m2],
                backbone=0,
                weights=[0.5, 0.5],
                only_unet=True,
            )
            self.assertTrue(validation.valid, validation.errors)
            job = MergeJobConfig(
                mode="legacy",
                model_paths=validation.normalized["model_paths"],
                model_names=validation.normalized["model_names"],
                backbone_idx=validation.normalized["backbone_idx"],
                output_base_name="RecoverSource",
                weights=validation.normalized["weights"],
                only_unet=bool(validation.normalized.get("only_unet")),
                component_policy=validation.normalized.get("component_policy"),
                execution={"mode": "low-memory", "progress": "quiet"},
                job_name="RecoverSource",
                job_description="Recovery export/run test",
            )
            artifacts = execute_merge_job(context.output_dir, context.metadata_dir, job)
            exported = root / "exported.yaml"

            with patch("xlfusion.app.resolve_app_context", return_value=context):
                code_export = app_main(
                    ["--recover-metadata", str(artifacts.metadata_folder), "--export-recovered", str(exported)]
                )
                self.assertEqual(code_export, 0)
                self.assertTrue(exported.exists())

                code_run = app_main(["--recover-metadata", str(artifacts.metadata_folder), "--run-recovered"])
                self.assertEqual(code_run, 0)

    def test_algebra_success_path_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            root.joinpath("config.yaml").write_text("app:\n  tool_name: XLFusion\n", encoding="utf-8")
            context = resolve_app_context(root, reporter=None)

            _save_model(context.models_dir / "a.safetensors", 1.0)
            _save_model(context.models_dir / "b.safetensors", 2.0)
            _save_model(context.models_dir / "c.safetensors", 3.0)

            with patch("xlfusion.app.resolve_app_context", return_value=context):
                code = app_main(
                    [
                        "--algebra",
                        "0",
                        "1",
                        "2",
                        "--alpha",
                        "0.25",
                        "--algebra-output",
                        "AlgebraTest",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue(list(context.output_dir.glob("AlgebraTest_*.safetensors")))

    def test_analysis_compare_exports_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            root.joinpath("config.yaml").write_text("app:\n  tool_name: XLFusion\n", encoding="utf-8")
            context = resolve_app_context(root, reporter=None)

            _save_model(context.models_dir / "a.safetensors", 1.0)
            _save_model(context.models_dir / "b.safetensors", 2.0)
            report = root / "report.json"

            with patch("xlfusion.app.resolve_app_context", return_value=context):
                code = app_main(["--compare", "0", "1", "--export-analysis", str(report)])
            self.assertEqual(code, 0)
            self.assertTrue(report.exists())


if __name__ == "__main__":
    unittest.main()

