import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from xlfusion.batch_processor import main as batch_main
from xlfusion.config import resolve_app_context


class BatchProcessorMainTests(unittest.TestCase):
    def test_missing_config_path_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "missing.yaml"
            code = batch_main([str(missing)])
        self.assertEqual(code, 1)

    def test_validate_only_success_does_not_execute(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "batch.yaml"
            cfg.write_text("global_settings: {output_base: out, continue_on_error: true, max_parallel: 1}\nbatch_jobs: []\n", encoding="utf-8")
            context = resolve_app_context(Path(td), reporter=None)

            fake_config = MagicMock()
            fake_config.templates = {}
            fake_validator = MagicMock()
            fake_validator.validate_config.return_value = True
            fake_validator.errors = []
            fake_validator.warnings = []

            with patch("xlfusion.batch_processor.load_batch_config", return_value=fake_config):
                with patch("xlfusion.batch_processor.resolve_app_context", return_value=context):
                    with patch("xlfusion.batch_processor.BatchValidator", return_value=fake_validator):
                        with patch("xlfusion.batch_processor.BatchProcessor") as processor_cls:
                            code = batch_main([str(cfg), "--validate-only"])

            self.assertEqual(code, 0)
            processor_cls.assert_not_called()

    def test_template_override_requires_known_template(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "batch.yaml"
            cfg.write_text("global_settings: {}\nbatch_jobs: []\n", encoding="utf-8")

            fake_config = MagicMock()
            fake_config.templates = {"known": {"config_template": {"mode": "legacy"}}}
            fake_config.batch_jobs = []

            with patch("xlfusion.batch_processor.load_batch_config", return_value=fake_config):
                code = batch_main([str(cfg), "--template", "missing"])

            self.assertEqual(code, 1)

    def test_execute_success_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "batch.yaml"
            cfg.write_text("global_settings: {output_base: out, continue_on_error: true, max_parallel: 1}\nbatch_jobs: []\n", encoding="utf-8")
            context = resolve_app_context(root, reporter=None)

            fake_config = MagicMock()
            fake_config.templates = {}
            fake_validator = MagicMock()
            fake_validator.validate_config.return_value = True
            fake_validator.errors = []
            fake_validator.warnings = []

            processor_instance = MagicMock()
            processor_instance.process_batch.return_value = {"failed_jobs": 0, "total_jobs": 0, "successful_jobs": 0}

            with patch("xlfusion.batch_processor.resolve_app_context", return_value=context):
                with patch("xlfusion.batch_processor.load_batch_config", return_value=fake_config):
                    with patch("xlfusion.batch_processor.BatchValidator", return_value=fake_validator):
                        with patch("xlfusion.batch_processor.BatchProcessor", return_value=processor_instance):
                            code = batch_main([str(cfg)])
            self.assertEqual(code, 0)

    def test_execute_failure_returns_one(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "batch.yaml"
            cfg.write_text("global_settings: {output_base: out, continue_on_error: true, max_parallel: 1}\nbatch_jobs: []\n", encoding="utf-8")
            context = resolve_app_context(root, reporter=None)

            fake_config = MagicMock()
            fake_config.templates = {}
            fake_validator = MagicMock()
            fake_validator.validate_config.return_value = True
            fake_validator.errors = []
            fake_validator.warnings = []

            processor_instance = MagicMock()
            processor_instance.process_batch.return_value = {"failed_jobs": 2, "total_jobs": 2, "successful_jobs": 0}

            with patch("xlfusion.batch_processor.resolve_app_context", return_value=context):
                with patch("xlfusion.batch_processor.load_batch_config", return_value=fake_config):
                    with patch("xlfusion.batch_processor.BatchValidator", return_value=fake_validator):
                        with patch("xlfusion.batch_processor.BatchProcessor", return_value=processor_instance):
                            code = batch_main([str(cfg)])
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
