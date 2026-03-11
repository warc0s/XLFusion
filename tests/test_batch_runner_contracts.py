import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xlfusion.batch_runner import BatchProcessor
from xlfusion.batch_schema import BatchConfig, BatchJob


class BatchRunnerContractTests(unittest.TestCase):
    def test_validate_only_does_not_execute_jobs(self) -> None:
        config = BatchConfig(
            global_settings={"output_base": "out", "continue_on_error": True, "max_parallel": 1},
            batch_jobs=[
                BatchJob(name="j1", mode="legacy", models=["a.safetensors", "b.safetensors"]),
            ],
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            processor = BatchProcessor(config, root, validate_only=True)
            with patch.object(processor, "_process_job") as process_job:
                results = processor.process_batch()
            process_job.assert_not_called()
            self.assertEqual(results["successful_jobs"], 1)
            self.assertEqual(results["failed_jobs"], 0)

    def test_continue_on_error_false_stops_after_failure(self) -> None:
        config = BatchConfig(
            global_settings={"output_base": "out", "continue_on_error": False, "max_parallel": 1},
            batch_jobs=[
                BatchJob(name="j1", mode="legacy", models=["a.safetensors", "b.safetensors"]),
                BatchJob(name="j2", mode="legacy", models=["a.safetensors", "b.safetensors"]),
            ],
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            processor = BatchProcessor(config, root, validate_only=False)
            with patch.object(processor, "_process_job", side_effect=[RuntimeError("boom"), None]) as process_job:
                results = processor.process_batch()
            self.assertEqual(process_job.call_count, 1)
            self.assertEqual(results["failed_jobs"], 1)
            self.assertEqual(len(results["jobs"]), 1)


if __name__ == "__main__":
    unittest.main()

