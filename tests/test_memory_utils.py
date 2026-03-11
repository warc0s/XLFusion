import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file as st_load
from safetensors.torch import save_file as st_save

import xlfusion.memory as memory_utils


class MemoryUtilsTests(unittest.TestCase):
    def test_estimate_memory_requirement_uses_file_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p1 = root / "a.bin"
            p2 = root / "b.bin"
            p3 = root / "c.bin"
            p1.write_bytes(b"a" * 1024)
            p2.write_bytes(b"b" * 2048)
            p3.write_bytes(b"c" * 4096)

            required = memory_utils.estimate_memory_requirement([p1, p2, p3], {0, 2})
            self.assertAlmostEqual(required, (1024 + 4096) / (1024**3), places=12)

    def test_get_available_memory_none_when_psutil_missing(self) -> None:
        with patch.object(memory_utils, "PSUTIL_AVAILABLE", False):
            self.assertIsNone(memory_utils.get_available_memory_gb())
            self.assertTrue(memory_utils.check_memory_availability(9999.0))

    def test_save_state_compacts_float32_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "state.safetensors"
            state = {
                "float32_2d": torch.ones((2, 2), dtype=torch.float32),
                "float32_1d": torch.ones((2,), dtype=torch.float32),
                "int64_2d": torch.ones((2, 2), dtype=torch.int64),
            }
            memory_utils.save_state(out, state, meta={"title": "compact"})
            loaded = st_load(str(out), device="cpu")
            self.assertEqual(loaded["float32_2d"].dtype, torch.float16)
            self.assertEqual(loaded["float32_1d"].dtype, torch.float32)
            self.assertEqual(loaded["int64_2d"].dtype, torch.int64)

    def test_load_state_preserves_on_disk_dtype(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.safetensors"
            st_save({"w": torch.ones((2, 2), dtype=torch.float16)}, str(path), metadata={"title": "dtype"})
            loaded = memory_utils.load_state(path)
            self.assertEqual(loaded["w"].dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()

