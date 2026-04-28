import runpy
import unittest
from unittest.mock import patch

from xlfusion import gui_panels
from xlfusion.gui_app import main as gui_main


class PackageEntrypointsTests(unittest.TestCase):
    def test_python_m_xlfusion_calls_app_main(self) -> None:
        with patch("xlfusion.app.main", return_value=0) as mocked:
            with self.assertRaises(SystemExit) as ctx:
                runpy.run_module("xlfusion.__main__", run_name="__main__")
        self.assertEqual(ctx.exception.code, 0)
        mocked.assert_called_once()

    def test_gui_entrypoint_delegates_to_launch_gui(self) -> None:
        with patch("xlfusion.gui_app.launch_gui") as mocked:
            gui_main()
        mocked.assert_called_once()

    def test_gui_panels_export_shared_preview_constants(self) -> None:
        self.assertIn("down_0_1", gui_panels.BLOCK_GROUPS)
        self.assertGreater(len(gui_panels.MODEL_COLORS), 1)


if __name__ == "__main__":
    unittest.main()
