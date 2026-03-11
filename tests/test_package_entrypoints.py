import runpy
import unittest
from unittest.mock import patch


class PackageEntrypointsTests(unittest.TestCase):
    def test_python_m_xlfusion_calls_app_main(self) -> None:
        with patch("xlfusion.app.main", return_value=0) as mocked:
            with self.assertRaises(SystemExit) as ctx:
                runpy.run_module("xlfusion.__main__", run_name="__main__")
        self.assertEqual(ctx.exception.code, 0)
        mocked.assert_called_once()


if __name__ == "__main__":
    unittest.main()
