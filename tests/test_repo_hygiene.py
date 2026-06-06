import unittest
from pathlib import Path

from xlfusion.version import __version__


class RepoHygieneTests(unittest.TestCase):
    def test_local_only_files_are_ignored(self) -> None:
        root = Path(__file__).resolve().parents[1]
        ignore_lines = {
            line.strip()
            for line in (root / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

        self.assertIn("AGENTS.md", ignore_lines)
        self.assertIn(".codex", ignore_lines)
        self.assertIn(".codex/", ignore_lines)
        self.assertIn(".claude/", ignore_lines)
        self.assertIn("config.yaml", ignore_lines)
        self.assertIn("workspace/models/**", ignore_lines)
        self.assertIn("workspace/loras/**", ignore_lines)
        self.assertIn("workspace/output/**", ignore_lines)
        self.assertIn("workspace/metadata/**", ignore_lines)
        self.assertIn("workspace/presets/**", ignore_lines)
        self.assertIn("*.safetensors", ignore_lines)
        self.assertIn("*.ckpt", ignore_lines)

    def test_config_template_exists(self) -> None:
        root = Path(__file__).resolve().parents[1]
        example = root / "config.yaml.example"

        self.assertTrue(example.exists())
        self.assertIn("model_output:", example.read_text(encoding="utf-8"))

    def test_license_and_package_metadata_are_present(self) -> None:
        root = Path(__file__).resolve().parents[1]
        license_text = (root / "LICENSE").read_text(encoding="utf-8")
        pyproject_text = (root / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn("MIT License", license_text)
        self.assertIn("Copyright (c) 2026 warc0s", license_text)
        self.assertIn(f'version = "{__version__}"', pyproject_text)
        self.assertIn('license = { file = "LICENSE" }', pyproject_text)
        self.assertIn('xlfusion = "xlfusion.app:main"', pyproject_text)


if __name__ == "__main__":
    unittest.main()
