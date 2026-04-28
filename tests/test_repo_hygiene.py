import unittest
from pathlib import Path


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
        self.assertIn("workspace/output/**", ignore_lines)

    def test_config_template_exists(self) -> None:
        root = Path(__file__).resolve().parents[1]
        example = root / "config.yaml.example"

        self.assertTrue(example.exists())
        self.assertIn("model_output:", example.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
