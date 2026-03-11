import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xlfusion.cli import (
    pick_backbone,
    prompt_component_scope,
    prompt_output_name,
    prompt_select,
    prompt_weights,
)


class CliPromptContractMoreTests(unittest.TestCase):
    def test_prompt_output_name_defaults(self) -> None:
        with patch("builtins.input", return_value=""):
            self.assertEqual(prompt_output_name("XLFusion"), "XLFusion")

    def test_prompt_weights_fills_missing_values(self) -> None:
        with patch("builtins.input", return_value="0.2"):
            weights = prompt_weights(["a", "b"], [0.7, 0.3])
        self.assertEqual(weights, [0.2, 0.3])

    def test_pick_backbone_tie_prompts(self) -> None:
        with patch("builtins.input", return_value="1"):
            idx = pick_backbone(["a", "b"], [1.0, 1.0])
        self.assertEqual(idx, 1)

    def test_prompt_select_uses_defaults_on_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = []
            for name in ["a.safetensors", "b.safetensors"]:
                p = root / name
                p.write_bytes(b"x" * 10)
                paths.append(p)
            with patch("builtins.input", return_value=""):
                selected = prompt_select(paths, "Pick", [0, 1])
        self.assertEqual(selected, [0, 1])

    def test_prompt_component_scope_legacy_defaults_to_only_unet(self) -> None:
        with patch("builtins.input", side_effect=["", "", ""]):
            only_unet, policy = prompt_component_scope("legacy")
        self.assertTrue(only_unet)
        self.assertEqual(policy["vae"], "exclude")
        self.assertEqual(policy["text_encoder"], "exclude")
        self.assertEqual(policy["other"], "exclude")


if __name__ == "__main__":
    unittest.main()

