import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file as st_save

from xlfusion.config import resolve_app_context
from xlfusion.app import main as app_main


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


class AppMainFlowsTests(unittest.TestCase):
    def test_gui_guardrail_rejects_combined_modes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            root.joinpath("config.yaml").write_text("app:\n  tool_name: XLFusion\n", encoding="utf-8")
            context = resolve_app_context(root, reporter=None)
            with patch("xlfusion.app.resolve_app_context", return_value=context):
                code = app_main(["--gui", "--batch", "dummy.yaml"])
        self.assertEqual(code, 1)

    def test_recovery_guardrail_rejects_combined_modes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            root.joinpath("config.yaml").write_text("app:\n  tool_name: XLFusion\n", encoding="utf-8")
            context = resolve_app_context(root, reporter=None)
            with patch("xlfusion.app.resolve_app_context", return_value=context):
                code = app_main(["--recover-metadata", "meta_1", "--batch", "dummy.yaml"])
        self.assertEqual(code, 1)

    def test_algebra_requires_models(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            root.joinpath("config.yaml").write_text("app:\n  tool_name: XLFusion\n", encoding="utf-8")
            context = resolve_app_context(root, reporter=None)
            with patch("xlfusion.app.resolve_app_context", return_value=context):
                code = app_main(["--algebra", "0", "1", "2"])
        self.assertEqual(code, 1)

    def test_interactive_legacy_happy_path_runs_to_completion(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            root.joinpath("config.yaml").write_text("app:\n  tool_name: XLFusion\n", encoding="utf-8")
            context = resolve_app_context(root, reporter=None)

            _save_model(context.models_dir / "a.safetensors", 1.0)
            _save_model(context.models_dir / "b.safetensors", 0.0)

            inputs = [
                "",  # preset source
                "0,1",  # select models
                "",  # mode (default legacy)
                "",  # weights (default suggestion)
                "", "", "",  # block multipliers (skip)
                "", "", "",  # cross-attn boosts (skip)
                "", "", "",  # component scope defaults
                "",  # output name (default)
                "", "", "",  # execution options defaults
                "",  # export preflight (skip)
                "",  # save preset (skip)
                "",  # proceed (default yes)
            ]

            with patch("xlfusion.app.resolve_app_context", return_value=context):
                with patch("builtins.input", side_effect=inputs):
                    code = app_main([])

            self.assertEqual(code, 0)
            outputs = sorted(context.output_dir.glob("*.safetensors"))
            metas = sorted(context.metadata_dir.glob("meta_*"))
            self.assertTrue(outputs)
            self.assertTrue(metas)


if __name__ == "__main__":
    unittest.main()
