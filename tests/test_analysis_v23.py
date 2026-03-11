import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file as st_save

from xlfusion.analyzer import CompatibilityAnalyzer, ModelDiffAnalyzer, RecommendationEngine


def _save_analysis_model(path: Path, structure: float, semantics: float, style: float, detail: float) -> None:
    st_save(
        {
            "model.diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.weight": torch.full((2, 2), structure),
            "model.diffusion_model.mid_block.attentions.0.transformer_blocks.0.attn2.to_v.weight": torch.full((2, 2), semantics),
            "model.diffusion_model.up_blocks.0.resnets.0.conv1.weight": torch.full((2, 2, 1, 1), style),
            "model.diffusion_model.up_blocks.3.resnets.0.conv1.weight": torch.full((2, 2, 1, 1), detail),
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.mlp.fc1.weight": torch.full((2, 2), semantics),
        },
        str(path),
    )


class ActionableAnalysisTests(unittest.TestCase):
    def test_diff_analysis_returns_region_and_submodule_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = root / "a.safetensors"
            b = root / "b.safetensors"
            _save_analysis_model(a, structure=1.0, semantics=1.0, style=0.5, detail=0.4)
            _save_analysis_model(b, structure=1.0, semantics=2.0, style=2.0, detail=2.5)

            report = ModelDiffAnalyzer().analyze_model_differences(a, b)

            self.assertIn("structure", report.region_summaries)
            self.assertIn("style", report.region_summaries)
            self.assertTrue(any(name.startswith("semantics:") for name in report.submodule_summaries))
            self.assertIn("style", report.dominance_summary)
            self.assertGreater(len(report.histograms), 0)

    def test_recommendation_profiles_are_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = root / "base.safetensors"
            style = root / "style.safetensors"
            _save_analysis_model(base, structure=3.0, semantics=2.0, style=0.5, detail=0.4)
            _save_analysis_model(style, structure=0.4, semantics=1.0, style=3.0, detail=2.8)

            recommendations = RecommendationEngine().generate_recommendations([base, style], "style_transfer")

            self.assertEqual(len(recommendations), 1)
            suggestion = recommendations[0].suggested_config
            self.assertEqual(suggestion["mode"], "hybrid")
            self.assertIn("hybrid_config", suggestion)
            self.assertIn("attn2_locks", suggestion)

    def test_compatibility_reports_risk_for_low_similarity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = root / "a.safetensors"
            b = root / "b.safetensors"
            _save_analysis_model(a, structure=1.0, semantics=1.0, style=1.0, detail=1.0)
            _save_analysis_model(b, structure=-1.0, semantics=-1.0, style=-1.0, detail=-1.0)

            report = CompatibilityAnalyzer().calculate_compatibility([a, b])

            self.assertLess(report.compatibility_score, 80.0)
            self.assertTrue(report.risk_alerts)


if __name__ == "__main__":
    unittest.main()
