"""Actionable analysis helpers for XLFusion."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json

import numpy as np
import torch

from .blocks import classify_component_key, classify_submodule_key, get_block_assignment
from .memory import load_state

REGION_ALIASES = {
    "down_0_1": "structure",
    "down_2_3": "structure",
    "mid": "semantics",
    "up_0_1": "style",
    "up_2_3": "detail",
}


@dataclass
class DiffAnalysisResult:
    model_a: str
    model_b: str
    overall_similarity: float
    region_summaries: Dict[str, Dict[str, float]]
    submodule_summaries: Dict[str, Dict[str, float]]
    histograms: Dict[str, Dict[str, List[float]]]
    dominance_summary: Dict[str, str]
    significant_changes: List[str]
    risk_alerts: List[str] = field(default_factory=list)


@dataclass
class CompatibilityReport:
    models: List[str]
    compatibility_score: float
    architecture_match: bool
    warnings: List[str]
    pairwise_similarity: Dict[Tuple[int, int], float] = field(default_factory=dict)
    risk_alerts: List[str] = field(default_factory=list)


@dataclass
class PredictionReport:
    fusion_config: Dict[str, Any]
    predicted_dominance: Dict[str, str]
    diversity_score: float
    warnings: List[str]
    recommended_backbone: Optional[str] = None


@dataclass
class Recommendation:
    profile: str
    priority: str
    message: str
    rationale: str
    suggested_config: Optional[Dict[str, Any]] = None


def _region_for_key(key: str) -> str:
    component = classify_component_key(key)
    if component != "unet":
        return component
    return REGION_ALIASES.get(get_block_assignment(key) or "other", "other")


def _safe_histogram(values: Sequence[float], *, bins: int = 5, start: float = 0.0, end: float = 1.0) -> Dict[str, List[float]]:
    if not values:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(values, bins=bins, range=(start, end))
    return {
        "edges": [float(edge) for edge in edges.tolist()],
        "counts": [int(count) for count in counts.tolist()],
    }


def _summarize_metric(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "count": float(arr.size),
    }


def _fingerprint_model(state: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    regions: Dict[str, List[float]] = {}
    submodules: Dict[str, List[float]] = {}

    for key, tensor in state.items():
        region = _region_for_key(key)
        submodule = classify_submodule_key(key)
        tensor_f = tensor.to(torch.float32)
        norm_value = float(torch.norm(tensor_f).item() / max(tensor_f.numel(), 1))
        regions.setdefault(region, []).append(norm_value)
        submodules.setdefault(f"{region}:{submodule}", []).append(norm_value)

    return {
        "regions": {name: _summarize_metric(values) for name, values in regions.items()},
        "submodules": {name: _summarize_metric(values) for name, values in submodules.items()},
    }


class ModelDiffAnalyzer:
    """Analyze detailed differences between two checkpoints."""

    def __init__(self, significant_threshold: float = 0.1) -> None:
        self.significant_threshold = significant_threshold

    def analyze_model_differences(self, model_a: Path, model_b: Path) -> DiffAnalysisResult:
        if not model_a.exists():
            raise FileNotFoundError(f"Model A not found: {model_a}")
        if not model_b.exists():
            raise FileNotFoundError(f"Model B not found: {model_b}")

        state_a = load_state(model_a)
        state_b = load_state(model_b)

        common_keys = sorted(set(state_a.keys()) & set(state_b.keys()))
        cosine_by_region: Dict[str, List[float]] = {}
        rel_change_by_region: Dict[str, List[float]] = {}
        cosine_by_submodule: Dict[str, List[float]] = {}
        rel_change_by_submodule: Dict[str, List[float]] = {}
        all_similarities: List[float] = []
        significant_changes: List[str] = []

        for key in common_keys:
            tensor_a = state_a[key]
            tensor_b = state_b[key]
            if tensor_a.shape != tensor_b.shape:
                continue
            tensor_a_f = tensor_a.flatten().to(torch.float32)
            tensor_b_f = tensor_b.flatten().to(torch.float32)
            if tensor_a_f.numel() == 0:
                continue

            cosine = torch.nn.functional.cosine_similarity(tensor_a_f.unsqueeze(0), tensor_b_f.unsqueeze(0)).item()
            relative_change = float(torch.norm(tensor_a_f - tensor_b_f).item() / max(torch.norm(tensor_a_f).item(), 1e-8))

            region = _region_for_key(key)
            submodule = f"{region}:{classify_submodule_key(key)}"
            cosine_by_region.setdefault(region, []).append(float(cosine))
            rel_change_by_region.setdefault(region, []).append(relative_change)
            cosine_by_submodule.setdefault(submodule, []).append(float(cosine))
            rel_change_by_submodule.setdefault(submodule, []).append(relative_change)
            all_similarities.append(float(cosine))

            if relative_change > self.significant_threshold:
                significant_changes.append(key)

        region_summaries: Dict[str, Dict[str, float]] = {}
        for region, values in cosine_by_region.items():
            region_summaries[region] = {
                "cosine_mean": _summarize_metric(values)["mean"],
                "relative_change_mean": _summarize_metric(rel_change_by_region.get(region, []))["mean"],
                "tensor_count": _summarize_metric(values)["count"],
            }

        submodule_summaries: Dict[str, Dict[str, float]] = {}
        for submodule, values in cosine_by_submodule.items():
            submodule_summaries[submodule] = {
                "cosine_mean": _summarize_metric(values)["mean"],
                "relative_change_mean": _summarize_metric(rel_change_by_submodule.get(submodule, []))["mean"],
                "tensor_count": _summarize_metric(values)["count"],
            }

        histograms = {
            region: {
                "cosine": _safe_histogram(cosine_by_region.get(region, []), bins=5, start=-1.0, end=1.0),
                "relative_change": _safe_histogram(rel_change_by_region.get(region, []), bins=5, start=0.0, end=1.0),
            }
            for region in sorted(region_summaries)
        }

        fingerprint_a = _fingerprint_model(state_a)
        fingerprint_b = _fingerprint_model(state_b)
        dominance_summary = self._build_dominance_summary(fingerprint_a, fingerprint_b)
        overall_similarity = float(np.mean(all_similarities)) if all_similarities else 0.0
        risk_alerts = []
        if overall_similarity < 0.35:
            risk_alerts.append("The models are very far apart; aggressive merges may become unstable.")
        if overall_similarity > 0.995:
            risk_alerts.append("The models are almost identical; the merge may add little value.")

        return DiffAnalysisResult(
            model_a=model_a.name,
            model_b=model_b.name,
            overall_similarity=overall_similarity,
            region_summaries=region_summaries,
            submodule_summaries=submodule_summaries,
            histograms=histograms,
            dominance_summary=dominance_summary,
            significant_changes=significant_changes,
            risk_alerts=risk_alerts,
        )

    def _build_dominance_summary(
        self,
        fingerprint_a: Dict[str, Dict[str, float]],
        fingerprint_b: Dict[str, Dict[str, float]],
    ) -> Dict[str, str]:
        summary: Dict[str, str] = {}
        feature_map = {
            "composition": "structure",
            "semantics": "semantics",
            "style": "style",
            "detail": "detail",
        }
        for label, region in feature_map.items():
            mean_a = fingerprint_a["regions"].get(region, {}).get("mean", 0.0)
            mean_b = fingerprint_b["regions"].get(region, {}).get("mean", 0.0)
            if abs(mean_a - mean_b) < 1e-6:
                summary[label] = "balanced"
            else:
                summary[label] = "model_a" if mean_a > mean_b else "model_b"
        return summary


class CompatibilityAnalyzer:
    """Assess structural and sampled numerical compatibility."""

    def calculate_compatibility(self, models: List[Path]) -> CompatibilityReport:
        warnings: List[str] = []
        structures = []
        for path in models:
            state = load_state(path)
            structures.append((path.name, {key: tuple(tensor.shape) for key, tensor in state.items()}))

        reference_name, reference = structures[0]
        architecture_match = True
        for name, structure in structures[1:]:
            if set(structure.keys()) != set(reference.keys()):
                architecture_match = False
                warnings.append(f"{name} differs structurally from {reference_name}")
                continue
            for key, shape in structure.items():
                if reference[key] != shape:
                    architecture_match = False
                    warnings.append(f"{name} has shape mismatches against {reference_name}")
                    break

        pairwise_similarity: Dict[Tuple[int, int], float] = {}
        similarities: List[float] = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                diff = ModelDiffAnalyzer().analyze_model_differences(models[i], models[j])
                pairwise_similarity[(i, j)] = diff.overall_similarity
                similarities.append(diff.overall_similarity)

        compatibility_score = 100.0
        if not architecture_match:
            compatibility_score -= 45.0
        if similarities:
            average_similarity = float(np.mean(similarities))
            if average_similarity < 0.4:
                compatibility_score -= 25.0
            elif average_similarity < 0.7:
                compatibility_score -= 10.0
        compatibility_score -= min(len(warnings) * 4.0, 20.0)
        compatibility_score = max(0.0, compatibility_score)

        risk_alerts: List[str] = []
        if similarities:
            avg = float(np.mean(similarities))
            if avg < 0.35:
                risk_alerts.append("Sampled pairwise similarity is very low across the selected models.")
            if avg > 0.995:
                risk_alerts.append("Selected models are almost identical and may not justify a merge.")
        if not architecture_match:
            risk_alerts.append("Architecture differences detected; expect fallback behaviour or skipped tensors.")

        return CompatibilityReport(
            models=[path.name for path in models],
            compatibility_score=compatibility_score,
            architecture_match=architecture_match,
            warnings=warnings,
            pairwise_similarity=pairwise_similarity,
            risk_alerts=risk_alerts,
        )


class FusionPredictor:
    """Turn a candidate config into a coarse dominance prediction."""

    def predict_fusion_characteristics(self, models: List[Path], config: Dict[str, Any]) -> PredictionReport:
        warnings: List[str] = []
        predicted_dominance: Dict[str, str] = {}
        mode = config.get("mode", "legacy")

        if mode == "legacy":
            weights = config.get("weights") or [1.0 / max(len(models), 1)] * len(models)
            total = sum(weights) or 1.0
            normalized = [float(weight) / total for weight in weights]
            dominant_idx = int(np.argmax(normalized))
            for block in ["down_0_1", "down_2_3", "mid", "up_0_1", "up_2_3"]:
                weight = normalized[dominant_idx]
                predicted_dominance[block] = f"{models[dominant_idx].name} ({weight:.1%})" if weight > 0.55 else "Balanced mix"
            diversity_score = 1.0 - max(normalized)
            recommended_backbone = models[dominant_idx].name
        elif mode == "perres":
            assignments = config.get("assignments") or {}
            for block in ["down_0_1", "down_2_3", "mid", "up_0_1", "up_2_3"]:
                idx = assignments.get(block, 0)
                predicted_dominance[block] = f"{models[idx].name} (100%)" if 0 <= idx < len(models) else "Unknown"
            diversity_score = len(set(assignments.values())) / max(len(models), 1) if assignments else 0.0
            recommended_backbone = models[next(iter(assignments.values()), 0)].name if models else None
        else:
            hybrid_config = config.get("hybrid_config") or {}
            for block in ["down_0_1", "down_2_3", "mid", "up_0_1", "up_2_3"]:
                block_weights = hybrid_config.get(block, {})
                if not block_weights:
                    predicted_dominance[block] = "Not configured"
                    continue
                dominant_idx = max(block_weights, key=block_weights.get)
                dominant_weight = block_weights[dominant_idx]
                predicted_dominance[block] = f"{models[dominant_idx].name} ({dominant_weight:.1%})" if dominant_weight > 0.55 else "Mixed"
            all_weights = [value for block in hybrid_config.values() for value in block.values()]
            diversity_score = 1.0 - max(all_weights) if all_weights else 0.0
            recommended_backbone = models[int(config.get("backbone", 0))].name if models else None

        if diversity_score < 0.15:
            warnings.append("The proposed configuration is very concentrated in a single model.")
        return PredictionReport(
            fusion_config=config,
            predicted_dominance=predicted_dominance,
            diversity_score=diversity_score,
            warnings=warnings,
            recommended_backbone=recommended_backbone,
        )


class RecommendationEngine:
    """Generate actionable merge starting points from model fingerprints."""

    GOALS = {
        "balanced": "Blend composition, semantics and style evenly.",
        "style_transfer": "Preserve structure while moving style aggressively.",
        "detail_recovery": "Recover detail-heavy up blocks without losing the base model.",
        "prompt_fidelity": "Prioritize text-conditioning and semantic stability.",
        "detail_enhance": "Legacy alias for detail_recovery.",
    }

    def _fingerprints_for_models(self, models: Sequence[Path]) -> List[Dict[str, Dict[str, float]]]:
        return [_fingerprint_model(load_state(model_path)) for model_path in models]

    def generate_recommendations(self, models: List[Path], goal: str = "balanced") -> List[Recommendation]:
        if goal == "detail_enhance":
            goal = "detail_recovery"
        if goal not in self.GOALS:
            raise ValueError(f"Unknown goal '{goal}'")

        fingerprints = self._fingerprints_for_models(models)
        region_means = []
        for fp in fingerprints:
            region_means.append({
                "structure": fp["regions"].get("structure", {}).get("mean", 0.0),
                "semantics": fp["regions"].get("semantics", {}).get("mean", 0.0),
                "style": fp["regions"].get("style", {}).get("mean", 0.0),
                "detail": fp["regions"].get("detail", {}).get("mean", 0.0),
            })

        structure_idx = int(np.argmax([item["structure"] for item in region_means])) if region_means else 0
        style_idx = int(np.argmax([item["style"] for item in region_means])) if region_means else 0
        detail_idx = int(np.argmax([item["detail"] for item in region_means])) if region_means else 0
        semantics_idx = int(np.argmax([item["semantics"] for item in region_means])) if region_means else 0

        recommendations: List[Recommendation] = []
        balanced_weights = [round(1.0 / len(models), 4)] * len(models)

        recommendations.append(
            Recommendation(
                profile="balanced",
                priority="medium",
                message="Balanced starting point with equal global weights.",
                rationale="Use this when no single model should dominate the merge.",
                suggested_config={
                    "mode": "legacy",
                    "weights": balanced_weights,
                    "backbone": structure_idx,
                },
            )
        )
        recommendations.append(
            Recommendation(
                profile="style_transfer",
                priority="high",
                message=f"Use {models[structure_idx].name} for structure and {models[style_idx].name} for style-sensitive up blocks.",
                rationale="Structure is stronger in down blocks, while style strength is higher in the up blocks.",
                suggested_config={
                    "mode": "hybrid",
                    "backbone": structure_idx,
                    "hybrid_config": {
                        "down_0_1": {structure_idx: 0.8, style_idx: 0.2},
                        "down_2_3": {structure_idx: 0.8, style_idx: 0.2},
                        "mid": {structure_idx: 0.6, style_idx: 0.4},
                        "up_0_1": {style_idx: 0.7, structure_idx: 0.3},
                        "up_2_3": {style_idx: 0.8, structure_idx: 0.2},
                    },
                    "attn2_locks": {"down": structure_idx, "mid": semantics_idx, "up": style_idx},
                },
            )
        )
        recommendations.append(
            Recommendation(
                profile="detail_recovery",
                priority="high",
                message=f"Use {models[detail_idx].name} for the detail-heavy up blocks.",
                rationale="The model with the strongest detail fingerprint should dominate the late up blocks.",
                suggested_config={
                    "mode": "perres",
                    "backbone": structure_idx,
                    "assignments": {
                        "down_0_1": structure_idx,
                        "down_2_3": structure_idx,
                        "mid": semantics_idx,
                        "up_0_1": detail_idx,
                        "up_2_3": detail_idx,
                    },
                    "attn2_locks": {"down": structure_idx, "mid": semantics_idx, "up": detail_idx},
                },
            )
        )
        recommendations.append(
            Recommendation(
                profile="prompt_fidelity",
                priority="high",
                message=f"Bias the semantic mid blocks and attention locks toward {models[semantics_idx].name}.",
                rationale="Prompt fidelity is primarily influenced by semantic and attention-heavy regions.",
                suggested_config={
                    "mode": "hybrid",
                    "backbone": semantics_idx,
                    "hybrid_config": {
                        "down_0_1": {structure_idx: 0.7, semantics_idx: 0.3},
                        "down_2_3": {structure_idx: 0.6, semantics_idx: 0.4},
                        "mid": {semantics_idx: 0.8, structure_idx: 0.2},
                        "up_0_1": {detail_idx: 0.6, semantics_idx: 0.4},
                        "up_2_3": {detail_idx: 0.6, semantics_idx: 0.4},
                    },
                    "attn2_locks": {"mid": semantics_idx},
                },
            )
        )

        if goal == "balanced":
            return [recommendations[0]]
        return [rec for rec in recommendations if rec.profile == goal] or recommendations


def generate_analysis_report(results: Dict[str, Any]) -> str:
    lines = [
        "=" * 80,
        "XLFusion V2.4 - Actionable Analysis Report",
        "=" * 80,
        "",
    ]

    if "diff_analysis" in results:
        diff: DiffAnalysisResult = results["diff_analysis"]
        lines.append(f"DIFFERENCE ANALYSIS: {diff.model_a} vs {diff.model_b}")
        lines.append(f"Overall similarity: {diff.overall_similarity:.4f}")
        lines.append(f"Significant tensor changes: {len(diff.significant_changes)}")
        if diff.risk_alerts:
            lines.append("Risk alerts:")
            for alert in diff.risk_alerts:
                lines.append(f"  - {alert}")
        lines.append("Dominance summary:")
        for label, winner in diff.dominance_summary.items():
            lines.append(f"  {label}: {winner}")
        lines.append("Region summaries:")
        for region, summary in sorted(diff.region_summaries.items()):
            lines.append(
                f"  {region}: cosine={summary['cosine_mean']:.4f}, "
                f"relative_change={summary['relative_change_mean']:.4f}, tensors={int(summary['tensor_count'])}"
            )
        lines.append("")

    if "compatibility" in results:
        compat: CompatibilityReport = results["compatibility"]
        lines.append("COMPATIBILITY ANALYSIS")
        lines.append(f"Models: {', '.join(compat.models)}")
        lines.append(f"Compatibility score: {compat.compatibility_score:.1f}/100")
        lines.append(f"Architecture match: {'yes' if compat.architecture_match else 'no'}")
        for warning in compat.warnings:
            lines.append(f"  warning: {warning}")
        for alert in compat.risk_alerts:
            lines.append(f"  risk: {alert}")
        lines.append("")

    if "prediction" in results:
        pred: PredictionReport = results["prediction"]
        lines.append("FUSION PREDICTION")
        lines.append(f"Diversity score: {pred.diversity_score:.3f}")
        if pred.recommended_backbone:
            lines.append(f"Suggested backbone: {pred.recommended_backbone}")
        for block, dominance in pred.predicted_dominance.items():
            lines.append(f"  {block}: {dominance}")
        for warning in pred.warnings:
            lines.append(f"  warning: {warning}")
        lines.append("")

    if "recommendations" in results:
        recs: List[Recommendation] = results["recommendations"]
        lines.append("RECOMMENDATIONS")
        for rec in recs:
            lines.append(f"  [{rec.priority}] {rec.profile}: {rec.message}")
            lines.append(f"    rationale: {rec.rationale}")
            if rec.suggested_config:
                lines.append(f"    config: {rec.suggested_config}")
        lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)


def export_analysis_json(results: Dict[str, Any], output_path: Path) -> None:
    def _to_jsonable(value: Any) -> Any:
        if hasattr(value, "__dataclass_fields__"):
            value = asdict(value)

        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            converted: Dict[str, Any] = {}
            for k, v in value.items():
                key = k if isinstance(k, str) else str(k)
                converted[key] = _to_jsonable(v)
            return converted
        if isinstance(value, (list, tuple)):
            return [_to_jsonable(item) for item in value]
        if isinstance(value, set):
            return [_to_jsonable(item) for item in sorted(value, key=lambda item: str(item))]

        try:  # Optional numpy support
            import numpy as np  # type: ignore

            if isinstance(value, np.ndarray):
                return value.tolist()
        except Exception:
            pass

        return str(value)

    output_path.write_text(json.dumps(_to_jsonable(results), indent=2), encoding="utf-8")
