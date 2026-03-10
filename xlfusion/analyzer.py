#!/usr/bin/env python3
"""
XLFusion V1.3 - Advanced Model Analyzer
========================================

Provides analysis, comparison, prediction and recommendation capabilities for SDXL model fusion.

Features:
- Model difference analysis with block-level statistics
- Compatibility checking between multiple models
- Fusion result prediction
- Intelligent configuration recommendations
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import torch
import numpy as np
from collections import defaultdict

from .memory import load_state
from .blocks import get_block_assignment, is_cross_attn_key, UNET_PREFIX


@dataclass
class DiffAnalysisResult:
    """Results from model difference analysis"""
    model_a: str
    model_b: str
    block_differences: Dict[str, Dict[str, float]]
    significant_changes: List[str]
    overall_similarity: float


@dataclass
class CompatibilityReport:
    """Compatibility analysis report for multiple models"""
    models: List[str]
    compatibility_score: float
    architecture_match: bool
    warnings: List[str]
    pairwise_similarity: Dict[Tuple[int, int], float] = field(default_factory=dict)


@dataclass
class PredictionReport:
    """Prediction of fusion characteristics"""
    fusion_config: Dict
    predicted_dominance: Dict[str, str]
    diversity_score: float
    warnings: List[str]


@dataclass
class Recommendation:
    """Single recommendation for fusion configuration"""
    type: str
    priority: str
    message: str
    suggested_config: Optional[Dict] = None


class ModelDiffAnalyzer:
    """Analyzes differences between two SDXL models."""
    
    def __init__(self, significant_threshold: float = 0.1):
        self.significant_threshold = significant_threshold
    
    def analyze_model_differences(
        self, 
        model_a: Path, 
        model_b: Path
    ) -> DiffAnalysisResult:
        """
        Analyze tensor-by-tensor differences between two models.
        
        Args:
            model_a: Path to first model
            model_b: Path to second model
            
        Returns:
            DiffAnalysisResult containing detailed difference analysis
        """
        # Validate paths exist
        if not model_a.exists():
            raise FileNotFoundError(f"Model A not found: {model_a}")
        if not model_b.exists():
            raise FileNotFoundError(f"Model B not found: {model_b}")
        
        print(f"Loading model A: {model_a.name}")
        state_a = load_state(model_a)
        
        print(f"Loading model B: {model_b.name}")
        state_b = load_state(model_b)
        
        print("Analyzing differences...")
        
        block_diffs = defaultdict(lambda: {
            'l1_distances': [],
            'l2_distances': [],
            'cosine_similarities': [],
            'relative_changes': []
        })
        
        significant_changes = []
        all_similarities = []
        
        common_keys = set(state_a.keys()) & set(state_b.keys())
        
        for i, key in enumerate(common_keys):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(common_keys)} tensors...")
            
            tensor_a = state_a[key]
            tensor_b = state_b[key]
            
            if tensor_a.shape != tensor_b.shape:
                continue
            
            diff = tensor_a - tensor_b
            
            l1_dist = torch.abs(diff).mean().item()
            l2_dist = torch.norm(diff).item() / diff.numel()
            
            flat_a = tensor_a.flatten().float()
            flat_b = tensor_b.flatten().float()
            cos_sim = torch.nn.functional.cosine_similarity(
                flat_a.unsqueeze(0), 
                flat_b.unsqueeze(0)
            ).item()
            
            relative_change = l2_dist / (torch.norm(tensor_a).item() / tensor_a.numel() + 1e-8)
            
            block = get_block_assignment(key)
            if block:
                block_diffs[block]['l1_distances'].append(l1_dist)
                block_diffs[block]['l2_distances'].append(l2_dist)
                block_diffs[block]['cosine_similarities'].append(cos_sim)
                block_diffs[block]['relative_changes'].append(relative_change)
            
            all_similarities.append(cos_sim)
            
            if relative_change > self.significant_threshold:
                significant_changes.append(key)
        
        block_statistics = {}
        for block, metrics in block_diffs.items():
            if metrics['l1_distances']:
                block_statistics[block] = {
                    'l1_mean': float(np.mean(metrics['l1_distances'])),
                    'l1_std': float(np.std(metrics['l1_distances'])),
                    'l2_mean': float(np.mean(metrics['l2_distances'])),
                    'l2_std': float(np.std(metrics['l2_distances'])),
                    'cosine_mean': float(np.mean(metrics['cosine_similarities'])),
                    'cosine_std': float(np.std(metrics['cosine_similarities'])),
                    'relative_change_mean': float(np.mean(metrics['relative_changes'])),
                    'relative_change_max': float(np.max(metrics['relative_changes'])),
                    'num_tensors': len(metrics['l1_distances'])
                }
        
        overall_similarity = float(np.mean(all_similarities)) if all_similarities else 0.0
        
        print(f"Analysis complete. Found {len(significant_changes)} significant changes.")
        
        return DiffAnalysisResult(
            model_a=model_a.name,
            model_b=model_b.name,
            block_differences=block_statistics,
            significant_changes=significant_changes,
            overall_similarity=overall_similarity
        )


class CompatibilityAnalyzer:
    """Analyzes compatibility between multiple models."""
    
    def calculate_compatibility(
        self, 
        models: List[Path]
    ) -> CompatibilityReport:
        """
        Calculate compatibility metrics for a set of models.
        
        Args:
            models: List of model paths
            
        Returns:
            CompatibilityReport with compatibility analysis
        """
        print(f"Analyzing compatibility of {len(models)} models...")
        
        warnings = []
        model_structures = []
        
        # Load model structures
        for model_path in models:
            print(f"  Loading structure of {model_path.name}")
            state = load_state(model_path)
            structure = {k: v.shape for k, v in state.items()}
            model_structures.append((model_path.name, structure))
            del state
        
        # Check architecture match
        architecture_match = True
        reference_keys = set(model_structures[0][1].keys())
        reference_shapes = model_structures[0][1]
        
        for name, structure in model_structures[1:]:
            if set(structure.keys()) != reference_keys:
                architecture_match = False
                warnings.append(f"{name} has different architecture (key mismatch)")
                continue
            
            for key in reference_keys:
                if structure[key] != reference_shapes[key]:
                    architecture_match = False
                    warnings.append(f"{name} has different shape for {key}")
                    break
        
        # Calculate pairwise similarity
        pairwise_similarity = {}
        similarities = []
        
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                print(f"  Calculating similarity: {models[i].name} vs {models[j].name}")
                
                state_i = load_state(models[i])
                state_j = load_state(models[j])
                
                common_keys = set(state_i.keys()) & set(state_j.keys())
                cos_sims = []
                
                for key in list(common_keys)[:100]:  # Sample for performance
                    if state_i[key].shape != state_j[key].shape:
                        continue
                    
                    flat_i = state_i[key].flatten().float()
                    flat_j = state_j[key].flatten().float()
                    
                    cos_sim = torch.nn.functional.cosine_similarity(
                        flat_i.unsqueeze(0),
                        flat_j.unsqueeze(0)
                    ).item()
                    cos_sims.append(cos_sim)
                
                avg_similarity = float(np.mean(cos_sims)) if cos_sims else 0.0
                pairwise_similarity[(i, j)] = avg_similarity
                similarities.append(avg_similarity)
                
                del state_i, state_j
        
        # Detect outliers
        if similarities:
            mean_similarity = np.mean(similarities)
            std_similarity = np.std(similarities)
            
            for (i, j), sim in pairwise_similarity.items():
                if sim < mean_similarity - 2 * std_similarity:
                    warnings.append(
                        f"{models[i].name} and {models[j].name} are outliers "
                        f"(similarity: {sim:.3f})"
                    )
        
        # Calculate compatibility score
        compatibility_score = 100.0
        if not architecture_match:
            compatibility_score -= 50.0
        if warnings:
            compatibility_score -= min(len(warnings) * 5.0, 30.0)
        if similarities and np.mean(similarities) < 0.7:
            compatibility_score -= 20.0
        
        compatibility_score = max(0.0, compatibility_score)
        
        print(f"Compatibility analysis complete. Score: {compatibility_score:.1f}/100")
        
        return CompatibilityReport(
            models=[m.name for m in models],
            compatibility_score=compatibility_score,
            architecture_match=architecture_match,
            warnings=warnings,
            pairwise_similarity=pairwise_similarity
        )


class FusionPredictor:
    """Predicts characteristics of fusion results."""
    
    def predict_fusion_characteristics(
        self,
        models: List[Path],
        config: Dict
    ) -> PredictionReport:
        """
        Predict characteristics of the fusion result.
        
        Args:
            models: List of model paths
            config: Fusion configuration
            
        Returns:
            PredictionReport with predictions and warnings
        """
        print("Predicting fusion characteristics...")
        
        warnings = []
        predicted_dominance = {}
        
        num_models = len(models)
        
        if num_models == 0:
            warnings.append("No models provided for prediction")
            return PredictionReport(
                fusion_config=config,
                predicted_dominance={},
                diversity_score=0.0,
                warnings=warnings
            )
        mode = config.get('mode', 'legacy')
        
        # Analyze based on mode
        if mode == 'legacy':
            weights = config.get('weights', [1.0] * num_models)
            
            if len(weights) != num_models:
                warnings.append(
                    f"Weight count mismatch: {len(weights)} weights for {num_models} models"
                )
                weights = [1.0] * num_models
            
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            blocks = ['down_0_1', 'down_2_3', 'mid', 'up_0_1', 'up_2_3']
            
            for block in blocks:
                dominant_idx = np.argmax(normalized_weights)
                dominant_weight = normalized_weights[dominant_idx]
                
                if dominant_weight > 0.7:
                    predicted_dominance[block] = f"{models[dominant_idx].name} ({dominant_weight:.1%})"
                elif dominant_weight > 0.5:
                    predicted_dominance[block] = f"{models[dominant_idx].name} (moderate {dominant_weight:.1%})"
                else:
                    predicted_dominance[block] = "Balanced mix"
            
            diversity_score = 1.0 - np.max(normalized_weights)
            
        elif mode == 'perres':
            assignments = config.get('assignments', {})
            blocks = ['down_0_1', 'down_2_3', 'mid', 'up_0_1', 'up_2_3']
            
            for block in blocks:
                if block in assignments:
                    model_idx = assignments[block]
                    if 0 <= model_idx < len(models):
                        predicted_dominance[block] = f"{models[model_idx].name} (100%)"
                    else:
                        warnings.append(f"Invalid assignment for {block}: index {model_idx}")
                        predicted_dominance[block] = "Unknown"
                else:
                    predicted_dominance[block] = "Not assigned"
            
            diversity_score = len(set(assignments.values())) / len(models) if assignments else 0.0
            
        elif mode == 'hybrid':
            hybrid_config = config.get('hybrid_config', {})
            blocks = ['down_0_1', 'down_2_3', 'mid', 'up_0_1', 'up_2_3']
            
            for block in blocks:
                if block in hybrid_config:
                    block_weights = hybrid_config[block]
                    if block_weights:
                        max_idx = max(block_weights.keys(), key=lambda k: block_weights[k])
                        max_weight = block_weights[max_idx]
                        if max_weight > 0.7:
                            predicted_dominance[block] = f"{models[max_idx].name} ({max_weight:.1%})"
                        else:
                            predicted_dominance[block] = "Mixed"
                    else:
                        predicted_dominance[block] = "Not configured"
                else:
                    predicted_dominance[block] = "Not configured"
            
            all_weights = []
            for block_weights in hybrid_config.values():
                all_weights.extend(block_weights.values())
            diversity_score = 1.0 - max(all_weights) if all_weights else 0.0
        else:
            warnings.append(f"Unknown mode: {mode}")
            diversity_score = 0.0
        
        # General warnings
        if diversity_score < 0.2:
            warnings.append(
                f"Low diversity: fusion may be dominated by single model"
            )
        
        if 'attn2_locks' in config:
            locks = config['attn2_locks']
            if isinstance(locks, dict):
                for block_type, lock_idx in locks.items():
                    if lock_idx < 0 or lock_idx >= num_models:
                        warnings.append(f"Invalid attn2_lock index for {block_type}: {lock_idx}")
        
        if 'backbone' in config:
            backbone = config['backbone']
            if isinstance(backbone, int) and (backbone < 0 or backbone >= num_models):
                warnings.append(f"Invalid backbone index: {backbone}")
        
        print(f"Prediction complete. Diversity score: {diversity_score:.3f}")
        
        return PredictionReport(
            fusion_config=config,
            predicted_dominance=predicted_dominance,
            diversity_score=diversity_score,
            warnings=warnings
        )


class RecommendationEngine:
    """Generates fusion recommendations based on model analysis."""
    
    GOALS = {
        'style_transfer': {
            'description': 'Transfer style while preserving structure',
            'focus': 'cross_attention'
        },
        'detail_enhance': {
            'description': 'Enhance details and quality',
            'focus': 'up_blocks'
        },
        'balanced': {
            'description': 'Balanced fusion of all characteristics',
            'focus': 'all'
        }
    }
    
    def generate_recommendations(
        self,
        models: List[Path],
        goal: str = 'balanced'
    ) -> List[Recommendation]:
        """
        Generate fusion recommendations based on model analysis and goal.
        
        Args:
            models: List of model paths
            goal: Fusion goal ('style_transfer', 'detail_enhance', 'balanced')
            
        Returns:
            List of Recommendation objects
        """
        print(f"Generating recommendations for goal: {goal}")
        
        recommendations = []
        
        if goal not in self.GOALS:
            recommendations.append(Recommendation(
                type="warning",
                priority="high",
                message=f"Unknown goal '{goal}'. Using 'balanced' instead.",
            ))
            goal = 'balanced'
        
        print("Analyzing model characteristics...")
        model_stats = []
        
        for model_path in models:
            state = load_state(model_path)
            
            cross_attn_keys = [k for k in state.keys() if is_cross_attn_key(k)]
            cross_attn_variance = []
            
            for key in cross_attn_keys[:10]:  # Sample for performance
                variance = torch.var(state[key]).item()
                cross_attn_variance.append(variance)
            
            avg_cross_attn_var = float(np.mean(cross_attn_variance)) if cross_attn_variance else 0.0
            
            model_stats.append({
                'name': model_path.name,
                'cross_attn_variance': avg_cross_attn_var,
                'total_keys': len(state.keys())
            })
            
            del state
        
        # Generate goal-specific recommendations
        if goal == 'style_transfer':
            style_model_idx = int(np.argmax([s['cross_attn_variance'] for s in model_stats]))
            
            recommendations.append(Recommendation(
                type="attn2_lock",
                priority="high",
                message=f"Lock cross-attention to {model_stats[style_model_idx]['name']} "
                        f"for style transfer",
                suggested_config={'attn2_locks': {'down': style_model_idx, 'mid': style_model_idx, 'up': style_model_idx}}
            ))
            
            if len(models) > 1:
                structure_idx = 1 - style_model_idx if len(models) == 2 else 0
                recommendations.append(Recommendation(
                    type="backbone",
                    priority="high",
                    message=f"Use {model_stats[structure_idx]['name']} as backbone "
                            f"to preserve structure",
                    suggested_config={'backbone': structure_idx}
                ))
        
        elif goal == 'detail_enhance':
            recommendations.append(Recommendation(
                type="config",
                priority="high",
                message="Use PerRes or Hybrid mode with detail model on up_blocks",
                suggested_config={
                    'mode': 'perres',
                    'assignments': {
                        'down_0_1': 0,
                        'down_2_3': 0,
                        'mid': 0,
                        'up_0_1': 1 if len(models) > 1 else 0,
                        'up_2_3': 1 if len(models) > 1 else 0
                    }
                }
            ))
            
            detail_model_idx = 0
            recommendations.append(Recommendation(
                type="backbone",
                priority="medium",
                message=f"Consider using {model_stats[detail_model_idx]['name']} "
                        f"as backbone for base quality",
                suggested_config={'backbone': detail_model_idx}
            ))
        
        else:  # balanced
            equal_weights = [1.0 / len(models)] * len(models)
            recommendations.append(Recommendation(
                type="config",
                priority="medium",
                message="Use equal weights for balanced fusion",
                suggested_config={'mode': 'legacy', 'weights': equal_weights}
            ))
        
        # General recommendations
        if len(models) > 3:
            recommendations.append(Recommendation(
                type="warning",
                priority="medium",
                message=f"Fusing {len(models)} models may dilute characteristics. "
                        f"Consider reducing to 2-3 models.",
            ))
        
        print(f"Generated {len(recommendations)} recommendations")
        
        return recommendations


def generate_analysis_report(results: Dict[str, Any]) -> str:
    """
    Format analysis results into readable text report.
    
    Args:
        results: Dictionary containing analysis results
        
    Returns:
        Formatted text report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("XLFusion V1.3 - Model Analysis Report")
    lines.append("=" * 80)
    lines.append("")
    
    if 'diff_analysis' in results:
        diff: DiffAnalysisResult = results['diff_analysis']
        lines.append(f"DIFFERENCE ANALYSIS: {diff.model_a} vs {diff.model_b}")
        lines.append(f"Overall Similarity: {diff.overall_similarity:.4f}")
        lines.append(f"Significant Changes: {len(diff.significant_changes)} tensors")
        lines.append("")
        lines.append("Block-wise Statistics:")
        for block, stats in diff.block_differences.items():
            lines.append(f"  {block}:")
            lines.append(f"    L2 Distance: {stats['l2_mean']:.6f} ± {stats['l2_std']:.6f}")
            lines.append(f"    Cosine Similarity: {stats['cosine_mean']:.4f} ± {stats['cosine_std']:.4f}")
            lines.append(f"    Relative Change: {stats['relative_change_mean']:.4f} (max: {stats['relative_change_max']:.4f})")
        lines.append("")
    
    if 'compatibility' in results:
        compat: CompatibilityReport = results['compatibility']
        lines.append("COMPATIBILITY ANALYSIS")
        lines.append(f"Models: {', '.join(compat.models)}")
        lines.append(f"Compatibility Score: {compat.compatibility_score:.1f}/100")
        lines.append(f"Architecture Match: {'✓' if compat.architecture_match else '✗'}")
        if compat.warnings:
            lines.append("Warnings:")
            for warning in compat.warnings:
                lines.append(f"  - {warning}")
        lines.append("")
    
    if 'prediction' in results:
        pred: PredictionReport = results['prediction']
        lines.append("FUSION PREDICTION")
        lines.append(f"Diversity Score: {pred.diversity_score:.3f}")
        lines.append("Predicted Dominance by Block:")
        for block, dominance in pred.predicted_dominance.items():
            lines.append(f"  {block}: {dominance}")
        if pred.warnings:
            lines.append("Warnings:")
            for warning in pred.warnings:
                lines.append(f"  - {warning}")
        lines.append("")
    
    if 'recommendations' in results:
        recs: List[Recommendation] = results['recommendations']
        lines.append("RECOMMENDATIONS")
        for rec in recs:
            priority_marker = "!!!" if rec.priority == "high" else "!!" if rec.priority == "medium" else "!"
            lines.append(f"  [{priority_marker}] {rec.type.upper()}: {rec.message}")
            if rec.suggested_config:
                lines.append(f"      Config: {rec.suggested_config}")
        lines.append("")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


def export_analysis_json(results: Dict[str, Any], output_path: Path) -> None:
    """
    Export analysis results to JSON file.
    
    Args:
        results: Dictionary containing analysis results
        output_path: Path to output JSON file
    """
    import json
    from dataclasses import asdict
    
    serializable_results = {}
    
    for key, value in results.items():
        if hasattr(value, '__dataclass_fields__'):
            serializable_results[key] = asdict(value)
        elif isinstance(value, list) and value and hasattr(value[0], '__dataclass_fields__'):
            serializable_results[key] = [asdict(item) for item in value]
        else:
            serializable_results[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Analysis exported to {output_path}")


if __name__ == "__main__":
    print("XLFusion V1.3 - Analyzer Module")
    print("Import this module to use analysis capabilities:")
    print("  from analyzer import ModelDiffAnalyzer, CompatibilityAnalyzer")
    print("  from analyzer import FusionPredictor, RecommendationEngine")
