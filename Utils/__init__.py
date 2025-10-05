"""
XLFusion Utils Module

This module contains all utility functions and classes for XLFusion,
organized by functionality for better maintainability.
"""

# Config utilities
from .config import load_config, ensure_dirs, list_safetensors, next_version_path

# Block utilities
from .blocks import (
    UNET_PREFIX,
    get_block_assignment,
    is_cross_attn_key,
    get_attn2_block_type,
    should_merge_key,
    group_for_key,
    is_cross_attn_key_legacy
)

# Merge utilities
from .merge import merge_hybrid, merge_perres, stream_weighted_merge_from_paths, validate_hybrid_config

# LoRA utilities
from .lora import (
    apply_single_lora,
    lora_pairs_from_state,
    parse_lora_alpha_rank,
    map_lora_key_to_base,
    DOWN_PAT,
    UP_PAT,
    ALPHA_KEYS,
    CROSS_TOKENS,
    CROSS_PROJ
)

# Memory utilities
from .memory import (
    estimate_memory_requirement,
    check_memory_availability,
    load_state_with_fallback,
    load_state,
    save_state,
    PSUTIL_AVAILABLE
)

# CLI utilities
from .cli import (
    prompt_select,
    prompt_weights,
    pick_backbone,
    prompt_block_merge,
    prompt_crossattn_boost,
    prompt_loras,
    prompt_hybrid_config,
    prompt_perres_assignments
)

# Analyzer (optional)
try:
    from .analyzer import (
        ModelDiffAnalyzer,
        CompatibilityAnalyzer,
        FusionPredictor,
        RecommendationEngine,
        generate_analysis_report,
        export_analysis_json
    )
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False

# Batch processor
try:
    from .batch_processor import BatchProcessor
    BATCH_PROCESSOR_AVAILABLE = True
except ImportError:
    BATCH_PROCESSOR_AVAILABLE = False

# Templates
try:
    from .templates import TEMPLATES
    TEMPLATES_AVAILABLE = True
except ImportError:
    TEMPLATES_AVAILABLE = False

__all__ = [
    # Config
    'load_config', 'ensure_dirs', 'list_safetensors', 'next_version_path',
    
    # Blocks
    'UNET_PREFIX', 'get_block_assignment', 'is_cross_attn_key', 
    'get_attn2_block_type', 'should_merge_key', 'group_for_key', 
    'is_cross_attn_key_legacy',
    
    # Merge
    'merge_hybrid', 'merge_perres', 'stream_weighted_merge_from_paths', 
    'validate_hybrid_config',
    
    # LoRA
    'apply_single_lora', 'lora_pairs_from_state', 'parse_lora_alpha_rank', 
    'map_lora_key_to_base', 'DOWN_PAT', 'UP_PAT', 'ALPHA_KEYS', 
    'CROSS_TOKENS', 'CROSS_PROJ',
    
    # Memory
    'estimate_memory_requirement', 'check_memory_availability', 
    'load_state_with_fallback', 'load_state', 'save_state', 'PSUTIL_AVAILABLE',
    
    # CLI
    'prompt_select', 'prompt_weights', 'pick_backbone', 'prompt_block_merge',
    'prompt_crossattn_boost', 'prompt_loras', 'prompt_hybrid_config', 
    'prompt_perres_assignments',
    
    # Analyzer
    'ANALYZER_AVAILABLE', 'ModelDiffAnalyzer', 'CompatibilityAnalyzer',
    'FusionPredictor', 'RecommendationEngine', 'generate_analysis_report',
    'export_analysis_json',
    
    # Batch processor
    'BATCH_PROCESSOR_AVAILABLE', 'BatchProcessor',
    
    # Templates
    'TEMPLATES_AVAILABLE', 'TEMPLATES'
]