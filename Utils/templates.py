#!/usr/bin/env python3
"""
XLFusion V1.2 - Predefined Templates
====================================

Templates for common model fusion patterns.
"""

from typing import Dict, Any


# Predefined templates for common fusion scenarios
DEFAULT_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "style_transfer": {
        "mode": "hybrid",
        "description": "Transfer artistic style while preserving content structure",
        "default_params": {
            "primary_weight": 0.7,
            "style_focus": "up_blocks",
            "content_focus": "down_blocks"
        },
        "config_template": {
            "backbone": 0,
            "hybrid_config": {
                "down_0_1": {"0": "{{primary_weight}}", "1": "{{1 - primary_weight}}"},
                "down_2_3": {"0": "{{primary_weight}}", "1": "{{1 - primary_weight}}"},
                "mid": {"0": 0.5, "1": 0.5},
                "up_0_1": {"1": "{{primary_weight}}", "0": "{{1 - primary_weight}}"},
                "up_2_3": {"1": "{{primary_weight}}", "0": "{{1 - primary_weight}}"}
            },
            "attn2_locks": {
                "down": 0,
                "mid": 0,
                "up": 1
            }
        }
    },

    "balanced_merge": {
        "mode": "legacy",
        "description": "Classic 50/50 weighted merge",
        "config_template": {
            "weights": [0.5, 0.5],
            "backbone": 0
        }
    },

    "detail_enhance": {
        "mode": "perres",
        "description": "Enhance details using complementary models",
        "config_template": {
            "backbone": 0,
            "assignments": {
                "down_0_1": 0,
                "down_2_3": 1,
                "mid": 0,
                "up_0_1": 1,
                "up_2_3": 1
            },
            "attn2_locks": {
                "down": 0,
                "mid": 0,
                "up": 1
            }
        }
    },

    "composition_focus": {
        "mode": "hybrid",
        "description": "Focus on composition and structure with style overlay",
        "default_params": {
            "base_weight": 0.8,
            "style_weight": 0.2
        },
        "config_template": {
            "backbone": 0,
            "hybrid_config": {
                "down_0_1": {"0": "{{base_weight}}", "1": "{{style_weight}}"},
                "down_2_3": {"0": "{{base_weight}}", "1": "{{style_weight}}"},
                "mid": {"0": "{{base_weight}}", "1": "{{style_weight}}"},
                "up_0_1": {"0": "{{base_weight}}", "1": "{{style_weight}}"},
                "up_2_3": {"1": "{{base_weight}}", "0": "{{style_weight}}"}
            },
            "attn2_locks": {
                "down": 0,
                "mid": 0,
                "up": 0
            }
        }
    },

    "texture_boost": {
        "mode": "hybrid",
        "description": "Boost textures and details in final output",
        "default_params": {
            "texture_weight": 0.6,
            "base_weight": 0.4
        },
        "config_template": {
            "backbone": 0,
            "hybrid_config": {
                "down_0_1": {"0": "{{base_weight}}", "1": "{{texture_weight}}"},
                "down_2_3": {"0": "{{base_weight}}", "1": "{{texture_weight}}"},
                "mid": {"0": "{{base_weight}}", "1": "{{texture_weight}}"},
                "up_0_1": {"0": "{{base_weight}}", "1": "{{texture_weight}}"},
                "up_2_3": {"1": "{{base_weight}}", "0": "{{texture_weight}}"}
            },
            "attn2_locks": {
                "down": 0,
                "mid": 0,
                "up": 1
            }
        }
    },

    "realism_blend": {
        "mode": "legacy",
        "description": "Blend realistic and stylized models with cross-attention boost",
        "default_params": {
            "realism_weight": 0.7,
            "style_weight": 0.3,
            "attention_boost": 1.15
        },
        "config_template": {
            "weights": ["{{realism_weight}}", "{{style_weight}}"],
            "backbone": 0,
            "crossattn_boosts": [
                {"down": "{{attention_boost}}", "mid": 1.0, "up": 1.0},
                {"down": 1.0, "mid": 1.0, "up": 1.0}
            ]
        }
    },

    "quality_upscale": {
        "mode": "perres",
        "description": "Use different models for different quality aspects",
        "config_template": {
            "backbone": 0,
            "assignments": {
                "down_0_1": 0,  # Base model for composition
                "down_2_3": 1,  # Quality model for details
                "mid": 1,       # Quality model for processing
                "up_0_1": 1,    # Quality model for reconstruction
                "up_2_3": 0     # Base model for final output
            }
        }
    }
}


def get_template(name: str) -> Dict[str, Any]:
    """Get a predefined template by name"""
    if name not in DEFAULT_TEMPLATES:
        raise ValueError(f"Template '{name}' not found. Available: {list(DEFAULT_TEMPLATES.keys())}")
    return DEFAULT_TEMPLATES[name].copy()


def list_templates() -> Dict[str, str]:
    """List all available templates with descriptions"""
    return {name: template["description"] for name, template in DEFAULT_TEMPLATES.items()}


def interpolate_template_params(template: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Interpolate parameters in template configuration"""
    import re

    def interpolate_value(value: Any) -> Any:
        if isinstance(value, str):
            # Replace {{param}} with actual values
            for param_name, param_value in params.items():
                placeholder = "{{" + param_name + "}}"
                if placeholder in value:
                    value = value.replace(placeholder, str(param_value))
            return value
        elif isinstance(value, dict):
            return {k: interpolate_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [interpolate_value(item) for item in value]
        else:
            return value

    return interpolate_value(template)


if __name__ == "__main__":
    # Print available templates
    print("Available XLFusion Templates:")
    print("=" * 40)
    for name, desc in list_templates().items():
        print(f"  {name}: {desc}")
    print()
    print("Usage in batch config:")
    print("  template: style_transfer")
    print("  template_params:")
    print("    primary_weight: 0.8")