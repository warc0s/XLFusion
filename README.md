# XLFusion V1.1

Professional SDXL checkpoint merger with advanced fusion capabilities. Supports three distinct merging modes for optimal model combination and customization.

## Features

### Merge Modes

**Legacy Mode**
- Classic weighted merge with configurable block-level control
- Down/Mid/Up block multipliers for fine-tuned blending
- Cross-attention boost support for enhanced prompt adherence
- LoRA baking integration with automatic scaling
- Memory-optimized streaming merge process

**PerRes Mode (Per-Resolution)**
- Resolution-based block assignment for precise control
- Independent model selection for each resolution tier:
  - Down 0,1: 64x, 32x resolutions (composition and structure)
  - Down 2,3: 16x, 8x resolutions (semantic details)
  - Mid: 8x latent space (abstract processing)
  - Up 0,1: 8x, 16x resolutions (reconstruction initiation)
  - Up 2,3: 32x, 64x resolutions (final style and textures)
- Optional cross-attention locks for consistency
- 100% block assignment without blending

**Hybrid Mode (NEW in V1.1)**
- Combines weighted blending with resolution-based control
- Configure different weights for each resolution block group
- Maximum flexibility: best of Legacy and PerRes modes
- Granular control over composition, semantics, and style
- Cross-attention boost and locks support
- Perfect for complex multi-model fusion scenarios

### Technical Capabilities

- **Model Compatibility**: All SDXL derivatives (NoobAI, Illustrious, Pony, etc.)
- **Format Support**: .safetensors input/output with A1111 compatibility
- **Configurable System**: Centralized YAML configuration for all settings
- **Metadata Preservation**: Comprehensive audit logs and embedded metadata
- **Automatic Versioning**: Prevents accidental overwrites with incremental naming

## Usage

1. Place checkpoint files (.safetensors) in the `models/` directory
2. For Legacy mode: Place LoRA files (.safetensors) in the `loras/` directory
3. (Optional) Customize settings in `config.yaml`
4. Run the script: `python XLFusion.py`
5. Select merge mode:
   - **[1] Legacy** - Classic weighted merge with LoRA support
   - **[2] PerRes** - Resolution-based block assignment
   - **[3] Hybrid** - Combined weighted + resolution control (V1.1)
6. Follow interactive prompts for your selected mode
7. Merged models are saved to `output/` with metadata logs in `metadata/`

### Hybrid Mode Usage (V1.1)

The new Hybrid mode offers maximum flexibility by combining the best of Legacy and PerRes modes:

1. **Model Selection**: Choose 2-4 models with complementary strengths
2. **Block Weight Configuration**: Set different weights for each resolution block:
   - Down 0,1 (64x, 32x) - Composition & Structure
   - Down 2,3 (16x, 8x) - Semantic Details
   - Mid (8x) - Abstract Processing
   - Up 0,1 (8x, 16x) - Reconstruction
   - Up 2,3 (32x, 64x) - Final Style & Textures
3. **Optional Features**: Cross-attention boost and locks for enhanced control
4. **Result**: Perfect fusion with granular control over each aspect

## Architecture

XLFusion V1.1 features a **modular architecture** that separates merge functionality into specialized components for better maintainability and code organization.

### Core Components

- **XLFusion.py** - Main application with CLI interface, configuration management, and orchestration
- **code-utils/** - Modular merge implementations:
  - **legacy_merge.py** - Traditional weighted merging with LoRA support
  - **perres_merge.py** - Resolution-based block assignment with CLI prompts
  - **hybrid_merge.py** - Combined weighted + resolution control with configuration UI

This modular design allows for:
- **Clean separation** of merge algorithms
- **Easy maintenance** and debugging of specific modes
- **Extensibility** for future merge techniques
- **Reusable components** across different interfaces

## Directory Structure

```
XLFusion/
├── XLFusion.py          # Main application script
├── config.yaml          # Centralized configuration
├── code_utils/          # Modular merge components
│   ├── legacy_merge.py  # Legacy mode implementation
│   ├── perres_merge.py  # PerRes mode implementation
│   ├── hybrid_merge.py  # Hybrid mode implementation
│   └── __init__.py      # Python package initialization
├── models/              # Input checkpoint files
├── loras/               # LoRA files (Legacy mode only)
├── output/              # Merged model outputs
└── metadata/            # Audit logs and metadata
```

## Output

- **Merged Models**: `XLFusion_V{number}.safetensors`
- **Metadata Logs**: `meta_XLFusion_V{number}.txt`
- **Compatibility**: AUTOMATIC1111, ComfyUI, and other SDXL-compatible software

## Configuration

XLFusion uses `config.yaml` for centralized configuration management. Key settings include:

### Model Output Settings
- **Base Name**: Customize the base name for merged models
- **Version Format**: Configure version numbering (V1, V2, v1, Ver1, etc.)
- **Auto-Versioning**: Automatic increment to prevent overwrites
- **Directory Paths**: Customize input/output folder locations

### Merge Mode Defaults
- **Legacy Mode**: Default multipliers for down/mid/up blocks and cross-attention boost
- **PerRes Mode**: Default cross-attention lock behavior
- **Hybrid Mode**: Default cross-attention boost and lock settings for optimal performance

### Advanced Configuration

#### PerRes Mode
- Assign different models to specific resolution blocks
- Configure cross-attention locks for improved prompt consistency
- Ideal for combining models with complementary strengths

#### Legacy Mode
- Weight-based merging with block-level multipliers
- Cross-attention boost for enhanced text adherence
- LoRA integration with customizable scaling

#### Hybrid Mode (V1.1)
- Configure weights per resolution block for maximum control
- Combine the flexibility of Legacy with precision of PerRes
- Perfect for complex fusion scenarios requiring different strengths per resolution
- Supports cross-attention boost and optional locks

## Contact

- **Portfolio**: [https://warcos.dev/](https://warcos.dev/)
- **LinkedIn**: [https://www.linkedin.com/in/marcosgarest/](https://www.linkedin.com/in/marcosgarest/)