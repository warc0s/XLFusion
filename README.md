# XLFusion V1.0

Professional SDXL checkpoint merger with advanced fusion capabilities. Supports two distinct merging modes for optimal model combination and customization.

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
5. Select merge mode and follow interactive prompts
6. Merged models are saved to `output/` with metadata logs in `metadata/`

## Directory Structure

```
XLFusion/
├── XLFusion.py          # Main script
├── config.yaml          # Centralized configuration
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

### Advanced Configuration

#### PerRes Mode
- Assign different models to specific resolution blocks
- Configure cross-attention locks for improved prompt consistency
- Ideal for combining models with complementary strengths

#### Legacy Mode
- Weight-based merging with block-level multipliers
- Cross-attention boost for enhanced text adherence
- LoRA integration with customizable scaling

## Contact

- **Portfolio**: [https://warcos.dev/](https://warcos.dev/)
- **LinkedIn**: [https://www.linkedin.com/in/marcosgarest/](https://www.linkedin.com/in/marcosgarest/)