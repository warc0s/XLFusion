#!/usr/bin/env python3
"""
XLFusion V1.1

Interactive SDXL checkpoint merger with three fusion modes:
- Legacy: Classic weighted merge with down/mid/up blocks
- PerRes: Resolution-based block control
- Hybrid: Combined weighted + resolution control (V1.1)

PerRes (per-resolution) mode:
- 100% assignment by block pairs based on resolution
- Down 0,1 (64x, 32x) and 2,3 (16x, 8x)
- Mid (8x latent)
- Up 0,1 (8x, 16x) and 2,3 (32x, 64x)
- Optional cross-attention (attn2) locks

Expected folder structure:
  ./models     -> base checkpoints .safetensors
  ./loras      -> LoRA files .safetensors (legacy mode only)
  ./output     -> merged outputs .safetensors
  ./metadata   -> audit logs meta_*.txt

Requirements:
  pip install torch safetensors
"""
from __future__ import annotations
import sys
import re
import gc
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import torch
import yaml
from safetensors.torch import load_file as st_load, save_file as st_save

# Import merge modules and common utilities
from code_utils.common import load_state, save_state
from code_utils.legacy_merge import (
    stream_weighted_merge_from_paths,
    apply_single_lora
)
from code_utils.perres_merge import merge_perres, prompt_perres_assignments
from code_utils.hybrid_merge import (
    merge_hybrid,
    prompt_hybrid_weights
)

# SDXL A1111 style naming constants
UNET_PREFIX = "model.diffusion_model."

# Configuration loader
def load_config() -> dict:
    """Load configuration from config.yaml"""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Fallback configuration if config.yaml doesn't exist
        return {
            "model_output": {
                "base_name": "XLFusion",
                "version_prefix": "V",
                "file_extension": ".safetensors",
                "output_dir": "output",
                "metadata_dir": "metadata",
                "auto_version": True
            },
            "directories": {
                "models": "models",
                "loras": "loras",
                "output": "output",
                "metadata": "metadata"
            },
            "merge_defaults": {
                "legacy": {
                    "cross_attention_boost": 1.0,
                    "down_blocks_multiplier": 1.0,
                    "mid_blocks_multiplier": 1.0,
                    "up_blocks_multiplier": 1.0
                },
                "perres": {
                    "cross_attention_locks": False
                },
                "hybrid": {
                    "cross_attention_boost": 1.0,
                    "cross_attention_locks": False
                }
            },
            "app": {
                "tool_name": "XLFusion",
                "version": "1.1"
            }
        }

# ---------------- I/O Utilities ----------------

def ensure_dirs(root: Path) -> Tuple[Path, Path, Path, Path]:
    config = load_config()
    dirs = config["directories"]

    models = root / dirs["models"]
    loras = root / dirs["loras"]
    output = root / dirs["output"]
    metadata = root / dirs["metadata"]
    for p in [models, loras, output, metadata]:
        p.mkdir(parents=True, exist_ok=True)
    return models, loras, output, metadata


def list_safetensors(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("*.safetensors") if p.is_file()])


def next_version_path(output_dir: Path) -> Tuple[Path, int]:
    config = load_config()
    output_cfg = config["model_output"]

    base_name = output_cfg["base_name"]
    version_prefix = output_cfg["version_prefix"]
    file_extension = output_cfg["file_extension"]

    pattern = re.compile(fr"{re.escape(base_name)}_{re.escape(version_prefix)}(\d+){re.escape(file_extension)}$")
    max_v = 0

    for p in output_dir.glob(f"{base_name}_{version_prefix}*{file_extension}"):
        m = pattern.search(p.name)
        if m:
            try:
                v = int(m.group(1))
                max_v = max(max_v, v)
            except ValueError:
                pass

    next_v = max_v + 1
    return output_dir / f"{base_name}_{version_prefix}{next_v}{file_extension}", next_v


# ---------------- Interactive CLI ----------------

def prompt_select(items: List[Path], title: str, default_idx: List[int]) -> List[int]:
    print(f"\n{title}")
    for i, p in enumerate(items):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  [{i}] {p.name}  ({size_mb:.1f} MB)")
    if not items:
        return []
    def_str = ",".join(str(i) for i in default_idx if 0 <= i < len(items))
    raw = input(f"Select indices separated by comma [{def_str}]: ").strip()
    if not raw:
        raw = def_str
    idx = []
    for tok in raw.split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            k = int(tok)
            if 0 <= k < len(items):
                idx.append(k)
        except ValueError:
            print(f"Warning: '{tok}' is not a valid index, skipping.")
    idx = sorted(list(dict.fromkeys(idx)))

    # Ensure at least one valid selection
    if not idx and def_str:
        print("No valid indices selected, using defaults.")
        for i in default_idx:
            if 0 <= i < len(items):
                idx.append(i)
        idx = sorted(list(dict.fromkeys(idx)))

    return idx


def prompt_weights(names: List[str], suggestion: List[float]) -> List[float]:
    print("\nUNet weights per checkpoint.")
    if not names:
        print("Error: No model names provided.")
        return []

    ws: List[float] = []
    for name, sug in zip(names, suggestion):
        raw = input(f"Weight for {name} [{sug}]: ").strip()
        if raw == "":
            ws.append(float(sug))
        else:
            try:
                ws.append(float(raw))
            except ValueError:
                print("Invalid input. Using suggestion.")
                ws.append(float(sug))

    # Ensure we have weights
    if not ws:
        print("No weights provided. Using uniform distribution.")
        ws = [1.0 for _ in names]

    s = sum(ws)
    if s <= 0:
        print("All weights were zero or invalid. Assigning uniform distribution.")
        ws = [1.0 for _ in ws]
        s = sum(ws)

    ws_norm = [w / s for w in ws]
    print("Normalized weights:")
    for name, w in zip(names, ws_norm):
        print(f"  {name}: {w:.4f}")
    return ws_norm


def pick_backbone(names: List[str], weights: Optional[List[float]] = None) -> int:
    if not names:
        print("Error: No model names provided for backbone selection.")
        return 0

    if weights and len(weights) == len(names):
        by_w = sorted(list(enumerate(weights)), key=lambda x: x[1], reverse=True)
        default_idx = by_w[0][0]
    else:
        default_idx = 0

    # Ensure default_idx is valid
    if default_idx >= len(names):
        default_idx = 0

    print(f"\nBackbone for CLIP and VAE")
    print("The backbone provides text and image encoders. Choose the most versatile model.")
    raw = input(f"Default: {names[default_idx]} [Enter to accept, or index 0..{len(names)-1}]: ").strip()
    if raw == "":
        return default_idx
    try:
        idx = int(raw)
        if 0 <= idx < len(names):
            return idx
        else:
            print(f"Index {idx} out of range. Using default.")
    except ValueError:
        print("Invalid input format. Using default.")
    return default_idx


def prompt_block_merge(names: List[str]) -> Optional[List[Dict[str, float]]]:
    print("\nMerge by down/mid/up blocks")
    raw = input("Enable block merge? [n] ").strip().lower()
    if raw not in ("y", "yes"):
        return None
    mults: List[Dict[str, float]] = []
    for name in names:
        s_down = input(f"Down multiplier for {name} [1.0]: ").strip()
        s_mid  = input(f"Mid multiplier for {name}  [1.0]: ").strip()
        s_up   = input(f"Up multiplier for {name}   [1.0]: ").strip()
        try:
            m_down = float(s_down) if s_down else 1.0
            m_mid  = float(s_mid) if s_mid else 1.0
            m_up   = float(s_up) if s_up else 1.0
        except ValueError:
            print("Invalid input. Using 1.0, 1.0, 1.0.")
            m_down, m_mid, m_up = 1.0, 1.0, 1.0
        mults.append(dict(down=m_down, mid=m_mid, up=m_up, other=1.0))
    return mults


def prompt_crossattn_boost(names: List[str]) -> Optional[List[Dict[str, float]]]:
    print("\nCross-attention boost")
    raw = input("Configure cross-attention boost? [n] ").strip().lower()
    if raw not in ("y", "yes"):
        return None
    print("Models:")
    for i, n in enumerate(names):
        print(f"  [{i}] {n}")
    raw_idx = input("Indices to boost separated by comma: ").strip()
    idxs: List[int] = []
    for tok in raw_idx.split(',') if raw_idx else []:
        tok = tok.strip()
        if not tok:
            continue
        try:
            k = int(tok)
            if 0 <= k < len(names):
                idxs.append(k)
        except ValueError:
            pass
    idxs = sorted(list(dict.fromkeys(idxs)))
    boosts = [dict(down=1.0, mid=1.0, up=1.0) for _ in names]
    if not idxs:
        print("No model selected for boost.")
        return None
    for k in idxs:
        raw_f = input(f"Factor for {names[k]} [1.20]: ").strip()
        try:
            factor = float(raw_f) if raw_f else 1.20
        except ValueError:
            print("Invalid input. Using 1.20.")
            factor = 1.20
        raw_where = input("Blocks to boost [down,mid]: ").strip().lower()
        if not raw_where:
            where = {"down", "mid"}
        else:
            where = {w.strip() for w in raw_where.split(',') if w.strip() in {"down", "mid", "up"}}
            if not where:
                where = {"down", "mid"}
        for g in where:
            boosts[k][g] = factor
    return boosts


def prompt_loras(lora_files: List[Path]) -> List[Tuple[Path, float]]:
    if not lora_files:
        return []
    print("\nLoRA selection to bake")
    for i, p in enumerate(lora_files):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  [{i}] {p.name}  ({size_mb:.1f} MB)")
    raw_idx = input("Select indices separated by comma [Enter for none]: ").strip()
    if not raw_idx:
        return []
    idxs = []
    for tok in raw_idx.split(','):
        tok = tok.strip()
        if tok == "":
            continue
        try:
            k = int(tok)
            if 0 <= k < len(lora_files):
                idxs.append(k)
        except ValueError:
            pass
    idxs = sorted(list(dict.fromkeys(idxs)))
    chosen = []
    for k in idxs:
        p = lora_files[k]
        raw_s = input(f"Scale for {p.name} [0.30]: ").strip()
        try:
            s = 0.30 if raw_s == "" else float(raw_s)
        except ValueError:
            print("Invalid input. Using 0.30.")
            s = 0.30
        chosen.append((p, s))
    return chosen


# All merge functions are now imported from code_utils modules

# ---------------- Orchestration ----------------

def main() -> int:
    try:
        root = Path(__file__).resolve().parent
    except Exception:
        # Fallback to current working directory if __file__ resolution fails
        root = Path.cwd()
        print("Warning: Using current working directory as project root.")
    models_dir, loras_dir, output_dir, metadata_dir = ensure_dirs(root)

    print("\n" + "="*60)
    print(" XLFusion V1.1 - Advanced SDXL checkpoint merger")
    print("="*60)
    print(f"\nDetected structure:")
    print(f"  models:   {models_dir}")
    print(f"  loras:    {loras_dir}")
    print(f"  output:   {output_dir}")
    print(f"  metadata: {metadata_dir}")

    model_files = list_safetensors(models_dir)
    if not model_files:
        print("\nNo checkpoints found in ./models")
        print("   Place at least one and run again.")
        return 1

    # Mode selection
    print("\n" + "="*60)
    print("FUSION MODE")
    print("="*60)
    print("[1] Legacy - Classic weighted merge (3 blocks + LoRAs)")
    print("[2] PerRes - Resolution-based control (no LoRAs)")
    print("[3] Hybrid - Combined weighting + resolution control (NEW)")
    print("\n    Legacy: Traditional weighted blending")
    print("    PerRes: Complete block assignment by resolution")
    print("    Hybrid: Best of both - weights applied per resolution")

    mode_raw = input("\nSelect mode [1]: ").strip()
    if mode_raw == "2":
        mode = "perres"
    elif mode_raw == "3":
        mode = "hybrid"
    else:
        mode = "legacy"

    if mode == "perres":
        # PERRES MODE
        print("\n" + "="*60)
        print("PERRES MODE ACTIVATED")
        print("="*60)

        # Model selection
        print("\nAvailable checkpoints:")
        for i, p in enumerate(model_files):
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  [{i}] {p.name}  ({size_mb:.1f} MB)")

        print("\nRecommendation: use 2-3 complementary models")
        print("   Ex: one generalist, one with good style, one realistic")

        default_model_idx = list(range(min(3, len(model_files))))
        selected_idx = prompt_select(model_files, "Select models to merge:", default_model_idx)

        if not selected_idx:
            print("\nNo models selected. Aborting.")
            return 1
        if len(selected_idx) < 2:
            print("\nPerRes requires at least 2 models. Switching to Legacy mode...")
            mode = "legacy"
        else:
            selected_models = [model_files[i] for i in selected_idx]
            model_names = [p.name for p in selected_models]

            # Configure PerRes assignments
            assignments, attn2_locks = prompt_perres_assignments(model_names)

            # Backbone
            backbone_idx = pick_backbone(model_names)

            # PerRes fusion
            merged = merge_perres(
                selected_models,
                assignments,
                backbone_idx,
                attn2_locks
            )

            # Prepare metadata
            out_path, version = next_version_path(output_dir)

            config = load_config()
            app_cfg = config["app"]
            output_cfg = config["model_output"]

            meta_embed = {
                "title": f"{output_cfg['base_name']}_{output_cfg['version_prefix']}{version}",
                "format": "sdxl-a1111-like",
                "merge_mode": "perres",
                "backbone": model_names[backbone_idx],
                "models": json.dumps(model_names, ensure_ascii=False),
                "assignments": json.dumps(assignments, ensure_ascii=False),
                "attn2_locks": json.dumps(attn2_locks, ensure_ascii=False) if attn2_locks else "",
                "created": datetime.now().isoformat(timespec='seconds'),
                "tool": "XLFusion",
            }

            # Save
            print(f"\nSaving model to {out_path.name}...")
            save_state(out_path, merged, meta_embed)

            # Audit log
            meta_txt = metadata_dir / f"meta_{output_cfg['base_name']}_{output_cfg['version_prefix']}{version}.txt"
            lines = [
                f"{app_cfg['tool_name']} V{app_cfg['version']} - PerRes Mode",
                f"Date: {datetime.now().isoformat(timespec='seconds')}",
                f"Output: {out_path.name}",
                "",
                "Base models:",
            ]
            for i, name in enumerate(model_names):
                lines.append(f"  [{i}] {name}")
            lines.append("")
            lines.append(f"Backbone (CLIP/VAE): {model_names[backbone_idx]}")
            lines.append("")
            lines.append("Resolution assignments:")
            lines.append(f"  Down 0,1: {model_names[assignments['down_0_1']]}")
            lines.append(f"  Down 2,3: {model_names[assignments['down_2_3']]}")
            lines.append(f"  Mid:      {model_names[assignments['mid']]}")
            lines.append(f"  Up 0,1:   {model_names[assignments['up_0_1']]}")
            lines.append(f"  Up 2,3:   {model_names[assignments['up_2_3']]}")

            if attn2_locks:
                lines.append("")
                lines.append("Cross-attention locks (attn2):")
                lines.append(f"  Down: {model_names[attn2_locks['down']]}")
                lines.append(f"  Mid:  {model_names[attn2_locks['mid']]}")
                lines.append(f"  Up:   {model_names[attn2_locks['up']]}")

            lines.append("")
            lines.append(f"Total keys: {len(merged)}")
            lines.append(f"UNet keys: {len([k for k in merged.keys() if k.startswith(UNET_PREFIX)])}")

            meta_txt.write_text("\n".join(lines), encoding="utf-8")
            print(f"Metadata log: {meta_txt.name}")

            print("\nPerRes fusion completed successfully")
            print("Validate with different seeds and prompts to verify consistency")
            return 0

    # HYBRID MODE
    elif mode == "hybrid":
        print("\n" + "="*60)
        print("HYBRID MODE ACTIVATED")
        print("="*60)
        print("Combines weighted blending with resolution-based control")
        print("Configure different weights for each resolution block")

        # Model selection
        print("\nAvailable checkpoints:")
        for i, p in enumerate(model_files):
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  [{i}] {p.name}  ({size_mb:.1f} MB)")

        print("\nRecommendation: use 2-4 models with complementary strengths")
        default_model_idx = list(range(min(3, len(model_files))))
        selected_idx = prompt_select(model_files, "Select models to merge:", default_model_idx)

        if not selected_idx:
            print("\nNo models selected. Aborting.")
            return 1
        if len(selected_idx) < 2:
            print("\nHybrid mode requires at least 2 models. Switching to Legacy mode...")
            mode = "legacy"
        else:
            selected_models = [model_files[i] for i in selected_idx]
            model_names = [p.name for p in selected_models]

            # Configure hybrid weights per resolution block
            block_weights = prompt_hybrid_weights(model_names)

            # Backbone selection
            backbone_idx = pick_backbone(model_names)

            # Cross-attention boost
            try:
                config = load_config()
                default_boost = config.get("merge_defaults", {}).get("hybrid", {}).get("cross_attention_boost", 1.0)
            except (KeyError, TypeError, AttributeError):
                default_boost = 1.0  # Fallback if hybrid config missing
            boost_str = input(f"\nCross-attention boost [{default_boost}]: ").strip()
            if boost_str:
                try:
                    cross_attention_boost = float(boost_str)
                    if cross_attention_boost < 0:
                        print("Warning: Negative boost value, using default.")
                        cross_attention_boost = default_boost
                except ValueError:
                    print("Invalid boost value, using default.")
                    cross_attention_boost = default_boost
            else:
                cross_attention_boost = default_boost

            # Optional cross-attention locks
            use_locks = input("\nUse cross-attention locks? [n]: ").strip().lower() in ['y', 'yes']
            attn2_locks = None
            if use_locks:
                print("\nCross-attention locks (override block weights for attn2 layers):")
                locks = {}
                for block_type in ["down", "mid", "up"]:
                    idx_str = input(f"  {block_type.capitalize()} attn2 model [0]: ").strip()
                    if idx_str.isdigit():
                        idx = int(idx_str)
                        if 0 <= idx < len(selected_models):
                            locks[block_type] = idx
                        else:
                            print(f"Invalid index {idx}, using 0")
                            locks[block_type] = 0
                    else:
                        locks[block_type] = 0
                attn2_locks = locks

            # Execute hybrid merge
            print(f"\nExecuting hybrid merge...")
            merged = merge_hybrid(
                selected_models,
                [1.0] * len(selected_models),  # Not used in hybrid, but kept for compatibility
                block_weights,
                backbone_idx,
                cross_attention_boost,
                attn2_locks
            )

            # Prepare metadata
            out_path, version = next_version_path(output_dir)

            app_cfg = config["app"]
            output_cfg = config["model_output"]

            meta_embed = {
                "title": f"{output_cfg['base_name']}_{output_cfg['version_prefix']}{version}",
                "format": "sdxl-a1111-like",
                "merge_mode": "hybrid",
                "backbone": model_names[backbone_idx],
                "models": json.dumps(model_names, ensure_ascii=False),
                "block_weights": json.dumps(block_weights, ensure_ascii=False),
                "cross_attention_boost": str(cross_attention_boost),
                "attn2_locks": json.dumps(attn2_locks, ensure_ascii=False) if attn2_locks else "",
                "created": datetime.now().isoformat(timespec='seconds'),
                "tool": "XLFusion",
            }

            # Save
            print(f"\nSaving model to {out_path.name}...")
            save_state(out_path, merged, meta_embed)

            # Audit log
            meta_txt = metadata_dir / f"meta_{output_cfg['base_name']}_{output_cfg['version_prefix']}{version}.txt"
            lines = [
                f"{app_cfg['tool_name']} V{app_cfg['version']} - Hybrid Mode",
                f"Date: {datetime.now().isoformat(timespec='seconds')}",
                f"Output: {out_path.name}",
                "",
                "Base models:",
            ]
            for i, name in enumerate(model_names):
                lines.append(f"  [{i}] {name}")
            lines.append("")
            lines.append(f"Backbone (CLIP/VAE): {model_names[backbone_idx]}")
            lines.append("")
            lines.append("Block weights:")
            for block, weights in block_weights.items():
                lines.append(f"  {block}: {[f'{w:.3f}' for w in weights]}")

            if cross_attention_boost != 1.0:
                lines.append(f"\nCross-attention boost: {cross_attention_boost}")

            if attn2_locks:
                lines.append("")
                lines.append("Cross-attention locks (attn2):")
                lines.append(f"  Down: {model_names[attn2_locks['down']]}")
                lines.append(f"  Mid:  {model_names[attn2_locks['mid']]}")
                lines.append(f"  Up:   {model_names[attn2_locks['up']]}")

            lines.append("")
            lines.append(f"Total keys: {len(merged)}")
            lines.append(f"UNet keys: {len([k for k in merged.keys() if k.startswith(UNET_PREFIX)])}")

            meta_txt.write_text("\n".join(lines), encoding="utf-8")
            print(f"Metadata log: {meta_txt.name}")

            print("\nHybrid fusion completed successfully")
            print("Validate with different seeds and prompts to verify consistency")
            return 0

    # LEGACY MODE
    if mode == "legacy":
        print("\nLegacy mode activated")

        # Checkpoint selection
        default_model_idx = [0, 1] if len(model_files) >= 2 else [0]
        selected_idx = prompt_select(model_files, "Available checkpoints:", default_model_idx)
        if not selected_idx:
            print("No checkpoint selected. Aborting.")
            return 1

        selected_models = [model_files[i] for i in selected_idx]
        model_names = [p.name for p in selected_models]

        # Weights
        if len(selected_models) == 1:
            suggestions = [1.0]
        elif len(selected_models) == 2:
            suggestions = [0.65, 0.35]
        elif len(selected_models) == 3:
            suggestions = [0.62, 0.28, 0.10]
        else:
            suggestions = [1.0 / len(selected_models) for _ in selected_models]
        weights = prompt_weights(model_names, suggestions)

        # Backbone
        backbone_idx = pick_backbone(model_names, weights)

        # Block merge
        block_multipliers = prompt_block_merge(model_names)

        # Cross-attention boost
        crossattn_boosts = prompt_crossattn_boost(model_names)

        print("\nFusion with memory containment...")
        merged, backbone_state = stream_weighted_merge_from_paths(
            selected_models, weights, backbone_idx,
            only_unet=True,
            block_multipliers=block_multipliers,
            crossattn_boosts=crossattn_boosts,
        )

        # LoRAs
        lora_files = list_safetensors(loras_dir)
        baked_info = []
        if lora_files:
            chosen_loras = prompt_loras(lora_files)
            if chosen_loras:
                for p, s in chosen_loras:
                    print(f"Baking {p.name} with scale {s}...")
                    applied, skipped = apply_single_lora(merged, p, s)
                    print(f"  Applied {applied}, skipped {skipped}")
                    baked_info.append((p.name, float(s), int(applied), int(skipped)))

        # Save
        out_path, version = next_version_path(output_dir)

        config = load_config()
        app_cfg = config["app"]
        output_cfg = config["model_output"]

        meta_embed = {
            "title": f"{output_cfg['base_name']}_{output_cfg['version_prefix']}{version}",
            "format": "sdxl-a1111-like",
            "unet_merge": "weighted_blocked_crossboost_stream",
            "backbone": model_names[backbone_idx],
            "bases": json.dumps([{ "file": n, "weight": float(w) } for n, w in zip(model_names, weights)], ensure_ascii=False),
            "block_multipliers": json.dumps(block_multipliers, ensure_ascii=False) if block_multipliers else "",
            "crossattn_boosts": json.dumps(crossattn_boosts, ensure_ascii=False) if crossattn_boosts else "",
            "loras": json.dumps([{ "file": n, "scale": s, "applied": a, "skipped": k } for n, s, a, k in baked_info], ensure_ascii=False),
            "created": datetime.now().isoformat(timespec='seconds'),
            "tool": "XLFusion",
        }

        print(f"\nSaving model to {out_path.name}...")
        save_state(out_path, merged, meta_embed)

        # Audit log
        meta_txt = metadata_dir / f"meta_{output_cfg['base_name']}_{output_cfg['version_prefix']}{version}.txt"
        lines = [
            f"{app_cfg['tool_name']} V{app_cfg['version']} - Legacy Mode",
            f"Date: {datetime.now().isoformat(timespec='seconds')}",
            f"Output: {out_path.name}",
            "",
            f"Backbone (CLIP/VAE): {model_names[backbone_idx]}",
            "",
            "Base weights UNet:",
        ]
        for n, w in zip(model_names, weights):
            lines.append(f"  {n}: {w:.6f}")

        if block_multipliers:
            lines.append("")
            lines.append("Block multipliers:")
            for n, m in zip(model_names, block_multipliers):
                lines.append(f"  {n}: down={m.get('down',1.0):.3f} mid={m.get('mid',1.0):.3f} up={m.get('up',1.0):.3f}")

        if crossattn_boosts:
            lines.append("")
            lines.append("Cross-attention boost:")
            for n, b in zip(model_names, crossattn_boosts):
                lines.append(f"  {n}: down={b.get('down',1.0):.3f} mid={b.get('mid',1.0):.3f} up={b.get('up',1.0):.3f}")

        if baked_info:
            lines.append("")
            lines.append("Baked LoRAs:")
            for n, s, a, k in baked_info:
                lines.append(f"  {n}  scale={s}  applied={a}  skipped={k}")

        lines.append("")
        lines.append(f"Total keys: {len(merged)}")
        lines.append(f"UNet keys: {len([k for k in merged.keys() if k.startswith(UNET_PREFIX)])}")

        meta_txt.write_text("\n".join(lines), encoding="utf-8")
        print(f"Metadata log: {meta_txt.name}")

        print("\nLegacy fusion completed")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)