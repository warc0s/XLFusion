"""
CLI utilities for XLFusion
Interactive prompt functions for user input
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def prompt_select(items: List[Path], title: str, default_idx: List[int]) -> List[int]:
    print(f"\n{title}")
    for i, p in enumerate(items):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  [{i}] {p.name}  ({size_mb:.1f} MB)")
    if not items:
        return []
    def_str = ",".join(str(i) for i in default_idx if 0 <= i < len(items))
    valid_defaults = [i for i in default_idx if 0 <= i < len(items)]
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
            pass
    return idx or valid_defaults


def prompt_weights(names: List[str], suggestion: List[float]) -> List[float]:
    def_str = ",".join(str(w) for w in suggestion)
    raw = input(f"Enter weights for {names} [{def_str}]: ").strip()
    if not raw:
        raw = def_str
    weights = []
    for tok in raw.split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            weights.append(float(tok))
        except ValueError:
            pass
    # Fill missing with suggestion
    while len(weights) < len(names):
        weights.append(suggestion[len(weights)])
    return weights[:len(names)]


def pick_backbone(names: List[str], weights: Optional[List[float]] = None) -> int:
    if weights is None:
        weights = [1.0] * len(names)
    max_w = max(weights)
    candidates = [i for i, w in enumerate(weights) if w == max_w]
    if len(candidates) == 1:
        return candidates[0]
    print("\nMultiple models have equal highest weight. Pick backbone:")
    for i in candidates:
        print(f"  [{i}] {names[i]}")
    while True:
        try:
            choice = int(input(f"Select backbone index [{candidates[0]}]: ").strip() or str(candidates[0]))
            if choice in candidates:
                return choice
        except ValueError:
            pass


def prompt_block_merge(names: List[str]) -> Optional[List[Dict[str, float]]]:
    """
    Solicita multiplicadores por bloque y por modelo para el modo Legacy.

    Soporta N modelos. Para cada bloque (down_0_1, down_2_3, mid, up_0_1, up_2_3)
    se piden pesos para cada modelo como pares idx:peso separados por coma.
    Si se deja vacío, se salta la configuración por bloques.

    Devuelve una lista de longitud N donde cada elemento es un diccionario
    {block_name -> multiplier} para ese modelo.
    """
    print("\nMultiplicadores por bloque (opcional):")
    print("Para cada bloque, introduce pesos por modelo en formato 0:1.0,1:0.5,2:0.0")
    print("Bloques disponibles: down_0_1, down_2_3, mid, up_0_1, up_2_3")

    blocks = ['down_0_1', 'down_2_3', 'mid', 'up_0_1', 'up_2_3']
    per_model: List[Dict[str, float]] = [{ } for _ in names]

    any_set = False
    for block in blocks:
        raw = input(f"Pesos para {block} (e.g., 0:1.0,1:0.5) [ENTER para omitir]: ").strip()
        if not raw:
            continue
        any_set = True
        for item in raw.split(','):
            if ':' not in item:
                continue
            idx_str, weight_str = item.split(':', 1)
            try:
                idx = int(idx_str.strip())
                w = float(weight_str.strip())
            except ValueError:
                continue
            if 0 <= idx < len(names):
                per_model[idx][block] = w

    return per_model if any_set else None


def prompt_crossattn_boost(names: List[str]) -> Optional[List[Dict[str, float]]]:
    """
    Solicita boosts para cross-attention por bloque grueso (down/mid/up) y por modelo.
    Soporta N modelos. Devuelve lista de longitud N con {block -> boost} por modelo.
    """
    print("\nCross-attention boost (opcional):")
    print("Para cada bloque (down, mid, up), introduce pares idx:boost (e.g., 0:1.0,1:0.8)")

    blocks = ['down', 'mid', 'up']
    per_model: List[Dict[str, float]] = [{ } for _ in names]
    any_set = False
    for block in blocks:
        raw = input(f"Boosts para {block} [ENTER para omitir]: ").strip()
        if not raw:
            continue
        any_set = True
        for item in raw.split(','):
            if ':' not in item:
                continue
            idx_str, boost_str = item.split(':', 1)
            try:
                idx = int(idx_str.strip())
                b = float(boost_str.strip())
            except ValueError:
                continue
            if 0 <= idx < len(names):
                per_model[idx][block] = b

    return per_model if any_set else None


def prompt_loras(lora_files: List[Path]) -> List[Tuple[Path, float]]:
    if not lora_files:
        return []
    
    print("\nAvailable LoRA files:")
    for i, p in enumerate(lora_files):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  [{i}] {p.name} ({size_mb:.1f} MB)")
    
    raw = input("Select LoRA indices with scale (e.g., 0:1.0,2:0.5): ").strip()
    if not raw:
        return []
    
    selected = []
    for item in raw.split(','):
        if ':' not in item:
            continue
        idx_str, scale_str = item.split(':', 1)
        try:
            idx = int(idx_str.strip())
            scale = float(scale_str.strip())
            if 0 <= idx < len(lora_files):
                selected.append((lora_files[idx], scale))
        except ValueError:
            continue
    
    return selected


def prompt_hybrid_config(model_names: List[str]) -> Tuple[Dict[str, Dict[str, float]], Optional[Dict[str, int]]]:
    print("\nHybrid mode configuration:")
    print("For each block group, specify model weights (sum should be ~1.0)")
    
    blocks = ['down_0_1', 'down_2_3', 'mid', 'up_0_1', 'up_2_3']
    config = {}
    
    for block in blocks:
        print(f"\n{block}:")
        print(f"Available models: {', '.join(f'{i}:{name}' for i, name in enumerate(model_names))}")
        
        while True:
            raw = input(f"Enter weights for {block} (e.g., 0:0.7,1:0.3): ").strip()
            if not raw:
                # Default to first model
                config[block] = {0: 1.0}
                break
            
            weights = {}
            valid = True
            for item in raw.split(','):
                if ':' not in item:
                    valid = False
                    break
                idx_str, weight_str = item.split(':', 1)
                try:
                    idx = int(idx_str.strip())
                    weight = float(weight_str.strip())
                    if 0 <= idx < len(model_names):
                        weights[idx] = weight
                    else:
                        valid = False
                        break
                except ValueError:
                    valid = False
                    break
            
            if valid and weights:
                config[block] = weights
                break
            print("Invalid format. Try again.")
    
    # Cross-attention locks
    print("\nCross-attention locks (optional):")
    print("Lock specific block types to specific models")
    raw = input("Enter locks (e.g., down:0,mid:1,up:2) or press Enter to skip: ").strip()
    
    locks = None
    if raw:
        locks = {}
        for item in raw.split(','):
            if ':' not in item:
                continue
            block_type, idx_str = item.split(':', 1)
            block_type = block_type.strip()
            try:
                idx = int(idx_str.strip())
                if block_type in ['down', 'mid', 'up'] and 0 <= idx < len(model_names):
                    locks[block_type] = idx
            except ValueError:
                continue
    
    return config, locks


def prompt_perres_assignments(model_names: List[str]) -> Tuple[Dict[str, int], Optional[Dict[str, int]]]:
    print("\nPerRes mode configuration:")
    print("Assign models to block groups (100% assignment)")
    
    blocks = ['down_0_1', 'down_2_3', 'mid', 'up_0_1', 'up_2_3']
    assignments = {}
    
    for block in blocks:
        print(f"\n{block}:")
        print(f"Available models: {', '.join(f'{i}:{name}' for i, name in enumerate(model_names))}")
        
        while True:
            raw = input(f"Assign model for {block} [0]: ").strip()
            if not raw:
                raw = "0"
            
            try:
                idx = int(raw)
                if 0 <= idx < len(model_names):
                    assignments[block] = idx
                    break
                else:
                    print(f"Invalid index. Use 0-{len(model_names)-1}")
            except ValueError:
                print("Invalid number. Try again.")
    
    # Cross-attention locks
    print("\nCross-attention locks (optional):")
    print("Lock specific block types to specific models")
    raw = input("Enter locks (e.g., down:0,mid:1,up:2) or press Enter to skip: ").strip()
    
    locks = None
    if raw:
        locks = {}
        for item in raw.split(','):
            if ':' not in item:
                continue
            block_type, idx_str = item.split(':', 1)
            block_type = block_type.strip()
            try:
                idx = int(idx_str.strip())
                if block_type in ['down', 'mid', 'up'] and 0 <= idx < len(model_names):
                    locks[block_type] = idx
            except ValueError:
                continue
    
    return assignments, locks
