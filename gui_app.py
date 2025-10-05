from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import ttk, messagebox

from Utils.config import ensure_dirs, list_safetensors, load_config
from Utils.merge import merge_hybrid, merge_perres, stream_weighted_merge_from_paths
from Utils.lora import apply_single_lora
from Utils.workflow import save_merge_results

BLOCK_GROUPS = ["down_0_1", "down_2_3", "mid", "up_0_1", "up_2_3"]
ATTN_BLOCKS = ["down", "mid", "up"]
MODEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _clear_children(widget: tk.Widget) -> None:
    for child in widget.winfo_children():
        child.destroy()


class LegacyConfigPanel(ttk.Frame):
    """Configuration panel for Legacy mode."""

    def __init__(self, master: tk.Widget) -> None:
        super().__init__(master)
        self.model_names: List[str] = []
        self.lora_paths: List[Path] = []

        self.weights_vars: List[tk.DoubleVar] = []
        self.backbone_var = tk.StringVar(value="0")
        self.backbone_options: List[str] = []

        self.blocks_enabled = tk.BooleanVar(value=False)
        self.cross_enabled = tk.BooleanVar(value=False)

        self.block_vars: Dict[tuple[int, str], tk.DoubleVar] = {}
        self.cross_vars: Dict[tuple[int, str], tk.DoubleVar] = {}
        self.lora_vars: List[Dict[str, tk.Variable]] = []

        self.weights_frame = ttk.LabelFrame(self, text="Global Weights")
        self.weights_frame.pack(fill="x", pady=4)

        self.backbone_frame = ttk.LabelFrame(self, text="Backbone Model")
        self.backbone_frame.pack(fill="x", pady=4)

        self.block_section = ttk.LabelFrame(self, text="Per-block multipliers (optional)")
        self.block_section.pack(fill="x", pady=4)
        block_toggle = ttk.Checkbutton(
            self.block_section,
            text="Enable per-block control",
            variable=self.blocks_enabled,
            command=self._update_block_state,
        )
        block_toggle.pack(anchor="w", padx=4, pady=(2, 4))
        self.block_table = ttk.Frame(self.block_section)
        self.block_table.pack(fill="x", padx=4, pady=4)

        self.cross_section = ttk.LabelFrame(self, text="Cross-attention boost (optional)")
        self.cross_section.pack(fill="x", pady=4)
        cross_toggle = ttk.Checkbutton(
            self.cross_section,
            text="Enable cross-attention control",
            variable=self.cross_enabled,
            command=self._update_cross_state,
        )
        cross_toggle.pack(anchor="w", padx=4, pady=(2, 4))
        self.cross_table = ttk.Frame(self.cross_section)
        self.cross_table.pack(fill="x", padx=4, pady=4)

        self.lora_section = ttk.LabelFrame(self, text="LoRA baking (optional)")
        self.lora_section.pack(fill="x", pady=4)

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    def refresh(
        self,
        model_names: List[str],
        lora_paths: List[Path],
        existing: Optional[Dict[str, object]] = None,
    ) -> None:
        self.model_names = model_names
        self.lora_paths = lora_paths
        self.blocks_enabled.set(False)
        self.cross_enabled.set(False)

        self._build_weights()
        self._build_backbone()
        self._build_block_table()
        self._build_cross_table()
        self._build_lora_table()

        if existing:
            weights = existing.get("weights")
            if isinstance(weights, list):
                for var, value in zip(self.weights_vars, weights):
                    var.set(float(value))

            backbone_idx = existing.get("backbone_idx")
            if isinstance(backbone_idx, int) and 0 <= backbone_idx < len(self.backbone_options):
                self.backbone_var.set(self.backbone_options[backbone_idx])

            blocks = existing.get("block_multipliers") or []
            if isinstance(blocks, list):
                self.blocks_enabled.set(True)
                for model_idx, data in enumerate(blocks):
                    for block, value in data.items():
                        key = (model_idx, block)
                        if key in self.block_vars:
                            self.block_vars[key].set(float(value))
                self._update_block_state()

            cross = existing.get("crossattn_boosts") or []
            if isinstance(cross, list):
                self.cross_enabled.set(True)
                for model_idx, data in enumerate(cross):
                    for block, value in data.items():
                        key = (model_idx, block)
                        if key in self.cross_vars:
                            self.cross_vars[key].set(float(value))
                self._update_cross_state()

            loras = existing.get("loras") or []
            if isinstance(loras, list):
                path_to_item = {item["path"]: item for item in self.lora_vars}
                for lora_entry in loras:
                    path_obj = None
                    scale_val = 1.0
                    if isinstance(lora_entry, tuple):
                        path_obj, scale_val = lora_entry
                    elif isinstance(lora_entry, dict):
                        path_name = lora_entry.get("file")
                        scale_val = lora_entry.get("scale", 1.0)
                        for p in self.lora_paths:
                            if p.name == path_name:
                                path_obj = p
                                break
                    if path_obj and path_obj in path_to_item:
                        item = path_to_item[path_obj]
                        item["enabled"].set(True)
                        item["scale"].set(float(scale_val))

    def _build_weights(self) -> None:
        _clear_children(self.weights_frame)
        self.weights_vars = []

        header = ttk.Frame(self.weights_frame)
        header.pack(fill="x", padx=4, pady=(4, 0))
        ttk.Label(header, text="Model", width=50, anchor="w").pack(side="left")
        ttk.Label(header, text="Weight", width=8).pack(side="right", padx=(4, 0))

        for idx, name in enumerate(self.model_names):
            row = ttk.Frame(self.weights_frame)
            row.pack(fill="x", padx=4, pady=2)
            var = tk.DoubleVar(value=1.0 if idx == 0 else 0.0)
            self.weights_vars.append(var)
            ttk.Label(row, text=f"[{idx}] {name}", anchor="w").pack(side="left", expand=True, fill="x")
            spin = ttk.Spinbox(
                row,
                from_=0.0,
                to=1.0,
                increment=0.05,
                textvariable=var,
                width=6,
                justify="right",
            )
            spin.pack(side="right")

        controls = ttk.Frame(self.weights_frame)
        controls.pack(fill="x", padx=4, pady=(4, 6))
        ttk.Button(controls, text="Normalize", command=self._normalize_weights).pack(side="left")

    def _build_backbone(self) -> None:
        _clear_children(self.backbone_frame)
        ttk.Label(
            self.backbone_frame,
            text="Select the model that will act as the main backbone",
        ).pack(anchor="w", padx=4, pady=(4, 2))

        values = [f"[{idx}] {name}" for idx, name in enumerate(self.model_names)]
        self.backbone_options = values
        if values:
            self.backbone_var.set(values[0])
        combo = ttk.Combobox(
            self.backbone_frame,
            values=values,
            textvariable=self.backbone_var,
            state="readonly",
        )
        combo.pack(anchor="w", padx=4, pady=(0, 4))

    def _build_block_table(self) -> None:
        _clear_children(self.block_table)
        self.block_vars = {}

        if not self.model_names:
            return

        total_columns = len(self.model_names) + 1
        for col in range(total_columns + 1):
            self.block_table.columnconfigure(col, weight=0)

        ttk.Label(self.block_table, text="Block", width=18).grid(row=0, column=0, padx=4, pady=2, sticky="w")
        for idx in range(len(self.model_names)):
            ttk.Label(self.block_table, text=f"M{idx}", width=8).grid(row=0, column=idx + 1, padx=4, pady=2)

        for r, block in enumerate(BLOCK_GROUPS, start=1):
            ttk.Label(self.block_table, text=block, width=18).grid(row=r, column=0, padx=4, pady=2, sticky="w")
            for c in range(len(self.model_names)):
                var = tk.DoubleVar(value=1.0 if c == 0 else 0.0)
                self.block_vars[(c, block)] = var
                spin = ttk.Spinbox(
                    self.block_table,
                    from_=0.0,
                    to=2.0,
                    increment=0.05,
                    textvariable=var,
                    width=6,
                    justify="right",
                )
                spin.grid(row=r, column=c + 1, padx=4, pady=2)

        ttk.Button(
            self.block_table,
            text="Reset",
            command=self._reset_block_table,
        ).grid(row=len(BLOCK_GROUPS) + 1, column=0, padx=4, pady=(4, 2), sticky="w")

        self._update_block_state()

    def _build_cross_table(self) -> None:
        _clear_children(self.cross_table)
        self.cross_vars = {}

        if not self.model_names:
            return

        total_columns = len(self.model_names) + 1
        for col in range(total_columns + 1):
            self.cross_table.columnconfigure(col, weight=0)

        ttk.Label(self.cross_table, text="Block", width=18).grid(row=0, column=0, padx=4, pady=2, sticky="w")
        for idx in range(len(self.model_names)):
            ttk.Label(self.cross_table, text=f"M{idx}", width=8).grid(row=0, column=idx + 1, padx=4, pady=2)

        for r, block in enumerate(ATTN_BLOCKS, start=1):
            ttk.Label(self.cross_table, text=block, width=18).grid(row=r, column=0, padx=4, pady=2, sticky="w")
            for c in range(len(self.model_names)):
                var = tk.DoubleVar(value=1.0 if c == 0 else 0.0)
                self.cross_vars[(c, block)] = var
                spin = ttk.Spinbox(
                    self.cross_table,
                    from_=0.0,
                    to=2.0,
                    increment=0.05,
                    textvariable=var,
                    width=6,
                    justify="right",
                )
                spin.grid(row=r, column=c + 1, padx=4, pady=2)

        ttk.Button(
            self.cross_table,
            text="Reset",
            command=self._reset_cross_table,
        ).grid(row=len(ATTN_BLOCKS) + 1, column=0, padx=4, pady=(4, 2), sticky="w")

        self._update_cross_state()

    def _build_lora_table(self) -> None:
        _clear_children(self.lora_section)
        ttk.Label(self.lora_section, text="Select optional LoRAs to bake").pack(
            anchor="w", padx=4, pady=(4, 2)
        )
        self.lora_vars = []

        if not self.lora_paths:
            ttk.Label(self.lora_section, text="No LoRA files found").pack(
                anchor="w", padx=4, pady=(0, 4)
            )
            return

        for path in self.lora_paths:
            row = ttk.Frame(self.lora_section)
            row.pack(fill="x", padx=4, pady=2)
            enabled = tk.BooleanVar(value=False)
            scale = tk.DoubleVar(value=1.0)
            ttk.Checkbutton(row, text=path.name, variable=enabled).pack(side="left", anchor="w")
            ttk.Label(row, text="Scale").pack(side="left", padx=(6, 2))
            spin = ttk.Spinbox(
                row,
                from_=0.0,
                to=2.0,
                increment=0.05,
                textvariable=scale,
                width=6,
                justify="right",
            )
            spin.pack(side="left")
            self.lora_vars.append({"path": path, "enabled": enabled, "scale": scale})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_weights(self) -> None:
        total = sum(var.get() for var in self.weights_vars)
        if total <= 0:
            return
        for var in self.weights_vars:
            var.set(round(var.get() / total, 4))

    def _reset_block_table(self) -> None:
        for (model_idx, block), var in self.block_vars.items():
            var.set(1.0 if model_idx == 0 else 0.0)

    def _reset_cross_table(self) -> None:
        for (model_idx, block), var in self.cross_vars.items():
            var.set(1.0 if model_idx == 0 else 0.0)

    def _update_block_state(self) -> None:
        state = "!disabled" if self.blocks_enabled.get() else "disabled"
        for widget in self.block_table.winfo_children():
            if isinstance(widget, ttk.Spinbox):
                widget.state([state])

    def _update_cross_state(self) -> None:
        state = "!disabled" if self.cross_enabled.get() else "disabled"
        for widget in self.cross_table.winfo_children():
            if isinstance(widget, ttk.Spinbox):
                widget.state([state])

    # ------------------------------------------------------------------
    # External accessors
    # ------------------------------------------------------------------
    def get_config(self) -> Dict[str, object]:
        weights = [float(var.get()) for var in self.weights_vars]
        backbone_idx = 0
        try:
            backbone_idx = int(self.backbone_var.get().split()[0].strip("[]"))
        except (ValueError, IndexError):
            backbone_idx = 0

        block_multipliers: Optional[List[Dict[str, float]]] = None
        if self.blocks_enabled.get():
            block_multipliers = []
            for model_idx in range(len(self.model_names)):
                row: Dict[str, float] = {}
                for block in BLOCK_GROUPS:
                    row[block] = float(self.block_vars[(model_idx, block)].get())
                block_multipliers.append(row)

        cross_boosts: Optional[List[Dict[str, float]]] = None
        if self.cross_enabled.get():
            cross_boosts = []
            for model_idx in range(len(self.model_names)):
                row: Dict[str, float] = {}
                for block in ATTN_BLOCKS:
                    row[block] = float(self.cross_vars[(model_idx, block)].get())
                cross_boosts.append(row)

        loras = []
        for item in self.lora_vars:
            if item["enabled"].get():
                loras.append((item["path"], float(item["scale"].get())))

        return {
            "weights": weights,
            "backbone_idx": backbone_idx,
            "block_multipliers": block_multipliers,
            "crossattn_boosts": cross_boosts,
            "loras": loras,
        }


class PerResConfigPanel(ttk.Frame):
    """Configuration panel for PerRes mode."""

    def __init__(self, master: tk.Widget) -> None:
        super().__init__(master)
        self.model_names: List[str] = []
        self.assignment_vars: Dict[str, tk.StringVar] = {}
        self.lock_vars: Dict[str, tk.StringVar] = {}
        self.lora_paths: List[Path] = []
        self.lora_vars: List[Dict[str, tk.Variable]] = []

        self.assign_frame = ttk.LabelFrame(self, text="Block assignment")
        self.assign_frame.pack(fill="x", pady=4)

        self.lock_frame = ttk.LabelFrame(self, text="Cross-attention locks (optional)")
        self.lock_frame.pack(fill="x", pady=4)

        self.lora_section = ttk.LabelFrame(self, text="LoRA baking (optional)")
        self.lora_section.pack(fill="x", pady=4)

    def refresh(
        self,
        model_names: List[str],
        lora_paths: List[Path],
        existing: Optional[Dict[str, object]] = None,
    ) -> None:
        self.model_names = model_names
        self.lora_paths = lora_paths
        self._build_assignments()
        self._build_locks()
        self._build_lora_table()

        if existing:
            assignments = existing.get("assignments") or {}
            for block, var in self.assignment_vars.items():
                idx = assignments.get(block)
                if isinstance(idx, int) and 0 <= idx < len(self.model_names):
                    var.set(f"{idx} - {self.model_names[idx]}")

            locks = existing.get("attn2_locks") or {}
            for block, var in self.lock_vars.items():
                idx = locks.get(block)
                if isinstance(idx, int) and 0 <= idx < len(self.model_names):
                    var.set(f"{idx} - {self.model_names[idx]}")

            loras = existing.get("loras") or []
            if isinstance(loras, list):
                path_to_item = {item["path"]: item for item in self.lora_vars}
                for lora_entry in loras:
                    path_obj = None
                    scale_val = 1.0
                    if isinstance(lora_entry, tuple):
                        path_obj, scale_val = lora_entry
                    elif isinstance(lora_entry, dict):
                        path_name = lora_entry.get("file")
                        scale_val = lora_entry.get("scale", 1.0)
                        for p in self.lora_paths:
                            if p.name == path_name:
                                path_obj = p
                                break
                    if path_obj and path_obj in path_to_item:
                        item = path_to_item[path_obj]
                        item["enabled"].set(True)
                        item["scale"].set(float(scale_val))

    def _build_assignments(self) -> None:
        _clear_children(self.assign_frame)
        self.assignment_vars = {}

        if not self.model_names:
            return

        ttk.Label(
            self.assign_frame,
            text="Assign each block to a model (100% contribution)",
        ).pack(anchor="w", padx=4, pady=(4, 2))

        options = [f"{idx} - {name}" for idx, name in enumerate(self.model_names)]
        for block in BLOCK_GROUPS:
            row = ttk.Frame(self.assign_frame)
            row.pack(fill="x", padx=4, pady=2)
            ttk.Label(row, text=block, width=18).pack(side="left")
            default = options[0] if options else ""
            var = tk.StringVar(value=default)
            combo = ttk.Combobox(row, values=options, textvariable=var, state="readonly")
            combo.pack(side="left", padx=(4, 0))
            self.assignment_vars[block] = var

    def _build_locks(self) -> None:
        _clear_children(self.lock_frame)
        self.lock_vars = {}

        if not self.model_names:
            return

        ttk.Label(
            self.lock_frame,
            text="Optional: lock cross-attention blocks to a model",
        ).pack(anchor="w", padx=4, pady=(4, 2))

        options = ["None"] + [f"{idx} - {name}" for idx, name in enumerate(self.model_names)]
        for block in ATTN_BLOCKS:
            row = ttk.Frame(self.lock_frame)
            row.pack(fill="x", padx=4, pady=2)
            ttk.Label(row, text=block, width=18).pack(side="left")
            var = tk.StringVar(value="None")
            combo = ttk.Combobox(row, values=options, textvariable=var, state="readonly")
            combo.pack(side="left", padx=(4, 0))
            self.lock_vars[block] = var

    def get_config(self) -> Dict[str, object]:
        assignments: Dict[str, int] = {}
        for block, var in self.assignment_vars.items():
            try:
                idx = int(var.get().split("-", 1)[0].strip())
            except (ValueError, IndexError):
                idx = 0
            assignments[block] = idx

        locks: Dict[str, int] = {}
        for block, var in self.lock_vars.items():
            raw = var.get()
            if raw == "None":
                continue
            try:
                idx = int(raw.split("-", 1)[0].strip())
            except (ValueError, IndexError):
                continue
            locks[block] = idx

        loras = []
        for item in self.lora_vars:
            if item["enabled"].get():
                loras.append((item["path"], float(item["scale"].get())))

        return {
            "assignments": assignments,
            "attn2_locks": locks or None,
            "loras": loras,
        }

    def _build_lora_table(self) -> None:
        _clear_children(self.lora_section)
        ttk.Label(self.lora_section, text="Select optional LoRAs to bake").pack(
            anchor="w", padx=4, pady=(4, 2)
        )
        self.lora_vars = []

        if not self.lora_paths:
            ttk.Label(self.lora_section, text="No LoRA files found").pack(
                anchor="w", padx=4, pady=(0, 4)
            )
            return

        for path in self.lora_paths:
            row = ttk.Frame(self.lora_section)
            row.pack(fill="x", padx=4, pady=2)
            enabled = tk.BooleanVar(value=False)
            scale = tk.DoubleVar(value=1.0)
            ttk.Checkbutton(row, text=path.name, variable=enabled).pack(side="left", anchor="w")
            ttk.Label(row, text="Scale").pack(side="left", padx=(6, 2))
            spin = ttk.Spinbox(
                row,
                from_=0.0,
                to=2.0,
                increment=0.05,
                textvariable=scale,
                width=6,
                justify="right",
            )
            spin.pack(side="left")
            self.lora_vars.append({"path": path, "enabled": enabled, "scale": scale})


class HybridConfigPanel(ttk.Frame):
    """Configuration panel for Hybrid mode."""

    def __init__(self, master: tk.Widget) -> None:
        super().__init__(master)
        self.model_names: List[str] = []
        self.weight_vars: Dict[str, List[tk.DoubleVar]] = {}
        self.lock_vars: Dict[str, tk.StringVar] = {}
        self.lora_paths: List[Path] = []
        self.lora_vars: List[Dict[str, tk.Variable]] = []

        self.weights_frame = ttk.LabelFrame(self, text="Per-block weights")
        self.weights_frame.pack(fill="x", pady=4)

        self.lock_frame = ttk.LabelFrame(self, text="Cross-attention locks (optional)")
        self.lock_frame.pack(fill="x", pady=4)

        self.lora_section = ttk.LabelFrame(self, text="LoRA baking (optional)")
        self.lora_section.pack(fill="x", pady=4)

    def refresh(
        self,
        model_names: List[str],
        lora_paths: List[Path],
        existing: Optional[Dict[str, object]] = None,
    ) -> None:
        self.model_names = model_names
        self.lora_paths = lora_paths
        self._build_weight_table()
        self._build_locks()
        self._build_lora_table()

        if existing:
            hybrid_cfg = existing.get("hybrid_config") or {}
            for block, weights in hybrid_cfg.items():
                vars_row = self.weight_vars.get(block)
                if not vars_row:
                    continue
                for idx, value in weights.items():
                    if isinstance(idx, int) and idx < len(vars_row):
                        vars_row[idx].set(float(value))

            locks = existing.get("attn2_locks") or {}
            for block, var in self.lock_vars.items():
                idx = locks.get(block)
                if isinstance(idx, int) and 0 <= idx < len(self.model_names):
                    var.set(f"{idx} - {self.model_names[idx]}")

            loras = existing.get("loras") or []
            if isinstance(loras, list):
                path_to_item = {item["path"]: item for item in self.lora_vars}
                for lora_entry in loras:
                    path_obj = None
                    scale_val = 1.0
                    if isinstance(lora_entry, tuple):
                        path_obj, scale_val = lora_entry
                    elif isinstance(lora_entry, dict):
                        path_name = lora_entry.get("file")
                        scale_val = lora_entry.get("scale", 1.0)
                        for p in self.lora_paths:
                            if p.name == path_name:
                                path_obj = p
                                break
                    if path_obj and path_obj in path_to_item:
                        item = path_to_item[path_obj]
                        item["enabled"].set(True)
                        item["scale"].set(float(scale_val))

    def _build_weight_table(self) -> None:
        _clear_children(self.weights_frame)
        self.weight_vars = {}

        if not self.model_names:
            return

        total_columns = len(self.model_names) + 2  # bloque + modelos + acciones
        for col in range(total_columns + 1):
            self.weights_frame.columnconfigure(col, weight=0)

        ttk.Label(self.weights_frame, text="Block", width=18).grid(row=0, column=0, padx=4, pady=2, sticky="w")
        for idx in range(len(self.model_names)):
            ttk.Label(self.weights_frame, text=f"M{idx}", width=8).grid(row=0, column=idx + 1, padx=4, pady=2)
        ttk.Label(self.weights_frame, text="Actions", width=10).grid(
            row=0, column=len(self.model_names) + 1, padx=4, pady=2
        )

        for r, block in enumerate(BLOCK_GROUPS, start=1):
            ttk.Label(self.weights_frame, text=block, width=18).grid(row=r, column=0, padx=4, pady=2, sticky="w")
            vars_row: List[tk.DoubleVar] = []
            for c in range(len(self.model_names)):
                var = tk.DoubleVar(value=1.0 if c == 0 else 0.0)
                vars_row.append(var)
                spin = ttk.Spinbox(
                    self.weights_frame,
                    from_=0.0,
                    to=1.0,
                    increment=0.05,
                    textvariable=var,
                    width=6,
                    justify="right",
                )
                spin.grid(row=r, column=c + 1, padx=4, pady=2)
            ttk.Button(
                self.weights_frame,
                text="Normalize",
                command=lambda b=block: self._normalize_block(b),
            ).grid(row=r, column=len(self.model_names) + 1, padx=4, pady=2)
            self.weight_vars[block] = vars_row

    def _build_locks(self) -> None:
        _clear_children(self.lock_frame)
        self.lock_vars = {}

        if not self.model_names:
            return

        ttk.Label(
            self.lock_frame,
            text="Optional: lock cross-attention blocks to a model",
        ).pack(anchor="w", padx=4, pady=(4, 2))

        options = ["None"] + [f"{idx} - {name}" for idx, name in enumerate(self.model_names)]
        for block in ATTN_BLOCKS:
            row = ttk.Frame(self.lock_frame)
            row.pack(fill="x", padx=4, pady=2)
            ttk.Label(row, text=block, width=18).pack(side="left")
            var = tk.StringVar(value="None")
            combo = ttk.Combobox(row, values=options, textvariable=var, state="readonly")
            combo.pack(side="left", padx=(4, 0))
            self.lock_vars[block] = var

    def _normalize_block(self, block: str) -> None:
        vars_row = self.weight_vars.get(block, [])
        total = sum(var.get() for var in vars_row)
        if total <= 0:
            if vars_row:
                vars_row[0].set(1.0)
            return
        for var in vars_row:
            var.set(round(var.get() / total, 4))

    def get_config(self) -> Dict[str, object]:
        hybrid_config: Dict[str, Dict[int, float]] = {}
        for block, vars_row in self.weight_vars.items():
            block_weights: Dict[int, float] = {}
            for idx, var in enumerate(vars_row):
                value = float(var.get())
                if value > 0:
                    block_weights[idx] = value
            if not block_weights:
                block_weights[0] = 1.0
            hybrid_config[block] = block_weights

        locks: Dict[str, int] = {}
        for block, var in self.lock_vars.items():
            raw = var.get()
            if raw == "None":
                continue
            try:
                idx = int(raw.split("-", 1)[0].strip())
            except (ValueError, IndexError):
                continue
            locks[block] = idx

        loras = []
        for item in self.lora_vars:
            if item["enabled"].get():
                loras.append((item["path"], float(item["scale"].get())))

        return {
            "hybrid_config": hybrid_config,
            "attn2_locks": locks or None,
            "loras": loras,
        }

    def _build_lora_table(self) -> None:
        _clear_children(self.lora_section)
        ttk.Label(self.lora_section, text="Select optional LoRAs to bake").pack(
            anchor="w", padx=4, pady=(4, 2)
        )
        self.lora_vars = []

        if not self.lora_paths:
            ttk.Label(self.lora_section, text="No LoRA files found").pack(
                anchor="w", padx=4, pady=(0, 4)
            )
            return

        for path in self.lora_paths:
            row = ttk.Frame(self.lora_section)
            row.pack(fill="x", padx=4, pady=2)
            enabled = tk.BooleanVar(value=False)
            scale = tk.DoubleVar(value=1.0)
            ttk.Checkbutton(row, text=path.name, variable=enabled).pack(side="left", anchor="w")
            ttk.Label(row, text="Scale").pack(side="left", padx=(6, 2))
            spin = ttk.Spinbox(
                row,
                from_=0.0,
                to=2.0,
                increment=0.05,
                textvariable=scale,
                width=6,
                justify="right",
            )
            spin.pack(side="left")
            self.lora_vars.append({"path": path, "enabled": enabled, "scale": scale})


class FusionGUI:
    """Main graphical interface of XLFusion V2.0."""

    def __init__(
        self,
        root_dir: Path,
        models_dir: Path,
        loras_dir: Path,
        output_dir: Path,
        metadata_dir: Path,
    ) -> None:
        self.root_dir = root_dir
        self.models_dir = models_dir
        self.loras_dir = loras_dir
        self.output_dir = output_dir
        self.metadata_dir = metadata_dir

        self.state: Dict[str, object] = {
            "model_indices": [],
            "model_paths": [],
            "model_names": [],
            "mode": "legacy",
            "legacy": {},
            "perres": {},
            "hybrid": {},
        }

        self.model_paths: List[Path] = []
        self.lora_paths: List[Path] = []

        self.root = tk.Tk()
        version = load_config()["app"].get("version", "2.0")
        self.root.title(f"XLFusion V{version} - Graphical Interface")
        self.root.geometry("980x720")
        self.root.minsize(840, 600)

        self.log_queue: Queue = Queue()
        self.worker: Optional[threading.Thread] = None
        self.is_running = False

        self.mode_var = tk.StringVar(value="legacy")

        self._build_layout()
        self._load_resources()
        self._show_step(0)
        self._update_nav_buttons()
        self.root.after(150, self._poll_queue)

    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=12, pady=(12, 4))
        ttk.Label(top, text="Guided Assistant", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(
            top,
            text="Complete the steps from left to right to configure and run a merge.",
        ).pack(anchor="w", pady=(2, 0))

        self.content_container = ttk.Frame(self.root)
        self.content_container.pack(fill="both", expand=True, padx=12, pady=12)

        self.content_canvas = tk.Canvas(self.content_container, highlightthickness=0)
        self.content_scrollbar = ttk.Scrollbar(
            self.content_container, orient="vertical", command=self.content_canvas.yview
        )
        self.content_canvas.configure(yscrollcommand=self.content_scrollbar.set)

        self.scroll_frame = ttk.Frame(self.content_canvas)
        self.scroll_window = self.content_canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")

        self.scroll_frame.bind(
            "<Configure>",
            lambda _: self.content_canvas.configure(scrollregion=self.content_canvas.bbox("all")),
        )
        self.content_canvas.bind(
            "<Configure>",
            lambda e: self.content_canvas.itemconfigure(self.scroll_window, width=e.width),
        )

        self.content_canvas.pack(side="left", fill="both", expand=True)
        self.content_scrollbar.pack(side="right", fill="y")

        self.content_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.content_canvas.bind("<Button-4>", self._on_mousewheel)
        self.content_canvas.bind("<Button-5>", self._on_mousewheel)

        self.steps: List[ttk.Frame] = [
            self._create_models_step(self.scroll_frame),
            self._create_mode_step(self.scroll_frame),
            self._create_config_step(self.scroll_frame),
            self._create_preview_step(self.scroll_frame),
            self._create_run_step(self.scroll_frame),
        ]
        for frame in self.steps:
            frame.pack_forget()

        nav = ttk.Frame(self.root)
        nav.pack(fill="x", padx=12, pady=(0, 12))
        self.prev_btn = ttk.Button(nav, text="Back", command=self._go_previous)
        self.prev_btn.pack(side="left")
        self.next_btn = ttk.Button(nav, text="Next", command=self._go_next)
        self.next_btn.pack(side="left", padx=(8, 0))
        self.run_btn = ttk.Button(nav, text="Start Merge", command=self._start_merge)
        self.close_btn = ttk.Button(nav, text="Close", command=self.root.destroy)
        self.close_btn.pack(side="right")

    # ------------------------------------------------------------------
    # Step builders
    # ------------------------------------------------------------------
    def _create_models_step(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent)
        ttk.Label(frame, text="Step 1: Model Library", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text="Select at least two checkpoints from the models/ folder.",
        ).pack(anchor="w", pady=(2, 8))

        table_frame = ttk.Frame(frame)
        table_frame.pack(fill="both", expand=True)

        columns = ("name", "size", "updated")
        self.model_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            selectmode="extended",
            height=14,
        )
        self.model_tree.heading("name", text="Model")
        self.model_tree.heading("size", text="Size (MB)")
        self.model_tree.heading("updated", text="Updated")
        self.model_tree.column("name", width=460, anchor="w")
        self.model_tree.column("size", width=110, anchor="center")
        self.model_tree.column("updated", width=180, anchor="center")

        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.model_tree.yview)
        self.model_tree.configure(yscrollcommand=scroll.set)
        self.model_tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.model_tree.bind("<<TreeviewSelect>>", lambda _: self._update_model_selection())

        controls = ttk.Frame(frame)
        controls.pack(fill="x", pady=8)
        ttk.Button(controls, text="Refresh", command=self._populate_model_tree).pack(side="left")
        ttk.Button(controls, text="Quick select (first two)", command=self._quick_select_models).pack(
            side="left", padx=(6, 0)
        )

        self.model_summary = ttk.Label(frame, text="0 models selected")
        self.model_summary.pack(anchor="w", pady=(4, 0))
        return frame

    def _create_mode_step(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent)
        ttk.Label(frame, text="Step 2: Mode Selection", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text="Choose the merge mode that best fits your goal.",
        ).pack(anchor="w", pady=(2, 8))

        modes = [
            ("legacy", "Legacy", "Classic weighted mix with per-block controls and LoRA"),
            ("perres", "PerRes", "100% assignment per resolution block"),
            ("hybrid", "Hybrid", "Combination of assignment and per-block weights"),
        ]

        for value, title, desc in modes:
            card = ttk.Frame(frame, padding=8, relief="ridge")
            card.pack(fill="x", pady=6)
            ttk.Radiobutton(
                card,
                text=title,
                value=value,
                variable=self.mode_var,
            ).pack(anchor="w")
            ttk.Label(card, text=desc, wraplength=720).pack(anchor="w", padx=16, pady=(2, 0))

        return frame

    def _create_config_step(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent)
        ttk.Label(frame, text="Step 3: Mode Configuration", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.config_hint = ttk.Label(frame, text="Complete the parameters of the selected mode.")
        self.config_hint.pack(anchor="w", pady=(2, 8))

        self.config_container = ttk.Frame(frame)
        self.config_container.pack(fill="both", expand=True)

        self.legacy_panel = LegacyConfigPanel(self.config_container)
        self.perres_panel = PerResConfigPanel(self.config_container)
        self.hybrid_panel = HybridConfigPanel(self.config_container)
        for panel in [self.legacy_panel, self.perres_panel, self.hybrid_panel]:
            panel.pack_forget()

        return frame

    def _create_preview_step(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent)
        ttk.Label(frame, text="Step 4: Preview", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text="Review the block distribution before running the merge.",
        ).pack(anchor="w", pady=(2, 8))

        preview_top = ttk.Frame(frame)
        preview_top.pack(fill="both", expand=True)

        self.preview_canvas = tk.Canvas(preview_top, height=260, background="#f8f9fa", highlightthickness=1)
        self.preview_canvas.pack(fill="x", pady=(0, 8))

        table_frame = ttk.Frame(preview_top)
        table_frame.pack(fill="both", expand=True)
        self.preview_table = ttk.Treeview(
            table_frame,
            columns=("block", "detalle"),
            show="headings",
            height=8,
        )
        self.preview_table.heading("block", text="Block")
        self.preview_table.heading("detalle", text="Configuration")
        self.preview_table.column("block", width=160, anchor="w")
        self.preview_table.column("detalle", width=600, anchor="w")
        self.preview_table.pack(fill="both", expand=True)

        return frame

    def _create_run_step(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent)
        ttk.Label(frame, text="Step 5: Run Merge", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text="Start the merge and monitor progress in real time.",
        ).pack(anchor="w", pady=(2, 8))

        self.run_summary = ttk.Label(frame, text="Ready to run")
        self.run_summary.pack(anchor="w", pady=(4, 8))

        self.progress = ttk.Progressbar(frame, mode="determinate")
        actions = ttk.Frame(frame)
        actions.pack(fill="x", pady=(0, 8))
        self.cancel_btn = ttk.Button(actions, text="Cancel", command=self._cancel_running, state="disabled")
        self.cancel_btn.pack(side="left")

        self.log_text = tk.Text(frame, height=14, state="disabled", wrap="word")
        self.log_text.pack(fill="both", expand=True)

        return frame

    # ------------------------------------------------------------------
    # Resource loading
    # ------------------------------------------------------------------
    def _load_resources(self) -> None:
        self.model_paths = list_safetensors(self.models_dir)
        self.lora_paths = list_safetensors(self.loras_dir)
        self._populate_model_tree()

    def _populate_model_tree(self) -> None:
        selection = set(self.model_tree.selection()) if hasattr(self, "model_tree") else set()
        children = self.model_tree.get_children()
        if children:
            self.model_tree.delete(*children)

        for idx, path in enumerate(self.model_paths):
            size_mb = path.stat().st_size / (1024 * 1024)
            updated = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            self.model_tree.insert(
                "",
                "end",
                iid=str(idx),
                values=(path.name, f"{size_mb:.2f}", updated),
            )

        for iid in selection:
            if iid in self.model_tree.get_children():
                self.model_tree.selection_add(iid)
        self._update_model_selection()

    def _update_model_selection(self) -> None:
        selected = [int(iid) for iid in self.model_tree.selection()]
        if not selected:
            self.model_summary.config(text="0 models selected")
            return
        names = [self.model_paths[i].name for i in selected]
        preview = ", ".join(names[:3])
        if len(names) > 3:
            preview += " ..."
        self.model_summary.config(text=f"{len(names)} models: {preview}")

    def _quick_select_models(self) -> None:
        self.model_tree.selection_remove(self.model_tree.selection())
        for idx in range(min(2, len(self.model_paths))):
            self.model_tree.selection_add(str(idx))
        self._update_model_selection()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def _show_step(self, index: int) -> None:
        if index < 0 or index >= len(self.steps):
            return
        for frame in self.steps:
            frame.pack_forget()
        self.steps[index].pack(fill="both", expand=True, padx=4, pady=4)
        self.content_canvas.yview_moveto(0)
        self.current_step = index

        if index == 2:
            self._prepare_config_panel()
        elif index == 3:
            self._update_preview()
        elif index == 4:
            self._update_run_summary()

        self._update_nav_buttons()

    def _update_nav_buttons(self) -> None:
        if self.current_step > 0:
            self.prev_btn.state(["!disabled"])
        else:
            self.prev_btn.state(["disabled"])

        if self.current_step < len(self.steps) - 1:
            self.next_btn.state(["!disabled"])
            if self.run_btn.winfo_manager():
                self.run_btn.pack_forget()
        else:
            self.next_btn.state(["disabled"])
            if not self.run_btn.winfo_manager():
                self.run_btn.pack(side="left", padx=(8, 0))
            if self.is_running:
                self.run_btn.state(["disabled"])
            else:
                self.run_btn.state(["!disabled"])

    def _on_mousewheel(self, event: tk.Event) -> str:
        if event.delta:
            self.content_canvas.yview_scroll(int(-event.delta / 120), "units")
        elif getattr(event, 'num', None) in (4, 5):
            direction = -1 if event.num == 4 else 1
            self.content_canvas.yview_scroll(direction, "units")
        return "break"

    def _go_next(self) -> None:
        if self.current_step >= len(self.steps) - 1:
            return
        if not self._validate_step(self.current_step):
            return
        self._show_step(self.current_step + 1)

    def _go_previous(self) -> None:
        if self.current_step <= 0:
            return
        self._show_step(self.current_step - 1)

    # ------------------------------------------------------------------
    # Step validation and preparation
    # ------------------------------------------------------------------
    def _validate_step(self, index: int) -> bool:
        if index == 0:
            selected_ids = [int(iid) for iid in self.model_tree.selection()]
            if len(selected_ids) < 2:
                messagebox.showwarning("Insufficient selection", "Select at least two models.")
                return False
            selected_ids.sort()
            self.state["model_indices"] = selected_ids
            self.state["model_paths"] = [self.model_paths[i] for i in selected_ids]
            self.state["model_names"] = [self.model_paths[i].name for i in selected_ids]
            return True

        if index == 1:
            mode = self.mode_var.get()
            self.state["mode"] = mode
            return True

        if index == 2:
            mode: str = self.state.get("mode", "legacy")
            model_count = len(self.state.get("model_indices", []))
            if model_count < 2:
                messagebox.showwarning("Incomplete configuration", "Select models before configuring.")
                return False

            if mode == "legacy":
                config = self.legacy_panel.get_config()
                if len(config["weights"]) != model_count or sum(config["weights"]) <= 0:
                    messagebox.showwarning("Invalid weights", "Enter valid weights for all models.")
                    return False
                backbone = config["backbone_idx"]
                if backbone < 0 or backbone >= model_count:
                    messagebox.showwarning("Invalid backbone", "Select a valid backbone.")
                    return False
                self.state["legacy"] = config
                return True

            if mode == "perres":
                config = self.perres_panel.get_config()
                assignments = config.get("assignments", {})
                if set(assignments.keys()) != set(BLOCK_GROUPS):
                    messagebox.showwarning("Incomplete blocks", "Assign all resolution blocks.")
                    return False
                if any(idx < 0 or idx >= model_count for idx in assignments.values()):
                    messagebox.showwarning("Invalid assignment", "Model index out of range.")
                    return False
                self.state["perres"] = config
                return True

            if mode == "hybrid":
                config = self.hybrid_panel.get_config()
                hybrid_cfg = config.get("hybrid_config", {})
                if set(hybrid_cfg.keys()) != set(BLOCK_GROUPS):
                    messagebox.showwarning("Incomplete blocks", "Configure all blocks.")
                    return False
                self.state["hybrid"] = config
                return True

        return True

    def _prepare_config_panel(self) -> None:
        mode: str = self.state.get("mode", "legacy")
        model_names: List[str] = list(self.state.get("model_names", []))

        for panel in [self.legacy_panel, self.perres_panel, self.hybrid_panel]:
            panel.pack_forget()

        if mode == "legacy":
            self.config_hint.config(text="Define weights, backbone and advanced options.")
            existing = self.state.get("legacy")
            self.legacy_panel.refresh(model_names, self.lora_paths, existing)
            self.legacy_panel.pack(fill="both", expand=True)
        elif mode == "perres":
            self.config_hint.config(text="Assign each resolution block to a model.")
            existing = self.state.get("perres")
            self.perres_panel.refresh(model_names, self.lora_paths, existing)
            self.perres_panel.pack(fill="both", expand=True)
        else:
            self.config_hint.config(text="Configure custom per-block weights.")
            existing = self.state.get("hybrid")
            self.hybrid_panel.refresh(model_names, self.lora_paths, existing)
            self.hybrid_panel.pack(fill="both", expand=True)

    def _update_preview(self) -> None:
        self.preview_canvas.delete("all")
        for row in self.preview_table.get_children():
            self.preview_table.delete(row)

        model_names: List[str] = list(self.state.get("model_names", []))
        mode: str = self.state.get("mode", "legacy")

        if not model_names:
            return

        if mode == "legacy":
            config = self.state.get("legacy", {})
            weights = config.get("weights", [1.0, 0.0])
            preview_data = {block: {i: weights[i] for i in range(len(model_names))} for block in BLOCK_GROUPS}
        elif mode == "perres":
            config = self.state.get("perres", {})
            assignments = config.get("assignments", {})
            preview_data = {block: {assignments.get(block, 0): 1.0} for block in BLOCK_GROUPS}
        else:
            config = self.state.get("hybrid", {})
            preview_data = config.get("hybrid_config", {})

        width = max(self.preview_canvas.winfo_width(), 860)
        height = 220
        margin = 40
        bar_width = (width - 2 * margin) / max(len(BLOCK_GROUPS), 1)

        for idx, block in enumerate(BLOCK_GROUPS):
            x0 = margin + idx * bar_width
            x1 = x0 + bar_width * 0.8
            contributions = preview_data.get(block, {})
            total = sum(contributions.values()) or 1.0
            y = height

            summary_parts = []
            for model_idx, weight in contributions.items():
                ratio = weight / total
                color = MODEL_COLORS[model_idx % len(MODEL_COLORS)]
                y1 = y - ratio * (height - 40)
                self.preview_canvas.create_rectangle(x0, y1, x1, y, fill=color, outline="black")
                label = f"{model_names[model_idx]} ({weight:.2f})"
                summary_parts.append(label)
                y = y1

            self.preview_canvas.create_text(
                (x0 + x1) / 2,
                height + 10,
                text=block,
                anchor="n",
            )

            summary = ", ".join(summary_parts) if summary_parts else "No data"
            self.preview_table.insert("", "end", values=(block, summary))

    def _update_run_summary(self) -> None:
        model_names = ", ".join(self.state.get("model_names", []))
        mode = self.state.get("mode", "legacy")
        self.run_summary.config(text=f"Mode: {mode} | Models: {model_names}")

    # ------------------------------------------------------------------
    # Merge execution
    # ------------------------------------------------------------------
    def _start_merge(self) -> None:
        if self.is_running:
            return
        mode = self.state.get("mode")
        if not mode:
            messagebox.showwarning("Incomplete configuration", "Select a valid mode.")
            return

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

        if not self.progress.winfo_manager():
            self.progress.pack(fill="x", pady=(0, 8))
        # reset progress to 0 until recibimos total
        self.progress.configure(mode="determinate", value=0, maximum=100)

        self.is_running = True
        self.run_btn.state(["disabled"])
        self.cancel_btn.state(["!disabled"])
        self.cancel_event = threading.Event()

        self.worker = threading.Thread(target=self._merge_worker, daemon=True)
        self.worker.start()

    def _merge_worker(self) -> None:
        try:
            model_paths: List[Path] = list(self.state.get("model_paths", []))
            model_names: List[str] = list(self.state.get("model_names", []))
            mode: str = self.state.get("mode", "legacy")

            if len(model_paths) < 2:
                raise RuntimeError("At least two models are required to merge.")

            self.log_queue.put(("info", f"Selected models: {', '.join(model_names)}"))

            def on_progress(kind: str, value: int) -> None:
                if kind == "total":
                    self.log_queue.put(("progress_total", value))
                elif kind == "tick":
                    self.log_queue.put(("progress_tick", value))

            if mode == "legacy":
                config = self.state.get("legacy", {})
                weights = config.get("weights", [])
                backbone_idx = int(config.get("backbone_idx", 0))
                block_multipliers = config.get("block_multipliers")
                cross_boosts = config.get("crossattn_boosts")
                merged, _stats = stream_weighted_merge_from_paths(
                    model_paths,
                    weights,
                    backbone_idx,
                    only_unet=True,
                    block_multipliers=block_multipliers,
                    crossattn_boosts=cross_boosts,
                    progress_cb=on_progress,
                    cancel_event=self.cancel_event,
                )
                for lora_path, scale in config.get("loras", []):
                    applied, skipped = apply_single_lora(merged, lora_path, scale)
                    self.log_queue.put(("info", f"LoRA {lora_path.name}: applied {applied}, skipped {skipped}"))

                yaml_kwargs: Dict[str, object] = {"weights": weights}
                if block_multipliers:
                    yaml_kwargs["block_multipliers"] = block_multipliers
                if cross_boosts:
                    yaml_kwargs["crossattn_boosts"] = cross_boosts
                lora_yaml = []
                for lora_path, scale in config.get("loras", []):
                    lora_yaml.append({"file": lora_path.name, "scale": scale})
                if lora_yaml:
                    yaml_kwargs["loras"] = lora_yaml

            elif mode == "perres":
                config = self.state.get("perres", {})
                assignments = config.get("assignments", {})
                attn2 = config.get("attn2_locks")
                backbone_idx = list(assignments.values())[0]
                merged = merge_perres(
                    model_paths,
                    assignments,
                    backbone_idx,
                    attn2,
                    progress_cb=on_progress,
                    cancel_event=self.cancel_event,
                )

                # Aplicar LoRAs si hay
                for lora_path, scale in config.get("loras", []):
                    applied, skipped = apply_single_lora(merged, lora_path, scale)
                    self.log_queue.put(("info", f"LoRA {lora_path.name}: applied {applied}, skipped {skipped}"))

                yaml_kwargs = {"assignments": assignments}
                if attn2:
                    yaml_kwargs["attn2_locks"] = attn2
                lora_yaml = []
                for lora_path, scale in config.get("loras", []):
                    lora_yaml.append({"file": lora_path.name, "scale": scale})
                if lora_yaml:
                    yaml_kwargs["loras"] = lora_yaml

            else:
                config = self.state.get("hybrid", {})
                hybrid_cfg = config.get("hybrid_config", {})
                attn2 = config.get("attn2_locks")
                backbone_idx = 0
                merged = merge_hybrid(
                    model_paths,
                    hybrid_cfg,
                    backbone_idx,
                    attn2,
                    progress_cb=on_progress,
                    cancel_event=self.cancel_event,
                )

                # Aplicar LoRAs si hay
                for lora_path, scale in config.get("loras", []):
                    applied, skipped = apply_single_lora(merged, lora_path, scale)
                    self.log_queue.put(("info", f"LoRA {lora_path.name}: applied {applied}, skipped {skipped}"))

                yaml_kwargs = {"hybrid_config": hybrid_cfg}
                if attn2:
                    yaml_kwargs["attn2_locks"] = attn2
                lora_yaml = []
                for lora_path, scale in config.get("loras", []):
                    lora_yaml.append({"file": lora_path.name, "scale": scale})
                if lora_yaml:
                    yaml_kwargs["loras"] = lora_yaml

            # Recolectar rutas de LoRA (si hay) para hashing en metadata
            lora_paths = [p for p, _s in config.get("loras", [])] if config.get("loras") else None
            output_path, metadata_folder, version = save_merge_results(
                self.output_dir,
                self.metadata_dir,
                merged,
                model_names,
                mode,
                backbone_idx,
                yaml_kwargs,
                model_paths=model_paths,
                lora_paths=lora_paths,
            )

            self.log_queue.put(("success", f"Merge completed: {output_path.name} (V{version})"))
            self.log_queue.put(("info", f"Metadata saved to {metadata_folder.name}"))
        except Exception as exc:  # pragma: no cover
            self.log_queue.put(("error", str(exc)))
        finally:
            self.log_queue.put(("done", ""))

    def _poll_queue(self) -> None:
        try:
            while True:
                level, message = self.log_queue.get_nowait()
                if level == "info":
                    self._append_log(message)
                elif level == "success":
                    self._append_log(message)
                    messagebox.showinfo("Merge completed", message)
                elif level == "error":
                    self._append_log(f"ERROR: {message}")
                    messagebox.showerror("Merge error", message)
                elif level == "progress_total":
                    try:
                        total = int(message)
                    except Exception:
                        total = 0
                    if total > 0:
                        self.progress.configure(mode="determinate", value=0, maximum=total)
                elif level == "progress_tick":
                    try:
                        inc = int(message)
                    except Exception:
                        inc = 1
                    self.progress.step(inc)
                elif level == "done":
                    self.is_running = False
                    # no indeterminate; simplemente asegurar que boton cancelar se desactiva
                    self.cancel_btn.state(["disabled"])
                    self.run_btn.state(["!disabled"])
                    self._update_nav_buttons()
        except Empty:
            pass
        finally:
            self.root.after(200, self._poll_queue)

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # ------------------------------------------------------------------
    def run(self) -> None:
        self.root.mainloop()

    def _cancel_running(self) -> None:
        if getattr(self, 'cancel_event', None) is not None:
            self.cancel_event.set()
            self._append_log("Cancellation requested by user...")


def launch_gui(root_dir: Path) -> None:
    models_dir, loras_dir, output_dir, metadata_dir = ensure_dirs(root_dir)
    app = FusionGUI(root_dir, models_dir, loras_dir, output_dir, metadata_dir)
    app.run()


if __name__ == "__main__":
    launch_gui(Path(__file__).resolve().parent)
