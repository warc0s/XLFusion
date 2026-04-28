from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import ttk

from .blocks import SDXL_ATTN_BLOCKS, SDXL_BLOCK_GROUPS

BLOCK_GROUPS = list(SDXL_BLOCK_GROUPS)
ATTN_BLOCKS = list(SDXL_ATTN_BLOCKS)
LEGACY_BLOCKS = list(SDXL_ATTN_BLOCKS)
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

        for r, block in enumerate(LEGACY_BLOCKS, start=1):
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
        ).grid(row=len(LEGACY_BLOCKS) + 1, column=0, padx=4, pady=(4, 2), sticky="w")

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
                for block in LEGACY_BLOCKS:
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
