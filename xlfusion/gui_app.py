from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import filedialog, ttk, messagebox

from .gui_panels import (
    BLOCK_GROUPS,
    MODEL_COLORS,
    HybridConfigPanel,
    LegacyConfigPanel,
    PerResConfigPanel,
)
from .config import AppContext, list_safetensors
from .execution import execution_options_to_dict
from .runtime import execute_merge_job
from .types import MergeJobConfig
from .presets import (
    batch_job_to_runtime_state,
    inspect_recovery_source,
    load_single_job_preset,
    save_single_job_preset,
)
from .validation import export_preflight_plan, format_preflight_plan, validate_merge_request


class FusionGUI:
    """Main graphical interface of XLFusion."""

    def __init__(
        self,
        context: AppContext,
    ) -> None:
        self.context = context
        self.root_dir = context.root_dir
        self.models_dir = context.models_dir
        self.loras_dir = context.loras_dir
        self.output_dir = context.output_dir
        self.metadata_dir = context.metadata_dir
        self.presets_dir = context.presets_dir

        self.state: Dict[str, object] = {
            "model_indices": [],
            "model_paths": [],
            "model_names": [],
            "mode": "legacy",
            "legacy": {},
            "perres": {},
            "hybrid": {},
            "output_name": self.context.config["model_output"].get("base_name", "XLFusion"),
            "execution": execution_options_to_dict(None),
        }

        self.model_paths: List[Path] = []
        self.lora_paths: List[Path] = []

        self.root = tk.Tk()
        tool_name = self.context.config["app"].get("tool_name", "XLFusion")
        self.root.title(f"{tool_name} - Graphical Interface")
        self.root.geometry("980x720")
        self.root.minsize(840, 600)

        self.log_queue: Queue = Queue()
        self.worker: Optional[threading.Thread] = None
        self.is_running = False

        self.mode_var = tk.StringVar(value="legacy")
        self.output_name_var = tk.StringVar(value=str(self.state["output_name"]))
        self.execution_mode_var = tk.StringVar(value="low-memory")
        self.progress_mode_var = tk.StringVar(value="auto")
        self.progress_every_var = tk.StringVar(value="250")
        self.include_vae_var = tk.BooleanVar(value=False)
        self.include_text_encoder_var = tk.BooleanVar(value=False)
        self.include_other_var = tk.BooleanVar(value=False)

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
            text=f"Select at least two checkpoints from {self.models_dir}.",
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
        header = ttk.Frame(frame)
        header.pack(fill="x")
        ttk.Label(header, text="Step 3: Mode Configuration", font=("Segoe UI", 12, "bold")).pack(anchor="w", side="left")
        actions = ttk.Frame(header)
        actions.pack(side="right")
        ttk.Button(actions, text="Load Preset", command=self._load_preset_dialog).pack(side="left", padx=(0, 4))
        ttk.Button(actions, text="Load Metadata", command=self._load_metadata_dialog).pack(side="left", padx=(0, 4))
        ttk.Button(actions, text="Save Preset", command=self._save_preset_dialog).pack(side="left")
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

        preflight_actions = ttk.Frame(frame)
        preflight_actions.pack(fill="x", pady=(8, 4))
        ttk.Label(preflight_actions, text="Fusion plan").pack(side="left")
        ttk.Button(preflight_actions, text="Export TXT", command=lambda: self._export_preflight("txt")).pack(
            side="right", padx=(4, 0)
        )
        ttk.Button(preflight_actions, text="Export JSON", command=lambda: self._export_preflight("json")).pack(
            side="right"
        )

        self.preflight_text = tk.Text(frame, height=12, state="disabled", wrap="word")
        self.preflight_text.pack(fill="both", expand=True)

        return frame

    def _create_run_step(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent)
        ttk.Label(frame, text="Step 5: Run Merge", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text="Start the merge and monitor progress in real time.",
        ).pack(anchor="w", pady=(2, 8))

        execution_frame = ttk.LabelFrame(frame, text="Execution")
        execution_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(execution_frame, text="Output base name").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(execution_frame, textvariable=self.output_name_var, width=32).grid(
            row=0, column=1, padx=4, pady=4, sticky="w"
        )
        ttk.Label(execution_frame, text="Execution mode").grid(row=1, column=0, padx=4, pady=4, sticky="w")
        ttk.Combobox(
            execution_frame,
            textvariable=self.execution_mode_var,
            values=["low-memory", "standard"],
            state="readonly",
            width=16,
        ).grid(row=1, column=1, padx=4, pady=4, sticky="w")
        ttk.Label(execution_frame, text="Progress output").grid(row=1, column=2, padx=4, pady=4, sticky="w")
        ttk.Combobox(
            execution_frame,
            textvariable=self.progress_mode_var,
            values=["auto", "simple", "quiet"],
            state="readonly",
            width=12,
        ).grid(row=1, column=3, padx=4, pady=4, sticky="w")
        ttk.Label(execution_frame, text="Simple interval").grid(row=1, column=4, padx=4, pady=4, sticky="w")
        ttk.Entry(execution_frame, textvariable=self.progress_every_var, width=8).grid(
            row=1, column=5, padx=4, pady=4, sticky="w"
        )
        ttk.Label(execution_frame, text="Non-UNet scope").grid(row=2, column=0, padx=4, pady=4, sticky="nw")
        scope_box = ttk.Frame(execution_frame)
        scope_box.grid(row=2, column=1, columnspan=5, padx=4, pady=4, sticky="w")
        ttk.Checkbutton(scope_box, text="Include VAE", variable=self.include_vae_var).pack(side="left")
        ttk.Checkbutton(scope_box, text="Include text encoder", variable=self.include_text_encoder_var).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(scope_box, text="Include other tensors", variable=self.include_other_var).pack(side="left", padx=(8, 0))

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
    def _build_validation(self):
        mode: str = self.state.get("mode", "legacy")
        model_paths: List[Path] = list(self.state.get("model_paths", []))
        only_unet, component_policy = self._get_component_scope_config(mode)
        block_mapping = str(self.state.get("block_mapping", "sdxl"))

        if mode == "legacy":
            config = self.legacy_panel.get_config()
            self.state["legacy"] = config
            return validate_merge_request(
                mode=mode,
                model_paths=model_paths,
                backbone=config.get("backbone_idx", 0),
                weights=config.get("weights"),
                block_multipliers=config.get("block_multipliers"),
                crossattn_boosts=config.get("crossattn_boosts"),
                loras=config.get("loras"),
                loras_dir=self.loras_dir,
                only_unet=only_unet,
                component_policy=component_policy,
                block_mapping=block_mapping,
            )

        if mode == "perres":
            config = self.perres_panel.get_config()
            self.state["perres"] = config
            assignments = config.get("assignments", {})
            backbone_idx = list(assignments.values())[0] if assignments else 0
            return validate_merge_request(
                mode=mode,
                model_paths=model_paths,
                backbone=backbone_idx,
                assignments=assignments,
                attn2_locks=config.get("attn2_locks"),
                loras=config.get("loras"),
                loras_dir=self.loras_dir,
                only_unet=only_unet,
                component_policy=component_policy,
                block_mapping=block_mapping,
            )

        config = self.hybrid_panel.get_config()
        self.state["hybrid"] = config
        return validate_merge_request(
            mode=mode,
            model_paths=model_paths,
            backbone=0,
            hybrid_config=config.get("hybrid_config"),
            attn2_locks=config.get("attn2_locks"),
            loras=config.get("loras"),
            loras_dir=self.loras_dir,
            only_unet=only_unet,
            component_policy=component_policy,
            block_mapping=block_mapping,
        )

    def _set_component_scope_for_mode(self, mode: str, config: Optional[Dict[str, object]] = None) -> None:
        config = config or {}
        if config.get("component_policy"):
            policy = config.get("component_policy") or {}
            self.include_vae_var.set(policy.get("vae") != "exclude")
            self.include_text_encoder_var.set(policy.get("text_encoder") != "exclude")
            self.include_other_var.set(policy.get("other") != "exclude")
            return
        only_unet = bool(config.get("only_unet")) if "only_unet" in config else (mode == "legacy")
        if only_unet:
            self.include_vae_var.set(False)
            self.include_text_encoder_var.set(False)
            self.include_other_var.set(False)
            return
        default_include = mode != "legacy"
        self.include_vae_var.set(default_include)
        self.include_text_encoder_var.set(default_include)
        self.include_other_var.set(default_include)

    def _get_component_scope_config(self, mode: str) -> tuple[bool, Dict[str, str]]:
        include_action = "merge" if mode == "legacy" else "backbone"
        component_policy = {
            "vae": include_action if self.include_vae_var.get() else "exclude",
            "text_encoder": include_action if self.include_text_encoder_var.get() else "exclude",
            "other": include_action if self.include_other_var.get() else "exclude",
        }
        only_unet = all(action == "exclude" for action in component_policy.values())
        return only_unet, component_policy

    def _get_execution_config(self) -> Dict[str, object]:
        try:
            log_every = max(1, int(self.progress_every_var.get().strip() or "250"))
        except ValueError:
            log_every = 250
        execution = {
            "mode": self.execution_mode_var.get() or "low-memory",
            "progress": self.progress_mode_var.get() or "auto",
            "log_every": log_every,
        }
        self.state["execution"] = execution
        self.state["output_name"] = self.output_name_var.get().strip() or self.context.config["model_output"].get("base_name", "XLFusion")
        return execution

    def _apply_runtime_state(self, runtime: Dict[str, object]) -> None:
        model_names = list(runtime.get("models", []))
        name_to_idx = {path.name: idx for idx, path in enumerate(self.model_paths)}
        missing = [name for name in model_names if name not in name_to_idx]
        if missing:
            raise ValueError(f"Missing models in workspace: {', '.join(missing)}")

        selected_ids = [name_to_idx[name] for name in model_names]
        self.model_tree.selection_remove(self.model_tree.selection())
        for idx in selected_ids:
            self.model_tree.selection_add(str(idx))
        self._update_model_selection()

        self.state["model_indices"] = selected_ids
        self.state["model_paths"] = [self.model_paths[i] for i in selected_ids]
        self.state["model_names"] = model_names
        self.state["block_mapping"] = str(runtime.get("block_mapping", "sdxl"))

        mode = str(runtime.get("mode", "legacy"))
        self.mode_var.set(mode)
        self.state["mode"] = mode
        self.state[mode] = runtime.get("config", {})
        self._set_component_scope_for_mode(mode, self.state[mode] if isinstance(self.state[mode], dict) else None)

        execution = runtime.get("execution", {}) or {}
        execution = execution_options_to_dict(execution)
        self.execution_mode_var.set(str(execution.get("mode", "low-memory")))
        self.progress_mode_var.set(str(execution.get("progress", "auto")))
        self.progress_every_var.set(str(execution.get("log_every", 250)))
        self.output_name_var.set(str(runtime.get("output_name") or self.context.config["model_output"].get("base_name", "XLFusion")))
        self.state["execution"] = execution
        self.state["output_name"] = self.output_name_var.get()

        self._prepare_config_panel()
        validation = self._build_validation()
        self.state["validation"] = validation
        self.state["preflight"] = validation.preflight
        self._update_preview()
        self._update_run_summary()

    def _load_preset_dialog(self) -> None:
        path = filedialog.askopenfilename(
            title="Load preset",
            initialdir=str(self.presets_dir),
            filetypes=[("YAML files", "*.yaml *.yml")],
        )
        if not path:
            return
        try:
            job = load_single_job_preset(Path(path))
            self._apply_runtime_state(batch_job_to_runtime_state(job))
        except Exception as exc:
            messagebox.showerror("Preset error", str(exc))

    def _load_metadata_dialog(self) -> None:
        path = filedialog.askdirectory(
            title="Load metadata folder",
            initialdir=str(self.metadata_dir),
            mustexist=True,
        )
        if not path:
            return
        try:
            inspection = inspect_recovery_source(Path(path), self.context)
            self._apply_runtime_state(batch_job_to_runtime_state(inspection.job))
            if inspection.warnings:
                messagebox.showwarning("Metadata warnings", "\n".join(inspection.warnings))
        except Exception as exc:
            messagebox.showerror("Metadata error", str(exc))

    def _save_preset_dialog(self) -> None:
        validation = self._build_validation()
        if not validation.valid:
            details = "\n".join(f"- {item.field}: {item.message}" for item in validation.errors)
            messagebox.showwarning("Invalid configuration", details)
            return
        path = filedialog.asksaveasfilename(
            title="Save preset",
            initialdir=str(self.presets_dir),
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml")],
        )
        if not path:
            return

        execution = self._get_execution_config()
        only_unet, component_policy = self._get_component_scope_config(str(self.state.get("mode", "legacy")))
        loras = [
            {"file": item["file"], "scale": item["scale"]}
            for item in validation.normalized.get("loras", [])
        ] or None
        try:
            save_single_job_preset(
                Path(path),
                mode=str(self.state.get("mode", "legacy")),
                model_names=list(validation.normalized.get("model_names", [])),
                backbone_idx=int(validation.normalized.get("backbone_idx", 0)),
                output_name=self.output_name_var.get().strip() or None,
                block_mapping=str(validation.normalized.get("block_mapping", "sdxl")),
                execution=execution,
                job_name=f"GUI_{self.state.get('mode', 'legacy')}",
                description="Saved from XLFusion GUI",
                weights=validation.normalized.get("weights"),
                assignments=validation.normalized.get("assignments"),
                hybrid_config=validation.normalized.get("hybrid_config"),
                attn2_locks=validation.normalized.get("attn2_locks"),
                block_multipliers=validation.normalized.get("block_multipliers"),
                crossattn_boosts=validation.normalized.get("crossattn_boosts"),
                loras=loras,
                only_unet=only_unet,
                component_policy=component_policy,
            )
        except Exception as exc:
            messagebox.showerror("Preset error", str(exc))
            return
        messagebox.showinfo("Preset saved", f"Preset saved to:\n{path}")

    def _set_preflight_text(self, content: str) -> None:
        self.preflight_text.configure(state="normal")
        self.preflight_text.delete("1.0", "end")
        self.preflight_text.insert("1.0", content)
        self.preflight_text.configure(state="disabled")

    def _export_preflight(self, fmt: str) -> None:
        plan = self.state.get("preflight")
        if not plan:
            messagebox.showwarning("No preflight", "Validate the configuration first to generate a fusion plan.")
            return

        filetypes = [("Text files", "*.txt")] if fmt == "txt" else [("JSON files", "*.json")]
        suffix = ".txt" if fmt == "txt" else ".json"
        path = filedialog.asksaveasfilename(
            title="Export fusion plan",
            defaultextension=suffix,
            filetypes=filetypes,
            initialdir=str(self.output_dir),
        )
        if not path:
            return

        export_preflight_plan(plan, Path(path))
        messagebox.showinfo("Exported", f"Fusion plan exported to:\n{path}")

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
            if len(self.state.get("model_indices", [])) < 2:
                messagebox.showwarning("Incomplete configuration", "Select models before configuring.")
                return False
            validation = self._build_validation()
            if not validation.valid:
                details = "\n".join(f"- {item.field}: {item.message}" for item in validation.errors)
                messagebox.showwarning("Invalid configuration", details)
                return False
            self.state["validation"] = validation
            self.state["preflight"] = validation.preflight
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
            self._set_component_scope_for_mode(mode, existing if isinstance(existing, dict) else None)
        elif mode == "perres":
            self.config_hint.config(text="Assign each resolution block to a model.")
            existing = self.state.get("perres")
            self.perres_panel.refresh(model_names, self.lora_paths, existing)
            self.perres_panel.pack(fill="both", expand=True)
            self._set_component_scope_for_mode(mode, existing if isinstance(existing, dict) else None)
        else:
            self.config_hint.config(text="Configure custom per-block weights.")
            existing = self.state.get("hybrid")
            self.hybrid_panel.refresh(model_names, self.lora_paths, existing)
            self.hybrid_panel.pack(fill="both", expand=True)
            self._set_component_scope_for_mode(mode, existing if isinstance(existing, dict) else None)

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

        validation = self.state.get("validation")
        if validation and validation.preflight:
            self.state["preflight"] = validation.preflight
            self._set_preflight_text(format_preflight_plan(validation.preflight))
        else:
            self._set_preflight_text("Complete the configuration to generate the fusion plan.")

    def _update_run_summary(self) -> None:
        plan = self.state.get("preflight")
        if plan:
            execution = self._get_execution_config()
            self.run_summary.config(
                text=(
                    f"Mode: {plan.mode} | Backbone: {plan.backbone_name} | "
                    f"Models used: {', '.join(plan.selected_models)} | "
                    f"Estimated memory: {plan.estimated_memory_gb:.2f} GB | "
                    f"Scope: {'UNet only' if plan.only_unet else 'UNet + selected components'} | "
                    f"Execution: {execution['mode']} | Output: {self.output_name_var.get().strip() or 'default'}"
                )
            )
            return
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
        validation = self._build_validation()
        if not validation.valid:
            details = "\n".join(f"- {item.field}: {item.message}" for item in validation.errors)
            messagebox.showwarning("Invalid configuration", details)
            return
        self.state["validation"] = validation
        self.state["preflight"] = validation.preflight
        execution = self._get_execution_config()
        self._update_run_summary()
        self._set_preflight_text(format_preflight_plan(validation.preflight))

        proceed = messagebox.askyesno(
            "Confirm merge",
            (
                f"Mode: {self.state.get('mode')}\n"
                f"Output: {self.output_name_var.get().strip() or 'default'}\n"
                f"Execution: {execution['mode']} / {execution['progress']}\n\n"
                "Do you want to start the merge?"
            ),
        )
        if not proceed:
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
            validation = self.state.get("validation")
            if validation is None or not validation.valid:
                raise RuntimeError("Validated configuration is not available.")

            model_paths: List[Path] = list(validation.normalized.get("model_paths", []))
            model_names: List[str] = list(validation.normalized.get("model_names", []))
            mode: str = self.state.get("mode", "legacy")
            execution = self._get_execution_config()
            output_name = self.output_name_var.get().strip() or None
            only_unet = bool(validation.normalized.get("only_unet"))
            component_policy = validation.normalized.get("component_policy")

            if len(model_paths) < 2:
                raise RuntimeError("At least two models are required to merge.")

            self.log_queue.put(("info", f"Selected models: {', '.join(model_names)}"))

            def on_progress(kind: str, value: int) -> None:
                if kind == "total":
                    self.log_queue.put(("progress_total", value))
                elif kind == "tick":
                    self.log_queue.put(("progress_tick", value))

            normalized = validation.normalized
            merge_job = MergeJobConfig(
                mode=mode,
                model_paths=model_paths,
                model_names=model_names,
                backbone_idx=int(normalized.get("backbone_idx", 0)),
                block_mapping=str(normalized.get("block_mapping", "sdxl")),
                output_base_name=output_name,
                weights=normalized.get("weights"),
                assignments=normalized.get("assignments"),
                hybrid_config=normalized.get("hybrid_config"),
                attn2_locks=normalized.get("attn2_locks"),
                block_multipliers=normalized.get("block_multipliers"),
                crossattn_boosts=normalized.get("crossattn_boosts"),
                loras=normalized.get("loras"),
                only_unet=only_unet,
                component_policy=component_policy,
                execution=execution,
                job_name=f"GUI_{mode}",
                job_description="Interactive GUI run",
            )

            result = execute_merge_job(
                self.output_dir,
                self.metadata_dir,
                merge_job,
                progress_cb=on_progress,
                cancel_event=self.cancel_event,
            )

            for report in result.lora_reports:
                self.log_queue.put(
                    (
                        "info",
                        f"LoRA {report.get('lora_file')}: applied {report.get('applied_pairs')}, skipped {report.get('skipped_pairs')}",
                    )
                )

            self.log_queue.put(("success", f"Merge completed: {result.output_path.name} (V{result.version})"))
            self.log_queue.put(("info", f"Metadata saved to {result.metadata_folder.name}"))
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
    from .config import resolve_app_context

    app = FusionGUI(resolve_app_context(root_dir))
    app.run()


def main() -> None:
    """Console-script entry point for the graphical interface."""
    launch_gui(Path(__file__).resolve().parent.parent)


if __name__ == "__main__":
    main()
