#!/usr/bin/env python3
"""Utility to build the GUI into a Windows executable using PyInstaller."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import PyInstaller.__main__  # type: ignore
except ImportError as exc:  # pragma: no cover
    print("PyInstaller is not installed. Run 'pip install pyinstaller' and try again.")
    sys.exit(1)


def _format_add_data(src: Path, dest: str) -> str:
    """Format the --add-data option respecting platform differences."""
    separator = ";" if sys.platform.startswith("win") else ":"
    return f"{src}{separator}{dest}"


def build(name: str, onefile: bool) -> None:
    root = Path(__file__).resolve().parents[1]
    gui_entry = root / "gui_app.py"
    if not gui_entry.exists():
        raise FileNotFoundError("gui_app.py not found at the project root")

    assets = [
        _format_add_data(root / "config.yaml", "."),
        _format_add_data(root / "batch_config_example.yaml", "."),
    ]

    args = [
        "--noconfirm",
        "--clean",
        "--name",
        name,
        "--windowed",
    ]

    if onefile:
        args.append("--onefile")

    for entry in assets:
        args.extend(["--add-data", str(entry)])

    args.append(str(gui_entry))

    PyInstaller.__main__.run(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the GUI with PyInstaller")
    parser.add_argument("--name", default="XLFusionGUI", help="Name of the resulting executable")
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="Generate a single-file executable (--onefile)",
    )
    opts = parser.parse_args()
    build(opts.name, opts.onefile)


if __name__ == "__main__":
    main()
