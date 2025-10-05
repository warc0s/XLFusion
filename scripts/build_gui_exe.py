#!/usr/bin/env python3
"""Utilidad para compilar la GUI en un ejecutable de Windows mediante PyInstaller."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import PyInstaller.__main__  # type: ignore
except ImportError as exc:  # pragma: no cover
    print("PyInstaller no esta instalado. Ejecuta 'pip install pyinstaller' y vuelve a intentarlo.")
    sys.exit(1)


def _format_add_data(src: Path, dest: str) -> str:
    """Formatea la opcion --add-data respetando diferencias de plataforma."""
    separator = ";" if sys.platform.startswith("win") else ":"
    return f"{src}{separator}{dest}"


def build(name: str, onefile: bool) -> None:
    root = Path(__file__).resolve().parents[1]
    gui_entry = root / "gui_app.py"
    if not gui_entry.exists():
        raise FileNotFoundError("No se encontro gui_app.py en la raiz del proyecto")

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
    parser = argparse.ArgumentParser(description="Compila la GUI con PyInstaller")
    parser.add_argument("--name", default="XLFusionGUI", help="Nombre del ejecutable resultante")
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="Genera un ejecutable unico (--onefile)",
    )
    opts = parser.parse_args()
    build(opts.name, opts.onefile)


if __name__ == "__main__":
    main()
