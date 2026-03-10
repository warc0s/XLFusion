#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from xlfusion.gui_app import launch_gui


if __name__ == "__main__":
    launch_gui(Path(__file__).resolve().parent)
