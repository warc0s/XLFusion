import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file as st_save

from Utils.merge import stream_weighted_merge_from_paths, merge_perres, merge_hybrid, MergeCancelled


def _make_small_state(val: float) -> dict:
    W = torch.full((2, 2), val, dtype=torch.float32)
    state = {
        "model.diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.weight": W.clone(),
        "model.diffusion_model.mid_block.attentions.0.transformer_blocks.0.attn2.to_v.weight": W.clone(),
        "model.diffusion_model.up_blocks.0.resnets.0.conv1.weight": W.clone(),
    }
    return state


class ProgressCancelTests(unittest.TestCase):
    def test_progress_ticks_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            m1 = tmp / "m1.safetensors"
            m2 = tmp / "m2.safetensors"
            st_save(_make_small_state(1.0), str(m1))
            st_save(_make_small_state(0.0), str(m2))

            seen = {"total": 0, "ticks": 0}

            def cb(kind: str, value: int) -> None:
                if kind == "total":
                    seen["total"] = int(value)
                elif kind == "tick":
                    seen["ticks"] += int(value)

            merged, _ = stream_weighted_merge_from_paths(
                [m1, m2], [0.5, 0.5], base_idx=0, progress_cb=cb
            )
            self.assertGreater(len(merged), 0)
            self.assertGreater(seen["total"], 0)
            # Al menos tantas ticks como claves del backbone (pueden añadirse más por otras claves)
            self.assertGreaterEqual(seen["ticks"], seen["total"])

    def test_cancel_hybrid(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            m1 = tmp / "m1.safetensors"
            m2 = tmp / "m2.safetensors"
            st_save(_make_small_state(1.0), str(m1))
            st_save(_make_small_state(2.0), str(m2))

            import threading
            ev = threading.Event()
            ev.set()  # cancelar inmediatamente

            cfg = {
                "down_0_1": {0: 1.0},
                "down_2_3": {0: 1.0},
                "mid": {0: 0.5, 1: 0.5},
                "up_0_1": {0: 1.0},
                "up_2_3": {0: 1.0},
            }
            with self.assertRaises(MergeCancelled):
                merge_hybrid([m1, m2], cfg, backbone_idx=0, cancel_event=ev)
