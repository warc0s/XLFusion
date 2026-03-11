import unittest

from xlfusion.execution import build_processing_order


class ExecutionOrderingTests(unittest.TestCase):
    def test_processing_order_deduplicates_without_sort(self) -> None:
        base = ["k1", "k2", "k1"]
        extras = [["k2", "k3"], ["k4", "k3"]]
        ordered = build_processing_order(base, extras, sort_keys=False)
        self.assertEqual(ordered, ["k1", "k2", "k3", "k4"])

    def test_processing_order_sorts_by_coarse_group(self) -> None:
        keys = [
            "model.diffusion_model.up_blocks.0.resnets.0.conv1.weight",
            "first_stage_model.decoder.conv_in.weight",
            "model.diffusion_model.mid_block.attentions.0.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.time_embed.0.weight",
        ]

        ordered = build_processing_order(keys, [], sort_keys=True, block_mapping="sdxl")
        idx = {key: ordered.index(key) for key in keys}

        self.assertLess(idx[keys[3]], idx[keys[2]])  # down before mid
        self.assertLess(idx[keys[2]], idx[keys[0]])  # mid before up
        self.assertLess(idx[keys[0]], idx[keys[4]])  # up before UNet other
        self.assertLess(idx[keys[0]], idx[keys[1]])  # up before non-UNet other


if __name__ == "__main__":
    unittest.main()
