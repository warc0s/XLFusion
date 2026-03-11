import unittest

from xlfusion.blocks import (
    classify_component_key,
    classify_submodule_key,
    get_attn2_block_type,
    get_block_assignment,
    get_block_mapping,
    group_for_key,
    is_cross_attn_key,
)


class BlocksContractTests(unittest.TestCase):
    def test_unet_block_assignment_and_attn_detection(self) -> None:
        key = "model.diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.weight"
        self.assertEqual(get_block_assignment(key), "down_0_1")
        self.assertEqual(group_for_key(key), "down")
        self.assertTrue(is_cross_attn_key(key))
        self.assertEqual(get_attn2_block_type(key), "down")

        key_mid = "model.diffusion_model.mid_block.attentions.0.transformer_blocks.0.attn2.to_v.weight"
        self.assertEqual(get_block_assignment(key_mid), "mid")
        self.assertEqual(group_for_key(key_mid), "mid")
        self.assertTrue(is_cross_attn_key(key_mid))
        self.assertEqual(get_attn2_block_type(key_mid), "mid")

        key_other = "model.diffusion_model.time_embed.0.weight"
        self.assertEqual(get_block_assignment(key_other), "other")
        self.assertEqual(group_for_key(key_other), "other")
        self.assertFalse(is_cross_attn_key(key_other))
        self.assertIsNone(get_attn2_block_type(key_other))

    def test_component_and_submodule_classification(self) -> None:
        unet_key = "model.diffusion_model.up_blocks.0.attentions.0.transformer_blocks.0.attn2.to_out.0.weight"
        self.assertEqual(classify_component_key(unet_key), "unet")
        self.assertEqual(classify_submodule_key(unet_key), "cross_attention")

        vae_key = "first_stage_model.decoder.conv_in.weight"
        self.assertEqual(classify_component_key(vae_key), "vae")

        text_encoder_key = "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"
        self.assertEqual(classify_component_key(text_encoder_key), "text_encoder")

        other_key = "unrelated.key"
        self.assertEqual(classify_component_key(other_key), "other")
        self.assertEqual(classify_submodule_key(other_key), "other")

    def test_unknown_block_mapping_raises(self) -> None:
        with self.assertRaises(KeyError):
            get_block_mapping("does_not_exist")


if __name__ == "__main__":
    unittest.main()

