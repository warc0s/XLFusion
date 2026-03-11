import unittest

from xlfusion.blocks import get_block_mapping


class BlockMappingRegistryTests(unittest.TestCase):
    def test_sdxl_mapping_is_available(self) -> None:
        mapping = get_block_mapping("sdxl")
        self.assertEqual(mapping.name, "sdxl")
        self.assertIn("down_0_1", mapping.block_groups)
        self.assertIn("mid", mapping.block_groups)


if __name__ == "__main__":
    unittest.main()

