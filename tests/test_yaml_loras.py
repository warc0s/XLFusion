import unittest
import yaml

from Utils.config import generate_batch_config_yaml


class YamlLorasTests(unittest.TestCase):
    def test_loras_in_legacy_yaml(self):
        loras = [{"file": "l1.safetensors", "scale": 0.5}]
        yml = generate_batch_config_yaml(
            mode="legacy",
            model_names=["a.safetensors", "b.safetensors"],
            backbone_idx=0,
            version=1,
            weights=[0.7, 0.3],
            loras=loras,
        )
        data = yaml.safe_load(yml)
        job = data["batch_jobs"][0]
        self.assertIn("loras", job)
        self.assertEqual(job["loras"], loras)

    def test_loras_in_perres_yaml(self):
        loras = [{"file": "l1.safetensors", "scale": 0.5}]
        yml = generate_batch_config_yaml(
            mode="perres",
            model_names=["a.safetensors", "b.safetensors"],
            backbone_idx=0,
            version=1,
            assignments={"down_0_1": 0, "down_2_3": 1, "mid": 0, "up_0_1": 1, "up_2_3": 0},
            loras=loras,
        )
        data = yaml.safe_load(yml)
        job = data["batch_jobs"][0]
        self.assertIn("loras", job)
        self.assertEqual(job["loras"], loras)

    def test_loras_in_hybrid_yaml(self):
        loras = [{"file": "l1.safetensors", "scale": 0.5}]
        yml = generate_batch_config_yaml(
            mode="hybrid",
            model_names=["a.safetensors", "b.safetensors"],
            backbone_idx=0,
            version=1,
            hybrid_config={
                "down_0_1": {0: 1.0},
                "down_2_3": {0: 1.0},
                "mid": {0: 1.0},
                "up_0_1": {0: 1.0},
                "up_2_3": {0: 1.0},
            },
            loras=loras,
        )
        data = yaml.safe_load(yml)
        job = data["batch_jobs"][0]
        self.assertIn("loras", job)
        self.assertEqual(job["loras"], loras)


if __name__ == "__main__":
    unittest.main()

