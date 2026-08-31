import unittest

import torch
from torch import nn

from src.utils.training_utils import _match_model_state_dict


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 2, 3),
            nn.ReLU(),
            nn.BatchNorm2d(2),
        )


class TrainingUtilsStateDictCompatibilityTest(unittest.TestCase):
    def test_matches_keys_with_backbone_and_feature_prefixes(self):
        model = DummyBackbone()
        model_state = model.state_dict()
        checkpoint = {
            "features_0.weight": model_state["backbone.0.weight"].clone(),
            "features_0.bias": model_state["backbone.0.bias"].clone(),
            "features_2.weight": model_state["backbone.2.weight"].clone(),
            "features_2.bias": model_state["backbone.2.bias"].clone(),
            "features_2.running_mean": model_state["backbone.2.running_mean"].clone(),
            "features_2.running_var": model_state["backbone.2.running_var"].clone(),
        }

        matched = _match_model_state_dict(model, checkpoint)

        self.assertEqual(
            set(matched.keys()),
            {
                "backbone.0.weight",
                "backbone.0.bias",
                "backbone.2.weight",
                "backbone.2.bias",
                "backbone.2.running_mean",
                "backbone.2.running_var",
            },
        )


if __name__ == "__main__":
    unittest.main()
