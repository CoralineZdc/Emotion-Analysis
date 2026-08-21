import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet 

from models import Model, RegressionHead


class EfficientNetB0(Model):
    def __init__(self, num_channels: int = 1, num_outputs: int = 1, dropout_rate: float = 0.3, freezed: bool = False):
        super().__init__(model_name="efficientnet", num_channels=num_channels, num_outputs=num_outputs, dropout_rate=dropout_rate, freezed=freezed)

    def initialize_backbone(self):
        # Initialize EfficientNet-B0 backbone
        backbone = EfficientNet.from_name('efficientnet-b0', in_channels=self.num_channels)  # Assuming grayscale images; change to 3 for RGB
        return backbone

    def set_classifier(self):
        in_features = self.backbone._fc.in_features

        self.backbone._fc = nn.Identity()  # Remove the original fully connected layer
        self.head = RegressionHead(in_features, 128, self.num_outputs, self.dropout_rate)

if __name__ == "__main__":
    print(f"--- Testing EfficientNetB0 ---")
    model = EfficientNetB0(num_channels=3, num_outputs=1, dropout_rate=0.3, freezed=True)
    print(f"Model architecture:\n{model}")
    dummy_input = torch.randn(4, 3, 48, 48)  # Batch of size 4, 3 channels, 48x48 images

    try:
        output = model(dummy_input)
        print(f"EfficientNetB0 Input: {dummy_input.shape} -> Output: {output.shape}")
        print("Success!")
    except Exception as e:
        print(f"Error with EfficientNetB0: {e}")