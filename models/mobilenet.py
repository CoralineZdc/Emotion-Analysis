import torch
from torch import nn
from torchvision import models

from models import Model, RegressionHead

class MobileNet(Model):
    def __init__(self, num_channels: int = 1, num_outputs: int = 1, dropout_rate: float = 0.3, freezed: bool = False):
        super().__init__(model_name="mobilenet", num_channels=num_channels, num_outputs=num_outputs, dropout_rate=dropout_rate, freezed=freezed)

    def initialize_backbone(self):
        # Initialize MobileNet backbone
        backbone = models.mobilenet_v2(weights=None)
        # Modify the first convolutional layer to accept the specified number of input channels
        backbone.features[0][0] = nn.Conv2d(self.num_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        return backbone

    def set_classifier(self):
        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Identity()  # Remove the original fully connected layer
        self.head = RegressionHead(in_features, 128, self.num_outputs, self.dropout_rate)


if __name__ == "__main__":
    print(f"--- Testing MobileNet ---")
    model = MobileNet(num_channels=1, num_outputs=1, dropout_rate=0.3, freezed=True)
    dummy_input = torch.randn(4, 1, 48, 48)  # Batch of size 4, 1 channel, 48x48 images
    
    try:
        output = model(dummy_input)
        print(f"MobileNet Input: {dummy_input.shape} -> Output: {output.shape}")
        print("Success!")
    except Exception as e:
        print(f"Error with MobileNet: {e}")