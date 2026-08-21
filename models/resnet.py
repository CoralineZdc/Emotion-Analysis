import torch
import torch.nn as nn
from torchvision import models 

from models import Model, RegressionHead


class ResNetRegression(Model):
    def __init__(self, model_name: str = "resnet18", num_channels: int = 1, num_outputs: int = 1, dropout_rate: float = 0.0, freezed: bool = False):
        super().__init__(model_name=model_name, num_channels=num_channels, num_outputs=num_outputs, dropout_rate=dropout_rate, freezed=freezed)
        self.model_name = model_name

    def initialize_backbone(self):
        MODELS = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
        }

        model_class = MODELS.get(self.model_name)
        if model_class is None:
            raise ValueError(f"Unsupported model name: {self.model_name}. Supported models are 'resnet18', 'resnet34', and 'resnet50'.")
        backbone = model_class(weights=None)  # Load the model without pre-trained weights

        # Modify the first convolutional layer to accept the specified number of input channels
        backbone.conv1 = nn.Conv2d(self.num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.bn1 = nn.BatchNorm2d(64)

        return backbone

    def set_classifier(self):
        in_features = self.backbone.fc.in_features
        hidden_features = 256
        out_features = self.num_outputs

        self.backbone.fc = nn.Identity()  # Remove the original fully connected layer
        self.head =  RegressionHead(in_features, hidden_features, out_features, self.dropout_rate)


if __name__ == "__main__":
    model_names = ["resnet18", "resnet34", "resnet50"]

    for model_name in model_names:
        print(f"\n--- Testing {model_name} ---")
        model = ResNetRegression(model_name=model_name, num_channels=1, num_outputs=1, freezed=True)
        dummy_input = torch.randn(4, 1, 48, 48)  # Batch of size 4, 1 channel, 48x48 images
        
        try:
            output = model(dummy_input)
            print(f"{model_name} Input: {dummy_input.shape} -> Output: {output.shape}")
            print("Success!")
        except Exception as e:
            print(f"Error with {model_name}: {e}")
