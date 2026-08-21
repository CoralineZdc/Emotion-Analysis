
import torch
import torch.nn as nn
from torchvision import models

from models import Model, RegressionHead

class VGGRegression(Model):
    def __init__(self, model_name: str = "vgg16", num_channels: int = 3, num_outputs: int = 1, dropout_rate: float = 0.3, freezed: bool = False):
        super().__init__(model_name=model_name, num_channels=num_channels, num_outputs=num_outputs, dropout_rate=dropout_rate, freezed=freezed)

    def initialize_backbone(self):
        MODELS = {
            "vgg11": models.vgg11,
            "vgg13": models.vgg13,
            "vgg16": models.vgg16,
            "vgg19": models.vgg19,
        }
        
        model_class = MODELS.get(self.model_name)
        if model_class is None:
            raise ValueError(f"Unsupported model name: {self.model_name}. Supported models are 'vgg11', 'vgg13', 'vgg16', and 'vgg19'.")
        backbone = model_class(weights=None)  # Load the model without pre-trained weights

        # Modify the first convolutional layer to accept the specified number of input channels
        backbone.features[0] = nn.Conv2d(self.num_channels, 64, kernel_size=3, padding=1)

        return backbone
    
    def set_classifier(self):
        in_features = self.backbone.classifier[6].in_features
        hidden_features = 4096
        out_features = self.num_outputs

        self.backbone.classifier[6] = nn.Identity()  # Remove the original fully connected layer
        self.head = RegressionHead(in_features, hidden_features, out_features, self.dropout_rate)


if __name__ == "__main__":
    model_names = ["vgg11", "vgg13", "vgg16", "vgg19"]

    for model_name in model_names:
        print(f"\n--- Testing {model_name} ---")
        model = VGGRegression(model_name=model_name, num_channels=1, num_outputs=1, dropout_rate=0.3, freezed=True)
        dummy_input = torch.randn(4, 1, 48, 48)  # Batch of size 4, 1 channel, 48x48 images
        
        try:
            output = model(dummy_input)
            print(f"{model_name} Input: {dummy_input.shape} -> Output: {output.shape}")
            print("Success!")
        except Exception as e:
            print(f"Error with {model_name}: {e}")
