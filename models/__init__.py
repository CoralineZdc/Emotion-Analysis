"""Model definitions for Age Estimation."""

from torch import nn
from abc import ABC, abstractmethod

class Model(nn.Module, ABC):

    def __init__(self, model_name: str, num_channels: int = 3, num_outputs: int = 1, dropout_rate: float = 0.3, freezed: bool = False):
        """
        Initializes the Model class.
        
        Args:
            model_name (str): Name of the model architecture (e.g., "vgg16").
            num_channels (int): Number of input channels (default: 1 for grayscale images).
            num_outputs (int): Number of outputs for regression (default: 1).
            dropout_rate (float): Dropout rate for regularization (default: 0.3).
            freezed (bool): Whether to freeze the backbone layers (default: False).
        """
        super().__init__()
        self.model_name = model_name
        self.num_channels = num_channels
        self.num_outputs = num_outputs
        self.dropout_rate = dropout_rate
        self.freezed = freezed

        self.backbone = self.initialize_backbone()
        self.set_classifier()
        if self.freezed:
            self.freeze_backbone()  # Freeze the backbone if specified

    @abstractmethod
    def initialize_backbone(self) -> nn.Module:
        """Must return the backbone model (e.g., VGG, ResNet, etc.) as an nn.Module."""
        pass

    @abstractmethod
    def set_classifier(self):
        """
        Must create self.head (nn.Module) and attach it.
        Can also modify self.backbone if needed (e.g., removing old FC).
        """
        pass

    def freeze_backbone(self):
        """
        Default implementation to freeze the backbone layers of the model, preventing them from being updated during training.
        Subclasses can override this method if they have a different way of freezing the backbone.
        """
        if self.backbone is None:
            raise RuntimeError("The backbone model is not defined. Please implement it in a subclass.")
        for param in self.backbone.parameters():
            param.requires_grad = False
        if self.head is not None:
            for param in self.head.parameters():
                param.requires_grad = True  # Ensure the head is trainable
        
    def forward(self, x):
        """Defines the forward pass of the model."""
        x = self.backbone(x)
        if self.head is None:
            raise RuntimeError("Head not defined. Please implement set_classifier() in the subclass.")
        return self.head(x)


class RegressionHead(nn.Sequential):
    def __init__(self, in_features, hidden_features, out_features, dropout_rate, activation=nn.SiLU):
        super().__init__(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, hidden_features),
            activation(inplace=True) if activation == nn.ReLU else activation(),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features)
        )