import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from models.base_model import BaseModel
from training.trainer_config import TrainerConfig


class ResNetModel(BaseModel):
    def __init__(self, config: TrainerConfig, num_classes=3, use_pretrained=True):
        """
        Initialize the ResNet50 model strategy.

        Args:
            config (TrainerConfig): Configuration for the model.
            num_classes (int): Number of output classes.
            use_pretrained (bool): Whether to use pretrained ImageNet weights.
        """
        self.weights = ResNet50_Weights.DEFAULT if use_pretrained else None
        self.num_classes = num_classes
        self.config = config

    def get_model(self) -> nn.Module:
        """
        Instantiate and return the ResNet50 model with modified output layer.

        Returns:
            nn.Module: A ResNet50 model adapted to the specified number of classes.
        """
        model = resnet50(weights=self.weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=self.config.dropout_rate),
            nn.Linear(in_features, self.num_classes)
        )
        if self.config.freeze_base:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model

    def get_transforms(self, train: bool = True) -> transforms.Compose:
        """
        Return the image transforms that match the model’s expected input format.

        Returns:
            transforms.Compose: A torchvision transform pipeline.
        """
        return super().get_transforms(train)

    def get_backbone(self) -> nn.Module:
        """
        Return the ResNet50 backbone without the classification head.

        Useful for feature extraction or hybrid pipelines.

        Returns:
            nn.Module: ResNet50 without the final fully connected layer.
        """
        model = resnet50(weights=self.weights)
        backbone = nn.Sequential(*(list(model.children())[:-1]))
        return backbone

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the input tensor using the model's backbone.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Extracted features from the model's backbone
        """
        x = self.get_backbone()(x)
        x = torch.flatten(x, 1)
        return x
