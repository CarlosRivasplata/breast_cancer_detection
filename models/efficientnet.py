import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from models.base_model import BaseModel
from training.trainer_config import TrainerConfig


class EfficientNetModel(BaseModel):
    def __init__(self, config: TrainerConfig, num_classes=3, use_pretrained=True, dropout_rate=0.5):
        """
        Initialize the EfficientNet-B0 model strategy.

        Args:
            num_classes (int): Number of output classes.
            use_pretrained (bool): Whether to use pretrained ImageNet weights.
        """
        self.weights = EfficientNet_B0_Weights.DEFAULT if use_pretrained else None
        self.num_classes = num_classes
        self.config = config

    def get_model(self) -> nn.Module:
        """
        Instantiate and return the EfficientNet-B0 model with modified classifier.

        Returns:
            nn.Module: An EfficientNet-B0 model adapted to the specified number of classes.
        """
        model = efficientnet_b0(weights=self.weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=self.config.dropout_rate),
            nn.Linear(in_features, self.num_classes)
        )
        if self.config.freeze_base:
            for param in model.parameters():
                param.requires_grad = False

            if getattr(self.config, "freeze_efficientnet_last_blocks", False):
                for block in model.features[-3:]:
                    for param in block.parameters():
                        param.requires_grad = True

            for param in model.classifier.parameters():
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
        Return the EfficientNet-B0 backbone without the classifier.

        Useful for feature extraction or hybrid pipelines.

        Returns:
            nn.Module: EfficientNet-B0 without the final classification layer.
        """
        model = efficientnet_b0(weights=self.weights)
        backbone = nn.Sequential(
            model.features,
            model.avgpool
        )
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
