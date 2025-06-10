import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import transforms
from models.base_model import BaseModel
from training.trainer_config import TrainerConfig


class MobileNetModel(BaseModel):
    def __init__(self, config: TrainerConfig, num_classes=3, use_pretrained=True, dropout_rate=0.5):
        """
        Initialize the MobileNetV3-Small model strategy.

        Args:
            config (TrainerConfig): Configuration for the model.
            num_classes (int): Number of output classes.
            use_pretrained (bool): Whether to use pretrained ImageNet weights.
        """
        self.weights = MobileNet_V3_Small_Weights.DEFAULT if use_pretrained else None
        self.num_classes = num_classes
        self.config = config

    def get_model(self) -> nn.Module:
        """
        Instantiate and return the MobileNetV3-Small model with modified classifier.

        Returns:
            nn.Module: A MobileNetV3-Small model adapted to the specified number of classes.
        """
        model = mobilenet_v3_small(weights=self.weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(p=self.config.dropout_rate),
            nn.Linear(in_features, self.num_classes)
        )

        if self.config.freeze_base:
            for param in model.features.parameters():
                param.requires_grad = False

            if getattr(self.config, "freeze_mobilenet_last_blocks", False):
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
        Return the MobileNetV3-Small backbone without the classifier.

        Useful for feature extraction or hybrid pipelines.

        Returns:
            nn.Module: MobileNetV3 without the classification head.
        """
        model = mobilenet_v3_small(weights=self.weights)
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
