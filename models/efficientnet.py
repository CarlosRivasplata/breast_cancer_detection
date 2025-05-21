import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from models.base_model import BaseModel


class EfficientNetModel(BaseModel):
    def __init__(self, num_classes=3, use_pretrained=True):
        """
        Initialize the EfficientNet-B0 model strategy.

        Args:
            num_classes (int): Number of output classes.
            use_pretrained (bool): Whether to use pretrained ImageNet weights.
        """
        self.weights = EfficientNet_B0_Weights.DEFAULT if use_pretrained else None
        self.num_classes = num_classes

    def get_model(self) -> nn.Module:
        """
        Instantiate and return the EfficientNet-B0 model with modified classifier.

        Returns:
            nn.Module: An EfficientNet-B0 model adapted to the specified number of classes.
        """
        model = efficientnet_b0(weights=self.weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
        return model

    def get_transforms(self) -> transforms.Compose:
        """
        Return the image transforms that match the model’s expected input format.

        Returns:
            transforms.Compose: A torchvision transform pipeline.
        """
        return super().get_transforms()


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
