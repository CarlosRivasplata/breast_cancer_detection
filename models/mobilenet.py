import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import transforms
from models.base_model import BaseModel


class MobileNetModel(BaseModel):
    def __init__(self, num_classes=3, use_pretrained=True, dropout_rate=0.5):
        """
        Initialize the MobileNetV3-Small model strategy.

        Args:
            num_classes (int): Number of output classes.
            use_pretrained (bool): Whether to use pretrained ImageNet weights.
        """
        self.weights = MobileNet_V3_Small_Weights.DEFAULT if use_pretrained else None
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

    def get_model(self) -> nn.Module:
        """
        Instantiate and return the MobileNetV3-Small model with modified classifier.

        Returns:
            nn.Module: A MobileNetV3-Small model adapted to the specified number of classes.
        """
        model = mobilenet_v3_small(weights=self.weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(in_features, self.num_classes)
        )
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
