from abc import ABC, abstractmethod
import torch.nn as nn
from torchvision import transforms

class BaseModel(ABC):
    @abstractmethod
    def get_model(self) -> nn.Module:
        """Return the actual PyTorch model (e.g., ResNet50, EfficientNet, etc.)"""
        pass

    @abstractmethod
    def get_transforms(self) -> transforms.Compose:
        """Return the appropriate torchvision transform pipeline."""
        pass

    def get_backbone(self) -> nn.Module:
        """
        Optional: Return model backbone without the classifier head.
        
        Useful for feature extraction or hybrid pipelines.
        """
        raise NotImplementedError("This model doesn't expose a backbone.")
