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
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def get_backbone(self) -> nn.Module:
        """
        Optional: Return model backbone without the classifier head.
        
        Useful for feature extraction or hybrid pipelines.
        """
        raise NotImplementedError("This model doesn't expose a backbone.")
