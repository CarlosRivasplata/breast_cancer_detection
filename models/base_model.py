from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torchvision import transforms

class BaseModel(ABC):
    @abstractmethod
    def get_model(self) -> nn.Module:
        """Return the actual PyTorch model (e.g., ResNet50, EfficientNet, etc.)"""
        pass

    @abstractmethod
    def get_transforms(self, train: bool = True) -> transforms.Compose:
        """Return the appropriate torchvision transform pipeline."""
        t = self.config.transforms
        transform_list = [transforms.Resize((t.resize, t.resize))]

        if train and t.use_augmentations:
            transform_list += [
                transforms.RandomHorizontalFlip(p=t.horizontal_flip_prob),
                transforms.RandomRotation(degrees=t.rotation_degrees),
                transforms.RandomAffine(degrees=0, translate=(t.translate, t.translate), scale=(t.scale_min, t.scale_max)),
                transforms.ColorJitter(brightness=t.brightness, contrast=t.contrast),
                transforms.GaussianBlur(kernel_size=3, sigma=(t.blur_sigma_min, t.blur_sigma_max))
            ]

        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[t.normalize_mean], std=[t.normalize_std])
        ]

        return transforms.Compose(transform_list)

    @abstractmethod
    def get_backbone(self) -> nn.Module:
        """
        Optional: Return model backbone without the classifier head.
        
        Useful for feature extraction or hybrid pipelines.
        """
        raise NotImplementedError("This model doesn't expose a backbone.")

    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the input tensor using the model's backbone.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Extracted features from the model's backbone
        """
        raise NotImplementedError("This model doesn't support feature extraction.")
