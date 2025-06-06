from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TransformConfig:
    use_augmentations: bool = True
    resize: int = 224
    horizontal_flip_prob: float = 0.5
    rotation_degrees: int = 5
    translate: float = 0.02
    scale_min: float = 0.98
    scale_max: float = 1.02
    brightness: float = 0.1
    contrast: float = 0.1
    blur_sigma_min: float = 0.1
    blur_sigma_max: float = 1.0
    normalize_mean: float = 0.5
    normalize_std: float = 0.5


@dataclass
class TrainerConfig:
    model_name: str
    dropout_rate: float = 0.3
    freeze_base: bool = False
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 70
    gamma: float = 2.0  # For FocalLoss only
    early_stop_patience: int = 5
    scheduler_patience: int = 3
    transforms: Optional[TransformConfig] = None
