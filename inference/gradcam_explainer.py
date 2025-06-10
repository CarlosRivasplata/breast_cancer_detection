import numpy as np
from PIL import Image
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class GradCAMExplainer:
    def __init__(self, predictor, target_layer=None):
        """
        GradCAM explainer for ResNet50, EfficientNetB0, and MobileNetV3.

        Args:
            predictor: CNNPredictor with .model, .device, and .transform
            target_layer (nn.Module): Optionally specify the exact target layer
        """
        self.predictor = predictor
        self.model = predictor.model
        self.device = predictor.device
        self.transform = predictor.transform

        self.model.to(self.device)
        self.model.eval()

        self.target_layer = target_layer or self._infer_target_layer()

    def _infer_target_layer(self):
        name = self.model.__class__.__name__.lower()

        if "resnet" in name:
            return self.model.layer4[-1]

        elif "efficientnet" in name:
            return self.model.features[-1]

        elif "mobilenet" in name:
            return self.model.features[-1]

        else:
            raise ValueError(f"Unsupported model type: {name}. Please provide target_layer explicitly.")

    def _unnormalize(self, tensor):
        """
        Undo normalization (assumes ImageNet mean/std).
        """
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        return (image * std + mean).clip(0, 1)

    def explain(self, image: Image.Image, class_idx: int = None) -> np.ndarray:
        """
        Generate GradCAM heatmap overlay for a given image.

        Args:
            image (PIL.Image): Input image
            class_idx (int, optional): Class index to explain. If None, top predicted class is used.

        Returns:
            np.ndarray: Heatmap-overlaid image in RGB format
        """
        resized_image = image.resize((224, 224))
        input_tensor = self.transform(resized_image).unsqueeze(0).to(self.device)

        rgb_img = self._unnormalize(input_tensor)

        with GradCAM(model=self.model, target_layers=[self.target_layer]) as cam:
            targets = [ClassifierOutputTarget(class_idx)] if class_idx is not None else None
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            return cam_image
