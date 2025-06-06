from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms import ToTensor
from inference.explainer_interface import ExplainerInterface
import torch
import numpy as np
from PIL import Image


class GradCAMExplainer(ExplainerInterface):
    def __init__(self, predictor, target_layer=None):
        """
        Initialize the GradCAM explainer.

        Args:
            predictor: A predictor object that contains the model and transforms
            target_layer: The target layer to generate GradCAM for. If None, uses the last layer of ResNet's layer4
        """
        self.predictor = predictor
        self.model = predictor.model
        self.device = predictor.device
        self.transform = predictor.transform

        if target_layer is None:
            self.target_layer = self.model.layer4[-1]
        else:
            self.target_layer = target_layer

    def explain(self, image: Image.Image) -> np.ndarray:
        """
        Generate a GradCAM explanation for the input image.

        Args:
            image (PIL.Image): Input image to explain

        Returns:
            PIL.Image: GradCAM visualization overlaid on the input image
        """
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        resized_image = image.resize((input_tensor.shape[-1], input_tensor.shape[-2]))  # Match model input size
        rgb_img = np.array(resized_image).astype(np.float32) / 255.0

        self.model.to(self.device)
        cam = GradCAM(model=self.model, target_layers=[self.target_layer], reshape_transform=None)
        
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        return cam_image