from abc import ABC, abstractmethod
from PIL import Image
import numpy as np


class ExplainerInterface(ABC):
    @abstractmethod
    def explain(self, image: Image.Image) -> np.ndarray:
        """
        Generate an explanation (e.g., heatmap) for the given input image.

        Args:
            image (PIL.Image): Input image

        Returns:
            np.ndarray: Explanation heatmap as a NumPy array
        """
        pass