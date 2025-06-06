from abc import ABC, abstractmethod
from PIL import Image
from typing import Union



class PredictorInterface(ABC):
    @abstractmethod
    def predict(self, image: Image.Image) -> dict:
        """
        Perform inference and return predicted label, confidence, and full probabilities.
        Returns:
            dict with keys: 'prediction', 'confidence', 'all_probs'
        """
        pass

    @abstractmethod
    def predict_from_path(self, image_path: Union[str, Image.Image]) -> dict:
        """
        Load and preprocess image from path, then run prediction.
        Should return the same output as `predict()`, with optional label mapping.
        """
        pass

    @abstractmethod
    def predict_directory(self):
        """
        Load and preprocess images from a given directory, then run prediction.
        """
        pass
