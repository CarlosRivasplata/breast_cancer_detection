from PIL import Image
import numpy as np
import joblib
from inference.predictor_interface import PredictorInterface


class ClassifierPredictor(PredictorInterface):
    """
    A classifier-based predictor that uses a trained model to make predictions on images.
    
    This class implements the PredictorInterface and provides functionality to:
    - Load a trained classifier model (e.g. SVM, XGBoost)
    - Transform input images and extract features
    - Make predictions and return probabilities
    """
    def __init__(self, strategy, model_path):
        """
        Initialize the classifier predictor with a model strategy and trained weights.

        Args:
            strategy: Model strategy object that provides feature extraction and transforms
            model_path (str): Path to the trained model weights file
        """
        self.strategy = strategy
        self.model = joblib.load(model_path)
        self.transform = strategy.get_transforms(train=False)

    def predict(self, image: Image.Image) -> dict:
        """
        Make a prediction on the input image.

        Args:
            image (PIL.Image): Input image to classify

        Returns:
            dict: Dictionary containing:
                - prediction (int): Predicted class index
                - confidence (float): Confidence score for the prediction
                - all_probs (numpy.ndarray): Probability distribution over all classes
        """
        x = self.transform(image).unsqueeze(0)
        features = self.strategy.extract_features(x).detach().numpy()
        probs = self.model.predict_proba(features)[0]
        pred = int(np.argmax(probs))
        return {
            "prediction": pred,
            "confidence": probs[pred],
            "all_probs": probs
        }
