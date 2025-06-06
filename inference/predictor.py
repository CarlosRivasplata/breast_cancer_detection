from inference.cnn_predictor import CNNPredictor
from inference.classifier_predictor import ClassifierPredictor


PREDICTOR_REGISTRY = {
    "ResNet50": CNNPredictor,
    "EfficientNetB0": CNNPredictor,
    "MobileNetV3": CNNPredictor,
    "SVM": ClassifierPredictor,
    "XGBoost": ClassifierPredictor,
}

def get_predictor(model_type: str, strategy, model_path: str, device=None):
    """
    Factory function to create and return the appropriate predictor instance.

    Args:
        model_type (str): Type of model to create (e.g. "ResNet50", "SVM", "XGBoost")
        strategy: Model strategy object that provides architecture/transforms
        model_path (str): Path to the trained model weights file
        device (torch.device, optional): Device to run inference on. Only used for CNN models.

    Returns:
        PredictorInterface: Instance of the appropriate predictor class

    Raises:
        ValueError: If model_type is not supported
    """
    if model_type not in PREDICTOR_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}")
    return PREDICTOR_REGISTRY[model_type](strategy, model_path, device)
