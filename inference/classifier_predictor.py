from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import csv
import joblib
import numpy as np
from PIL import Image

from inference.predictor_interface import PredictorInterface
from utils.dicom import load_image


class ClassifierPredictor(PredictorInterface):
    """
    Predictor for classifiers like SVM or XGBoost using handcrafted features.
    """
    def __init__(self, strategy, model_path, label_list=None):
        """
        Args:
            strategy: A BaseClassifier instance (e.g., SVMClassifier or XGBoostClassifier)
            model_path (str): Path to the saved joblib file
        """
        self.strategy = strategy
        data = joblib.load(model_path)
        if isinstance(data, dict) and "model" in data and "scaler" in data:
            self.strategy.model = data["model"]
            self.strategy.scaler = data["scaler"]
        else:
            # Legacy: just a model, no scaler
            self.strategy.model = data
            self.strategy.scaler = None
        self.label_list = label_list or ["CLASS_0", "CLASS_1", "CLASS_2"]

    def predict_features(self, features) -> dict:
        """
        Predict the class given a feature vector (1D or 2D array).
        Args:
            features (array-like): Feature vector (1D or 2D)
        Returns:
            dict: {
                "prediction": int,
                "class_label": str,
                "confidence": float,
                "all_probs": np.ndarray
            }
        """
        features = np.asarray(features)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if self.strategy.scaler is not None:
            features_scaled = self.strategy.scaler.transform(features)
        else:
            features_scaled = features
        probs = self.strategy.model.predict_proba(features_scaled)[0]
        pred = int(np.argmax(probs))
        class_label = self.label_list[pred] if pred < len(self.label_list) else str(pred)
        return {
            "prediction": pred,
            "class_label": class_label,
            "confidence": round(float(probs[pred]), 3),
            "all_probs": np.round(probs, 3)
        }

    def predict(self, image: Image.Image) -> dict:
        """
        Predict the class of a single PIL image using handcrafted features.
        Uses only mean, std, width, height.
        """
        img_gray = image.convert("L")
        img_array = np.array(img_gray).astype(np.float32)
        img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array) + 1e-8)
        features = np.array([img_array.mean(), img_array.std(), img_array.shape[1], img_array.shape[0]])
        return self.predict_features(features)
    
    def predict_from_path(self, image_path: Union[str, Image.Image]) -> dict:
        """
        Load and preprocess image from path, then run prediction.
        Should return the same output as `predict()`, with optional label mapping.

        Args:
            image_path (Union[str, Image.Image]): Path to image file or PIL Image object

        Returns:
            dict: Dictionary containing:
                - prediction (str): Predicted class label
                - confidence (float): Confidence score for the prediction
                - all_probs (numpy.ndarray): Probability distribution over all classes
        """
        if isinstance(image_path, str):
            image = load_image(image_path)
        else:
            image = image_path

        result = self.predict(image)
        if self.label_list:
            result["class_label"] = self.label_list[result["prediction"]]
        return result

    def predict_directory(
        self,
        dir_path: str,
        correlation_id: str,
        output_path: str,
        output_file_name: str = "inference.csv",
        recursive: bool = False,
        allowed_exts={".dcm", ".jpg", ".jpeg", ".png"}
    ) -> list[Dict]:
        """
        Predict on all images in a directory and write results to a timestamped CSV.

        Args:
            dir_path (str): Directory containing images.
            correlation_id (str): Unique ID for experiment folder.
            output_path (str): Base output directory.
            output_file_name (str): Base name of output file (timestamp will be appended).
            recursive (bool): Whether to include subfolders.
            allowed_exts (set): Allowed image extensions.

        Returns:
            List[dict]: List of prediction result dicts.
        """
        results = []
        dir_path = Path(dir_path)
        out_dir = Path(output_path) / correlation_id
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = output_file_name.rsplit(".", 1)[0]
        final_name = f"{base_name}_{timestamp}.csv"
        csv_path = out_dir / final_name

        files = dir_path.rglob("*") if recursive else dir_path.glob("*")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["filename", "filepath", "prediction", "class_label", "confidence"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for file in files:
                if file.suffix.lower() in allowed_exts:
                    try:
                        result = self.predict_from_path(str(file))
                        result["filename"] = file.name
                        result["filepath"] = str(file.resolve())

                        writer.writerow({
                            "filename": result["filename"],
                            "filepath": result["filepath"],
                            "prediction": result["prediction"],
                            "class_label": result.get("class_label", "N/A"),
                            "confidence": round(float(result["confidence"]), 3)
                        })

                        results.append(result)
                    except Exception as e:
                        print(f"[WARN] Failed to predict {file.name}: {e}")

        print(f"[INFO] Results saved to {csv_path}")
        return results