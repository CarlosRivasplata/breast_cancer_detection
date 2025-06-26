import streamlit as st
import torch
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
import pydicom
from io import BytesIO
from collections import defaultdict
import joblib
from utils.constants import IMAGES_ABS_PATH
from utils.feature_extraction import extract_features
from tqdm import tqdm

from models.resnet import ResNetModel
from models.efficientnet import EfficientNetModel
from models.mobilenet import MobileNetModel
from training.trainer_config import TrainerConfig
from inference.cnn_predictor import CNNPredictor
from inference.gradcam_explainer import GradCAMExplainer
from training.trainer_config import TransformConfig
from models.classifiers.svm_classifier import SVMClassifier
from models.classifiers.xgboost_classifier import XGBoostClassifier
from inference.classifier_predictor import ClassifierPredictor

import sys
import traceback


project_root = Path("../").resolve()  # one level up from /notebooks/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Page config
st.set_page_config(
    page_title="Breast Cancer Detection Dashboard",
    layout="wide"
)

# Title and description
st.markdown("# Breast Cancer Detection Dashboard")
st.markdown("""
### This dashboard allows you to:
- Run inference with multiple trained CNN models (ResNet50, EfficientNetB0, MobileNetV3)
- Upload and analyze medical images
- View individual and ensemble predictions
- Visualize model attention using GRAD-CAM for each model
""")

def load_image(uploaded_file) -> Image.Image:
    """
    Load a medical image from an uploaded file (.dcm, .jpg, .png) as a PIL Image.
    Handles both Streamlit UploadedFile and standard file objects.
    """
    filename = getattr(uploaded_file, 'name', None)
    if filename is None and hasattr(uploaded_file, 'name'):
        filename = uploaded_file.name
    elif filename is None:
        filename = ''  # fallback

    # Read file content as bytes
    if hasattr(uploaded_file, 'getvalue'):
        file_bytes = uploaded_file.getvalue()
    else:
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)  # reset pointer for re-use

    if filename.lower().endswith(".dcm"):
        dicom_data = pydicom.dcmread(BytesIO(file_bytes))
        image_array = dicom_data.pixel_array.astype(np.float32)
        image_array -= image_array.min()
        image_array /= (image_array.max() + 1e-5)
        image_array *= 255.0
        image_array = image_array.clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(image_array).convert("RGB")
        return pil_img
    else:
        return Image.open(BytesIO(file_bytes)).convert("RGB")

def calculate_ensemble_prediction(predictions):
    """
    Calculate ensemble prediction by averaging probabilities from all models.
    
    Args:
        predictions (dict): Dictionary of predictions from each model
        
    Returns:
        dict: Ensemble prediction with averaged probabilities
    """
    ensemble_probs = defaultdict(float)
    num_models = len(predictions)
    
    for model_pred in predictions.values():
        for i, prob in enumerate(model_pred['all_probs']):
            ensemble_probs[label_list[i]] += prob / num_models
    
    # Convert to list in the same order as the original label list
    ensemble_probs_list = [ensemble_probs[label] for label in label_list]
    
    # Get the class with highest probability
    max_prob_idx = np.argmax(ensemble_probs_list)
    predicted_class = label_list[max_prob_idx]
    confidence = ensemble_probs_list[max_prob_idx]
    
    return {
        'class_label': predicted_class,
        'confidence': confidence,
        'all_probs': ensemble_probs_list,
        'label_list': label_list
    }

# Sidebar for model loading
st.sidebar.header("Model Configuration")
model_paths = {
    "ResNet50": st.sidebar.file_uploader("Upload ResNet50 Model Weights (.pth)", type=['pth']),
    "EfficientNetB0": st.sidebar.file_uploader("Upload EfficientNetB0 Model Weights (.pth)", type=['pth']),
    "MobileNetV3": st.sidebar.file_uploader("Upload MobileNetV3 Model Weights (.pth)", type=['pth'])
}

# Always use 3 classes
label_list = ["BENIGN", "BENIGN_WITHOUT_CALLBACK", "MALIGNANT"]
num_classes = 3

# Initialize session state
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'explainers' not in st.session_state:
    st.session_state.explainers = {}

# Model loading
for model_type, model_path in model_paths.items():
    if model_path is not None and model_type not in st.session_state.models:
        with st.spinner(f"Loading {model_type}..."):
            try:
                # Save uploaded model temporarily
                temp_path = Path(f"temp_{model_type.lower()}.pth")
                with open(temp_path, "wb") as f:
                    f.write(model_path.getvalue())
                
                # Get model strategy and initialize predictor
                transform_config = TransformConfig(
                    use_augmentations=False,  # No augmentations for inference
                    resize=224,
                    normalize_mean=0.5,
                    normalize_std=0.5
                )
                config = TrainerConfig(
                    model_name=model_type,
                    transforms=transform_config
                )
                if model_type == "ResNet50":
                    strategy = ResNetModel(config, num_classes=num_classes)
                elif model_type == "EfficientNetB0":
                    strategy = EfficientNetModel(config, num_classes=num_classes)
                else:  # MobileNetV3
                    strategy = MobileNetModel(config, num_classes=num_classes)
                
                # Initialize predictor and explainer
                predictor = CNNPredictor(strategy, str(temp_path), label_list=label_list)
                st.session_state.models[model_type] = predictor
                st.session_state.explainers[model_type] = GradCAMExplainer(predictor)
                st.success(f"{model_type} loaded successfully!")
            except Exception as e:
                st.error(f"Error loading {model_type}: {str(e)}")
            finally:
                if temp_path.exists():
                    temp_path.unlink()

# Sidebar for classic classifier model loading
st.sidebar.header("Classic Classifier Configuration")
classic_models = {}
classic_model_types = ["SVM", "XGBoost"]
for model_type in classic_model_types:
    classic_models[model_type] = st.sidebar.file_uploader(f"Upload {model_type} Model (.pkl)", type=["pkl"], key=f"{model_type.lower()}_pkl")
parquet_file = st.sidebar.file_uploader("Upload .parquet file with image paths", type=["parquet"], key="parquet_file")

classic_predictors = {}
if any(classic_models.values()):
    if parquet_file is not None:
        try:
            df = pd.read_parquet(parquet_file)
            CORRECT_IMAGES_ROOT = "D:/TFM/breast_cancer_detection/data/CBIS-DDSM/CBIS-DDSM"
            df['full_image_path'] = df['image_path'].apply(lambda x: os.path.normpath(os.path.join(CORRECT_IMAGES_ROOT, x)))
            df['exists'] = df['full_image_path'].apply(os.path.exists)
            df = df[df['exists']].copy()
            if df.empty:
                st.warning("No valid image paths found in the .parquet file.")
                debug_df = pd.read_parquet(parquet_file)
                debug_df['resolved_path'] = debug_df['image_path'].apply(lambda x: os.path.normpath(os.path.join(IMAGES_ABS_PATH, x)))
                debug_df['exists'] = debug_df['resolved_path'].apply(os.path.exists)
                st.markdown('#### Debug: Resolved Paths and Existence')
                st.dataframe(debug_df[['image_path', 'resolved_path', 'exists']].head(10))
            else:
                # Extract features for classic classifiers
                with st.spinner("Extracting features..."):
                    features_df = extract_features(df)
                # Instantiate predictors for each classic model and run predictions with spinner
                with st.spinner("Running classic classifier predictions..."):
                    for model_type, model_file in classic_models.items():
                        if model_file is not None:
                            if model_type == "SVM":
                                strategy = SVMClassifier()
                            elif model_type == "XGBoost":
                                strategy = XGBoostClassifier()
                            else:
                                continue
                            # Save uploaded model temporarily
                            temp_path = Path(f"temp_{model_type.lower()}_classic.pkl")
                            with open(temp_path, "wb") as f:
                                f.write(model_file.getvalue())
                            classic_predictors[model_type] = ClassifierPredictor(strategy, str(temp_path), label_list=label_list)
                            temp_path.unlink()  # Remove temp file after loading
                    st.markdown("---")
                    st.markdown("## Classic Classifier Predictions (SVM/XGBoost)")
                    results = []
                    target_features = ["assessment", "breast_density", "subtlety", "mean_intensity", "std_intensity", "width", "height"]
                    for idx, row in tqdm(features_df.iterrows(), total=len(features_df), desc="Classic classifier predictions"):
                        img_path = row["full_image_path"]
                        row_result = {"Image": os.path.basename(img_path)}
                        for model_type, predictor in classic_predictors.items():
                            try:
                                feats = row[target_features].values.reshape(1, -1)
                                pred = predictor.strategy.model.predict(feats)[0]
                                proba = predictor.strategy.model.predict_proba(feats)[0]
                                row_result[f"{model_type} Prediction"] = label_list[pred]
                                row_result[f"{model_type} Probabilities"] = proba
                            except Exception as e:
                                row_result[f"{model_type} Error"] = str(e)
                        # CNN block for .parquet images
                        if st.session_state.models:
                            try:
                                with open(img_path, "rb") as f:
                                    image = load_image(f)
                                for model_type, predictor in st.session_state.models.items():
                                    pred = predictor.predict(image)
                                    row_result[f"{model_type} Prediction"] = pred["class_label"]
                                    row_result[f"{model_type} Confidence"] = pred["confidence"]
                            except Exception as e:
                                row_result["CNN Error"] = str(e)
                        results.append(row_result)
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df)
        except Exception as e:
            st.error(f"Error processing classic classifiers: {str(e)}")
            st.error(traceback.format_exc())

# Main content area
if not st.session_state.models:
    st.warning("Please upload models in the sidebar to begin.")
else:
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Medical Images",
        type=['dcm', 'jpg', 'png'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Process each uploaded file
        for uploaded_file in uploaded_files:
            st.markdown(f"### Analyzing: {uploaded_file.name}")
            
            # Load and display image
            try:
                image = load_image(uploaded_file)
                
                # Run predictions with all models first
                predictions = {}
                for model_type, predictor in st.session_state.models.items():
                    predictions[model_type] = predictor.predict(image)
                
                # Create columns for image and GRAD-CAM
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div style="display: flex; justify-content: center;"><h3 style="text-align: center; margin-bottom: 0;">Original Image</h3></div>', unsafe_allow_html=True)
                    st.image(image, width=300)
                
                with col2:
                    # Generate and display GRAD-CAM visualizations
                    st.markdown('<div style="display: flex; justify-content: center;"><h3 style="text-align: center; margin-bottom: 0;">Model Attention Maps (GRAD-CAM)</h3></div>', unsafe_allow_html=True)
                    for model_type, explainer in st.session_state.explainers.items():
                        with st.spinner(f"Generating GRAD-CAM for {model_type}..."):
                            cam_image = explainer.explain(image, predictions[model_type]['prediction'])
                            st.markdown(f'<div style="display: flex; justify-content: center;"><h4 style="text-align: center; margin-bottom: 0;">{model_type} GRAD-CAM</h4></div>', unsafe_allow_html=True)
                            st.image(cam_image, width=300)
                
                # Calculate ensemble prediction
                ensemble_pred = calculate_ensemble_prediction(predictions)
                
                # Display ensemble prediction
                st.markdown("### Ensemble Prediction")
                ensemble_df = pd.DataFrame({
                    'Class': ensemble_pred['label_list'],
                    'Probability': ensemble_pred['all_probs']
                })
                
                # Define colors for each class
                colors = {
                    'BENIGN': '#2ecc71',  # Green
                    'BENIGN_WITHOUT_CALLBACK': '#f1c40f',  # Yellow
                    'MALIGNANT': '#e74c3c'  # Red
                }
                
                # Create bar chart for ensemble prediction
                fig, ax = plt.subplots(figsize=(10, 4))
                bars = ax.bar(
                    ensemble_df['Class'],
                    ensemble_df['Probability'],
                    color=[colors[cls] for cls in ensemble_df['Class']]
                )
                
                # Customize the plot
                ax.set_ylim(0, 1)
                ax.set_ylabel('Probability', fontsize=12)
                ax.set_title('Ensemble Class Probabilities', fontsize=14)
                ax.tick_params(axis='both', which='major', labelsize=10)
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height,
                        f'{height:.2%}',
                        ha='center',
                        va='bottom',
                        fontsize=10
                    )
                
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                # Display ensemble confidence
                st.markdown("### Ensemble Prediction Confidence")
                st.metric(
                    "Prediction",
                    f"{ensemble_pred['confidence']:.2%}",
                    f"Class: {ensemble_pred['class_label']}"
                )
                
                # Display individual model predictions
                st.markdown("### Individual Model Predictions")
                for model_type, pred in predictions.items():
                    st.markdown(f"#### {model_type}")
                    st.metric(
                        "Prediction",
                        f"{pred['confidence']:.2%}",
                        f"Class: {pred['class_label']}"
                    )
                
                st.markdown("---")
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                st.markdown("---")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Powered by PyTorch")
