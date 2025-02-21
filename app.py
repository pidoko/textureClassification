import logging
from typing import Tuple
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import gradio as gr
from config import OUTPUT_DIR, IMAGE_SIZE, DISTANCES, ANGLES, LBP_RADIUS, LBP_POINTS, LBP_METHOD

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths to feature datasets
glcm_csv_path = Path(OUTPUT_DIR) / "texture_features_glcm.csv"
lbp_csv_path = Path(OUTPUT_DIR) / "texture_features_lbp.csv"


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load GLCM and LBP datasets separately with validation."""
    if not glcm_csv_path.exists() or not lbp_csv_path.exists():
        raise FileNotFoundError("One or both dataset files (GLCM or LBP) are missing.")

    df_glcm = pd.read_csv(glcm_csv_path)
    df_lbp = pd.read_csv(lbp_csv_path)

    return df_glcm, df_lbp


def preprocess_data(df: pd.DataFrame):
    """Splits data into train/test sets and applies standardization."""
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_models(X_train, y_train):
    """Trains multiple classifiers using GridSearchCV."""
    param_grids = {
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
        },
        "SVM": {
            "model": SVC(random_state=42),
            "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
        },
        "k-NN": {
            "model": KNeighborsClassifier(),
            "params": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
        },
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {"C": [0.1, 1, 10]},
        },
    }

    best_models = {}
    confusion_matrices = {}

    for name, config in param_grids.items():
        logging.info(f"Training {name}...")
        grid_search = GridSearchCV(config["model"], config["params"], cv=5, scoring="accuracy", n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_models[name] = grid_search.best_estimator_

    return best_models


def plot_confusion_matrices(y_test, models, X_test, filename_prefix):
    """Plots and saves confusion matrices for multiple models."""
    for name, model in models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        output_path = Path(OUTPUT_DIR) / f"{filename_prefix}_conf_matrix_{name}.png"
        plt.savefig(output_path)
        plt.close()

        logging.info(f"Confusion matrix for {name} saved to {output_path}")


def classify_texture(image, feature_type, model_name):
    """Classifies an input image using the selected feature type and model."""
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.equalizeHist(image)
    image = cv2.resize(image, IMAGE_SIZE)

    if feature_type == "GLCM":
        from feature import compute_glcm_features  # Import only when needed

        features = compute_glcm_features(image).reshape(1, -1)
        features = scaler_glcm.transform(features)
        prediction = best_models_glcm[model_name].predict(features)[0]

    elif feature_type == "LBP":
        from feature import compute_lbp_features

        features = compute_lbp_features(image).reshape(1, -1)
        features = scaler_lbp.transform(features)
        prediction = best_models_lbp[model_name].predict(features)[0]

    return prediction

try:
    # Load datasets
    df_glcm, df_lbp = load_datasets()

    # Preprocess GLCM and LBP datasets
    X_train_glcm, X_test_glcm, y_train_glcm, y_test_glcm, scaler_glcm = preprocess_data(df_glcm)
    X_train_lbp, X_test_lbp, y_train_lbp, y_test_lbp, scaler_lbp = preprocess_data(df_lbp)

    # Train models separately for GLCM and LBP
    best_models_glcm = train_models(X_train_glcm, y_train_glcm)
    best_models_lbp = train_models(X_train_lbp, y_train_lbp)

    # Plot confusion matrices
    plot_confusion_matrices(y_test_glcm, best_models_glcm, X_test_glcm, "GLCM")
    plot_confusion_matrices(y_test_lbp, best_models_lbp, X_test_lbp, "LBP")

    # Define Hugging Face App Title
    title = "Texture Classification Using GLCM and LBP"

    title += "\n\nSelect an appropriate image, choose a feature extraction method (GLCM or LBP), and pick a classifier to predict the texture category."

    title += "\n\nAppropriate image has little to no noise (only relevant texture)"
    
    # Gradio Interface
    interface = gr.Interface(
        fn=classify_texture,
        inputs=[
            gr.Image(type="numpy"),
            gr.Radio(["GLCM", "LBP"], label="Feature Type"),
            gr.Dropdown(choices=list(best_models_glcm.keys()), label="Select Classifier"),
        ],
        outputs=gr.Label(),
        title=title,
    )

    logging.info("Launching Gradio interface...")
    interface.launch()

except Exception as e:
    logging.error(f"An error occurred: {e}")
    exit(1)
