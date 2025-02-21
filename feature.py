# Program implementing GLCM to extract contrast, correlation, energy and homogeneity,
# and LBP to create a histogram to capture micro patterns.

import logging
import numpy as np
import skimage.feature as sf
import cv2
from glob import glob
import pandas as pd
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from pathlib import Path
from typing import List
from config import BASE_PATH, IMAGE_SIZE, DISTANCES, ANGLES, LBP_RADIUS, LBP_POINTS, LBP_METHOD, TEXTURE_CLASSES, OUTPUT_DIR

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """
    Loads an image in grayscale and applies histogram equalization.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        np.ndarray: Preprocessed grayscale image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        logging.warning(f"Skipping unreadable image: {image_path}")
        return None

    image = cv2.equalizeHist(image)
    image = cv2.resize(image, IMAGE_SIZE)
    return image

# Base path
base_path = "C:/Users/peter_idoko.VACFSS/Documents/VSCode/textureClassification"

# GLCM parameters: distances and angles
# Pixel offset to capture finer and coarse textures
distances = [1, 2, 3, 4]

# Captures the four primary orientations in a 2D image
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# LBP parameters: radius, points, method
LBP_RADIUS = 3  # Defines neighborhood radius
LBP_POINTS = min(8 * LBP_RADIUS, 24)  # Number of points for LBP
LBP_METHOD = "uniform"  # Rotation-invariant LBP

# Resize images to a smaller size to speed up GLCM processing
IMAGE_SIZE = (64, 64) # 64x64 pixels

def compute_glcm_features(image: np.ndarray) -> np.ndarray:
    """
    Computes GLCM features: contrast, correlation, energy, and homogeneity.
    
    Args:
        image (np.ndarray): Grayscale image.
    
    Returns:
        np.ndarray: Feature vector containing GLCM properties.
    """
    glcm = graycomatrix(image, distances=DISTANCES, angles=ANGLES, levels=256, symmetric=True, normed=True)

    features = np.hstack([
        graycoprops(glcm, prop).flatten()
        for prop in ["contrast", "correlation", "energy", "homogeneity"]
    ])

    return features

def compute_lbp_features(image: np.ndarray) -> np.ndarray:
    """
    Computes LBP histogram features.
    
    Args:
        image (np.ndarray): Grayscale image.
    
    Returns:
        np.ndarray: Normalized LBP histogram.
    """
    lbp = local_binary_pattern(image, P=LBP_POINTS, R=LBP_RADIUS, method=LBP_METHOD)

    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-8)  # Avoid division by zero
    return lbp_hist


def extract_features() -> None:
    """
    Extracts GLCM and LBP features from images and saves them as CSV files.
    """
    glcm_features_list, lbp_features_list, labels = [], [], []

    for class_label in TEXTURE_CLASSES:
        class_path = Path(BASE_PATH) / class_label
        image_files = list(class_path.glob("*.jpg")) + \
                      list(class_path.glob("*.png")) + \
                      list(class_path.glob("*.webp")) + \
                      list(class_path.glob("*.tiff"))

        for image_path in image_files:
            image = load_and_preprocess_image(str(image_path))
            if image is None:
                continue

            glcm_features = compute_glcm_features(image)
            lbp_features = compute_lbp_features(image)

            glcm_features_list.append(glcm_features)
            lbp_features_list.append(lbp_features)
            labels.append(class_label)

    save_features_to_csv(glcm_features_list, lbp_features_list, labels)


def save_features_to_csv(glcm_features_list: List[np.ndarray], lbp_features_list: List[np.ndarray], labels: List[str]) -> None:
    """
    Saves extracted features to CSV files.
    
    Args:
        glcm_features_list (List[np.ndarray]): List of GLCM feature vectors.
        lbp_features_list (List[np.ndarray]): List of LBP feature vectors.
        labels (List[str]): Corresponding class labels.
    """
    glcm_columns = [
        f"{prop}_{d}_{int(np.degrees(a))}"
        for prop in ["contrast", "correlation", "energy", "homogeneity"]
        for d in DISTANCES
        for a in ANGLES
    ]
    lbp_columns = [f"lbp_{i}" for i in range(LBP_POINTS + 2)]

    df_glcm = pd.DataFrame(glcm_features_list, columns=glcm_columns)
    df_lbp = pd.DataFrame(lbp_features_list, columns=lbp_columns)

    df_glcm["label"] = labels
    df_lbp["label"] = labels

    # Ensure output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    glcm_csv_path = Path(OUTPUT_DIR) / "texture_features_glcm.csv"
    lbp_csv_path = Path(OUTPUT_DIR) / "texture_features_lbp.csv"

    df_glcm.to_csv(glcm_csv_path, index=False)
    df_lbp.to_csv(lbp_csv_path, index=False)

    logging.info(f"GLCM features saved to {glcm_csv_path}")
    logging.info(f"LBP features saved to {lbp_csv_path}")


if __name__ == "__main__":
    extract_features()
