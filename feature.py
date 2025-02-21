# Program implementing GLCM to extract contrast, correlation, energy and homogeneity,
# and LBP to create a histogram to capture micro patterns.

import os
import numpy as np
import skimage.feature as sf
import cv2
from glob import glob
import pandas as pd

# Base path
base_path = "C:/Users/peter_idoko.VACFSS/Documents/VSCode/textureClassification"

# GLCM parameters: distances and angles
# Pixel offset to capture finer and coarse textures
distances = [1, 2, 3, 4]

# Captures the four primary orientations in a 2D image
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# Resize images to a smaller size to speed up GLCM processing
IMAGE_SIZE = (64, 64) # 64x64 pixels

# Function to compute GLCM features
def compute_glcm_features(image):
    image = cv2.resize(image, IMAGE_SIZE)
    glcm = sf.graycomatrix(image, distances = distances, angles = angles, levels = 256,
                           symmetric = True, normed = True)
    # Flatten 2D NumPy array into 1D array for Machine Learning classifier models.
    contrast = sf.graycoprops(glcm, 'contrast').flatten()
    correlation = sf.graycoprops(glcm, 'correlation').flatten()
    energy = sf.graycoprops(glcm, 'energy').flatten()
    homogeneity = sf.graycoprops(glcm, 'homogeneity').flatten()
    # Concatenate 1D arrays into single 1D feature vector
    return np.hstack([contrast, correlation, energy, homogeneity])

# Prepare dataset
glcm_features_list = []
labels = []

for class_label in ["wood", "brick", "stone"]:
    class_path = os.path.join(base_path, class_label)
    image_files = glob(os.path.join(class_path, "*.jpg")) + glob(os.path.join(class_path, "*.png")) + glob(os.path.join(class_path, "*.webp")) + glob(os.path.join(class_path, "*.tiff"))

    for image_path in image_files:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        # Normalize each image
        image = cv2.equalizeHist(image)

        # Compute features for each image
        glcm_features = compute_glcm_features(image)

        glcm_features_list.append(glcm_features)
        labels.append(class_label)

# Create DataFrames for GLCM and LBP
glcm_columns = [f"contrast_{d}_{int(np.degrees(a))}" for d in distances for a in angles] + \
               [f"correlation_{d}_{int(np.degrees(a))}" for d in distances for a in angles] + \
               [f"energy_{d}_{int(np.degrees(a))}" for d in distances for a in angles] + \
               [f"homogeneity_{d}_{int(np.degrees(a))}" for d in distances for a in angles]

df_glcm = pd.DataFrame(glcm_features_list, columns=glcm_columns)
df_glcm['label'] = labels

# Save CSV files
glcm_csv_path = os.path.join(base_path, "texture_features_glcm.csv")

df_glcm.to_csv(glcm_csv_path, index=False)

print(f"Feature extraction complete. GLCM data saved to {glcm_csv_path}")
