# Program implementing GLCM to extract contrast, correlation, energy and homogeneity,
# and LBP to create a histogram to capture micro patterns.

import os
import numpy as np
import skimage.feature as sf
import cv2

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