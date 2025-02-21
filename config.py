from pathlib import Path
import numpy as np

# Base path for dataset
BASE_PATH = Path("C:/Users/peter_idoko.VACFSS/Documents/VSCode/textureClassification")

# Output directory for CSV files
OUTPUT_DIR = BASE_PATH / "features"

# Image processing parameters
IMAGE_SIZE = (64, 64)  # Resize images for consistency

# GLCM Parameters
DISTANCES = [1, 2, 3, 4]
ANGLES = [0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi]  # 0°, 45°, 90°, 135°

# LBP Parameters
LBP_RADIUS = 3
LBP_POINTS = min(8 * LBP_RADIUS, 24)
LBP_METHOD = "uniform"

# Texture classes
TEXTURE_CLASSES = ["wood", "brick", "stone"]
