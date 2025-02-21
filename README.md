# GLCM & LBP Texture Classification
## Machine Learning Pipeline for Texture Analysis using GLCM & LBP Features
### Overview
This project implements Gray Level Co-occurrence Matrix (GLCM) and Local Binary Patterns (LBP) for texture classification using machine learning models. The system extracts texture features from images, trains classifiers (SVM, Random Forest, k-NN, Logistic Regression), and provides a Gradio-based UI for image classification.

### Features
#### Feature Extraction
GLCM: Captures contrast, correlation, energy, and homogeneity at different distances & angles.
LBP: Captures micro-patterns via a histogram-based texture representation.

#### Machine Learning Pipeline
Trains and optimizes SVM, Random Forest, k-NN, and Logistic Regression.
Uses GridSearchCV for hyperparameter tuning.
Implements feature scaling using StandardScaler.

#### Visualization & Evaluation
Generates confusion matrices for model performance analysis.
Saves trained models for future use.

#### Interactive UI with Gradio
Allows users to upload an image and classify it using GLCM or LBP features.

### Project Structure
📁 textureClassification/
│── 📁 features/                   # Extracted feature CSVs & confusion matrix plots
│── 📁 models/                     # Saved ML models
│── 📁 sample_images/               # Example textures for testing
│── 📜 config.py                    # Config file (paths, parameters)
│── 📜 feature_extraction.py         # Feature extraction pipeline (GLCM & LBP)
│── 📜 app.py                        # Model training, evaluation & Gradio UI
│── 📜 requirements.txt               # Python dependencies
│── 📜 README.md                      # Project documentation

### Setup & Installation
#### Install Dependencies
```
pip install -r requirements.txt
```

#### Dataset Structure
📁 dataset/
    ├── 📁 wood/
    │   ├── image1.jpg
    │   ├── image2.png
    ├── 📁 brick/
    │   ├── image1.jpg
    │   ├── image2.png
    ├── 📁 stone/
    │   ├── image1.jpg
    │   ├── image2.png

### Feature Extraction
#### To extract GLCM and LBP features from images, run:
```
python feature_extraction.py
```
#### Output:
Saves GLCM features to features/texture_features_glcm.csv
Saves LBP features to features/texture_features_lbp.csv

### Train & Evaluate Models
#### To train ML classifiers on extracted features, run:
```
python app.py
```
#### This will:
Load GLCM & LBP datasets
Train SVM, Random Forest, k-NN, and Logistic Regression
Generate confusion matrices for evaluation
Launch an interactive classification UI

### Gradio Interface Usage: 
Upload an image.
Select GLCM or LBP for feature extraction.
Choose a classifier (e.g., SVM, Random Forest).
Get the predicted texture class.

### Model Performance
#### Each classifier is evaluated using a confusion matrix:
Model	(GLCM)	(LBP) Accuracy
SVM	    92.5%	90.1%
RFst	89.7%	87.3%
k-NN	85.4%	83.9%
LRgn	81.2%	80.5%

### Configuration
#### Modify config.py to adjust:

BASE_PATH = "dataset/"
OUTPUT_DIR = "features/"

IMAGE_SIZE = (64, 64)

DISTANCES = [1, 2, 3, 4]
ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]

LBP_RADIUS = 3
LBP_POINTS = min(8 * LBP_RADIUS, 24)
LBP_METHOD = "uniform"

TEXTURE_CLASSES = ["wood", "brick", "stone"]

### Future Improvements
#### Deploy as a Web API (FastAPI/Flask)
#### Add More Textures for Classification
#### Optimize Feature Engineering for Better Performance

### License
#### This project is licensed under the MIT License.

### Author
Peter Chibuikem Idoko
VACFSS - IT Coordinator | AI/ML Engineer

### Need Help?
For questions or collaboration: Email: pidoko@hotmail.com
GitHub: https://github.com/pidoko/textureClassification
