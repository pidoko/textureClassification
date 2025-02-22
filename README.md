https://huggingface.co/spaces/pidoko/textureClassification

# GLCM & LBP Texture Classification
## Machine Learning Pipeline for Texture Analysis using GLCM & LBP Features
### Overview
This project implements Gray Level Co-occurrence Matrix (GLCM) and Local Binary Patterns (LBP) for texture classification using machine learning models. The system extracts texture features from images, trains classifiers (SVM, Random Forest, k-NN, Logistic Regression), and provides a Gradio-based UI for image classification. 

Selecting the appropriate pictures for the dataset proved to be the most important part of this project because there needed to be little to no noise, and the pictures needed to be mutually exclusive such that the model could effectively learn the difference between wood, brick and stone.

Also, I made the decision to work in grayscale because I think colour adds more noise than help with classification in this project. This is because wood, brick and stone can all be any colour based on how they are used in the real world.

GLCM was selected because it captures texture spatial patterns well, works well with grayscale images and provides meaningful statistical metrics which are useful for research. GLCM captures the spatial relationships between pixel intensities, and computes a matrix that describes how often pairs of pixel intensities appear at set distances and directions. From this matrix, statistical features like contrast, energy, homogeneity, and correlation are extracted.  

LBP was selected because it captures local texture information in a simple yet effective way. LBP encodes each pixel by comparing it to its neighbouring pixel, if a neighbouring pixel is brighter, it is assigned a 1, otherwise it's a 0. If the pixel is compared against the eight neighbouring pixels in a 3x3 window, then the eight bits of 1s and 0s from the comparison form an 8-bit binary number whih is converted to a decimal value. Then, a histogram is built with the frequency of each LBP decimal value after the entire image is processed. LBP is computationally efficient, robust to texture orientation changes, and works well for fine texture details like wood grain.

SVM was selected because it works well with high-dimensional data, can handle both linear and non-linear classification, and generalizes effectively. SVM finds the best possible boundary (hyperplane) that separates different classes by maximizing the margin between them. If the data is not clearly separable in its original form, SVM uses mathematical functions called kernels (like RBF or polynomial) to transform it into a higher-dimensional space where separation is easier. Because it focuses only on important data points (support vectors), SVM is resistant to noise and is useful for classifying textures like wood, brick, and stone.

kNN was selected because it is a simple, easy-to-understand method that does not need training and works well with complex decision boundaries. kNN classifies a new data point by checking which labeled points in the dataset are closest to it and assigns the most common class among its k-nearest neighbors. The choice of k affects how the model behaves—smaller values make it more sensitive to details, while larger values smooth out classifications. Since texture differences can be subtle, kNN is useful for recognizing patterns that do not follow a simple rule.

Random Forest was selected because it is a powerful ensemble method that reduces overfitting, handles complex datasets, and provides insights into important features. It builds multiple decision trees using different parts of the data and averages their predictions, making it more accurate and less likely to be misled by noise. Because it captures both simple and complex patterns, Random Forest is effective for classifying textures with different surface details, like distinguishing between the roughness of stone and the grain of wood.

Logistic Regression was selected because it is a simple and efficient model that serves as a good starting point, is easy to interpret, and provides probability scores for predictions. It works by creating a mathematical function that separates classes using a straight-line boundary and applies a sigmoid function to map the results to probabilities between 0 and 1. Logistic Regression is useful when the data is linearly separable and is often used as a baseline model before testing more advanced approaches. Its ability to provide confidence scores helps in cases where textures might be similar, allowing for better decision-making.

### Features
#### Feature Extraction
GLCM: Captures contrast, correlation, energy, and homogeneity at different distances & angles.

LBP: Captures micro-patterns via a histogram-based texture representation.

#### Machine Learning Pipeline
Trains and optimizes SVM, Random Forest, k-NN, and Logistic Regression.

Uses GridSearchCV for hyperparameter tuning because it tests each parameter combination on different subsets of the data, leading to more reliable results.

Implements feature scaling using StandardScaler because many machine learning algorithms, such as SVM and kNN, rely on distance-based calculations, and if features have different scales, the model might give more importance to features with larger values.

#### Visualization & Evaluation
Generates confusion matrices for model performance analysis because they help analyze where the model is making errors and whether certain classes are being misclassified more often than others.

Saves trained models for future use.

#### Interactive UI with Gradio
Allows users to upload an image and classify it using GLCM or LBP features.

### Project Structure
textureClassification/ │── features/ # Extracted feature CSVs & confusion matrix plots │── models/ # Saved ML models │── sample_images/ # Example textures for testing │── config.py # Config file (paths, parameters) │── feature_extraction.py # Feature extraction pipeline (GLCM & LBP) │── app.py # Model training, evaluation & Gradio UI │── requirements.txt # Python dependencies │── README.md # Project documentation

### Setup & Installation
#### Install Dependencies
```
pip install -r requirements.txt
```

#### Dataset Structure
dataset/ ├── wood/ │ ├── image1.jpg │ ├── image2.png ├── brick/ │ ├── image1.jpg │ ├── image2.png ├── stone/ │ ├── image1.jpg │ ├── image2.png

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
| Model  | Accuracy (GLCM) | Accuracy (LBP) |
|--------|---------------|---------------|
| SVM    | 92.5%        | 90.1%        |
| RFst   | 89.7%        | 87.3%        |
| k-NN   | 85.4%        | 83.9%        |
| LRgn   | 81.2%        | 80.5%        |

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
