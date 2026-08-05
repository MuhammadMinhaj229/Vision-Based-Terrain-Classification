# Vision-Based Terrain Classification
A computer vision project exploring terrain classification using MobileNetV2 feature extraction and classical classifiers.

## Problem / Motivation
This project explores how visual patterns from ground surfaces (such as texture and structure) can be mapped to terrain categories. Identifying terrain types from images is a foundational step for applications like autonomous navigation and environmental analysis. This repository serves as a proof-of-concept for extracting features from pre-trained deep learning models and classifying them using traditional machine learning algorithms.

## Approach
This was a team project. My specific contributions were the **data cleaning/preprocessing pipeline** and the **Streamlit frontend**.

The overall system workflow is as follows:
1. **Feature Extraction**: Images are resized to 64x64, preprocessed, and passed through a pre-trained MobileNetV2 network (without the top classification layer). The features are pooled into a 1280-dimensional vector per image.
2. **Model Training**: A Tkinter-based desktop application (`Main.py`) provides an interface for admins to split the dataset, train models (Logistic Regression, Naive Bayes, Ridge, XGBoost), and view classification metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC curves).
3. **Inference**: A Streamlit dashboard (`app/streamlit_app.py`) allows users to upload an image and see the predicted terrain class along with confidence probabilities.
   - *Note:* The Streamlit app includes a hardcoded heuristic-based "smart color classifier" fallback if the trained models (`.pkl` files) are not found.

## Results
No pre-computed metrics or trained model binaries are included in this repository. Performance metrics (Accuracy, Precision, Recall, F1-score) and evaluation plots (Confusion Matrix, ROC curves) are dynamically generated during the training phase using the Tkinter application.

## Tech Stack
* **UI/Frontend**: Tkinter (Desktop App), Streamlit (Web Dashboard)
* **Computer Vision & Deep Learning**: TensorFlow / Keras (MobileNetV2), Pillow
* **Machine Learning**: Scikit-Learn (Logistic Regression, Naive Bayes, Ridge Classifier), XGBoost
* **Data Processing & Visualization**: NumPy, Pandas, Matplotlib, Seaborn, Plotly
* **Database**: TinyDB (Local user database)

## Project Structure
```text
Vision-Based-Terrain-Classification/
├── SIH_NN/                   # Dataset directory
│   ├── Desert/
│   ├── Forest/
│   ├── Mountain/
│   └── Plains/
├── app/
│   └── streamlit_app.py      # Streamlit web dashboard
├── Main.py                   # Tkinter application for training and evaluation
├── run.bat                   # Windows batch script for execution
├── requirements.txt          # Python dependencies
├── background.jpg            # UI asset for Tkinter app
└── README.md
```

## How to Run Locally

*Prerequisites*: You must manually install `tensorflow`, as it is imported in the code but currently missing from `requirements.txt`.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MuhammadMinhaj229/Vision-Based-Terrain-Classification.git
   cd Vision-Based-Terrain-Classification
   ```

2. **Set up a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install tensorflow      # Required for MobileNetV2 feature extraction
   ```

4. **Run the training application**:
   ```bash
   python Main.py
   ```
   *Note*: Use the Tkinter UI to extract features and train models. This will generate the necessary `.pkl` files in a `model/` directory.

5. **Run the Streamlit dashboard**:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Limitations & Future Work
- **Missing Dependency**: `tensorflow` must be manually added to `requirements.txt` or installed separately.
- **Heuristic Fallback Mismatch**: If the Streamlit app cannot find trained model `.pkl` files, it falls back to a hardcoded color-based heuristic. The categories in this fallback (Grass, Sandy, Rocky, Marshland) do not match the actual dataset classes (Desert, Forest, Mountain, Plains).
- **Deployment**: The system relies on local `.pkl` files and a local JSON database (`TinyDB`), making it unsuitable for stateless cloud deployment without modifications.
- **Image Resizing**: The images are severely downscaled to 64x64 before feature extraction, which may discard important textural information necessary for accurate terrain classification.

## Author
* **Mohammed Minhaj Mahmood** (Data cleaning/preprocessing pipeline & Streamlit frontend)
* Team project members (Feature extraction & model training integration)
