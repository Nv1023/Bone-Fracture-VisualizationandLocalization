# Bone Fracture Localization 🦴🔍

An AI-powered system for detecting and localizing bone fractures in X-ray images using deep learning and computer vision techniques.

## 🌟 Project Overview

This project implements an automated bone fracture detection system using advanced deep learning models. The system can analyze X-ray images and accurately identify and localize fractures, providing valuable assistance for medical professionals in radiological diagnosis.

## 🎯 Objective

Develop a robust machine learning solution that can:
- **Detect fractures** in X-ray images with high accuracy
- **Localize fracture regions** using bounding box predictions
- **Assist medical professionals** in rapid diagnosis
- **Reduce diagnostic errors** through AI-powered analysis

## 🛠️ Technology Stack

### Core Libraries
- **PyTorch** - Deep learning framework
- **Torchvision** - Computer vision models and utilities
- **OpenCV (cv2)** - Image processing operations
- **PIL (Pillow)** - Image manipulation and loading
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis

### Machine Learning & Vision
- **Faster R-CNN** - Object detection architecture
- **Albumentations** - Advanced image augmentation
- **Scikit-learn** - Model evaluation and data splitting

### Model Architecture
- **FastRCNNPredictor** - Custom prediction head for fracture detection
- **Pre-trained backbone** - Transfer learning from ImageNet weights

## 📊 Dataset

### MURA-v1.1 Dataset
This project uses the MURA (MUsculoskeletal RAdiographs) dataset for training and evaluation.

**Download Instructions:**
1. Visit the dataset page: [MURA-v1.1 Dataset](https://www.kaggle.com/datasets/cjinny/mura-v11/data?select=MURA-v1.1)
2. Create a Kaggle account if you don't have one
3. Accept the dataset terms and conditions
4. Download the complete MURA-v1.1 dataset
5. Extract to your project directory

### Dataset Structure
```
MURA-v1.1/
├── train/
│   ├── XR_ELBOW/
│   ├── XR_FINGER/
│   ├── XR_FOREARM/
│   ├── XR_HAND/
│   ├── XR_HUMERUS/
│   ├── XR_SHOULDER/
│   └── XR_WRIST/
└── valid/
    ├── XR_ELBOW/
    ├── XR_FINGER/
    ├── XR_FOREARM/
    ├── XR_HAND/
    ├── XR_HUMERUS/
    ├── XR_SHOULDER/
    └── XR_WRIST/
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 10GB+ storage space

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/bone-fracture-localization.git
cd bone-fracture-localization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python pillow numpy pandas scikit-learn
pip install albumentations
```

### Dataset Setup

```bash
# Create data directory
mkdir data
cd data

# Download and extract MURA dataset
# Place the downloaded MURA-v1.1 folder here
# Final structure: data/MURA-v1.1/train/ and data/MURA-v1.1/valid/
```

### Project Structure

```
bone-fracture-localization/
├── data/
│   └── MURA-v1.1/          # Downloaded dataset
├── models/                  # Saved model checkpoints
├── notebooks/              # Jupyter notebooks for analysis
├── src/
│   ├── dataset.py          # Custom dataset class
│   ├── model.py            # Model architecture
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation utilities
│   └── utils.py            # Helper functions
├── requirements.txt        # Project dependencies
└── main.py                 # Main execution script
```

## 🏃‍♂️ Quick Start

### 1. Data Preparation
```python
# Load and preprocess the MURA dataset
python src/prepare_data.py --data_path data/MURA-v1.1
```

### 2. Model Training
```python
# Train the fracture detection model
python src/train.py --epochs 50 --batch_size 16 --lr 0.001
```

### 3. Model Evaluation
```python
# Evaluate model performance
python src/evaluate.py --model_path models/best_model.pth
```

### 4. Inference
```python
# Run inference on new X-ray images
python src/predict.py --image_path path/to/xray.jpg --model_path models/best_model.pth
```

## 🔧 Key Features

### Image Preprocessing
- **Normalization** - Standardize pixel values
- **Resizing** - Consistent input dimensions
- **Augmentation** - Rotation, flipping, contrast adjustment
- **Noise reduction** - Image enhancement techniques

### Model Architecture
- **Faster R-CNN backbone** for object detection
- **Custom ROI head** for fracture classification
- **Transfer learning** from pre-trained weights
- **Multi-scale feature extraction**

### Data Augmentation
```python
# Albumentations pipeline for robust training
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.3),
    A.Normalize(),
    ToTensorV2()
])
```

## 📈 Model Performance

### Evaluation Metrics
- **mAP (mean Average Precision)** - Overall detection accuracy
- **Precision** - True positive rate
- **Recall** - Sensitivity to fractures
- **F1-Score** - Balanced precision-recall metric

### Expected Results
- Training accuracy: ~85-90%
- Validation accuracy: ~80-85%
- Inference time: ~0.5-1.0 seconds per image

## 🎨 Visualization

The system provides:
- **Bounding box visualization** around detected fractures
- **Confidence score display** for each detection
- **Heatmap overlays** showing attention regions
- **Comparative analysis** between original and processed images

## 💻 Hardware Requirements

### Minimum Requirements
- **CPU**: 4-core processor
- **RAM**: 8GB
- **Storage**: 15GB available space
- **GPU**: Optional but recommended

### Recommended Requirements
- **CPU**: 8-core processor
- **RAM**: 16GB+
- **Storage**: 25GB+ available space
- **GPU**: NVIDIA GTX 1060 or better with 6GB+ VRAM

## 🔄 Training Process

### Hyperparameters
```python
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 50
WEIGHT_DECAY = 1e-4
STEP_SIZE = 20
GAMMA = 0.1
```

### Training Pipeline
1. **Data Loading** - Custom dataset class with augmentations
2. **Model Initialization** - Faster R-CNN with custom predictor
3. **Training Loop** - Forward pass, loss calculation, backpropagation
4. **Validation** - Model evaluation on validation set
5. **Checkpointing** - Save best performing models

### API Integration
The model can be integrated into:
- **REST APIs** for web applications
- **Desktop applications** for radiologists
- **Mobile apps** for point-of-care diagnosis
- **DICOM systems** for hospital integration

## 🔄 Future Enhancements

- **3D X-ray analysis** for complex fractures
- **Real-time processing** optimization
- **Multi-class fracture classification** (hairline, compound, etc.)
- **Integration with PACS systems**
- **Mobile application development**
- **Uncertainty quantification** for predictions


## ⚠️ Disclaimer

This system is designed for research and educational purposes. It should not be used as the sole diagnostic tool for medical decisions. Always consult with qualified medical professionals for accurate diagnosis and treatment.

---

*Advancing medical imaging through AI-powered fracture detection* 🏥✨
