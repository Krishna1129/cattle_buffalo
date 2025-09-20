# 🐮 Cow-Buffalo Classifier with Breed Detection

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A sophisticated deep learning-based system for classifying cattle images into cows and buffaloes, along with breed identification. Perfect for farmers, veterinarians, and livestock specialists! 🚀

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technical Workflow](#technical-workflow)
- [Technologies Used](#technologies-used)
- [Model Performance](#model-performance)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Future Enhancements](#future-enhancements)

## 🎯 Project Overview
This advanced computer vision project employs deep learning to:
1. Classify cattle images into two categories: cows 🐄 and buffaloes 🦬
2. Identify specific breeds within each category
3. Provide quick and accurate results through a user-friendly interface

## ⭐ Key Features
- Dual-stage classification (animal type + breed)
- High accuracy classification model
- Real-time processing
- User-friendly interface
- Support for multiple image formats
- Detailed prediction confidence scores

## 🏗️ Architecture
The project follows a two-stage classification architecture:

### Stage 1: Cow/Buffalo Classification
```
Input Image → Image Preprocessing → CNN Model → Animal Type Prediction
```

### Stage 2: Breed Classification
```
Classified Image → Breed-Specific Model → Breed Prediction
```

The system uses a hierarchical classification approach where:
1. First model determines if the image contains a cow or buffalo
2. Based on the result, appropriate breed classifier is activated
3. Final results are combined and presented to the user

## 🔄 Technical Workflow
1. **Image Input**: System accepts images in various formats (JPG, PNG)
2. **Preprocessing**:
   - Image resizing to 224x224 pixels
   - Normalization
   - Data augmentation (during training)
3. **Primary Classification**:
   - Cow/Buffalo detection using CNN
   - Confidence score calculation
4. **Breed Classification**:
   - Breed-specific feature extraction
   - Breed prediction with confidence scores
5. **Result Generation**:
   - Combining both classification results
   - Presenting final output to user

## 🛠️ Technologies Used
### Core Libraries
- **PyTorch**: Deep learning framework for model development
  - Used for creating and training CNN models
  - Provides efficient GPU utilization
  - Handles data batching and transformation

- **Python**: Primary programming language
  - Version 3.7+ required
  - Handles core application logic
  - Manages data processing pipelines

### Supporting Libraries
- **Pillow (PIL)**:
  - Image processing and manipulation
  - Support for various image formats
  - Efficient image transformations

- **Streamlit**:
  - Web interface development
  - Interactive UI components
  - Real-time result display

- **Flask**: 
  - API development
  - Backend server management
  - Request handling

## 📊 Model Performance
### Cow/Buffalo Classification Model
- **Training Accuracy**: 98%
- **Test Accuracy**: 97%
- **Training Duration**: 50 epochs
- **Hardware Used**: NVIDIA RTX 3060 GPU

### Breed Classification Model
- **Training Accuracy**: 94%
- **Test Accuracy**: 89%
- **Training Duration**: 100 epochs
- **Hardware Used**: NVIDIA RTX 3060 GPU

## 💻 Installation & Setup
1. Clone the repository:
```bash
git clone https://github.com/Krishna1129/cattle_buffalo.git
cd cow-buffalo-classifier
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download pre-trained models:
- Place `best_cow_buffalo_none_classifier.pth` in the `models` directory
- Place `breed_classifier.pth` in the `models` directory

## 🎮 Usage Guide
1. Start the application:
```bash
python cattle_with_breed_classifier.py
```

2. Upload an image through the interface
3. Wait for the model to process the image
4. View results showing:
   - Animal type (Cow/Buffalo)
   - Breed identification
   - Confidence scores

## 🚀 Future Enhancements
- [ ] Add support for more cattle breeds
- [ ] Implement real-time video classification
- [ ] Develop mobile application
- [ ] Add disease detection capabilities
- [ ] Integrate with livestock management systems

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/Krishna1129/cattle_buffalo/issues).

## ⭐ Show your support
Give a ⭐️ if this project helped you!
