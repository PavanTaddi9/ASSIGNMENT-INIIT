# FACE EMOTION RECOGNITION

**INTRA IITM HACKATHON(2024) - 8/100

## Project Overview

Emotion recognition from facial images is a challenging task, especially when the dataset suffers from class imbalance issues (i.e., some emotions appear far more frequently than others). To tackle this, we used a **ResNet34** architecture combined with an **autoencoder** for class imbalance Laerning. This solution improves the model's performance across all classes by balancing learning class specific features

## Model Architecture

1. **ResNet34 Backbone**:
    - The core of the model is based on the ResNet34 architecture, known for its performance in image classification tasks. This backbone provides robust feature extraction that enables effective emotion recognition.

2. **Autoencoder for Class Imbalance**:
    - To handle class imbalance, an autoencoder was integrated into the training process. The autoencoder learns to balance representations  by reconstructing data points from underrepresented classes more accurately, which helps the ResNet34 model to get class specific       features.

L. Wang, L. Zhang, X. Qi and Z. Yi, "Deep Attention-Based Imbalanced Image Classification," in IEEE Transactions on Neural Networks and      Learning Systems, vol. 33, no. 8, pp. 3320-3330, Aug. 2022, doi: 10.1109/TNNLS.2021.3051721.







## Dependencies

- Python 3.8+
- PyTorch
- NumPy
- OpenCV




