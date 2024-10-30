# FACE EMOTION RECOGNITION

**INTRA IITM HACKATHON - 8th Place**

Welcome to the **Face Emotion Recognition** repository! This project was developed as part of the Intra IITM Hackathon, where it achieved 6th place. The goal of this project is to build a model that recognizes emotions from facial images accurately, even when dealing with class imbalances in the dataset.

## Project Overview

Emotion recognition from facial images is a challenging task, especially when the dataset suffers from class imbalance issues (i.e., some emotions appear far more frequently than others). To tackle this, we used a **ResNet34** architecture combined with an **autoencoder** for class imbalance Laerning. This solution improves the model's performance across all classes by balancing learning for underrepresented emotions.Explored diffferent loss functions

## Model Architecture

1. **ResNet34 Backbone**:
    - The core of the model is based on the ResNet34 architecture, known for its performance in image classification tasks. This backbone provides robust feature extraction that enables effective emotion recognition.

2. **Autoencoder for Class Imbalance**:
    - To handle class imbalance, an autoencoder was integrated into the training process. The autoencoder learns to balance representations by reconstructing data points from underrepresented classes more accurately, which helps the ResNet34 model to get class specific features.






## Dependencies

- Python 3.8+
- PyTorch
- NumPy
- OpenCV
- VisionTransformers



