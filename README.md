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

## Key Features

- **High Accuracy Emotion Recognition**: Predicts emotions from facial expressions using transfer learning and deep feature extraction.
- **Class Imbalance Learning**: The autoencoder component helps the model deal with class imbalance, ensuring fairer representation of all emotion classes.
- **Efficient and Scalable**: The model is designed to be efficient, making it suitable for real-time applications or further experimentation.


## Usage

To use this model for your own face emotion recognition tasks, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/face-emotion-recognition.git
    cd face-emotion-recognition
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Training the Model**:
    Run the training script to train the model on your dataset.
    ```bash
    python train.py --dataset /path/to/your/dataset
    ```

4. **Evaluating the Model**:
    Evaluate the model's performance using the evaluation script.
    ```bash
    python evaluate.py --model path/to/saved/model
    ```

## Dependencies

- Python 3.8+
- PyTorch
- NumPy
- OpenCV
- Matplotlib
- [Any other dependencies relevant to your codebase]

## Contributing

Contributions are welcome! Please submit a pull request if you would like to improve the model or add new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Thank you for checking out the Face Emotion Recognition project! If you have any questions or feedback, please feel free to open an issue or contact us.
